"""后台维护调度测试：持久化时间戳、任务到期与失败重试。"""

import asyncio
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_chatlearning_test"


@pytest.fixture(scope="module")
def plugin_module():
    """将插件目录临时注册为包，以便加载带相对导入的 main.py。"""
    pkg = types.ModuleType(PACKAGE)
    pkg.__path__ = [str(PLUGIN_ROOT)]
    sys.modules[PACKAGE] = pkg
    spec = importlib.util.spec_from_file_location(
        f"{PACKAGE}.main", PLUGIN_ROOT / "main.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    for name in [n for n in sys.modules if n == PACKAGE or n.startswith(PACKAGE + ".")]:
        sys.modules.pop(name, None)


class FakeWordStock:
    def __init__(self, freed: int | None = 0):
        self.freed = freed
        self.calls: list[tuple] = []

    async def cleanup_low_freq(self, days=30, min_freq=2):
        self.calls.append(("cleanup_low_freq", days, min_freq))
        return 1

    async def build_index(self):
        self.calls.append(("build_index",))

    async def cleanup_versions(self, retention_hours=2):
        self.calls.append(("cleanup_versions", retention_hours))
        return self.freed


def make_plugin(plugin_module, tmp_path, wordstock):
    plugin = plugin_module.ChatLearningPlugin.__new__(plugin_module.ChatLearningPlugin)
    plugin._cfg = {
        "auto_schedule_enable": False,
        "auto_cleanup_days": 30,
        "version_cleanup_interval_hours": 6,
        "version_retention_hours": 2,
    }
    plugin.wordstock = wordstock
    plugin._data_root = str(tmp_path)
    plugin._maintenance_state_path = str(tmp_path / "maintenance_state.json")
    return plugin


def write_state(path: Path, last_cleanup: float, last_version_cleanup: float) -> None:
    path.write_text(
        json.dumps(
            {
                "last_cleanup": last_cleanup,
                "last_version_cleanup": last_version_cleanup,
            }
        ),
        encoding="utf-8",
    )


async def run_one_iteration(plugin, monkeypatch, iterations: int = 1) -> None:
    """让前 N 轮 sleep 立即返回，下一轮抛 CancelledError 退出循环。"""
    sleeps = {"count": 0}

    async def fake_sleep(_seconds):
        sleeps["count"] += 1
        if sleeps["count"] > iterations:
            raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    with pytest.raises(asyncio.CancelledError):
        await plugin._periodic_maintenance()


@pytest.mark.asyncio
async def test_missing_state_runs_version_cleanup(plugin_module, tmp_path, monkeypatch):
    stock = FakeWordStock(freed=0)
    plugin = make_plugin(plugin_module, tmp_path, stock)

    await run_one_iteration(plugin, monkeypatch)

    assert stock.calls == [("cleanup_versions", 2)]
    state = json.loads((tmp_path / "maintenance_state.json").read_text("utf-8"))
    assert state["last_version_cleanup"] > 0
    # 首次启动不立即执行 24h 低频清理
    assert state["last_cleanup"] > 0


@pytest.mark.asyncio
async def test_fresh_state_skips_all_cleanup(plugin_module, tmp_path, monkeypatch):
    now = time.time()
    write_state(tmp_path / "maintenance_state.json", now, now)
    stock = FakeWordStock()
    plugin = make_plugin(plugin_module, tmp_path, stock)

    await run_one_iteration(plugin, monkeypatch)

    assert stock.calls == []


@pytest.mark.asyncio
async def test_due_low_freq_cleanup_saves_state(plugin_module, tmp_path, monkeypatch):
    now = time.time()
    old = now - 25 * 3600
    write_state(tmp_path / "maintenance_state.json", old, now)
    stock = FakeWordStock()
    plugin = make_plugin(plugin_module, tmp_path, stock)

    await run_one_iteration(plugin, monkeypatch)

    assert stock.calls == [("cleanup_low_freq", 30, 2), ("build_index",)]
    state = json.loads((tmp_path / "maintenance_state.json").read_text("utf-8"))
    assert state["last_cleanup"] > old
    assert state["last_version_cleanup"] == now


@pytest.mark.asyncio
async def test_failed_version_cleanup_keeps_old_timestamp(
    plugin_module, tmp_path, monkeypatch
):
    now = time.time()
    old = now - 10 * 3600
    write_state(tmp_path / "maintenance_state.json", now, old)
    stock = FakeWordStock(freed=None)
    plugin = make_plugin(plugin_module, tmp_path, stock)

    await run_one_iteration(plugin, monkeypatch)

    assert stock.calls == [("cleanup_versions", 2)]
    state = json.loads((tmp_path / "maintenance_state.json").read_text("utf-8"))
    assert state["last_version_cleanup"] == old


@pytest.mark.asyncio
async def test_failed_version_cleanup_backs_off(plugin_module, tmp_path, monkeypatch):
    now = time.time()
    old = now - 10 * 3600
    write_state(tmp_path / "maintenance_state.json", now, old)
    stock = FakeWordStock(freed=None)
    plugin = make_plugin(plugin_module, tmp_path, stock)

    # Two iterations: the first fails, the second must wait for the backoff.
    await run_one_iteration(plugin, monkeypatch, iterations=2)

    assert stock.calls == [("cleanup_versions", 2)]


@pytest.mark.asyncio
async def test_zero_freed_still_counts_as_success(plugin_module, tmp_path, monkeypatch):
    now = time.time()
    old = now - 10 * 3600
    write_state(tmp_path / "maintenance_state.json", now, old)
    stock = FakeWordStock(freed=0)
    plugin = make_plugin(plugin_module, tmp_path, stock)

    await run_one_iteration(plugin, monkeypatch)

    state = json.loads((tmp_path / "maintenance_state.json").read_text("utf-8"))
    assert state["last_version_cleanup"] > old


@pytest.mark.asyncio
async def test_disabled_version_interval_does_nothing(
    plugin_module, tmp_path, monkeypatch
):
    now = time.time()
    write_state(tmp_path / "maintenance_state.json", now, 0)
    stock = FakeWordStock()
    plugin = make_plugin(plugin_module, tmp_path, stock)
    plugin._cfg["version_cleanup_interval_hours"] = 0

    await run_one_iteration(plugin, monkeypatch)

    assert stock.calls == []
