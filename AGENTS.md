# AGENTS.md

## Architecture

```
astrbot_plugin_chatlearning/
├── main.py                   # Plugin entry point (ChatLearningPlugin Star)
├── filter/
│   └── filter_engine.py      # FilterEngine: text cleaning, sensitive words, blocked users
├── learning/
│   └── collector.py          # Collector: time-window Q&A chain builder
├── reply/
│   └── responder.py          # Responder: vector matching, LLM semantic check, weighted random pick
├── storage/
│   └── wordstock.py          # WordStock: LanceDB vector storage (question-answer pairs)
├── migrate.py                # SQLite → LanceDB data migration script
├── metadata.yaml             # Plugin metadata (name, version, supported platforms)
└── tests/
    └── test_wordstock.py     # WordStock unit tests
```

### Data Flow

```
Group Message
    │
    ▼
FilterEngine.clean_at_mentions()   ← strip @user(QQ) → @user
    │
    ▼
FilterEngine.should_record()       ← sensitive words / blocked users
    │
    ├── [learn] Collector.feed()   ← time-window Q&A chaining
    │        │
    │        ▼
    │   WordStock.add_question()   ← store with embedding vector
    │
    └── [reply] Responder.try_reply()
             │
             ├── WordStock.search_similar()   ← vector cosine search
             ├── LLM semantic check (optional)
             └── weighted random pick → plain_result
```

### Key Modules

| Module | Class | Role |
|--------|-------|------|
| `main.py` | `ChatLearningPlugin` | AstrBot Star, registers commands & event handler |
| `filter/filter_engine.py` | `FilterEngine` | Text sanitisation, sensitive word & user blocking |
| `learning/collector.py` | `Collector` | Builds Q&A pairs from message streams using time windows |
| `reply/responder.py` | `Responder` | Matches queries to stored Q&A via vectors, with LLM fallback |
| `storage/wordstock.py` | `WordStock` | LanceDB-backed vector store for question-answer pairs |

## Python Runtime

- Use `uv run python` (or `uv run <script>.py`) to execute Python scripts.
- Use `uv run pytest` to run tests.

## Type Checking

- Use `ty` for static type checking (standalone binary).

## Formatting

- Use `ruff format` to format code.
- Use `ruff check` to lint.

## Commit Convention

- Commit messages **must** be written in English.
- Follow conventional commits: `feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:` etc.
