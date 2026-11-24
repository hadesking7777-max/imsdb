# IMSDb Local Screenplay Database and Offline Chatbot

A fully local system with two parts:

1. **The archive (Phase 1).** A permission-aware crawler discovers, downloads, preserves and parses screenplays
   from [The Internet Movie Script Database (IMSDb)](https://imsdb.com/) into a **local MongoDB** database.
2. **ScriptBot (Phase 2).** An offline chatbot answers questions about that archive in **English or Chinese**,
   using a local language model. It runs **on a GPU by default and falls back to the CPU** when no GPU is usable.

No containers, cloud services or remote AI APIs are used. The crawler never calls an AI model. After a one-time
download of Python packages and model files, everything works without an internet connection.

> **Source permissions.** IMSDb hosts scripts under a fair-use claim and grants no licence. The system ships
> at the `sample` level (at most 10 records). On 2026-11-23 the project owner recorded `bulk` authorization and
> the full archive was collected (see [docs/SOURCE_PERMISSIONS.md](docs/SOURCE_PERMISSIONS.md)).

**This README is the developer's entry point.** It explains how the project is put together, how to run it,
and how to change each part safely. Deeper references:

| Document | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Crawl pipeline, MongoDB collections and fields, record statuses, parsing stages |
| [docs/CHATBOT.md](docs/CHATBOT.md) | ScriptBot design, models, speed measurements, limitations |
| [docs/SOURCE_PERMISSIONS.md](docs/SOURCE_PERMISSIONS.md) | The source permission assessment and its history |
| [docs/SAMPLE_RECORDS.md](docs/SAMPLE_RECORDS.md) | One real example document per collection |
| `runtime/README.txt` | Why an app-local `msvcp140.dll` is shipped for Windows |

## Contents

1. [Quick start](#1-quick-start)
2. [How the system fits together](#2-how-the-system-fits-together)
3. [Requirements and installation](#3-requirements-and-installation)
4. [Configuration](#4-configuration)
5. [Running the archive (Phase 1)](#5-running-the-archive-phase-1)
6. [Running ScriptBot (Phase 2)](#6-running-scriptbot-phase-2)
7. [GPU by default, CPU fallback](#7-gpu-by-default-cpu-fallback)
8. [Repository layout](#8-repository-layout)
9. [Module guide](#9-module-guide)
10. [How to make common changes](#10-how-to-make-common-changes)
11. [Testing](#11-testing)
12. [Coding conventions and project rules](#12-coding-conventions-and-project-rules)
13. [Data layout, backup and recovery](#13-data-layout-backup-and-recovery)
14. [Troubleshooting](#14-troubleshooting)
15. [Known limitations](#15-known-limitations)
16. [History: the first live sample run](#16-history-the-first-live-sample-run)

---

## 1. Quick start

**What this repository contains.** All source code, tests, documentation, configuration templates, the
app-local Windows C++ runtime (`runtime/win64/msvcp140.dll`) and the Vulkan GPU build of the chatbot runtime
(`runtime/gpu/vulkan`). It does **not** contain the collected screenplays (raw pages, database, backups,
exports): they are copyrighted by their owners and are not redistributed. It also does not contain the model
files (5.7 GB, larger than GitHub allows); `models fetch` downloads them from Hugging Face at pinned, checksum-
verified revisions. Starting from a fresh clone therefore means: install, then collect your own archive (with
permission from the source, section 5) or restore a backup you already own (section 13).

On a machine that already has Python 3.12+ and a local MongoDB (section 3):

```bash
python scripts/setup.py --chatbot          # .venv + all packages (CPU runtime + Vulkan GPU runtime)
.venv\Scripts\activate                     # Windows   (Linux/macOS: source .venv/bin/activate)
python scripts/verify_environment.py       # checks Python, packages, MongoDB, models, GPU/CPU
python scripts/init_database.py            # collections, validators, indexes, permission record

# the archive
python -m src.main inspect-source          # read robots.txt and policy pages
python -m src.main crawl --limit 10        # collect up to 10 records (sample level)
python -m src.main stats                   # what is in the database

# the chatbot
python -m src.main models fetch            # one-time model download (~5.7 GB)
python -m src.main chat-index              # build the passage search index (~2 min)
python -m src.main device                  # shows whether the GPU or the CPU will be used, and why
python -m src.main serve --open            # chat page at http://localhost:8765
```

On the development computer MongoDB is a portable install: start it first with
`python scripts/local_mongodb.py start`.

## 2. How the system fits together

```
                         PHASE 1: THE ARCHIVE                               PHASE 2: SCRIPTBOT
 imsdb.com                                                    question (EN/中文) from browser / terminal
    │  HTTPX, 1 request per 5 s, robots.txt,                                │
    │  permission gate, retries, size cap                                  ▼
    ▼                                                          src/chatbot/engine.py
 src/crawler ──► data/raw (original HTML, SHA-256) ◄── reparse      1 plan     (Qwen3-8B, JSON grammar)
    │                                                               2 look up  (catalogue + MongoDB facts)
    ▼                                                               3 search   (MongoDB text index)
 src/parsers  (HTML → exact text → scenes, dialogue, characters)   4 re-rank  (Qwen3-Embedding-0.6B)
    │                                                               5 answer   (Qwen3-8B, streamed, cited)
    ▼                                                               6 verify   (quotes checked word for word)
 src/services/ingestion.py ──► MongoDB imsdb_local ──────────────────────┘
                               movies, scripts, scenes, dialogues,   chat_chunks (search passages)
                               characters, crawl_* , raw_artifacts    built by `chat-index`
```

* **Phase 1** is a queue-driven crawler. Every fetched page is stored unchanged in `data/raw/` (content-addressed
  by SHA-256), then parsed into MongoDB. Parsing can be re-run from the raw archive at any time without network
  access (`reparse`). Nothing from the source is silently discarded: uncertain parses are flagged, not dropped.
* **Phase 2** reads only from MongoDB and the local model files. It never writes to the Phase 1 collections;
  its own data is the `chat_chunks` / `chat_chunk_state` collections, rebuilt incrementally by `chat-index`.
* Both phases share `src/config.py` (settings), `src/database/connection.py` (local-only MongoDB access) and the
  command-line entry point `src/main.py`.

## 3. Requirements and installation

| Component | Version | Notes |
|---|---|---|
| Python | 3.12+ | CPython from python.org, a package manager, or pyenv |
| MongoDB Community Server | 7.0+ (tested with **9.0.4**) | Runs locally and listens on `127.0.0.1` only |
| MongoDB Database Tools | 100.x (optional) | `mongodump`/`mongorestore` for backups; a pure-Python BSON fallback is built in |
| Disk | ~2 GB for the archive, +6 GB for the chatbot models | |
| Memory (chatbot) | 16 GB RAM recommended | About 12 GB used when running on the CPU |
| GPU (chatbot, optional) | Any GPU with a Vulkan driver; ≥ 8 GB video memory for the whole model | Smaller GPUs get part of the model; no GPU = CPU |

### 3.1 Python

* **Windows:** `winget install Python.Python.3.12`, or the python.org installer (tick "Add to PATH").
* **macOS:** `brew install python@3.12`
* **Linux (Debian/Ubuntu):** `sudo apt install python3.12 python3.12-venv`

### 3.2 MongoDB Community Server (local)

**Windows, as a service (recommended):**
```powershell
winget install MongoDB.Server          # installs the "MongoDB" Windows service (needs admin / UAC)
winget install MongoDB.DatabaseTools   # mongodump / mongorestore
net start MongoDB                      # net stop MongoDB
```
The default `mongod.cfg` binds to `127.0.0.1`.

**Windows, portable (no admin rights; this is how the development machine is set up):**
1. Download the ZIP archives for the server and Database Tools from mongodb.com and check their SHA-256 checksums.
2. Extract them, e.g. to `D:\tools\mongodb\server-9.0.4` and `D:\tools\mongodb\database-tools-100.20.1`.
3. Create `mongod.cfg`:
   ```yaml
   storage: { dbPath: D:\tools\mongodb\data\db }
   systemLog: { destination: file, path: D:\tools\mongodb\log\mongod.log, logAppend: true }
   net: { bindIp: 127.0.0.1, port: 27017 }
   ```
4. Put `MONGOD_PATH`, `MONGOD_CONFIG` and `MONGODB_TOOLS_DIR` in `.env`, then run
   `python scripts/local_mongodb.py start` (also `stop` and `status`).
   If `mongod.exe` crashes immediately (event log: faulting module `MSVCP140.dll`), the Visual C++ runtime is
   older than 14.44: run `vc_redist.x64.exe` from the server's `bin` folder as Administrator.

**macOS:** `brew tap mongodb/brew && brew install mongodb-community mongodb-database-tools && brew services start mongodb-community`

**Linux (Ubuntu/Debian):** add MongoDB's `repo.mongodb.org` apt source (MongoDB's install guide), then
`sudo apt install -y mongodb-org mongodb-database-tools && sudo systemctl enable --now mongod`.
`/etc/mongod.conf` binds to `127.0.0.1` by default; keep it that way.

**Optional authentication.** In `mongosh`:
```javascript
use admin
db.createUser({user: "imsdb", pwd: passwordPrompt(), roles: [{role: "readWrite", db: "imsdb_local"},
               {role: "backup", db: "admin"}, {role: "restore", db: "admin"}]})
```
Enable `security.authorization: enabled`, restart, and set `MONGODB_USERNAME`, `MONGODB_PASSWORD` and
`MONGODB_AUTH_SOURCE=admin` in `.env`. Credentials are never hard-coded or logged; backup tools receive the
password through a temporary config file, not the command line.

### 3.3 Project setup

```bash
python scripts/setup.py                       # archive only: .venv + requirements.txt + .env
python scripts/setup.py --chatbot             # + chatbot runtime: CPU build + Vulkan GPU build (default)
python scripts/setup.py --chatbot --gpu-backend cuda   # + the CUDA build for NVIDIA GPUs (~510 MB)
python scripts/setup.py --chatbot --gpu-backend none   # CPU build only
```

| What `setup.py` installs | Where | From |
|---|---|---|
| Archive packages (httpx, beautifulsoup4, lxml, pydantic, pymongo, python-dotenv, pytest) | `.venv` | `requirements.txt` |
| Chatbot runtime, CPU build (llama-cpp-python 0.3.36 + numpy) | `.venv` | `requirements-chatbot.txt` |
| Chatbot runtime, Vulkan GPU build (42 MB, SHA-256 pinned) | `runtime/gpu/vulkan` | `requirements-gpu-vulkan.txt` |
| Chatbot runtime, CUDA GPU build (optional, 510 MB, SHA-256 pinned) | `runtime/gpu/cuda` | `requirements-gpu-cuda.txt` |

Everything is a prebuilt wheel: no compiler, CUDA toolkit or other system software is installed. A GPU build
that fails to install does not stop the setup; the chatbot then runs on the CPU. Afterwards:

```bash
python scripts/verify_environment.py   # Python, packages, folders, disk, MongoDB (local + reachable), tools,
                                       # chatbot runtime, compute device, models, search index
python scripts/init_database.py        # collections, validators, indexes, default permission record
```

## 4. Configuration

Settings come from environment variables or `.env` (copied from `.env.example` by `setup.py`), are validated by
`src/config.py` (`Settings`), and are immutable at run time. Invalid values stop the program with exit code 2.

**Archive and database**

| Variable | Default | Meaning |
|---|---|---|
| `MONGODB_URI` | `mongodb://127.0.0.1:27017/` | **Local only.** Remote hosts and `mongodb+srv://` are rejected |
| `MONGODB_DB` | `imsdb_local` | Database name |
| `MONGODB_USERNAME` / `MONGODB_PASSWORD` / `MONGODB_AUTH_SOURCE` | empty / empty / `admin` | Optional authentication |
| `MONGODB_TOOLS_DIR` | empty | Folder with `mongodump`/`mongorestore` if they are not on `PATH` |
| `MONGOD_PATH`, `MONGOD_CONFIG` | empty | Portable MongoDB, used by `scripts/local_mongodb.py` |
| `IMSDB_DATA_DIR`, `IMSDB_LOG_DIR` | `data`, `logs` | Local storage locations |
| `IMSDB_LOG_LEVEL` | `INFO` | Console and file log level |
| `IMSDB_USER_AGENT` | `imsdb-local-research/1.0 (…)` | Honest crawler identity; never contains personal data |
| `IMSDB_MIN_REQUEST_INTERVAL` | `5.0` | Seconds between requests to the same host (cannot be set below 1.0) |
| `IMSDB_CONCURRENCY` | `2` | Worker threads (the per-host interval still applies across all of them) |
| `IMSDB_MAX_RETRIES`, `IMSDB_BACKOFF_BASE`, `IMSDB_BACKOFF_MAX` | `4`, `2.0`, `120` | HTTP retry policy |
| `IMSDB_MAX_RETRY_AFTER` | `600` | Longer `Retry-After` values reschedule the URL instead of blocking a worker |
| `IMSDB_MAX_QUEUE_ATTEMPTS` | `5` | Queue-level attempts before a URL is marked `failed` |
| `IMSDB_REQUEST_TIMEOUT`, `IMSDB_CONNECT_TIMEOUT` | `30`, `10` | Seconds |
| `IMSDB_LEASE_SECONDS` | `1800` | How long a worker's claim on a queue item lasts before it is recovered |
| `IMSDB_MAX_RESPONSE_BYTES` | `20000000` | Responses larger than this are refused |

**Chatbot**

| Variable | Default | Meaning |
|---|---|---|
| `IMSDB_LLM_DEVICE` | `auto` | **`auto` = GPU if one is usable, otherwise CPU**; `gpu` = GPU or fail; `cpu` = never try the GPU |
| `IMSDB_LLM_GPU_LAYERS` | `-1` | Chat-model layers on the GPU: `-1` = as many as fit, `0` = none, `n` = exactly n |
| `IMSDB_GPU_RUNTIME_DIR` | `runtime/gpu` | Folder holding the GPU builds (`vulkan/`, `cuda/`) |
| `IMSDB_GPU_BACKENDS` | `cuda,vulkan` | GPU builds to try, in order (only installed ones are tried) |
| `IMSDB_MODELS_DIR` | `data/models` | Model files |
| `IMSDB_CHAT_MODEL` / `IMSDB_EMBED_MODEL` | `qwen3-8b-q4km` / `qwen3-embedding-0.6b-q8` | Keys of `MODELS` in `src/chatbot/model_store.py` |
| `IMSDB_LLM_THREADS` | `0` | CPU threads (0 = number of physical cores) |
| `IMSDB_LLM_CONTEXT` | `8192` | Answer context size in tokens |
| `IMSDB_CHAT_HOST` / `IMSDB_CHAT_PORT` | `127.0.0.1` / `8765` | Web page address; only loopback addresses are accepted |
| `IMSDB_MSVC_RUNTIME_DIR` | `runtime/win64` | Folder with the app-local `msvcp140.dll` (Windows) |

## 5. Running the archive (Phase 1)

```bash
python -m src.main inspect-source        # robots.txt + policy pages -> evidence + assessment
python -m src.main discover              # listing pages only (no screenplay downloads); --quick = 1 request
python -m src.main crawl --limit 10      # collect up to 10 records (within the permission quota)
python -m src.main crawl --all           # every discovered record - requires 'bulk' permission
python -m src.main resume                # continue the latest interrupted/crashed job
python -m src.main stats                 # dashboard + coverage report   (--json for machines)
python -m src.main validate              # consistency report -> data/exports/reports/  (--deep: all checks)
python -m src.main export --format json  # or jsonl; --no-text to omit screenplay text
python -m src.main backup                # verified local backup -> data/backups/
```
Further commands: `init-db`, `permit`, `reparse` (offline re-parse from the raw archive; `--listings` for
discovery pages), `restore`, `backup --verify DIR`, `backup --list`, `crawl --refresh-days N` (incremental sync),
`crawl --retry-failed`, `discover --refresh`. `python -m src.main <command> --help` lists every option.

**Exit codes** (`src/errors.py`, `ExitCode`): `0` ok · `1` error · `2` usage/config · `3` validation failed ·
`4` permission denied · `5` database unavailable · `6` network/access restriction · `7` backup failed ·
`130` interrupted.

**Interrupting:** press Ctrl+C once. Workers finish their current page, release claimed items and mark the job
`interrupted`; run `resume` to continue. Pressing Ctrl+C twice aborts immediately; leases expire and the next
run recovers them.

**Permission management (operator):**
```bash
python -m src.main permit --level bulk --granted-by "Name, role" \
    --evidence "Written permission from ... dated ..., stored at ..." --confirm
python -m src.main permit --level discovery_only --granted-by "me" --evidence "pausing collection" --confirm
python -m src.main permit --level sample --sample-limit 25 --granted-by "me" --evidence "..." --confirm
```
Every change is appended to `source_permissions.history`. Collected records keep the permission level and
version that applied when they were collected (`rights.*`).

## 6. Running ScriptBot (Phase 2)

```bash
python -m src.main models fetch             # once, needs internet: downloads + SHA-256-verifies the two models
python -m src.main models status --verify   # check the model files
python -m src.main chat-index               # once, and after every crawl/reparse (incremental, ~2 min in full)
python -m src.main device                   # GPU or CPU, and why  (--json for details)
python -m src.main serve --open             # web page at http://localhost:8765 (this computer only)
python -m src.main chat                     # the same chatbot in the terminal
python -m src.main ask "How does Alien end?" --json   # one question, with plan, sources and timings
```
`--no-rerank` (on `serve`, `chat`, `ask`) skips the embedding re-ranker: about 10 s faster on a CPU, slightly
less precise. Questions can be asked in English or Chinese. See [docs/CHATBOT.md](docs/CHATBOT.md) for what it
can answer and how the answer pipeline works.

## 7. GPU by default, CPU fallback

The chatbot is configured to run on a GPU (`IMSDB_LLM_DEVICE=auto`, the default) and to fall back to the CPU
automatically. The logic is in `src/chatbot/device.py`:

```
 start-up (serve / chat / ask)
    │
    ├─ IMSDB_LLM_DEVICE=cpu? ──────────────────────────────────────────────► CPU build (.venv)
    │
    ├─ for each installed GPU build in IMSDB_GPU_BACKENDS (cuda, then vulkan):
    │     test it in a CHILD PROCESS: import it, list devices, run the embedding
    │     model fully on the GPU once (a crash here cannot take the chatbot down)
    │        passed ─► how many layers fit in free GPU memory? ─► ≥ 1 ─► use this GPU build
    │        failed ─► try the next build
    │
    ├─ none passed: IMSDB_LLM_DEVICE=auto ──────────────────────────────────► CPU build (.venv)
    │               IMSDB_LLM_DEVICE=gpu  ──────────────────────────────────► error, with the reason
    │
    └─ loading the model on the GPU still fails (e.g. out of memory) ───────► reloaded on the CPU
```

* **Why two builds.** llama-cpp-python ships separate builds for CPU, Vulkan and CUDA, all named `llama_cpp`.
  Only one can be imported per process, so the CPU build lives in `.venv` (always present, always works) and GPU
  builds live in `runtime/gpu/<backend>/`. `device.activate()` puts the chosen folder first on `sys.path`
  before the first import.
* **Vulkan (default GPU build)** works with NVIDIA, AMD and Intel GPUs and needs only the normal graphics driver.
  **CUDA** (optional) is NVIDIA-only and needs driver 555 or newer; it is usually faster on NVIDIA cards.
* **Partial offload.** With `IMSDB_LLM_GPU_LAYERS=-1` the number of layers is computed from the free GPU memory,
  the model size, the KV caches and the embedding model (`plan_gpu_layers`); the rest runs on the CPU.
* **Where to see the decision:** `python -m src.main device`, the `compute device` line of
  `verify_environment.py`, the log line `compute device: …`, and the status pill on the web page
  ("Ready · N films · GPU/CPU"; hover for the reason).
* **On the development computer** (Ryzen 5 3600, no working GPU: its old GeForce GT 710 is no longer
  present) the Vulkan build is installed and tested at every start-up (≈0.3 s), finds no GPU, and the CPU build
  is used. The GPU branch is covered by unit tests with simulated devices (`tests/test_device.py`); it has not
  been run on real GPU hardware in this project yet. The first time it runs on a GPU machine, check
  `python -m src.main device`.

## 8. Repository layout

```
.
├── README.md                    this guide
├── pyproject.toml               project metadata, pytest configuration and markers
├── requirements.txt             archive dependencies
├── requirements-chatbot.txt     chatbot runtime, CPU build (into .venv)
├── requirements-gpu-vulkan.txt  chatbot runtime, Vulkan GPU build (into runtime/gpu/vulkan)
├── requirements-gpu-cuda.txt    chatbot runtime, CUDA GPU build (optional, into runtime/gpu/cuda)
├── .env.example                 every setting with its default and an explanation
├── docs/                        ARCHITECTURE, CHATBOT, SOURCE_PERMISSIONS, SAMPLE_RECORDS
├── scripts/
│   ├── setup.py                 one-shot setup (venv, packages, GPU build, .env, folders)
│   ├── verify_environment.py    environment check (exit 1 on a failed required check)
│   ├── init_database.py         collections, validators, indexes, permission record
│   ├── local_mongodb.py         start/stop/status for a portable mongod
│   ├── generate_sample_records.py  regenerates docs/SAMPLE_RECORDS.md from the database
│   └── check_docstrings.py      enforces the documentation convention (section 12)
├── src/
│   ├── main.py                  CLI entry point: `python -m src.main <command>`
│   ├── config.py                Settings (environment / .env), validated with Pydantic
│   ├── errors.py                exception types and CLI exit codes
│   ├── crawler/                 URL rules, polite HTTP fetcher, discovery, worker pool
│   ├── parsers/                 pure functions: listing/detail HTML, screenplay text, scenes, dialogue
│   ├── storage/raw_store.py     content-addressed archive of raw responses
│   ├── database/                connection (local only), Pydantic models, indexes/validators, repositories
│   ├── services/                ingestion, permissions, validation, stats, export, backup, deduplication
│   ├── utils/                   time, hashing, logging, text decoding/normalisation
│   └── chatbot/                 ScriptBot (Phase 2), see section 9.7
├── tests/                       pytest suite; fixtures/ holds synthetic HTML in IMSDb's page structure
├── runtime/
│   ├── win64/msvcp140.dll       newer Microsoft C++ runtime, loaded app-locally (Windows; not in git)
│   └── gpu/<backend>/           GPU builds of llama-cpp-python (installed by setup.py; not in git)
├── data/                        (not in git) raw/, processed/, exports/, backups/, models/, index/
└── logs/                        (not in git) imsdb.log (JSON lines), chatbot logs
```

## 9. Module guide

Every module, class and function has a Google-style docstring describing its purpose, arguments and return
value, so the code itself is the detailed reference. This section is the map: what each package is
responsible for, its main entry points, and what you must keep in mind when changing it.

### 9.1 `src/main.py`, `src/config.py`, `src/errors.py`: entry point and shared plumbing

* `main.py` builds the `argparse` CLI (`build_parser`). Each command is a function `cmd_<name>(rt, args) -> int`
  registered with `add(name, func, help)`. Chatbot commands are registered by `src/chatbot/cli.py:register`.
* `Runtime` holds one command's state: `settings`, a lazily connected `mongo`, a lazily built ingestion context
  (`ctx`), a shared `stop_event`, and the Ctrl+C handlers (first press = graceful stop, second = abort).
* `main()` maps exceptions to exit codes: every `IMSDbError` subclass in `errors.py` carries its own `ExitCode`.
* `config.py`: `Settings` (frozen Pydantic model) + `_ENV_MAP` (environment variable → field) + `load_settings()`.
  Validators enforce the local-only rules (loopback chat host, minimum request interval).

### 9.2 `src/crawler/`: what to fetch and how

| Module | Responsibility | Key names |
|---|---|---|
| `url_manager.py` | Canonical URLs and scope. Classifies every URL into a `PageType`; only allow-listed types are crawlable | `normalize_url`, `classify_url`, `is_crawlable`, `url_id`, `PageType` |
| `fetcher.py` | Polite HTTP: per-host `RateLimiter`, retries with backoff and jitter, `Retry-After`, size cap, same-host redirects, conditional requests | `Fetcher.fetch` → `FetchResult` |
| `discovery.py` | Seed URLs and the queue items derived from listing pages | `seed_items`, `listing_followups`, `detail_items` |
| `scheduler.py` | Worker threads over the persistent `crawl_queue`: lease-based claims, record budget, graceful stop | `CrawlScheduler.run`, `RecordBudget` |

Keep in mind: 401/403/407/451 responses mark the item `blocked` and **stop the crawl**; the crawler must not try to
get around access restrictions (section 12).

### 9.3 `src/parsers/`: pure parsing (no network, no database)

| Module | Input → output |
|---|---|
| `movie_parser.py` | Listing page HTML → `ListingPage` of `ListingEntry`; movie detail HTML → `MovieDetail` (writers, genres, dates, script link, draft info) |
| `screenplay_parser.py` | Script page HTML → `ExtractedScript` (exact text + bold/italic spans, extraction method); text → `ParsedScreenplay` (layout profile, labelled `Element`s with character offsets and confidence) |
| `scene_parser.py` | Scene headings (`INT./EXT.`, location, time of day) and scene segmentation |
| `dialogue_parser.py` | Character cues (`NAME (V.O.) (CONT'D)`), dialogue blocks with parentheticals, per-character aggregates |

`PARSER_VERSION` in `src/parsers/__init__.py` is stored on every parsed record; bump it when parsing output
changes so `reparse` knows which records are outdated. The stages are described in
[docs/ARCHITECTURE.md §4](docs/ARCHITECTURE.md#4-screenplay-parsing).

### 9.4 `src/storage/raw_store.py`: the raw archive

`RawStore.save()` writes each response body once to `data/raw/<host>/<page_type>/<sha[:2]>/<sha256>.html`
(never overwritten) and appends a line to the daily JSONL manifest. `read()` / `verify()` re-check the SHA-256.
This is what makes offline `reparse` and "never silently discard source content" possible.

### 9.5 `src/database/`: MongoDB access

| Module | Responsibility |
|---|---|
| `connection.py` | `Mongo.connect(settings)`; `assert_local_uri` rejects any non-local server |
| `models.py` | One Pydantic model per collection (`ScriptDoc`, `SceneDoc`, `QueueItem`, …), status enums, deterministic id helpers (`movie_id_for`, `script_id_for`, …) |
| `indexes.py` | `ensure_schema(db)`: creates collections, `$jsonSchema` validators and indexes; idempotent |
| `repositories.py` | **All reads and writes go through these classes** (`MovieRepository`, `ScriptRepository`, `QueueRepository`, `JobRepository`, `PermissionRepository`, …), bundled as `Repositories` |

MongoDB runs standalone (no transactions), so every write sequence is ordered to be safely repeatable after a
crash: new derived records are upserted first, and leftovers from older parses are deleted afterwards.

### 9.6 `src/services/`: application logic

| Module | Responsibility |
|---|---|
| `ingestion.py` | `IngestionService`: `inspect_source`, `discover`, `crawl`, `resume`, `reparse`. Dispatches queue items by page type (`handle_listing`, `handle_detail`, `handle_script`) and stores results through the repositories |
| `permissions.py` | Permission levels (`discovery_only` < `sample` < `bulk`), robots.txt policy, `PermissionGate` (every fetch and every record is checked against it) |
| `documents.py` | Turns parser output into validated documents (`build_script_bundle`, `movie_detail_fields`, rights metadata) |
| `validation.py` | Consistency checks → `ValidationReport` (Markdown + JSON in `data/exports/reports/`) |
| `stats.py` | Counts, coverage and the CLI dashboard |
| `deduplication.py` | Duplicate titles/contents, URL canonicalization merges |
| `export.py` | JSON / JSON Lines export with a provenance manifest |
| `backup.py` | Verified backups (mongodump or a Python BSON fallback, plus a ZIP of `data/raw`), verify, restore |

### 9.7 `src/chatbot/`: ScriptBot

| Module | Responsibility | Key names |
|---|---|---|
| `cli.py` | Commands `models`, `device`, `chat-index`, `ask`, `chat`, `serve` | `register` |
| `engine.py` | The pipeline: plan → resolve films → facts / passages → answer → quote check; sessions and follow-ups | `ChatEngine.ask`, `Plan`, `ChatResult`, `SessionStore` |
| `prompts.py` | Planner instructions and JSON schema, answer instructions (English and Chinese) | `PLAN_SCHEMA`, `INTENTS`, `planner_system`, `ANSWER_SYSTEM` |
| `llm.py` | Model wrappers (two chat contexts sharing weights: planner + answer; embedder), loading with GPU/CPU fallback | `LocalChatModel.generate`, `LocalEmbedder.embed`, `load_local_models` |
| `device.py` | GPU detection, child-process self-test, layer planning, CPU fallback | `select_device`, `activate`, `DeviceChoice` |
| `runtime.py` | Safe import of llama-cpp-python on Windows (app-local `msvcp140.dll`) | `import_llama_cpp`, `prepare_windows_runtime` |
| `model_store.py` | Pinned model manifest, resumable download, SHA-256 verification | `MODELS`, `ModelSpec`, `fetch_model`, `require_model` |
| `catalog.py` | In-memory film catalogue; title resolution (IMSDb's "X, The", franchises, typos, Chinese aliases) | `FilmCatalog.resolve` |
| `data/title_aliases.json` | Verified Chinese → English film titles | |
| `facts.py` | Exact data from MongoDB: film facts, characters, film lists, archive totals | `film_facts`, `character_facts`, `list_films` |
| `chunks.py` | Builds the `chat_chunks` search collection from scenes (~1,200-character passages) | `build_chunks`, `CHUNKER_VERSION` |
| `search.py` | Keyword search (MongoDB `$text`), opening/ending passages, re-ranking (reciprocal-rank fusion) | `keyword_search`, `rerank`, `Passage` |
| `templates.py` | Answers made without the model: lists, archive totals, full-script refusal | |
| `verify.py` | Checks every quoted English line in an answer against the passages | `unverified_quotes` |
| `server.py` | Standard-library HTTP server: page, `/api/health`, `/api/chat` (Server-Sent Events), `/api/passage`, `/api/reset`; Host/Origin checks, strict CSP | `make_server`, `EngineHolder` |
| `static/index.html` | The whole web page (HTML/CSS/JS inline, no external resources), English/中文 interface | |

The engine depends only on two small interfaces, `ChatModel.generate` and `Embedder.embed` (Protocols in
`llm.py`), which is how the tests replace the real models with fakes.

## 10. How to make common changes

Each recipe lists the files to touch, in order. Always finish with `python -m pytest` and
`python scripts/check_docstrings.py`.

### Archive

**Add a CLI command.** Write `cmd_<name>(rt: Runtime, args) -> int` in `src/main.py` (or in
`src/chatbot/cli.py` for chatbot commands), register it in `build_parser` (`register`) with `add(...)`, add its
options, return an `ExitCode`. Raise an `IMSDbError` subclass for expected failures so the right exit code is
used. Add a test that calls `main([...])` and document the command in sections 5/6.

**Add a setting.** Add the field (with type, default and bounds) to `Settings` in `src/config.py`, map its
environment variable in `_ENV_MAP`, add a validator if needed, document it in `.env.example` and in section 4.

**Change how screenplays are parsed.** Change the heuristics in `src/parsers/screenplay_parser.py` (line
labelling is in `_Classifier`), `scene_parser.py` or `dialogue_parser.py`. Add a fixture or a test case in
`tests/test_parsers.py` that shows the problem first. Bump `PARSER_VERSION` in `src/parsers/__init__.py`. Then
re-process stored data offline: `python -m src.main reparse`, `python -m src.main validate --deep`, and
`python -m src.main chat-index` (the chatbot index follows changed scripts automatically).

**Add a field to a stored document.** Add it to the Pydantic model in `src/database/models.py`; fill it in
`src/services/documents.py` (scripts/scenes/dialogues) or the relevant repository; if it is required, add it to
the `$jsonSchema` in `src/database/indexes.py` (existing documents are validated with `moderate` level); add an
index there if you query by it; add a check to `src/services/validation.py` if consistency matters; update
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and regenerate `docs/SAMPLE_RECORDS.md`
(`python scripts/generate_sample_records.py`).

**Crawl a new kind of page.** This changes what is collected from the source, so check the permission record
first. Add a `PageType` and its URL pattern in `src/crawler/url_manager.py` (`classify_url`, and `is_crawlable`
for the allow-list), create queue items for it in `src/crawler/discovery.py`, add a handler to the dispatch in
`IngestionService` (`src/services/ingestion.py`), a parser in `src/parsers/`, a model and repository in
`src/database/`, and tests using the `FakeSite` fixture in `tests/conftest.py` (tests never contact IMSDb).

**Change crawl speed or retries.** Use the settings (`IMSDB_MIN_REQUEST_INTERVAL`, `IMSDB_CONCURRENCY`,
`IMSDB_MAX_RETRIES`, …). The 1-second floor (`MIN_ALLOWED_INTERVAL` in `config.py`) is deliberate.

### Chatbot

**Change what the chatbot says or how it plans.** Edit `src/chatbot/prompts.py`. Keep the fixed instructions at
the start of each prompt: the model caches that prefix between questions, and changing it per question costs
~15 s on a CPU. The answer rules exist in English and Chinese (`ANSWER_SYSTEM_EN` / `ANSWER_SYSTEM_ZH`);
change both. Test with `python -m src.main ask "…" --json` in both languages.

**Add a new kind of question (intent).** Add the name to `INTENTS` in `prompts.py` (it is also the JSON-schema
enum), describe it with an example in `planner_system`, handle it in `ChatEngine._dispatch` in `engine.py` (use
`templates.py` if it can be answered exactly without the model), and add a test in `tests/test_chatbot.py` with
`FakeChat` returning the new plan.

**Change retrieval.** Search and re-ranking live in `search.py` (`keyword_search`, `rerank`, `RERANK_CHARS`); the
number of passages given to the model is `ChatEngine(max_passages=4, rerank_candidates=10)`. If you change how
passages are cut (`chunks.py`: `CHUNK_CHARS`, `split_chunks`, the text index weights), bump `CHUNKER_VERSION`
and run `python -m src.main chat-index`; only changed scripts are re-processed.

**Use a different model.** Add a `ModelSpec` to `MODELS` in `model_store.py` with a pinned Hugging Face revision,
the exact size and SHA-256 (from the Hugging Face API), the licence, `n_layers` and `kv_bytes_per_token` (used
for GPU memory planning), then set `IMSDB_CHAT_MODEL` / `IMSDB_EMBED_MODEL` and run `models fetch`. The prompt
format (`chatml` in `llm.py`) and the empty `<think>` block are specific to Qwen3; another model family needs
its own chat template there.

**Add Chinese film titles.** Add `"中文名": "English title as in IMSDb"` to
`src/chatbot/data/title_aliases.json` (no code change, restart the chatbot).

**Change the web page.** Everything is in `src/chatbot/static/index.html`. The Content-Security-Policy in
`server.py` forbids external scripts, styles and fonts on purpose (the page must work offline). All text from
the model is HTML-escaped before the small Markdown renderer runs; keep it that way.

**GPU/CPU behaviour.** Settings: `IMSDB_LLM_DEVICE`, `IMSDB_LLM_GPU_LAYERS`, `IMSDB_GPU_BACKENDS`. Code:
`device.py` (decision), `llm.py` (`choose_device`, `load_local_models`, load-time fallback). To support another
GPU build (e.g. a newer CUDA), add a requirements file with a SHA-256 pin, add it to `GPU_REQUIREMENTS` in
`scripts/setup.py` and to the allowed names in `Settings._check_backends`.

**Upgrade llama-cpp-python.** Keep the CPU build and the GPU builds on the same version. Test the CPU build on
the target processor first: newer Windows wheels have crashed with "illegal instruction" (0xc000001d) because
they use instructions the processor lacks (e.g. 0.3.38 on the Ryzen 5 3600). Update the pins and hashes in all three requirements files.

## 11. Testing

```bash
python -m pytest                          # the whole suite (MongoDB tests are skipped if no server is running)
python -m pytest -m "not mongodb"         # skip MongoDB integration tests
python -m pytest tests/test_chatbot.py    # one area
TEST_MONGODB_TOOLS_DIR=/path/to/tools/bin python -m pytest   # also run the mongodump/mongorestore test
python scripts/check_docstrings.py        # documentation check (also run by tests/test_docstrings.py)
```

| Test file | Covers |
|---|---|
| `test_crawler.py` | URL normalisation and classification, fetcher retries/backoff/`Retry-After`, rate limiting, robots.txt, permission gate, scheduler, interruption and crash recovery |
| `test_parsers.py` | Listing/detail parsing, screenplay extraction, scenes, cues, dialogue, uncertainty flags |
| `test_database.py` | Validators, indexes, repositories, idempotent writes |
| `test_ingestion.py` | End-to-end discover/crawl/reparse against a fake site, versioning, statuses, validation |
| `test_backup.py` | Backup, verification, tamper detection, restore |
| `test_chatbot.py` | Engine with fake models, catalogue, chunking, search, templates, quote check, HTTP server and its security checks, model download |
| `test_device.py` | GPU/CPU decision, fallbacks, layer planning, the real Vulkan build's self-test |
| `test_docstrings.py` | Every function documented (section 12) |

* Tests never contact IMSDb or download models. HTML fixtures in `tests/fixtures/` copy IMSDb's page
  *structure* with original, synthetic screenplay content; `FakeSite` (in `conftest.py`) serves them through a
  mock HTTP transport.
* MongoDB tests (marker `mongodb`) create a throwaway database `imsdb_test_<random>` and drop it afterwards.
* Chatbot tests use `FakeChat` / `FakeEmbedder` instead of the real models, so they run in seconds.

## 12. Coding conventions and project rules

**Documentation.** Every module, class and function has a Google-style docstring: a one-line summary, then
`Args:` (every parameter), `Returns:` (or `Yields:`) and, where useful, `Raises:`. Dataclasses and models list
their fields under `Attributes:`. Inline `#` comments explain non-obvious logic only.
`python scripts/check_docstrings.py` enforces this and `tests/test_docstrings.py` runs it, so an undocumented
function fails the test suite.

```python
def detect_language(text: str) -> str:
    """Decide whether to answer in Chinese or English.

    An explicit request ("用英文回答", "answer in Chinese") wins; otherwise the question is Chinese when it
    has at least two CJK characters and at least as many CJK characters as Latin words.

    Args:
        text: The user's question.

    Returns:
        "zh" for Chinese, "en" for English.
    """
```

**Style.** Python 3.12, type hints everywhere, `from __future__ import annotations`, lines up to 120 characters,
`pathlib` for paths, timezone-aware UTC times (`src/utils/clock.py`), `logging` (never `print`) in library code.

**Project rules. These are requirements, not preferences:**
* Everything stays local: MongoDB on localhost only, the chat server on loopback only, no cloud services, no
  containers. The code rejects non-local MongoDB URIs and chat hosts.
* No AI calls during ingestion. The chatbot uses only local models; nothing is sent to any AI service.
* Respect the source: honour robots.txt and the permission level; no bulk crawl without recorded authorization;
  never bypass authentication, anti-bot measures, access controls or rate limits; no proxy rotation or identity
  spoofing. The User-Agent is honest and contains no personal data.
* Never silently discard source content: keep the raw response, flag uncertain parses, record failures.
* No hard-coded credentials; secrets only come from `.env` / the environment and are never logged.
* No system-wide installs for the chatbot: prebuilt wheels in `.venv` / `runtime/`, models in `data/models`.

## 13. Data layout, backup and recovery

```
data/raw/imsdb.com/<page_type>/<sha[:2]>/<sha256>.html   original response bodies (content-addressed, never overwritten)
data/raw/_manifest/<date>.jsonl                          one line per fetch: url, status, headers, sha256, path, time
data/processed/                                          offloaded element arrays for unusually large scripts
data/exports/                                            exports + validation reports
data/backups/backup-<UTC stamp>/                         verified backups
data/models/                                             chatbot model files (GGUF)
logs/imsdb.log                                           JSON-lines application log (rotating, 10 x 10 MB)
```
`scripts.text` is the canonical extracted text; scenes, dialogue blocks and elements point into it with
character offsets. The collections and their fields are documented in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```bash
python -m src.main backup                      # mongodump archive (or python BSON) + raw ZIP + manifest, then verifies
python -m src.main backup --verify data/backups/backup-20261117T120000Z
python -m src.main restore data/backups/backup-20261117T120000Z --confirm            # into MONGODB_DB
python -m src.main restore data/backups/backup-... --confirm --target-db imsdb_restored   # side by side
python -m src.main restore data/backups/backup-... --confirm --drop                   # replace collections
```
* Each backup has a `manifest.json` (document counts, size and SHA-256 of every file, tool versions, restore
  steps) and `manifest.json.sha256`. Restore refuses unverified backups. Raw files that differ from the archive
  are never overwritten; the archived copy goes to `data/raw/_restore_conflicts/`.
* Manual restore: `mongorestore --uri=mongodb://127.0.0.1:27017/ --archive=mongodb.archive.gz --gzip
  --nsInclude='imsdb_local.*'`, then unzip `raw-archive.zip` into `data/raw/`.
* The chatbot's `chat_chunks` index is derived data: after a restore, rebuild it with `chat-index`.
* Backups stay on local disk; copy the backup folder to an external drive yourself for an off-machine copy.

## 14. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Exit code 5, "database unavailable" | MongoDB is not running: `python scripts/local_mongodb.py start` (portable) or start the service |
| `mongod.exe` exits immediately (Windows) | Old Visual C++ runtime; run `vc_redist.x64.exe` from MongoDB's `bin` folder as Administrator |
| Chatbot: "The chat search index has not been built yet" | `python -m src.main chat-index` |
| Chatbot: "Model file … is missing or incomplete" | `python -m src.main models fetch` (resumes partial downloads) |
| Chatbot crashes with `0xc0000005` when loading (Windows) | System `msvcp140.dll` older than 14.40 and the app-local copy is missing from `runtime/win64/` (see `runtime/README.txt`) |
| Chatbot crashes with `0xc000001d` | The llama-cpp-python build uses CPU instructions this processor lacks; stay on the pinned 0.3.36 |
| `device` says "no GPU build … is installed" | `python scripts/setup.py --chatbot` (installs the Vulkan build) |
| `device` says "no GPU found" although there is one | Update the graphics driver (Vulkan support comes with it). On NVIDIA, try `--gpu-backend cuda` |
| GPU runs out of memory | Lower `IMSDB_LLM_GPU_LAYERS` (e.g. 20) or `IMSDB_LLM_CONTEXT`; the loader also falls back to the CPU by itself |
| "Cannot listen on 127.0.0.1:8765" | ScriptBot is already running, or use `serve --port 8770` |
| Answers are slow (1–1.5 min) | Expected on a CPU; a GPU makes generation many times faster. `--no-rerank` saves ~10 s |

## 15. Known limitations

* **Authorization:** bulk collection was recorded by the project owner on 2026-11-23; the approval document itself
  is not yet attached to the permission record (`permit --level bulk --evidence "<path>" --confirm`).
* **Source data:** directors, summaries and IMDb ids are not published on IMSDb detail pages, so these fields stay
  empty and are never invented. PDF-only scripts are recorded but not downloaded. TV transcripts and writer
  pages are out of scope.
* **Parsing is heuristic.** Unusual layouts get lower confidence and are flagged (`uncertain`, `quality_flags`)
  rather than hidden. Dual (side-by-side) dialogue is detected but not split into two speakers.
* **Discovery completeness** is measured against IMSDb's own listings; unlisted records cannot be found.
* **No transactions:** MongoDB runs standalone, so writes are ordered and idempotent instead.
* **Chatbot:** an 8-billion-parameter model can misread a scene or mistranslate a line (answers cite their
  passages, and unverified quotes are flagged); keyword retrieval works best with names, places or objects;
  the stored screenplays are drafts and can differ from the released films; on a CPU an answer takes 1–1.5
  minutes. The GPU path has not yet been run on real GPU hardware (section 7).

## 16. History: the first live sample run

The first live run (2026-11-17) used the default `sample` permission, a 5 s request interval and the honest
User-Agent: `inspect-source` (5 requests), `discover` (48 requests, 1,295 distinct movie records), `crawl --limit 3`
(6 requests), `validate --deep` (passed), backup and side-by-side restore (identical counts). It surfaced five
issues the synthetic fixtures had not modelled, all fixed with regression tests and applied offline from the raw
archive: a site-banner `<h1>` hiding listing entries; a false redirect-loop detection on trailing slashes;
IMSDb's double-encoded links (`%2526`) creating alias records (merged; original hrefs kept in `source_hrefs`);
speech on the same line as a parenthetical; and a title-page credit block read as a character cue. Details and
sample documents: [docs/SAMPLE_RECORDS.md](docs/SAMPLE_RECORDS.md). The full archive was collected after bulk
authorization on 2026-11-23.
