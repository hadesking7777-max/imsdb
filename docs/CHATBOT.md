# ScriptBot: the offline screenplay chatbot

ScriptBot answers questions about the screenplays in the local IMSDb archive, in **English or Chinese**.
It runs entirely on this computer. After a one-time model download, no internet connection is used.
It runs on a **GPU by default** and falls back to the **CPU** automatically when no GPU is usable (section 4).

## 1. Quick start

```bash
# once: install the runtime (CPU build + Vulkan GPU build) and assemble the models from the repository's
# pieces in data/models/parts (downloaded from Hugging Face only if the pieces are missing)
python scripts/setup.py --chatbot
python -m src.main models fetch
python -m src.main device                    # shows whether the GPU or the CPU will be used, and why

# once, and again after new crawls or re-parsing (~2 minutes, offline)
python -m src.main chat-index

# every time
python scripts/local_mongodb.py start        # if MongoDB is not already running
python -m src.main serve --open              # opens http://localhost:8765
```

Other ways to use it:

```bash
python -m src.main chat                      # interactive chat in the terminal
python -m src.main ask "How does Alien end?"  # one question; add --json for plan, sources and timings
python -m src.main models status --verify    # check the model files and their SHA-256 checksums
python scripts/verify_environment.py         # includes chatbot checks (runtime, models, index)
```

## 2. What it can answer

| Kind of question | Example | How it is answered |
|---|---|---|
| Film details | "Who wrote Alien, and who are its main characters?" | Exact data from MongoDB, written up by the model |
| Characters | "《低俗小说》里朱尔斯说了多少句台词？" | Character records (line and word counts, first lines) plus matching scenes |
| Scenes, plot, quotes | "How does The Shawshank Redemption end?" | Screenplay passages found by search, cited as [1], [2] … |
| Lists | "List all horror films", "列出昆汀·塔伦蒂诺编剧的电影" | Exact list from MongoDB (instant, no model) |
| The archive | "What is in the archive?" | Exact totals (instant, no model) |
| Follow-ups | "And who says it?" | The previous film is carried over |

ScriptBot does **not** output full screenplays. Quotes are kept short, in line with the usage restriction
recorded on every record (educational and research use; the scripts belong to their studios and writers).

## 3. How it works

```
question (EN/中文)
   │
   ├─ 1. plan      Qwen3-8B, JSON-constrained: intent, films (as English titles), character,
   │               genre/writer/year, opening/ending, English search terms        ~8–10 s
   ├─ 2. look up   film catalogue (title variants, IMSDb's "X, The", franchises, typos) and
   │               exact facts from movies / scripts / characters / dialogues     < 1 s
   ├─ 3. search    MongoDB text index over 237,173 scene passages (chat_chunks),
   │               limited to the film when one is named                          < 1 s
   ├─ 4. re-rank   Qwen3-Embedding-0.6B compares the top 10 passages with the original
   │               question (works across English and Chinese)                    ~7–10 s
   └─ 5. answer    Qwen3-8B writes the answer from the facts and 4 passages, citing them;
                   streamed word by word                                          ~45–70 s
```

* **Grounding.** The answer model is told to use only the supplied facts and excerpts, to say when they
  do not contain the answer, to attribute dialogue to the capitalized name above it, and to point out
  when a famous line is missing from the stored draft (e.g. Titanic's draft has no "king of the world").
* **Quote check.** After each answer, every English quotation (3+ words) is compared word for word with
  the passages and facts the model was given (`src/chatbot/verify.py`). Quotes that are not found are
  shown under the answer as a warning. In testing this caught invented lines that sounded plausible.
* **Exact answers without the model.** Film lists and archive totals are produced from the database
  directly, in the question's language. Requests for a full screenplay are declined the same way.
* **Unknown films.** If a named film is not in the archive, ScriptBot says so instead of quoting
  scenes from other films. Films without a stored screenplay (PDF-only, empty pages) are explained.
* **Language.** Chinese questions get Chinese answers (Simplified unless the question is Traditional);
  quoted lines keep the original English with a Chinese translation. "用英文回答" / "answer in Chinese"
  override the detection.

### Why keyword search plus re-ranking, not a full vector index

Embedding every passage would have been the textbook approach, but on this computer the embedding model
processes about 180 tokens per second: the ~45 million tokens of scene text would take about 60 hours.
Keyword search over a MongoDB text index takes about two minutes to build and covers every word. Chinese
questions are translated into English search terms by the planner, and the multilingual re-ranker then
picks the passages that best match the original question.

## 4. Models and runtime

| Component | File | Size | Licence |
|---|---|---|---|
| Chat model | Qwen3-8B, Q4_K_M quantization (`Qwen/Qwen3-8B-GGUF`) | 5.03 GB | Apache-2.0 |
| Re-ranker | Qwen3-Embedding-0.6B, Q8_0 (`Qwen/Qwen3-Embedding-0.6B-GGUF`) | 0.64 GB | Apache-2.0 |
| Runtime, CPU build | llama-cpp-python 0.3.36 (prebuilt wheel, AVX2), in `.venv` | 10 MB | MIT |
| Runtime, GPU build | llama-cpp-python 0.3.36 Vulkan wheel, in `runtime/gpu/vulkan` (CUDA 12.5 optional) | 42 MB | MIT |

* Model files are pinned to an exact Hugging Face revision and verified by SHA-256 (`src/chatbot/model_store.py`).
  The download resumes if interrupted.
* **llama-cpp-python is pinned to 0.3.36.** Its Windows wheel targets AVX2 CPUs. Newer wheels (0.3.38)
  and some older ones use instructions this CPU (Ryzen 5 3600) lacks and crash with `0xc000001d`.
* **Windows C++ runtime.** This computer's `C:\Windows\System32\msvcp140.dll` (14.34) is too old for the
  runtime. A newer Microsoft-signed copy (14.51) is kept in `runtime/win64/` and loaded app-locally by
  `src/chatbot/runtime.py`; nothing is installed system-wide. See `runtime/README.txt`.
* **GPU by default, CPU fallback** (`src/chatbot/device.py`, setting `IMSDB_LLM_DEVICE=auto`). At start-up each
  installed GPU build is tested in a separate process (import, device list, one embedding run fully on the GPU).
  The first that passes is used, with as many chat-model layers on the GPU as fit in its free memory
  (`IMSDB_LLM_GPU_LAYERS=-1`); otherwise the CPU build in `.venv` is used. If loading on the GPU fails anyway
  (e.g. out of memory), the model is loaded again on the CPU. `IMSDB_LLM_DEVICE=cpu` skips the GPU test;
  `IMSDB_LLM_DEVICE=gpu` turns a missing GPU into an error. The decision is shown by
  `python -m src.main device` and in the page's status pill (hover for the reason).
* The development computer has no working GPU, so the measurements below are CPU measurements and the GPU path
  has only been exercised with simulated devices in the tests. With the whole model on a modern GPU, writing an
  answer is typically many times faster than on this CPU.
* Two model contexts share the chat model's weights: one for planning, one for answers. Each keeps its fixed
  instructions cached, which saves ~15 s per question. Memory use is about 12 GB.

## 5. Speed on this computer (Ryzen 5 3600, 64 GB, no GPU: CPU fallback)

| Step | Measured |
|---|---|
| Model loading | ~10 s, plus ~27 s warming the prompt caches (the page opens immediately; models load in the background) |
| Planning | 8–10 s |
| Search + re-ranking | 7–14 s |
| Writing the answer | 45–70 s (about 6 words per second, shown as they are written) |
| Lists / archive totals | instant after planning |

Ways to make it faster: a GPU (used automatically once present: any GPU with a current driver works with the
installed Vulkan build; `python scripts/setup.py --chatbot --gpu-backend cuda` adds the CUDA build for NVIDIA),
`--no-rerank` (saves ~10 s, slightly less precise), or fewer CPU-heavy programs running.

## 6. Web page

* Served by Python's standard library on `127.0.0.1:8765` (this computer only). Requests with a foreign
  `Host` header (DNS rebinding) or a foreign `Origin` (other websites) are rejected.
* The page loads nothing from the internet (strict Content-Security-Policy). Interface language follows
  the browser, with an English/中文 switch.
* Sources under each answer can be expanded to read the full passage. Citations [n] jump to their source.
* **Stop** cancels an answer immediately; closing the tab does the same.

## 7. Keeping it up to date

After `crawl --refresh-days`, `reparse`, or any change to the scripts, run `python -m src.main chat-index`.
It only re-processes scripts whose content changed. `chat-index --rebuild` rebuilds everything.

## 8. Limitations

* An 8-billion-parameter model makes mistakes: it can misread a scene or mistranslate a line. Every answer
  cites its passages, so check the sources when it matters.
* Keyword search needs words that occur in the screenplay. Vague questions ("the sad scene") work less well
  than ones with names, places or objects; asking about a specific film helps a lot.
* The screenplays are drafts, sometimes very different from the released films.
* Chinese film titles are translated by the model. Most well-known titles work; others can be added to
  `src/chatbot/data/title_aliases.json` (`"中文名": "English title"`).
