# CLAUDE.md — Blender JSON Scraper

## Project Purpose
C++ + Python pipeline to scrape Blender node JSON files from GitHub and produce curated JSONL training datasets for LLMs. A Blender 5.0+ addon for LLM-assisted node generation is also included.

---

## File Structure

```
c:\AI\
├── blender_scraper/
│   ├── src/
│   │   ├── main.cpp            # CLI entry point; parses args, loads config, runs pipeline
│   │   ├── scraper.hpp         # Core orchestrator: 4-phase pipeline
│   │   ├── github_client.hpp   # GitHub REST API v3; parallel search agents
│   │   ├── http_client.hpp     # libcurl RAII wrapper (GET/POST, rate-limit headers)
│   │   ├── apify_client.hpp    # Apify fallback web-scraping API client
│   │   ├── rate_limiter.hpp    # Token-bucket rate limiter (5 req/s default)
│   │   ├── token_rotator.hpp   # GitHub PAT pool; auto-rotates on 429/403
│   │   └── build_dataset.cpp   # Standalone: walks output/ tree → JSONL
│   ├── filter_dataset.py       # Scores + deduplicates JSONL by Blender relevance
│   ├── config.json             # Runtime config (git-ignored)
│   ├── config.example.json     # Config template
│   ├── CMakeLists.txt          # Requires C++20, libcurl, nlohmann/json
│   ├── build.sh                # Unix build
│   └── build_msvc.bat          # Windows MSVC build
├── qwen_node_assistant.py      # Blender 5.0+ addon: Ollama LLM → node trees
├── claude.html.html            # Web UI artifact
├── gemini.html                 # Web UI artifact
└── gpt.html                    # Web UI artifact
```

---

## 4-Phase Scraping Pipeline

```
Phase 1  DISCOVER REPOS     GitHub search (10 topics × 10 keywords × 9 year ranges)
                             4 parallel agents; atomic index; merged under mutex
         ↓
Phase 2  ENUMERATE FILES     GET /repos/{owner}/{repo}/contents — recursive, .json only
                             Collects: path, sha, download_url, html_url
         ↓
Phase 3  DOWNLOAD & SAVE     8-thread pool; HTTP GET raw content → validate JSON → write
                             Output: output/json_files/{owner}/{repo}/{path}
                             Sidecar: output/json_files/{owner}/{repo}/{path}.meta.json
         ↓
Phase 4  BUILD DATASET       build_dataset executable walks output/ tree
                             Pairs .json + .meta.json → one JSONL record per file
                             Output: output/training_dataset.jsonl
```

**Post-processing** (`filter_dataset.py`):
- Scores each record: node type matches (+5), tree type ids (+3), path keywords (+2), generic keys (+1–3)
- Hard-discard: Unity/Unreal/Minecraft signatures
- Deduplicates by `repo|path` key
- Threshold: `--min-score 2` (default)
- Output: `output/blender_nodes_dataset.jsonl`

---

## Key Components

### `scraper.hpp` — BlenderJsonScraper
Central orchestrator. Owns the thread pool and calls GitHubClient agents. Writes manifest and statistics.

### `github_client.hpp` — GitHubClient
Each parallel agent owns its own instance + RateLimiter. Calls TokenRotator for token selection. Methods: `search_blender_repos()`, `list_json_files()`, `download_file()`.

### `token_rotator.hpp` — TokenRotator
Manages a pool of GitHub PATs. Rotates on 403/429. If all exhausted, sleeps until earliest epoch reset from `X-RateLimit-Reset` header.

### `rate_limiter.hpp` — RateLimiter
Token-bucket: `acquire()` blocks until a token is available. `handle_rate_limit()` sleeps to the epoch reset time.

### `http_client.hpp` — HttpClient
Non-copyable libcurl RAII wrapper. Parses `X-RateLimit-Remaining` and `X-RateLimit-Reset` from response headers.

---

## Configuration

**Precedence (high → low):** CLI args → env vars (`GITHUB_TOKEN`, `APIFY_TOKEN`) → `config.json` → hardcoded defaults

```json
{
  "github_tokens":  ["ghp_..."],
  "apify_token":    "...",
  "output_dir":     "output",
  "max_repos":      500,
  "threads":        8,
  "search_agents":  4,
  "rps":            5,
  "use_apify":      false,
  "validate_json":  true,
  "min_json_size":  10
}
```

---

## Executables

| Binary | Entry | Usage |
|--------|-------|-------|
| `blender_scraper` | `main.cpp` | `blender_scraper [--config cfg.json] [--github-token T] [--output dir]` |
| `build_dataset` | `build_dataset.cpp` | `build_dataset [input_dir] [output.jsonl]` |

**Python scripts:**
- `filter_dataset.py [input.jsonl] [output.jsonl] [--min-score N] [--verbose]`
- `qwen_node_assistant.py` — install as Blender addon; connects to local Ollama

---

## Dependencies

| Dep | Purpose |
|-----|---------|
| libcurl | All HTTP (GitHub API, Apify, raw downloads) |
| nlohmann/json | JSON parse/serialize (auto-fetched by CMake if missing) |
| CMake 3.16+, C++20 | Build toolchain |
| Python 3.6+ stdlib only | filter_dataset.py, qwen_node_assistant.py |

---

## Build

```bash
# Unix/WSL
cd blender_scraper && bash build.sh

# Windows
build_msvc.bat
```

Output: `blender_scraper/build/blender_scraper` and `build_dataset`

---

## Data Flow Summary

```
GitHub API → json_files/{owner}/{repo}/**/*.json
           + .meta.json sidecars
           ↓
build_dataset → training_dataset.jsonl (all records)
           ↓
filter_dataset.py → blender_nodes_dataset.jsonl (curated, Blender-specific)
           ↓
Ready for OpenAI fine-tuning / Hugging Face datasets
```
