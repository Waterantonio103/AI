#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS
#include "scraper.hpp"
#include <nlohmann/json.hpp>
#include <curl/curl.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <filesystem>

using json = nlohmann::json;
namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Print usage
// ---------------------------------------------------------------------------
static void print_usage(const char* prog) {
    std::cout << R"(
Blender JSON Scraper — collect .json files from Blender GitHub repositories
for AI model training.

Usage:
  )" << prog << R"( [options]

Options:
  --config <file>     Path to config.json (default: config.json)
  --github-token <t>  GitHub Personal Access Token (overrides config)
  --apify-token <t>   Apify API token (overrides config, enables Apify mode)
  --output <dir>      Output directory (default: output)
  --max-repos <n>     Maximum repositories to scan (default: 500)
  --threads <n>       Download threads (default: 8)
  --rps <n>           GitHub API requests per second (default: 5)
  --no-validate       Skip JSON validation (save all files including malformed)
  --query-group <g>   Limit Phase 1 to a query slice (for GitHub Actions matrix):
                        all             (default) — run all 100 queries
                        topics          — 10 topic: queries only
                        keywords-early  — keyword × 2008-2019
                        keywords-mid    — keyword × 2020-2022
                        keywords-late   — keyword × 2023-2026
  --help              Show this help

Config file (config.json) format:
  {
    "github_token": "ghp_...",
    "apify_token":  "apify_...",
    "output_dir":   "output",
    "max_repos":    500,
    "threads":      8,
    "rps":          5,
    "use_apify":    false,
    "validate_json": true,
    "min_json_size": 10
  }

Environment variables (fallback):
  GITHUB_TOKEN   — GitHub PAT
  APIFY_TOKEN    — Apify API token

Output structure:
  output/
  ├── manifest.jsonl              — index of all discovered files
  ├── training_dataset.jsonl      — consolidated JSONL for AI training
  ├── json_files/                 — downloaded .json files (structured)
  │   └── <owner>/<repo>/<path>
  └── metadata/                   — per-scrape run metadata

Notes:
  • Requires a GitHub PAT with at least public_repo read scope.
  • With a free GitHub account: 5000 API req/hour authenticated.
  • The Apify token is optional — enables web scraping for repos the
    GitHub Search API misses (large repos, rate-limited queries).
  • Training dataset uses JSONL format (one JSON object per line),
    compatible with OpenAI fine-tuning and Hugging Face datasets.
)";
}

// ---------------------------------------------------------------------------
// Load config from JSON file
// ---------------------------------------------------------------------------
static scraper::Config load_config(const std::string& path) {
    scraper::Config cfg;
    if (fs::exists(path)) {
        std::ifstream ifs(path);
        auto j = json::parse(ifs);
        // Accept either a single "github_token" string or a "github_tokens" array
        if (j.contains("github_tokens") && j["github_tokens"].is_array()) {
            for (auto& t : j["github_tokens"])
                cfg.github_tokens.push_back(t.get<std::string>());
        } else if (j.contains("github_token")) {
            cfg.github_tokens.push_back(j["github_token"].get<std::string>());
        }
        if (j.contains("apify_token"))   cfg.apify_token   = j["apify_token"];
        if (j.contains("output_dir"))    cfg.output_dir    = j["output_dir"];
        if (j.contains("max_repos"))     cfg.max_repos     = j["max_repos"];
        if (j.contains("threads"))       cfg.threads       = j["threads"];
        if (j.contains("rps"))            cfg.rps            = j["rps"];
        if (j.contains("search_agents")) cfg.search_agents  = j["search_agents"];
        if (j.contains("use_apify"))     cfg.use_apify     = j["use_apify"];
        if (j.contains("validate_json")) cfg.validate_json = j["validate_json"];
        if (j.contains("min_json_size")) cfg.min_json_size = j["min_json_size"];
        if (j.contains("query_group"))   cfg.query_group   = j["query_group"];
    }
    // Environment variable fallbacks
    if (cfg.github_tokens.empty()) {
        const char* env = std::getenv("GITHUB_TOKEN");
        if (env) cfg.github_tokens.push_back(env);
    }
    if (cfg.apify_token.empty()) {
        const char* env = std::getenv("APIFY_TOKEN");
        if (env) { cfg.apify_token = env; cfg.use_apify = true; }
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// Write an example config.json if it doesn't exist
// ---------------------------------------------------------------------------
static void write_example_config(const std::string& path) {
    if (fs::exists(path)) return;
    json example = {
        {"github_tokens", {"ghp_TOKEN_1", "ghp_TOKEN_2", "ghp_TOKEN_3"}},
        {"apify_token",   ""},
        {"output_dir",    "output"},
        {"max_repos",     500},
        {"threads",       8},
        {"rps",           5},
        {"use_apify",     false},
        {"validate_json", true},
        {"min_json_size", 10}
    };
    std::ofstream ofs(path);
    ofs << example.dump(2) << "\n";
    std::cout << "[info] Created example config: " << path << "\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    // Initialize curl globally (once per process)
    curl_global_init(CURL_GLOBAL_DEFAULT);
    struct CurlCleanup { ~CurlCleanup() { curl_global_cleanup(); } } _cleanup;

    std::string config_path = "config.json";
    bool show_help = false;

    // --- Parse CLI arguments ---
    scraper::Config override_cfg;
    bool has_override = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            show_help = true;
        } else if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--github-token" && i + 1 < argc) {
            // Can be specified multiple times to add tokens to the rotation pool
            override_cfg.github_tokens.push_back(argv[++i]); has_override = true;
        } else if (arg == "--apify-token" && i + 1 < argc) {
            override_cfg.apify_token = argv[++i];
            override_cfg.use_apify   = true; has_override = true;
        } else if (arg == "--output" && i + 1 < argc) {
            override_cfg.output_dir = argv[++i]; has_override = true;
        } else if (arg == "--max-repos" && i + 1 < argc) {
            override_cfg.max_repos = std::stoi(argv[++i]); has_override = true;
        } else if (arg == "--threads" && i + 1 < argc) {
            override_cfg.threads = std::stoi(argv[++i]); has_override = true;
        } else if (arg == "--rps" && i + 1 < argc) {
            override_cfg.rps = std::stoi(argv[++i]); has_override = true;
        } else if (arg == "--no-validate") {
            override_cfg.validate_json = false; has_override = true;
        } else if (arg == "--query-group" && i + 1 < argc) {
            override_cfg.query_group = argv[++i]; has_override = true;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            show_help = true;
        }
    }

    if (show_help) { print_usage(argv[0]); return 0; }

    write_example_config(config_path);

    // Load base config then apply CLI overrides
    auto cfg = load_config(config_path);
    if (has_override) {
        if (!override_cfg.github_tokens.empty()) {
            // CLI tokens replace config tokens entirely
            cfg.github_tokens = override_cfg.github_tokens;
        }
        if (!override_cfg.apify_token.empty())  { cfg.apify_token = override_cfg.apify_token; cfg.use_apify = true; }
        if (!override_cfg.output_dir.empty() && override_cfg.output_dir != "output")
            cfg.output_dir = override_cfg.output_dir;
        if (override_cfg.max_repos != 500) cfg.max_repos     = override_cfg.max_repos;
        if (override_cfg.threads   != 8)   cfg.threads       = override_cfg.threads;
        if (override_cfg.rps       != 5)   cfg.rps           = override_cfg.rps;
        if (!override_cfg.validate_json)   cfg.validate_json = false;
        if (!override_cfg.query_group.empty() && override_cfg.query_group != "all")
            cfg.query_group = override_cfg.query_group;
    }

    // Remove placeholder tokens
    cfg.github_tokens.erase(
        std::remove_if(cfg.github_tokens.begin(), cfg.github_tokens.end(),
            [](const std::string& t) {
                return t.empty() || t.find("YOUR_TOKEN") != std::string::npos;
            }),
        cfg.github_tokens.end());

    // --- Validate required config ---
    if (cfg.github_tokens.empty()) {
        std::cerr << "\n[error] At least one GitHub token is required.\n"
                  << "  Option 1 — config.json:  \"github_tokens\": [\"ghp_...\", \"ghp_...\"]\n"
                  << "  Option 2 — CLI:          --github-token ghp_1 --github-token ghp_2\n"
                  << "  Option 3 — env var:      set GITHUB_TOKEN=ghp_...\n"
                  << "  Create tokens at: https://github.com/settings/tokens\n\n";
        return 1;
    }

    // --- Print effective config ---
    std::cout << "\n=== Blender JSON Scraper ===\n"
              << "  Output dir:  " << cfg.output_dir    << "\n"
              << "  Tokens:         " << cfg.github_tokens.size() << " (rotates on rate-limit)\n"
              << "  Max repos:      " << cfg.max_repos      << "\n"
              << "  Search agents:  " << cfg.search_agents  << " (parallel Phase 1)\n"
              << "  DL threads:     " << cfg.threads        << "\n"
              << "  API rps:     " << cfg.rps           << "\n"
              << "  Apify:       " << (cfg.use_apify ? "enabled" : "disabled") << "\n"
              << "  Validate:    " << (cfg.validate_json ? "yes" : "no") << "\n"
              << "\n";

    try {
        scraper::BlenderJsonScraper s(cfg);
        s.run();
    } catch (const std::exception& e) {
        std::cerr << "[fatal] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
