#pragma once
#include "github_client.hpp"
#include "apify_client.hpp"
#include "rate_limiter.hpp"
#include "token_rotator.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <sstream>
#include <set>

namespace scraper {

namespace fs = std::filesystem;
using json   = nlohmann::json;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
struct Config {
    std::vector<std::string> github_tokens;  // One or more PATs (rotated on rate-limit)
    std::string apify_token;                 // Optional: Apify API token
    std::string output_dir     = "output";
    int         max_repos      = 500;
    int         threads        = 8;          // Phase 2/3 download threads
    int         search_agents  = 4;          // Phase 1 parallel search agents
    int         rps            = 5;
    bool        use_apify      = false;
    bool        validate_json  = true;
    int         min_json_size  = 10;
    // Query group for GitHub Actions matrix parallelism:
    //   "all" (default), "topics", "keywords-early", "keywords-mid", "keywords-late"
    std::string query_group    = "all";
};

// ---------------------------------------------------------------------------
// Simple thread pool
// ---------------------------------------------------------------------------
class ThreadPool {
public:
    explicit ThreadPool(int num_threads) {
        for (int i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mu_);
                        cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                        if (stop_ && tasks_.empty()) return;
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                }
            });
        }
    }

    template<class F>
    void enqueue(F&& f) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            tasks_.emplace(std::forward<F>(f));
        }
        cv_.notify_one();
    }

    void wait_all() {
        // Spin until all tasks are done — simple approach
        while (true) {
            {
                std::lock_guard<std::mutex> lock(mu_);
                if (tasks_.empty()) break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& w : workers_) if (w.joinable()) w.join();
    }

private:
    std::vector<std::thread>          workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex                        mu_;
    std::condition_variable           cv_;
    bool                              stop_{false};
};

// ---------------------------------------------------------------------------
// Statistics tracker
// ---------------------------------------------------------------------------
struct Stats {
    std::atomic<int> repos_found{0};
    std::atomic<int> repos_scanned{0};
    std::atomic<int> json_files_found{0};
    std::atomic<int> json_files_saved{0};
    std::atomic<int> json_files_skipped{0};
    std::atomic<long> bytes_saved{0};

    void print() const {
        std::cout << "\n=== Scrape Complete ===\n"
                  << "  Repos found:        " << repos_found    << "\n"
                  << "  Repos scanned:      " << repos_scanned  << "\n"
                  << "  JSON files found:   " << json_files_found<< "\n"
                  << "  JSON files saved:   " << json_files_saved<< "\n"
                  << "  JSON files skipped: " << json_files_skipped << "\n"
                  << "  Total bytes saved:  " << bytes_saved     << "\n";
    }
};

// ---------------------------------------------------------------------------
// Main orchestrator
// ---------------------------------------------------------------------------
class BlenderJsonScraper {
public:
    explicit BlenderJsonScraper(Config cfg)
        : cfg_(std::move(cfg))
        , limiter_(cfg_.rps)
        , rotator_(cfg_.github_tokens)
        , github_(rotator_, limiter_)
        , apify_(cfg_.apify_token)
    {
        fs::create_directories(cfg_.output_dir);
        fs::create_directories(cfg_.output_dir + "/json_files");
        fs::create_directories(cfg_.output_dir + "/metadata");
    }

    // -----------------------------------------------------------------------
    // Full pipeline:
    //   1. Discover Blender repos on GitHub
    //   2. For each repo, list all .json files
    //   3. Download and validate each .json file
    //   4. Save with metadata (repo, path, stars, etc.)
    // -----------------------------------------------------------------------
    void run() {
        auto t_start = std::chrono::steady_clock::now();

        // --- Phase 1: Discover repos ---
        std::cout << "[scraper] Phase 1: Searching for Blender repositories...\n";
        auto repos = discover_repos();
        stats_.repos_found = (int)repos.size();
        std::cout << "[scraper] Found " << repos.size() << " unique repos\n";

        // --- Phase 2: Enumerate JSON files per repo ---
        std::cout << "[scraper] Phase 2: Enumerating JSON files...\n";
        std::vector<GitHubFile> all_files = enumerate_json_files(repos);
        stats_.json_files_found = (int)all_files.size();
        std::cout << "[scraper] Found " << all_files.size() << " .json files total\n";

        // Save file manifest before downloading (checkpoint)
        save_manifest(all_files);

        // --- Phase 3: Download in parallel ---
        std::cout << "[scraper] Phase 3: Downloading with " << cfg_.threads << " threads...\n";
        download_all(all_files);

        // --- Phase 4: Build consolidated training dataset ---
        std::cout << "[scraper] Phase 4: Building training dataset...\n";
        build_dataset();

        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - t_start).count();
        std::cout << "[scraper] Total time: " << elapsed << "s\n";
        stats_.print();
    }

private:
    Config        cfg_;
    RateLimiter   limiter_;
    TokenRotator  rotator_;
    GitHubClient  github_;
    ApifyClient   apify_;
    Stats         stats_;
    std::mutex    io_mu_;

    // -----------------------------------------------------------------------
    // Phase 1: Repo discovery — GitHub API + optional Apify fallback
    // -----------------------------------------------------------------------
    std::vector<std::string> discover_repos() {
        std::vector<std::string> repos;
        try {
            github_.search_blender_repos(repos, cfg_.max_repos, cfg_.search_agents,
                                         cfg_.query_group);
        } catch (const TokensExhaustedException&) {
            std::cerr << "[scraper] Tokens exhausted during search — "
                      << "proceeding with " << repos.size()
                      << " repos found so far.\n";
        }

        // If Apify is enabled and we want more coverage, use it as a supplement
        if (cfg_.use_apify && !cfg_.apify_token.empty()
            && (int)repos.size() < cfg_.max_repos) {
            std::cout << "[scraper] Supplementing with Apify GitHub scraper...\n";
            auto apify_items = apify_.scrape_github_search("blender .blend file");
            std::set<std::string> seen(repos.begin(), repos.end());
            for (auto& item : apify_items) {
                std::string url = item.value("url", "");
                // Extract "owner/repo" from GitHub URLs
                auto repo = extract_repo_from_url(url);
                if (!repo.empty() && seen.find(repo) == seen.end()) {
                    seen.insert(repo);
                    repos.push_back(repo);
                }
            }
        }

        return repos;
    }

    // -----------------------------------------------------------------------
    // Phase 2: For each repo, list JSON files using GitHub Trees API
    // -----------------------------------------------------------------------
    std::vector<GitHubFile> enumerate_json_files(const std::vector<std::string>& repos) {
        std::vector<GitHubFile> all_files;
        std::mutex              files_mu;

        // Process repos in a thread pool
        ThreadPool pool(cfg_.threads);
        std::atomic<int> done{0};
        int total = (int)repos.size();

        for (auto& repo : repos) {
            pool.enqueue([&, repo]() {
                // Each thread needs its own HttpClient (curl handle is not thread-safe)
                RateLimiter  local_limiter(cfg_.rps / cfg_.threads + 1);
                GitHubClient local_gh(rotator_, local_limiter);
                try {
                    auto files = local_gh.list_json_files(repo);
                    {
                        std::lock_guard<std::mutex> lock(files_mu);
                        for (auto& f : files) all_files.push_back(f);
                    }
                    stats_.repos_scanned++;
                } catch (const std::exception& e) {
                    std::cerr << "[scraper] repo " << repo << " error: " << e.what() << "\n";
                }
                int d = ++done;
                if (d % 10 == 0 || d == total)
                    std::cout << "[scraper] Enumerated " << d << "/" << total << " repos\n";
            });
        }
        pool.wait_all();
        return all_files;
    }

    // -----------------------------------------------------------------------
    // Phase 3: Parallel download
    // -----------------------------------------------------------------------
    void download_all(std::vector<GitHubFile>& files) {
        ThreadPool pool(cfg_.threads);
        std::atomic<int> done{0};
        int total = (int)files.size();

        for (auto& file : files) {
            pool.enqueue([&, &file = file]() {
                // Skip files already on disk from a previous run
                std::string out_path = cfg_.output_dir + "/json_files/"
                                     + sanitize_path(file.repo_full_name) + "/"
                                     + sanitize_path(file.path);
                if (fs::exists(out_path)) {
                    stats_.json_files_skipped++;
                    ++done;
                    return;
                }

                // Threads share the rotator (thread-safe) but each has its own
                // HttpClient and rate limiter to avoid curl handle contention.
                RateLimiter  local_limiter(cfg_.rps / cfg_.threads + 1);
                GitHubClient local_gh(rotator_, local_limiter);

                bool ok = local_gh.download_file(file);
                if (ok) {
                    if (save_json_file(file)) {
                        stats_.json_files_saved++;
                        stats_.bytes_saved += (long)file.raw_content.size();
                    } else {
                        stats_.json_files_skipped++;
                    }
                } else {
                    stats_.json_files_skipped++;
                }

                int d = ++done;
                if (d % 500 == 0 || d == total)
                    std::cout << "[scraper] Processed " << d << "/" << total
                              << " files (" << stats_.json_files_saved << " new, "
                              << stats_.json_files_skipped << " skipped)\n";
            });
        }
        pool.wait_all();
    }

    // -----------------------------------------------------------------------
    // Save a single .json file to disk with sidecar metadata
    // -----------------------------------------------------------------------
    bool save_json_file(const GitHubFile& file) {
        // Skip tiny files
        if ((int)file.raw_content.size() < cfg_.min_json_size)
            return false;

        // Validate JSON if requested
        if (cfg_.validate_json) {
            try {
                auto parsed = json::parse(file.raw_content);
                (void)parsed;
            } catch (...) {
                return false; // malformed
            }
        }

        // Build output path: output/json_files/<owner>/<repo>/<path...>
        std::string safe_path = cfg_.output_dir + "/json_files/"
                              + sanitize_path(file.repo_full_name) + "/"
                              + sanitize_path(file.path);
        fs::create_directories(fs::path(safe_path).parent_path());

        {
            std::ofstream ofs(safe_path, std::ios::binary);
            if (!ofs) return false;
            ofs.write(file.raw_content.data(), (std::streamsize)file.raw_content.size());
        }

        // Write sidecar metadata
        json meta;
        meta["repo"]         = file.repo_full_name;
        meta["path"]         = file.path;
        meta["sha"]          = file.sha;
        meta["html_url"]     = file.html_url;
        meta["size_bytes"]   = file.raw_content.size();
        meta["source"]       = "github";

        std::string meta_path = safe_path + ".meta.json";
        std::ofstream mofs(meta_path);
        mofs << meta.dump(2);

        return true;
    }

    // -----------------------------------------------------------------------
    // Phase 4: Build a single JSONL training file from all collected data
    // -----------------------------------------------------------------------
    void build_dataset() {
        std::string dataset_path = cfg_.output_dir + "/training_dataset.jsonl";
        std::ofstream ofs(dataset_path);
        if (!ofs) {
            std::cerr << "[scraper] Cannot create dataset file\n";
            return;
        }

        int records = 0;
        // Walk the output directory
        for (auto& entry : fs::recursive_directory_iterator(cfg_.output_dir + "/json_files")) {
            if (!entry.is_regular_file()) continue;
            auto ext = entry.path().extension().string();
            if (ext != ".json") continue;
            // Skip sidecar meta files
            if (entry.path().string().find(".meta.json") != std::string::npos) continue;

            // Find its sidecar
            std::string meta_path = entry.path().string() + ".meta.json";
            json meta = json::object(); // default to empty object, never null
            if (fs::exists(meta_path)) {
                std::ifstream mifs(meta_path);
                try {
                    json parsed = json::parse(mifs);
                    if (parsed.is_object()) meta = parsed;
                } catch (...) {}
            }

            // Read JSON content
            std::ifstream ifs(entry.path(), std::ios::binary);
            std::string content((std::istreambuf_iterator<char>(ifs)),
                                 std::istreambuf_iterator<char>());
            if (content.empty()) continue;
            json content_json;
            try { content_json = json::parse(content); } catch (...) { continue; }
            if (content_json.is_null()) continue;

            // Build training record
            json record;
            record["source"]   = "github";
            record["repo"]     = meta.contains("repo")     ? meta["repo"]     : "";
            record["path"]     = meta.contains("path")     ? meta["path"]     : "";
            record["html_url"] = meta.contains("html_url") ? meta["html_url"] : "";
            record["content"]  = content_json;

            ofs << record.dump() << "\n";
            ++records;
        }

        std::cout << "[scraper] Training dataset: " << dataset_path
                  << " (" << records << " records)\n";
    }

    // -----------------------------------------------------------------------
    // Save a manifest of all discovered files (checkpoint before download)
    // -----------------------------------------------------------------------
    void save_manifest(const std::vector<GitHubFile>& files) {
        std::string path = cfg_.output_dir + "/manifest.jsonl";
        std::ofstream ofs(path);
        for (auto& f : files) {
            json rec;
            rec["repo"]         = f.repo_full_name;
            rec["path"]         = f.path;
            rec["sha"]          = f.sha;
            rec["download_url"] = f.download_url;
            rec["html_url"]     = f.html_url;
            ofs << rec.dump() << "\n";
        }
        std::cout << "[scraper] Manifest saved: " << path << "\n";
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------
    static std::string sanitize_path(const std::string& s) {
        std::string out;
        for (char c : s) {
            if (c == '/' || c == '\\' || std::isalnum(c) || c == '.' || c == '-' || c == '_')
                out += c;
            else
                out += '_';
        }
        return out;
    }

    static std::string extract_repo_from_url(const std::string& url) {
        // https://github.com/owner/repo/...  ->  owner/repo
        const std::string prefix = "https://github.com/";
        if (url.substr(0, prefix.size()) != prefix) return "";
        std::string rest = url.substr(prefix.size());
        auto slash = rest.find('/');
        if (slash == std::string::npos) return "";
        auto owner = rest.substr(0, slash);
        rest = rest.substr(slash + 1);
        auto end = rest.find('/');
        std::string repo = (end == std::string::npos) ? rest : rest.substr(0, end);
        if (owner.empty() || repo.empty()) return "";
        return owner + "/" + repo;
    }
};

} // namespace scraper
