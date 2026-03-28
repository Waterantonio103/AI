#pragma once
#include "http_client.hpp"
#include "rate_limiter.hpp"
#include "token_rotator.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <set>
#include <unordered_set>
#include <functional>
#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <sstream>

namespace scraper {

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// Represents one file item discovered on GitHub
// ---------------------------------------------------------------------------
struct GitHubFile {
    std::string repo_full_name;   // "owner/repo"
    std::string path;             // path inside repo
    std::string sha;
    std::string download_url;
    std::string html_url;
    std::string raw_content;      // populated after download
};

// ---------------------------------------------------------------------------
// GitHub REST API v3 client
// Docs: https://docs.github.com/en/rest
//
// One instance per thread — HttpClient is not thread-safe.
// ---------------------------------------------------------------------------
class GitHubClient {
public:
    static constexpr int MAX_RETRIES = 5;

    explicit GitHubClient(TokenRotator& rotator, RateLimiter& limiter)
        : rotator_(rotator), limiter_(limiter) {}

    // -----------------------------------------------------------------------
    // Search for Blender repositories in parallel.
    //
    // - Builds a full query list (topic + keyword × year-slice).
    // - Dispatches queries to `num_agents` worker threads via atomic index.
    // - Each agent owns its own HttpClient + RateLimiter (no sharing).
    // - Results are merged into `repos` under a mutex — zero overlap.
    // -----------------------------------------------------------------------
    void search_blender_repos(std::vector<std::string>& repos,
                              int max_repos  = 1000,
                              int num_agents = 4,
                              const std::string& query_group = "all") {

        // --- Build query list ---
        const std::vector<std::string> topic_queries = {
            "topic:blender",
            "topic:blender-addon",
            "topic:blender3d",
            "topic:blenderassets",
            "topic:geometry-nodes",
            "topic:shader-nodes",
            "topic:cycles",
            "topic:eevee",
            "topic:blender-python",
            "topic:node-editor",
        };
        const std::vector<std::string> keyword_queries = {
            "blender nodes",
            "blender shader",
            "blender geometry nodes",
            "blender material",
            "blender addon",
            "blender python nodes",
            "ShaderNodeTree",
            "GeometryNodeTree",
            "bl_idname blender",
            "nodetree blender",
        };
        const std::vector<std::pair<std::string,std::string>> years = {
            {"2008-01-01","2014-12-31"},
            {"2015-01-01","2017-12-31"},
            {"2018-01-01","2019-12-31"},
            {"2020-01-01","2020-12-31"},
            {"2021-01-01","2021-12-31"},
            {"2022-01-01","2022-12-31"},
            {"2023-01-01","2023-12-31"},
            {"2024-01-01","2024-12-31"},
            {"2025-01-01","2026-12-31"},
        };

        // --- Split year ranges into thirds for matrix parallelism ---
        // keywords-early: indices 0-2   (2008-2019)
        // keywords-mid:   indices 3-5   (2020-2022)
        // keywords-late:  indices 6-8   (2023-2026)
        std::vector<std::string> all_queries;
        bool include_topics   = (query_group == "all" || query_group == "topics");
        bool include_kw_early = (query_group == "all" || query_group == "keywords-early");
        bool include_kw_mid   = (query_group == "all" || query_group == "keywords-mid");
        bool include_kw_late  = (query_group == "all" || query_group == "keywords-late");

        if (include_topics)
            for (auto& q : topic_queries)
                all_queries.push_back(q);

        for (auto& kw : keyword_queries) {
            for (int yi = 0; yi < (int)years.size(); ++yi) {
                bool include = (yi <= 2 && include_kw_early)
                            || (yi >= 3 && yi <= 5 && include_kw_mid)
                            || (yi >= 6 && include_kw_late);
                if (include)
                    all_queries.push_back(kw + " created:" + years[yi].first
                                          + ".." + years[yi].second);
            }
        }

        int total_q = (int)all_queries.size();
        std::cout << "[search] " << total_q << " queries across "
                  << num_agents << " parallel agents\n";

        // --- Shared state (mutex-protected) ---
        std::mutex         shared_mu;
        std::set<std::string> seen(repos.begin(), repos.end());
        std::atomic<int>   query_idx{0};   // next query to claim
        std::atomic<int>   completed_q{0};
        std::atomic<bool>  hard_stop{false};

        // --- Worker lambda — each thread runs this ---
        auto worker = [&](int agent_id) {
            // Each agent gets its own HTTP client and rate limiter
            RateLimiter  local_lim(2); // 2 req/s per agent — safe under 30/min
            GitHubClient local_gh(rotator_, local_lim);

            while (!hard_stop.load()) {
                int idx = query_idx.fetch_add(1);
                if (idx >= total_q) break;

                {
                    std::lock_guard<std::mutex> lk(shared_mu);
                    if ((int)repos.size() >= max_repos) { hard_stop = true; break; }
                }

                const std::string& q = all_queries[idx];
                int done = completed_q.load();
                std::cout << "[agent " << agent_id << "] Query "
                          << (done + 1) << "/" << total_q << ": " << q << "\n";

                // Run this query, collect into a local buffer
                std::vector<std::string> local_results;
                try {
                    local_gh.run_repo_search_query(q, local_results, max_repos);
                } catch (const TokensExhaustedException&) {
                    hard_stop = true;
                    break; // never throw from a worker thread — std::terminate kills the process
                           // hard_stop signals the main thread to throw after all workers join
                }

                // Merge into shared repos — deduplicate under lock
                {
                    std::lock_guard<std::mutex> lk(shared_mu);
                    for (auto& r : local_results) {
                        if ((int)repos.size() >= max_repos) break;
                        if (seen.insert(r).second)
                            repos.push_back(r);
                    }
                    ++completed_q;
                    std::cout << "[search] " << repos.size()
                              << " unique repos ("
                              << completed_q.load() << "/" << total_q
                              << " queries done)\n";
                }
            }
        };

        // --- Launch agents ---
        int agents = std::min(num_agents, total_q);
        std::vector<std::thread> threads;
        threads.reserve(agents);
        for (int i = 0; i < agents; ++i)
            threads.emplace_back(worker, i);

        // Collect exceptions from threads
        // (TokensExhaustedException needs to propagate to the caller)
        bool tokens_exhausted = false;
        for (auto& t : threads) {
            // threads::join() can't propagate exceptions directly,
            // so we use the hard_stop flag + re-check after join
            t.join();
        }
        if (hard_stop.load() && (int)repos.size() < max_repos) {
            // Only throw if we stopped due to token exhaustion, not max_repos cap
            throw TokensExhaustedException(0);
        }
    }

    // Single query against /search/repositories — fills local_results.
    // Called by worker threads. Thread-safe (uses its own http_).
    void run_repo_search_query(const std::string& query,
                               std::vector<std::string>& local_results,
                               int max_repos) {
        int page = 1;
        while ((int)local_results.size() < max_repos) {
            std::this_thread::sleep_for(std::chrono::seconds(3));

            std::ostringstream url;
            url << "/search/repositories?q=" << url_encode(query)
                << "&per_page=100&page=" << page
                << "&sort=updated&order=desc";

            auto resp = search_get_with_retry(url.str());
            if (!resp.ok()) break;

            json body;
            try { body = resp.json_body(); } catch (...) { break; }
            if (!body.contains("items") || body["items"].empty()) break;

            for (const auto& item : body["items"]) {
                std::string full = item.value("full_name", "");
                if (!full.empty())
                    local_results.push_back(full);
            }

            int total = body.value("total_count", 0);
            if (page * 100 >= total || page >= 10) break;
            ++page;
        }
    }

    // -----------------------------------------------------------------------
    // For a given repo, find all .json files and return their metadata.
    // Uses the Git Trees API (recursive) — one request per repo.
    // -----------------------------------------------------------------------
    std::vector<GitHubFile> list_json_files(const std::string& repo) {
        std::vector<GitHubFile> files;

        // Get default branch name
        auto repo_resp = api_get("/repos/" + repo);
        if (!repo_resp.ok()) return files;
        std::string branch;
        try {
            branch = repo_resp.json_body().value("default_branch", "main");
        } catch (...) {
            branch = "main";
        }

        // Fetch entire file tree recursively (single API call)
        auto tree_resp = api_get("/repos/" + repo
                                 + "/git/trees/" + branch + "?recursive=1");
        if (!tree_resp.ok()) return files;

        json tree;
        try { tree = tree_resp.json_body(); } catch (...) { return files; }
        if (!tree.contains("tree")) return files;

        for (const auto& node : tree["tree"]) {
            std::string path = node.value("path", "");
            // Match any .json file
            if (path.size() >= 5 && path.substr(path.size() - 5) == ".json") {
                GitHubFile f;
                f.repo_full_name = repo;
                f.path           = path;
                f.sha            = node.value("sha", "");
                f.download_url   = "https://raw.githubusercontent.com/"
                                   + repo + "/" + branch + "/" + url_encode(path);
                f.html_url       = "https://github.com/" + repo + "/blob/"
                                   + branch + "/" + path;
                files.push_back(std::move(f));
            }
        }
        return files;
    }

    // -----------------------------------------------------------------------
    // Download raw content of a file. Returns true on success.
    // -----------------------------------------------------------------------
    bool download_file(GitHubFile& file) {
        for (int attempt = 0; attempt < MAX_RETRIES; ++attempt) {
            limiter_.acquire();
            try {
                auto resp = http_.get(file.download_url, auth_headers());
                if (resp.ok()) {
                    file.raw_content = resp.body;
                    return true;
                }
                if (resp.is_429() || resp.is_403()) {
                    limiter_.handle_rate_limit(resp.rate_limit_reset());
                    continue;
                }
                return false;
            } catch (const std::exception& e) {
                std::cerr << "[github] download error: " << e.what() << "\n";
                std::this_thread::sleep_for(std::chrono::seconds(1 << attempt));
            }
        }
        return false;
    }

    // -----------------------------------------------------------------------
    // Search GitHub code API, invoke callback for each result item.
    // Automatically paginates up to max_results (GitHub caps at 1000).
    // -----------------------------------------------------------------------
    void search_code(const std::string& query,
                     std::function<void(const json&)> callback,
                     int max_results = 1000) {
        int page    = 1;
        int fetched = 0;

        while (fetched < max_results) {
            // Search API hard limit: 30 req/min per token.
            // 4-second gap = 15 req/min — half the limit, safe margin.
            std::this_thread::sleep_for(std::chrono::seconds(4));

            std::ostringstream url;
            url << "/search/code?q=" << url_encode(query)
                << "&per_page=100&page=" << page;

            auto resp = search_get_with_retry(url.str());
            if (!resp.ok()) break;

            json body;
            try { body = resp.json_body(); } catch (...) { break; }
            if (!body.contains("items") || body["items"].empty()) break;

            for (const auto& item : body["items"]) {
                if (fetched >= max_results) break;
                callback(item);
                ++fetched;
            }

            int total = body.value("total_count", 0);
            if (fetched >= total || fetched >= 1000) break;
            ++page;
        }
    }

private:
    TokenRotator& rotator_;
    RateLimiter&  limiter_;
    HttpClient    http_;


    std::unordered_map<std::string, std::string> auth_headers() const {
        return {
            {"Authorization",        "Bearer " + rotator_.current()},
            {"Accept",               "application/vnd.github+json"},
            {"User-Agent",           "blender-json-scraper/1.0"},
            {"X-GitHub-Api-Version", "2022-11-28"}
        };
    }

    HttpResponse api_get(const std::string& path) {
        return api_get_with_retry(path);
    }

    // Search API variant: if the reset is < 90s away, just sleep it rather
    // than rotating tokens (per-minute limits reset quickly).
    HttpResponse search_get_with_retry(const std::string& path) {
        for (int attempt = 0; attempt < MAX_RETRIES; ++attempt) {
            limiter_.acquire();
            try {
                auto resp = http_.get("https://api.github.com" + path, auth_headers());

                if (resp.rate_limit_remaining() >= 0)
                    rotator_.update_remaining(resp.rate_limit_remaining(),
                                              resp.rate_limit_reset());

                if (resp.ok()) return resp;

                // Only act on actual error responses — never on successful 200s
                if (resp.is_429() || resp.is_403()) {
                    long reset = resp.rate_limit_reset();
                    long now   = static_cast<long>(std::chrono::duration_cast<std::chrono::seconds>(
                                     std::chrono::system_clock::now().time_since_epoch()).count());
                    long wait  = reset - now + 2;

                    std::cerr << "[search] Rate limited (HTTP " << resp.status_code
                              << "), reset in " << wait << "s\n";

                    if (wait > 0 && wait <= 120) {
                        // Short reset — sleep and retry with same token
                        std::this_thread::sleep_for(std::chrono::seconds(wait));
                    } else {
                        // Long reset — rotate to next token
                        rotator_.on_rate_limited(reset);
                    }
                    continue;
                }
                if (resp.status_code == 422) {
                    std::cerr << "[github] unsupported search query (422): " << path << "\n";
                    return resp;
                }
                std::cerr << "[github] HTTP " << resp.status_code << ": " << path << "\n";
                return resp;
            } catch (const TokensExhaustedException&) {
                throw; // must propagate to discover_repos — do not swallow
            } catch (const std::exception& e) {
                std::cerr << "[github] search error: " << e.what() << "\n";
                std::this_thread::sleep_for(std::chrono::seconds(1 << attempt));
            }
        }
        return {};
    }

    HttpResponse api_get_with_retry(const std::string& path) {
        for (int attempt = 0; attempt < MAX_RETRIES; ++attempt) {
            limiter_.acquire();
            try {
                auto resp = http_.get("https://api.github.com" + path, auth_headers());

                // Update rotator with current rate-limit state
                if (resp.rate_limit_remaining() >= 0)
                    rotator_.update_remaining(resp.rate_limit_remaining(),
                                              resp.rate_limit_reset());

                if (resp.ok()) return resp;

                // Only rotate/sleep on actual error responses
                if (resp.is_429() || resp.is_403()) {
                    rotator_.on_rate_limited(resp.rate_limit_reset());
                    continue;
                }
                if (resp.status_code == 422) {
                    std::cerr << "[github] unsupported query (422): " << path << "\n";
                    return resp;
                }
                std::cerr << "[github] HTTP " << resp.status_code << ": " << path << "\n";
                return resp;
            } catch (const TokensExhaustedException&) {
                throw;
            } catch (const std::exception& e) {
                std::cerr << "[github] request error: " << e.what() << "\n";
                std::this_thread::sleep_for(std::chrono::seconds(1 << attempt));
            }
        }
        return {};
    }

    // Percent-encode a string for URL query parameters
    static std::string url_encode(const std::string& s) {
        std::ostringstream out;
        for (unsigned char c : s) {
            if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
                out << c;
            } else if (c == ' ') {
                out << '+';
            } else {
                out << '%'
                    << std::hex << std::uppercase
                    << ((c >> 4) & 0xF)
                    << (c & 0xF);
            }
        }
        return out.str();
    }
};

} // namespace scraper
