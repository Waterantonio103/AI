#pragma once
#define NOMINMAX  // prevent Windows.h min/max macros (pulled in by curl headers)
#include "http_client.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <thread>
#include <chrono>
#include <functional>

namespace scraper {

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// Apify REST API client
// Docs: https://docs.apify.com/api/v2
//
// This client:
//  1. Starts an Apify actor run (e.g. "apify/github-scraper")
//  2. Polls until the run finishes
//  3. Downloads items from the run's default dataset
// ---------------------------------------------------------------------------
class ApifyClient {
public:
    static constexpr const char* BASE_URL    = "https://api.apify.com/v2";
    // Apify actor IDs for GitHub-related scraping
    static constexpr const char* GITHUB_ACTOR       = "apify/github-scraper";
    static constexpr const char* WEB_SCRAPER_ACTOR  = "apify/web-scraper";

    explicit ApifyClient(const std::string& api_token)
        : token_(api_token) {}

    // -----------------------------------------------------------------------
    // Run the GitHub scraper actor for a list of repositories and return
    // all dataset items (parsed JSON).
    //
    // repos     — list of "owner/repo" strings
    // file_exts — filter to these extensions (e.g. {".json"})
    // -----------------------------------------------------------------------
    std::vector<json> scrape_repos_for_json(
        const std::vector<std::string>& repos,
        const std::vector<std::string>& file_exts = {".json"})
    {
        // Build actor input matching apify/github-scraper schema
        json input;
        input["searchType"] = "repositories";
        input["maxItems"]   = 50000;
        // Provide start URLs as repo pages
        json start_urls = json::array();
        for (auto& repo : repos) {
            start_urls.push_back({{"url", "https://github.com/" + repo}});
        }
        input["startUrls"] = start_urls;

        // Build a glob pattern to target JSON files
        json globs = json::array();
        for (auto& ext : file_exts)
            globs.push_back({{"glob", "**/*" + ext}});
        input["globs"] = globs;

        return run_actor_and_collect(GITHUB_ACTOR, input);
    }

    // -----------------------------------------------------------------------
    // Run the generic web scraper to pull .json file links from GitHub
    // search result pages (useful when the API is rate-limited).
    //
    // search_query — GitHub search query, e.g. "extension:blend topic:blender"
    // -----------------------------------------------------------------------
    std::vector<json> scrape_github_search(const std::string& search_query,
                                           int max_pages = 10) {
        std::string search_url = "https://github.com/search?type=repositories&q="
                               + url_encode(search_query);

        json input;
        input["startUrls"]          = json::array({{{"url", search_url}}});
        input["maxRequestsPerCrawl"] = max_pages * 30;  // repos per page ≈ 30
        input["pageFunction"]        = R"JS(
async function pageFunction(context) {
    const { $, request, log } = context;
    const results = [];
    // On a search results page, grab each repo link
    $('a[href*="/blob/"], a[href$=".json"]').each((i, el) => {
        const href = $(el).attr('href');
        if (href && href.endsWith('.json')) {
            results.push({ url: 'https://github.com' + href, source: request.url });
        }
    });
    // On a repo page, list JSON files
    $('a.js-navigation-open[href*=".json"]').each((i, el) => {
        const href = $(el).attr('href');
        if (href) results.push({ url: 'https://github.com' + href, source: request.url });
    });
    return results;
}
)JS";

        return run_actor_and_collect(WEB_SCRAPER_ACTOR, input);
    }

    // -----------------------------------------------------------------------
    // Generic: start any actor, wait for completion, return dataset items.
    // -----------------------------------------------------------------------
    std::vector<json> run_actor_and_collect(const std::string& actor_id,
                                            const json& input,
                                            int timeout_sec = 3600) {
        // 1. Start the run
        std::string run_id = start_actor_run(actor_id, input);
        if (run_id.empty()) {
            std::cerr << "[apify] Failed to start actor " << actor_id << "\n";
            return {};
        }
        std::cout << "[apify] Started actor run " << run_id << "\n";

        // 2. Poll until finished
        std::string dataset_id = wait_for_run(run_id, timeout_sec);
        if (dataset_id.empty()) {
            std::cerr << "[apify] Run did not finish in time\n";
            return {};
        }

        // 3. Download dataset
        return download_dataset(dataset_id);
    }

    // -----------------------------------------------------------------------
    // Download all items from a named dataset
    // -----------------------------------------------------------------------
    std::vector<json> download_dataset(const std::string& dataset_id,
                                       int batch_size = 1000) {
        std::vector<json> all_items;
        int offset = 0;
        while (true) {
            std::string url = std::string(BASE_URL) + "/datasets/"
                            + dataset_id + "/items?format=json"
                            + "&offset=" + std::to_string(offset)
                            + "&limit=" + std::to_string(batch_size)
                            + "&clean=true";

            auto resp = http_.get(url, auth_headers());
            if (!resp.ok()) {
                std::cerr << "[apify] Dataset fetch failed: " << resp.status_code << "\n";
                break;
            }

            auto batch = resp.json_body();
            if (!batch.is_array() || batch.empty()) break;

            for (auto& item : batch) all_items.push_back(item);
            if ((int)batch.size() < batch_size) break;
            offset += batch_size;
        }
        std::cout << "[apify] Downloaded " << all_items.size() << " dataset items\n";
        return all_items;
    }

    // -----------------------------------------------------------------------
    // List all named datasets (useful to resume interrupted runs)
    // -----------------------------------------------------------------------
    std::vector<json> list_datasets() {
        auto resp = http_.get(std::string(BASE_URL) + "/datasets?token=" + token_, {});
        if (!resp.ok()) return {};
        auto body = resp.json_body();
        if (body.contains("data") && body["data"].contains("items"))
            return body["data"]["items"].get<std::vector<json>>();
        return {};
    }

private:
    std::string token_;
    HttpClient  http_;

    std::unordered_map<std::string,std::string> auth_headers() const {
        return {
            {"Authorization", "Bearer " + token_},
            {"Content-Type",  "application/json"}
        };
    }

    // Start an actor run, return the runId
    std::string start_actor_run(const std::string& actor_id, const json& input) {
        std::string url = std::string(BASE_URL) + "/acts/" + actor_id + "/runs?token=" + token_;
        try {
            auto resp = http_.post(url, input, {});
            if (!resp.ok()) {
                std::cerr << "[apify] start_actor HTTP " << resp.status_code
                          << ": " << resp.body.substr(0, 200) << "\n";
                return "";
            }
            auto body = resp.json_body();
            return body["data"].value("id", "");
        } catch (const std::exception& e) {
            std::cerr << "[apify] start_actor error: " << e.what() << "\n";
            return "";
        }
    }

    // Poll run status until SUCCEEDED/FAILED, return defaultDatasetId
    std::string wait_for_run(const std::string& run_id, int timeout_sec) {
        auto deadline = std::chrono::steady_clock::now()
                      + std::chrono::seconds(timeout_sec);
        int poll_interval = 5;

        while (std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::seconds(poll_interval));
            poll_interval = poll_interval * 2 < 60 ? poll_interval * 2 : 60;

            std::string url = std::string(BASE_URL) + "/actor-runs/" + run_id
                            + "?token=" + token_;
            try {
                auto resp = http_.get(url, {});
                if (!resp.ok()) continue;
                auto body  = resp.json_body();
                std::string status = body["data"].value("status", "");
                std::cout << "[apify] Run " << run_id << " status: " << status << "\n";

                if (status == "SUCCEEDED")
                    return body["data"].value("defaultDatasetId", "");
                if (status == "FAILED" || status == "ABORTED" || status == "TIMED-OUT") {
                    std::cerr << "[apify] Run ended with status: " << status << "\n";
                    return "";
                }
            } catch (const std::exception& e) {
                std::cerr << "[apify] poll error: " << e.what() << "\n";
            }
        }
        return "";
    }

    static std::string url_encode(const std::string& s) {
        std::ostringstream out;
        for (unsigned char c : s) {
            if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~')
                out << c;
            else if (c == ' ')
                out << '+';
            else
                out << '%' << std::hex << std::uppercase
                    << ((c >> 4) & 0xF) << (c & 0xF);
        }
        return out.str();
    }
};

} // namespace scraper
