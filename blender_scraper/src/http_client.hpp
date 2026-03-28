#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <sstream>

namespace scraper {

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// Response from an HTTP request
// ---------------------------------------------------------------------------
struct HttpResponse {
    long        status_code{0};
    std::string body;
    std::unordered_map<std::string, std::string> headers;

    bool ok()     const { return status_code >= 200 && status_code < 300; }
    bool is_429() const { return status_code == 429; }
    bool is_403() const { return status_code == 403; }

    json json_body() const { return json::parse(body); }

    // GitHub rate-limit reset epoch from response headers
    long rate_limit_reset() const {
        auto it = headers.find("x-ratelimit-reset");
        if (it != headers.end()) return std::stol(it->second);
        return 0;
    }
    int rate_limit_remaining() const {
        auto it = headers.find("x-ratelimit-remaining");
        if (it != headers.end()) return std::stoi(it->second);
        return -1;
    }
};

// ---------------------------------------------------------------------------
// Minimal libcurl RAII wrapper — NOT thread-safe per instance.
// Create one per thread.
// ---------------------------------------------------------------------------
class HttpClient {
public:
    HttpClient() {
        curl_ = curl_easy_init();
        if (!curl_) throw std::runtime_error("curl_easy_init failed");
    }
    ~HttpClient() { if (curl_) curl_easy_cleanup(curl_); }

    HttpClient(const HttpClient&)            = delete;
    HttpClient& operator=(const HttpClient&) = delete;

    // Perform a GET. Throws on libcurl error.
    HttpResponse get(const std::string& url,
                     const std::unordered_map<std::string,std::string>& headers = {}) {
        return request("GET", url, headers, "");
    }

    // Perform a POST with JSON body.
    HttpResponse post(const std::string& url,
                      const json& body,
                      const std::unordered_map<std::string,std::string>& extra_headers = {}) {
        std::string body_str = body.dump();
        auto hdrs = extra_headers;
        hdrs["Content-Type"] = "application/json";
        return request("POST", url, hdrs, body_str);
    }

private:
    CURL* curl_{nullptr};

    static size_t write_cb(char* ptr, size_t size, size_t nmemb, void* userdata) {
        auto* s = static_cast<std::string*>(userdata);
        s->append(ptr, size * nmemb);
        return size * nmemb;
    }

    static size_t header_cb(char* buffer, size_t size, size_t nitems, void* userdata) {
        auto* hdrs = static_cast<std::unordered_map<std::string,std::string>*>(userdata);
        std::string line(buffer, size * nitems);
        // strip CRLF
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
            line.pop_back();
        auto colon = line.find(':');
        if (colon != std::string::npos) {
            std::string key = line.substr(0, colon);
            std::string val = line.substr(colon + 1);
            // trim leading space from val
            auto start = val.find_first_not_of(' ');
            if (start != std::string::npos) val = val.substr(start);
            // lowercase key
            for (auto& c : key) c = static_cast<char>(std::tolower(c));
            (*hdrs)[key] = val;
        }
        return size * nitems;
    }

    HttpResponse request(const std::string& method,
                         const std::string& url,
                         const std::unordered_map<std::string,std::string>& headers,
                         const std::string& body) {
        HttpResponse resp;
        curl_easy_reset(curl_);

        curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, write_cb);
        curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &resp.body);
        curl_easy_setopt(curl_, CURLOPT_HEADERFUNCTION, header_cb);
        curl_easy_setopt(curl_, CURLOPT_HEADERDATA, &resp.headers);
        curl_easy_setopt(curl_, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl_, CURLOPT_TIMEOUT, 30L);
        curl_easy_setopt(curl_, CURLOPT_SSL_VERIFYPEER, 1L);

        // Build header list
        curl_slist* hlist = nullptr;
        for (auto& [k, v] : headers) {
            std::string h = k + ": " + v;
            hlist = curl_slist_append(hlist, h.c_str());
        }
        if (hlist) curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, hlist);

        if (method == "POST") {
            curl_easy_setopt(curl_, CURLOPT_POST, 1L);
            curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, body.c_str());
            curl_easy_setopt(curl_, CURLOPT_POSTFIELDSIZE, (long)body.size());
        } else if (method != "GET") {
            curl_easy_setopt(curl_, CURLOPT_CUSTOMREQUEST, method.c_str());
        }

        CURLcode res = curl_easy_perform(curl_);
        if (hlist) curl_slist_free_all(hlist);

        if (res != CURLE_OK)
            throw std::runtime_error(std::string("curl: ") + curl_easy_strerror(res));

        curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &resp.status_code);
        return resp;
    }
};

} // namespace scraper
