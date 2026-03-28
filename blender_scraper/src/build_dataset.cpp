#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <atomic>

namespace fs = std::filesystem;
using json   = nlohmann::json;

// ---------------------------------------------------------------------------
// Standalone dataset builder.
// Walks output/json_files/, reads every .json + its .meta.json sidecar,
// and writes a clean JSONL file ready for AI training.
//
// Usage:
//   build_dataset [json_files_dir] [output_jsonl]
//   build_dataset output/json_files training_dataset.jsonl
// ---------------------------------------------------------------------------

struct Record {
    std::string repo;
    std::string path;
    std::string html_url;
    json        content;
};

// ---------------------------------------------------------------------------
// Attempt to load and parse one .json file + its sidecar.
// Returns false if the file should be skipped.
// ---------------------------------------------------------------------------
static bool load_record(const fs::path& json_path, Record& out) {
    // --- Read content ---
    std::ifstream ifs(json_path, std::ios::binary);
    if (!ifs) return false;
    std::string raw((std::istreambuf_iterator<char>(ifs)),
                     std::istreambuf_iterator<char>());
    if (raw.size() < 10) return false;  // trivially empty

    json content;
    try { content = json::parse(raw); }
    catch (...) { return false; }
    if (content.is_null()) return false;

    // --- Read sidecar meta ---
    json meta = json::object();
    fs::path meta_path = fs::path(json_path.string() + ".meta.json");
    if (fs::exists(meta_path)) {
        std::ifstream mifs(meta_path);
        try {
            json parsed = json::parse(mifs);
            if (parsed.is_object()) meta = parsed;
        } catch (...) {}
    }

    out.repo     = meta.contains("repo")     && meta["repo"].is_string()
                   ? meta["repo"].get<std::string>() : "";
    out.path     = meta.contains("path")     && meta["path"].is_string()
                   ? meta["path"].get<std::string>() : json_path.filename().string();
    out.html_url = meta.contains("html_url") && meta["html_url"].is_string()
                   ? meta["html_url"].get<std::string>() : "";
    out.content  = std::move(content);
    return true;
}

int main(int argc, char* argv[]) {
    std::string input_dir   = "output/json_files";
    std::string output_file = "output/training_dataset.jsonl";

    if (argc >= 2) input_dir   = argv[1];
    if (argc >= 3) output_file = argv[2];

    if (!fs::exists(input_dir)) {
        std::cerr << "[error] Input directory not found: " << input_dir << "\n";
        return 1;
    }

    std::ofstream ofs(output_file);
    if (!ofs) {
        std::cerr << "[error] Cannot write to: " << output_file << "\n";
        return 1;
    }

    std::cout << "Building dataset from: " << input_dir << "\n"
              << "Output:               " << output_file << "\n\n";

    int scanned  = 0;
    int written  = 0;
    int skipped  = 0;
    long bytes   = 0;

    for (auto& entry : fs::recursive_directory_iterator(input_dir)) {
        if (!entry.is_regular_file()) continue;

        std::string name = entry.path().filename().string();
        std::string ext  = entry.path().extension().string();

        // Skip sidecar meta files and non-json files
        if (name.find(".meta.json") != std::string::npos) continue;
        if (ext != ".json") continue;

        ++scanned;

        Record rec;
        if (!load_record(entry.path(), rec)) {
            ++skipped;
            continue;
        }

        // Build training record
        json record;
        record["source"]   = "github";
        record["repo"]     = rec.repo;
        record["path"]     = rec.path;
        record["html_url"] = rec.html_url;
        record["content"]  = rec.content;

        std::string line = record.dump();
        ofs << line << "\n";
        ++written;
        bytes += (long)line.size();

        if (scanned % 5000 == 0)
            std::cout << "  Processed " << scanned << " files ("
                      << written << " written, " << skipped << " skipped)...\n";
    }

    ofs.flush();
    std::cout << "\n=== Dataset Build Complete ===\n"
              << "  Files scanned:  " << scanned  << "\n"
              << "  Records written:" << written  << "\n"
              << "  Files skipped:  " << skipped  << "\n"
              << "  Output size:    " << bytes / 1024 / 1024 << " MB\n"
              << "  Output file:    " << output_file << "\n";
    return 0;
}
