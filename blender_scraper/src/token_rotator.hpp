#pragma once
#include <string>
#include <vector>
#include <mutex>
#include <iostream>
#include <chrono>
#include <thread>
#include <algorithm>
#include <stdexcept>

namespace scraper {

// Thrown when every token is exhausted and proceed_on_exhaust is true
struct TokensExhaustedException : std::runtime_error {
    long earliest_reset;
    explicit TokensExhaustedException(long reset)
        : std::runtime_error("All tokens exhausted"), earliest_reset(reset) {}
};

// ---------------------------------------------------------------------------
// Manages a pool of GitHub tokens and rotates between them automatically.
// When a token hits its rate limit, it is parked until its reset epoch,
// and the next available token is returned.
// If ALL tokens are exhausted, sleeps until the earliest reset.
// ---------------------------------------------------------------------------
class TokenRotator {
public:
    explicit TokenRotator(std::vector<std::string> tokens,
                          bool proceed_on_exhaust = true)
        : proceed_on_exhaust_(proceed_on_exhaust)
    {
        if (tokens.empty()) throw std::runtime_error("No GitHub tokens provided");
        for (auto& t : tokens) {
            slots_.push_back({t, 0, 5000});
        }
        std::cout << "[tokens] Loaded " << slots_.size() << " GitHub token(s)\n";
    }

    // Returns the currently active token (thread-safe).
    std::string current() {
        std::lock_guard<std::mutex> lock(mu_);
        return slots_[idx_].token;
    }

    // Called when a response comes back — update remaining count for current token.
    void update_remaining(int remaining, long reset_epoch) {
        std::lock_guard<std::mutex> lock(mu_);
        slots_[idx_].remaining  = remaining;
        slots_[idx_].reset_epoch = reset_epoch;
    }

    // Called when the current token is rate-limited (403/429 or remaining==0).
    // Rotates to the next available token. Sleeps if all are exhausted.
    void on_rate_limited(long reset_epoch) {
        size_t best = 0;
        long   best_reset = 0;
        long   wait = 0;

        {
            std::lock_guard<std::mutex> lock(mu_);

            // Park this token with its reset epoch
            slots_[idx_].remaining   = 0;
            slots_[idx_].reset_epoch = reset_epoch > 0
                ? reset_epoch
                : now_epoch() + 3600; // fallback: 1 hour

            // Try to find another token with remaining quota
            for (size_t i = 1; i <= slots_.size(); ++i) {
                size_t candidate = (idx_ + i) % slots_.size();
                if (is_available(slots_[candidate])) {
                    std::cerr << "[tokens] Token " << idx_ << " exhausted — "
                              << "switching to token " << candidate << "\n";
                    idx_ = candidate;
                    return;
                }
            }

            // All tokens exhausted — find the one that resets soonest
            best       = 0;
            best_reset = slots_[0].reset_epoch;
            for (size_t i = 1; i < slots_.size(); ++i) {
                if (slots_[i].reset_epoch < best_reset) {
                    best_reset = slots_[i].reset_epoch;
                    best = i;
                }
            }
            wait = best_reset - now_epoch() + 2;
        } // lock released before any sleep or throw

        if (proceed_on_exhaust_) {
            // Don't sleep — let the caller proceed with partial results
            std::cerr << "[tokens] All " << slots_.size()
                      << " token(s) exhausted (next reset in " << wait
                      << "s). Proceeding with data collected so far.\n";
            throw TokensExhaustedException(best_reset);
        }

        if (wait > 0) {
            std::cerr << "[tokens] All " << slots_.size()
                      << " token(s) exhausted. Sleeping " << wait
                      << "s until token " << best << " resets.\n";
            std::this_thread::sleep_for(std::chrono::seconds(wait));
        }
        {
            std::lock_guard<std::mutex> lock(mu_);
            idx_ = best;
            slots_[idx_].remaining = 5000;
        }
        std::cerr << "[tokens] Resuming with token " << best << "\n";
    }

    int token_count() const { return (int)slots_.size(); }

private:
    struct Slot {
        std::string token;
        long        reset_epoch{0};
        int         remaining{5000};
    };

    std::vector<Slot> slots_;
    size_t            idx_{0};
    std::mutex        mu_;
    bool              proceed_on_exhaust_{true};

    static long now_epoch() {
        return static_cast<long>(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    }

    bool is_available(const Slot& s) const {
        if (s.remaining > 0) return true;
        return now_epoch() >= s.reset_epoch;
    }
};

} // namespace scraper
