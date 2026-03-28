#pragma once
#include <chrono>
#include <mutex>
#include <thread>
#include <ctime>
#include <iostream>
#include <atomic>

namespace scraper {

// ---------------------------------------------------------------------------
// Token-bucket rate limiter — shared across threads.
// Supports GitHub-style epoch-based reset.
// ---------------------------------------------------------------------------
class RateLimiter {
public:
    explicit RateLimiter(int requests_per_second = 5)
        : rps_(requests_per_second) {}

    // Block until a request token is available.
    void acquire() {
        std::unique_lock<std::mutex> lock(mu_);
        auto now  = clock::now();
        auto wait = last_request_ + std::chrono::milliseconds(1000 / rps_) - now;
        if (wait > std::chrono::milliseconds(0)) {
            lock.unlock();
            std::this_thread::sleep_for(wait);
            lock.lock();
        }
        last_request_ = clock::now();
        ++total_requests_;
    }

    // Called when GitHub returns 429 or X-RateLimit-Remaining == 0.
    // Sleeps until the reset epoch (UTC seconds).
    void handle_rate_limit(long reset_epoch) {
        if (reset_epoch <= 0) {
            std::cerr << "[rate] Hit limit, sleeping 60s\n";
            std::this_thread::sleep_for(std::chrono::seconds(60));
            return;
        }
        long now_epoch = static_cast<long>(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        long wait_sec = reset_epoch - now_epoch + 2; // +2s buffer
        if (wait_sec > 0) {
            std::cerr << "[rate] GitHub rate limit hit. Sleeping " << wait_sec << "s until reset.\n";
            std::this_thread::sleep_for(std::chrono::seconds(wait_sec));
        }
    }

    long total_requests() const { return total_requests_.load(); }

private:
    using clock = std::chrono::steady_clock;
    int                    rps_;
    clock::time_point      last_request_{};
    std::mutex             mu_;
    std::atomic<long>      total_requests_{0};
};

} // namespace scraper
