#pragma once
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <cstddef>

namespace sanos {

static constexpr std::size_t CACHELINE = 64;

class ThreadPool {
public:
    explicit ThreadPool(int n_threads = 0) {
        if (n_threads <= 0)
            n_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
        n_workers_ = n_threads;
        workers_.resize(n_workers_);
        for (int i = 0; i < n_workers_; ++i)
            workers_[i] = std::thread([this, i]() { worker_loop(i); });
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            shutdown_ = true;
        }
        cv_start_.notify_all();
        for (auto& t : workers_) t.join();
    }

    void parallel_for(int n, const std::function<void(int)>& func) {
        if (n <= 0) return;
        if (n == 1) { func(0); return; }

        func_ = &func;
        task_count_ = n;
        next_task_.val.store(0, std::memory_order_relaxed);
        done_count_.val.store(0, std::memory_order_relaxed);

        {
            std::lock_guard<std::mutex> lk(mu_);
            generation_++;
        }
        cv_start_.notify_all();

        run_tasks();

        while (done_count_.val.load(std::memory_order_acquire) < n)
            std::this_thread::yield();
    }

    int size() const { return n_workers_; }

private:
    // Pad each atomic to its own cache line to prevent false sharing.
    // Without this, concurrent fetch_add on next_task_ and done_count_
    // bounce the same cache line between cores, adding ~50ns per access.
    struct alignas(CACHELINE) PaddedAtomic {
        std::atomic<int> val{0};
    };

    void run_tasks() {
        while (true) {
            int idx = next_task_.val.fetch_add(1, std::memory_order_relaxed);
            if (idx >= task_count_) break;
            (*func_)(idx);
            done_count_.val.fetch_add(1, std::memory_order_release);
        }
    }

    void worker_loop(int /*id*/) {
        int seen_gen = 0;
        while (true) {
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_start_.wait(lk, [&]() { return generation_ != seen_gen || shutdown_; });
                if (shutdown_) return;
                seen_gen = generation_;
            }
            run_tasks();
        }
    }

    std::vector<std::thread> workers_;
    int n_workers_ = 0;

    std::mutex mu_;
    std::condition_variable cv_start_;
    bool shutdown_ = false;
    int generation_ = 0;

    const std::function<void(int)>* func_ = nullptr;
    int task_count_ = 0;
    PaddedAtomic next_task_;   // own cache line
    PaddedAtomic done_count_;  // own cache line
};

} // namespace sanos
