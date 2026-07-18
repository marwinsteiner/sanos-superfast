#pragma once
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>

namespace sanos {

// Lightweight thread pool: threads spin-wait on tasks, zero allocation dispatch.
class ThreadPool {
public:
    explicit ThreadPool(int n_threads = 0) {
        if (n_threads <= 0)
            n_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
        n_workers_ = n_threads;
        workers_.resize(n_workers_);
        for (int i = 0; i < n_workers_; ++i) {
            workers_[i] = std::thread([this, i]() { worker_loop(i); });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            shutdown_ = true;
        }
        cv_start_.notify_all();
        for (auto& t : workers_) t.join();
    }

    // Run func(0), func(1), ..., func(n-1) in parallel across the pool.
    // Blocks until all complete. Zero allocation if n <= pool size.
    void parallel_for(int n, const std::function<void(int)>& func) {
        if (n <= 0) return;
        if (n == 1) { func(0); return; }

        func_ = &func;
        task_count_ = n;
        next_task_.store(0, std::memory_order_relaxed);
        done_count_.store(0, std::memory_order_relaxed);

        // Wake workers
        {
            std::lock_guard<std::mutex> lk(mu_);
            generation_++;
        }
        cv_start_.notify_all();

        // Caller also participates
        run_tasks();

        // Wait for all tasks to complete
        while (done_count_.load(std::memory_order_acquire) < n) {
            std::this_thread::yield();
        }
    }

    int size() const { return n_workers_; }

private:
    void run_tasks() {
        while (true) {
            int idx = next_task_.fetch_add(1, std::memory_order_relaxed);
            if (idx >= task_count_) break;
            (*func_)(idx);
            done_count_.fetch_add(1, std::memory_order_release);
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
    std::atomic<int> next_task_{0};
    std::atomic<int> done_count_{0};
};

} // namespace sanos
