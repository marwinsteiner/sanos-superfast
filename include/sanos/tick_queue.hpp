#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace sanos {

// Single tick update message. Fits in 32 bytes — half a cache line.
struct TickUpdate {
    int16_t  expiry_idx;
    int16_t  strike_idx;
    uint32_t seq;         // sequence number for ordering
    double   new_bid;
    double   new_ask;
};

// Lock-free single-producer single-consumer (SPSC) ring buffer.
//
// The producer (market data thread) writes tick updates without blocking.
// The consumer (fitting thread) drains all pending updates in a batch
// before re-solving affected expiries.
//
// Cache line padding on head/tail prevents false sharing between the
// producer writing head and consumer reading tail (or vice versa).
// The ring capacity is always a power of two for fast modulo (bitwise AND).
template <int Capacity = 4096>
class TickQueue {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    static constexpr int MASK = Capacity - 1;

public:
    // Producer: enqueue a tick. Returns false if full (back-pressure).
    bool push(int16_t expiry, int16_t strike, double bid, double ask) {
        int h = head_.val.load(std::memory_order_relaxed);
        int t = tail_.val.load(std::memory_order_acquire);
        if (h - t >= Capacity) return false;  // full

        auto& slot = buf_[h & MASK];
        slot.expiry_idx = expiry;
        slot.strike_idx = strike;
        slot.new_bid    = bid;
        slot.new_ask    = ask;
        slot.seq        = static_cast<uint32_t>(h);

        head_.val.store(h + 1, std::memory_order_release);
        return true;
    }

    // Consumer: pop one tick. Returns false if empty.
    bool pop(TickUpdate& out) {
        int t = tail_.val.load(std::memory_order_relaxed);
        int h = head_.val.load(std::memory_order_acquire);
        if (t >= h) return false;  // empty

        out = buf_[t & MASK];
        tail_.val.store(t + 1, std::memory_order_release);
        return true;
    }

    // Consumer: drain all pending ticks into a flat array.
    // Returns count drained. out must have space for Capacity elements.
    int drain(TickUpdate* out) {
        int t = tail_.val.load(std::memory_order_relaxed);
        int h = head_.val.load(std::memory_order_acquire);
        int count = h - t;
        for (int i = 0; i < count; ++i)
            out[i] = buf_[(t + i) & MASK];
        tail_.val.store(h, std::memory_order_release);
        return count;
    }

    int pending() const {
        return head_.val.load(std::memory_order_acquire)
             - tail_.val.load(std::memory_order_acquire);
    }

private:
    struct alignas(64) PaddedIdx {
        std::atomic<int> val{0};
    };

    TickUpdate buf_[Capacity];
    PaddedIdx  head_;  // written by producer, read by consumer
    PaddedIdx  tail_;  // written by consumer, read by producer
};

} // namespace sanos
