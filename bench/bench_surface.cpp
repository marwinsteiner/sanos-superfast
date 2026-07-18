#include "sanos/surface.hpp"
#include <cmath>
#include <cstdio>
#include <chrono>
#include <random>
#include <vector>
#include <string>
#include <numeric>

// Generate synthetic SPX-like option data
struct SyntheticSPX {
    struct Expiry {
        std::string label;
        double sqrtT;
        std::vector<double> strikes;
        std::vector<double> bids;
        std::vector<double> asks;
    };
    std::vector<Expiry> expiries;
};

static SyntheticSPX generate_spx_data(int n_expiries = 15, int strikes_per = 60, unsigned seed = 42) {
    std::mt19937 rng(seed);
    SyntheticSPX data;

    // Typical SPX expiries: 1d, 2d, 3d, 7d, 14d, 21d, 30d, 60d, 90d, 120d, 180d, 270d, 365d, 545d, 730d
    double dte_days[] = {1, 2, 3, 7, 14, 21, 30, 60, 90, 120, 180, 270, 365, 545, 730};
    int n_exp = std::min(n_expiries, 15);

    double base_vol = 0.18;  // ~18% ATM vol for SPX

    for (int j = 0; j < n_exp; ++j) {
        SyntheticSPX::Expiry exp;
        double T = dte_days[j] / 365.0;
        exp.sqrtT = std::sqrt(T);
        exp.label = std::to_string(static_cast<int>(dte_days[j])) + "DTE";

        // ATM vol with term structure: slightly decreasing then flat
        double atm_vol = base_vol * (1.0 + 0.3 * std::exp(-T * 2.0));

        // Generate strikes: log-spaced around ATM (K=1 for pure strikes)
        double vol_range = atm_vol * exp.sqrtT * 3.0;  // +/- 3 sigma
        double k_lo = std::exp(-vol_range);
        double k_hi = std::exp(vol_range);
        k_lo = std::max(k_lo, 0.5);
        k_hi = std::min(k_hi, 1.5);

        int ns = strikes_per;
        exp.strikes.resize(ns);
        for (int i = 0; i < ns; ++i) {
            double t = static_cast<double>(i) / (ns - 1);
            exp.strikes[i] = k_lo + (k_hi - k_lo) * t;
        }

        // Generate bid/ask from BS with skew + spread
        std::normal_distribution<double> noise(0.0, 0.001);
        exp.bids.resize(ns);
        exp.asks.resize(ns);

        for (int i = 0; i < ns; ++i) {
            double K = exp.strikes[i];
            double log_m = std::log(K);

            // Skew: vol increases for lower strikes
            double skew = -0.15 * log_m / exp.sqrtT;
            double local_vol = atm_vol + skew;
            local_vol = std::max(local_vol, 0.05);

            double total_var = local_vol * local_vol * T;
            double sqrt_v = std::sqrt(total_var);
            double d1 = (-log_m + 0.5 * total_var) / sqrt_v;
            double d2 = d1 - sqrt_v;
            double call_mid = 0.5 * std::erfc(-d1 * 0.7071067811865475)
                            - K * 0.5 * std::erfc(-d2 * 0.7071067811865475);

            // Spread proportional to vega (wider OTM)
            double vega = std::exp(-0.5 * d1 * d1) * 0.3989422804014327 * exp.sqrtT;
            double half_spread = std::max(0.0005, vega * 0.02);

            double bid = call_mid - half_spread + noise(rng);
            double ask = call_mid + half_spread + noise(rng);

            // Enforce constraints
            double intr = std::max(1.0 - K, 0.0);
            bid = std::max(bid, intr + 1e-6);
            ask = std::min(ask, 1.0 - 1e-6);
            ask = std::max(ask, bid + 1e-6);

            exp.bids[i] = bid;
            exp.asks[i] = ask;
        }

        data.expiries.push_back(std::move(exp));
    }
    return data;
}

static void bench_full_calibration(const SyntheticSPX& data, int n_runs = 50) {
    printf("=== Full Calibration Benchmark ===\n");
    printf("Expiries: %d, total strikes: ", static_cast<int>(data.expiries.size()));
    int total = 0;
    for (auto& e : data.expiries) total += static_cast<int>(e.strikes.size());
    printf("%d\n\n", total);

    // --- Cold calibration (new Surface each time) ---
    std::vector<double> cold_times;
    for (int run = 0; run < n_runs; ++run) {
        sanos::Surface surf;
        for (auto& e : data.expiries)
            surf.add_expiry(e.label, e.sqrtT,
                           e.strikes.data(), static_cast<int>(e.strikes.size()),
                           e.bids.data(), e.asks.data());
        double us = surf.calibrate();
        cold_times.push_back(us);

        if (run == 0) {
            printf("First run (cold): %.1f us (%.3f ms)\n", us, us / 1000.0);
            for (int j = 0; j < surf.n_expiries(); ++j) {
                auto& f = surf.fit(j);
                auto& m = surf.market(j);
                double max_err = 0.0;
                for (int i = 0; i < m.n(); ++i) {
                    double err = std::abs(f.fitted[i] - m.mids[i]) / m.spreads[i];
                    max_err = std::max(max_err, err);
                }
                printf("  %8s: N=%3d, max_err/spread=%.4f, atm_vol=%.4f\n",
                       m.label.c_str(), f.N(), max_err, f.atm_vol);
            }
        }
    }

    std::sort(cold_times.begin(), cold_times.end());

    printf("\nCold calibration over %d runs:\n", n_runs);
    printf("  Median: %8.1f us (%6.3f ms)\n", cold_times[cold_times.size()/2], cold_times[cold_times.size()/2]/1000.0);
    printf("  P95:    %8.1f us (%6.3f ms)\n", cold_times[static_cast<int>(cold_times.size()*0.95)], cold_times[static_cast<int>(cold_times.size()*0.95)]/1000.0);
    printf("  Min:    %8.1f us (%6.3f ms)\n", cold_times.front(), cold_times.front()/1000.0);

    // --- Warm recalibration (same Surface, persistent pool, cached kernels) ---
    sanos::Surface surf_warm;
    for (auto& e : data.expiries)
        surf_warm.add_expiry(e.label, e.sqrtT,
                       e.strikes.data(), static_cast<int>(e.strikes.size()),
                       e.bids.data(), e.asks.data());
    surf_warm.calibrate();  // first call: cold, creates pool

    std::vector<double> warm_times;
    for (int run = 0; run < n_runs; ++run) {
        // Simulate small quote perturbation (mids shift slightly)
        double us = surf_warm.calibrate();  // warm: pool exists, kernels cached
        warm_times.push_back(us);
    }

    std::sort(warm_times.begin(), warm_times.end());
    printf("\nWarm recalibration over %d runs (pool + kernel cached):\n", n_runs);
    printf("  Median: %8.1f us (%6.3f ms)\n", warm_times[warm_times.size()/2], warm_times[warm_times.size()/2]/1000.0);
    printf("  P95:    %8.1f us (%6.3f ms)\n", warm_times[static_cast<int>(warm_times.size()*0.95)], warm_times[static_cast<int>(warm_times.size()*0.95)]/1000.0);
    printf("  Min:    %8.1f us (%6.3f ms)\n\n", warm_times.front(), warm_times.front()/1000.0);
}

static void bench_tick_update(const SyntheticSPX& data, int n_ticks = 1000) {
    printf("=== Tick Update Benchmark ===\n");

    sanos::Surface surf;
    for (auto& e : data.expiries) {
        surf.add_expiry(e.label, e.sqrtT,
                       e.strikes.data(), static_cast<int>(e.strikes.size()),
                       e.bids.data(), e.asks.data());
    }
    surf.calibrate();

    std::mt19937 rng(123);
    std::vector<double> times;

    for (int t = 0; t < n_ticks; ++t) {
        int exp_idx = rng() % surf.n_expiries();
        int strike_idx = rng() % surf.market(exp_idx).n();

        auto& m = surf.market(exp_idx);
        double mid = m.mids[strike_idx];
        double spread = m.spreads[strike_idx];

        // Simulate small price move
        std::normal_distribution<double> move(0.0, spread * 0.1);
        double delta = move(rng);
        double new_bid = m.bids[strike_idx] + delta;
        double new_ask = m.asks[strike_idx] + delta;

        double us = surf.tick_update(exp_idx, strike_idx, new_bid, new_ask);
        times.push_back(us);
    }

    std::sort(times.begin(), times.end());
    double median = times[times.size() / 2];
    double p95 = times[static_cast<int>(times.size() * 0.95)];
    double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();

    printf("Over %d ticks:\n", n_ticks);
    printf("  Mean:   %8.1f us (%6.3f ms)\n", mean, mean / 1000.0);
    printf("  Median: %8.1f us (%6.3f ms)\n", median, median / 1000.0);
    printf("  P95:    %8.1f us (%6.3f ms)\n", p95, p95 / 1000.0);
    printf("  Min:    %8.1f us (%6.3f ms)\n", times.front(), times.front() / 1000.0);
    printf("  Max:    %8.1f us (%6.3f ms)\n\n", times.back(), times.back() / 1000.0);
}

static void bench_surface_query(const SyntheticSPX& data) {
    printf("=== Surface Query Benchmark ===\n");

    sanos::Surface surf;
    for (auto& e : data.expiries) {
        surf.add_expiry(e.label, e.sqrtT,
                       e.strikes.data(), static_cast<int>(e.strikes.size()),
                       e.bids.data(), e.asks.data());
    }
    surf.calibrate();

    // Query a grid of 100 strikes x 15 expiries
    int n_strikes = 100;
    int n_exp = surf.n_expiries();
    std::vector<double> query_strikes(n_strikes);
    for (int i = 0; i < n_strikes; ++i) {
        query_strikes[i] = 0.8 + 0.4 * i / (n_strikes - 1);
    }

    int n_queries = n_strikes * n_exp;
    auto t0 = std::chrono::high_resolution_clock::now();

    volatile double sink = 0.0;
    for (int run = 0; run < 100; ++run) {
        for (int j = 0; j < n_exp; ++j) {
            double T = surf.market(j).T();
            for (int i = 0; i < n_strikes; ++i) {
                sink += surf.vol(T, query_strikes[i]);
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double per_query = total_us / (100.0 * n_queries);

    printf("Grid: %d strikes x %d expiries = %d queries\n", n_strikes, n_exp, n_queries);
    printf("  Per query:  %.2f us\n", per_query);
    printf("  Full grid:  %.1f us (%.3f ms)\n\n", per_query * n_queries, per_query * n_queries / 1000.0);
}

int main() {
    printf("sanos-superfast benchmark\n");
    printf("========================\n\n");

    auto data = generate_spx_data(15, 60);

    bench_full_calibration(data);
    bench_tick_update(data);
    bench_surface_query(data);

    return 0;
}
