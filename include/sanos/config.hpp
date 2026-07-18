#pragma once

namespace sanos {

struct SurfaceConfig {
    // Smoothness: eta in [0, 1). 0 = linear, 0.25 recommended default.
    double eta = 0.25;

    // Minimum implied vol floor
    double floor_vol = 0.01;

    // Model strike grid parameters
    double max_dx = 0.10;     // max gap between model strikes near ATM
    double wing_dx = 0.50;    // max gap in wings (|K-1| > wing_boundary)
    double wing_boundary = 0.15; // where wings begin (distance from ATM)
    double min_dx = 1e-6;     // min gap (drop closer strikes)
    double min_k  = 0.1;      // left boundary strike (pure)
    double max_k  = 3.0;      // right boundary strike (pure)

    // QP solver parameters
    double smoothness_penalty = 1e-4;  // lambda * ||q||^2 regularizer
    int    max_qp_iters       = 200;   // max active-set pivots
    double qp_tol             = 1e-10; // KKT tolerance

    // Bid/ask handling
    enum class BidAskMode { Penalty, Constraint };
    BidAskMode bid_ask_mode = BidAskMode::Penalty;
    double penalty_epsilon  = 0.01;

    // Spread weighting
    bool   spread_weighted = true;
    double max_inv_spread  = 100.0;

    // Vol mode
    double vol_fac = 0.5;  // so eta = vol_fac^2 = 0.25

    // Kernel caching: skip kernel+Hessian rebuild if ATM var changed < this fraction
    double kernel_cache_tol = 0.01;  // 1% relative change

    // Parallelism
    int n_threads = 0;  // 0 = auto (hardware_concurrency), 1 = single-threaded
};

} // namespace sanos
