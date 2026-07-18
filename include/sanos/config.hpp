#pragma once

namespace sanos {

struct SurfaceConfig {
    // Smoothness: eta in [0, 1). 0 = linear, 0.25 recommended default.
    double eta = 0.25;

    // Minimum implied vol floor
    double floor_vol = 0.01;

    // Model strike grid parameters
    double max_dx = 0.05;     // max gap between model strikes
    double min_dx = 1e-6;     // min gap (drop closer strikes)
    double min_k  = 0.1;      // left boundary strike (pure)
    double max_k  = 3.0;      // right boundary strike (pure)

    // QP solver parameters
    double smoothness_penalty = 1e-6;  // lambda * ||q||^2 regularizer
    int    max_qp_iters       = 200;   // max active-set pivots
    double qp_tol             = 1e-10; // KKT tolerance

    // Bid/ask handling: "penalty" or "constraint"
    // penalty: 0.01 * |mid-fit| + max(fit-ask, 0) + max(bid-fit, 0)
    // constraint: hard bid <= fit <= ask
    enum class BidAskMode { Penalty, Constraint };
    BidAskMode bid_ask_mode = BidAskMode::Penalty;
    double penalty_epsilon  = 0.01;  // weight on mid-error vs bid/ask violation

    // Spread weighting
    bool   spread_weighted = true;
    double max_inv_spread  = 100.0;

    // Vol mode for kernel: "atm" uses eta * sqrt(atm_var) / sqrtT
    // This is the only mode we support for speed
    double vol_fac = 0.5;  // so eta = vol_fac^2 = 0.25
};

} // namespace sanos
