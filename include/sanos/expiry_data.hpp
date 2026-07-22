#pragma once
#include "types.hpp"
#include "config.hpp"
#include "bs_kernel.hpp"
#include "qp_solver.hpp"
#include <string>

namespace sanos {

// Per-expiry market data in Structure-of-Arrays (SOA) layout.
// All per-strike arrays are stored in a single contiguous allocation
// at fixed stride offsets. This means accessing all bids, all asks, etc.
// reads sequential memory — one cache line services 8 consecutive doubles
// instead of jumping between 9 separate heap allocations.
struct ExpiryMarket {
    std::string label;
    double sqrtT = 0.0;

    // SOA flat block: 9 arrays of n doubles in one contiguous allocation.
    // Layout: [strikes | bids | asks | mids | spreads | weights | iv_bids | iv_asks | w_prev]
    static constexpr int N_FIELDS = 9;
    AVec<double> block_;
    int n_ = 0;

    double* strikes()  { return block_.data(); }
    double* bids()     { return block_.data() + n_; }
    double* asks()     { return block_.data() + 2 * n_; }
    double* mids()     { return block_.data() + 3 * n_; }
    double* spreads()  { return block_.data() + 4 * n_; }
    double* weights()  { return block_.data() + 5 * n_; }
    double* iv_bids()  { return block_.data() + 6 * n_; }
    double* iv_asks()  { return block_.data() + 7 * n_; }
    double* w_prev()   { return block_.data() + 8 * n_; }

    const double* strikes()  const { return block_.data(); }
    const double* bids()     const { return block_.data() + n_; }
    const double* asks()     const { return block_.data() + 2 * n_; }
    const double* mids()     const { return block_.data() + 3 * n_; }
    const double* spreads()  const { return block_.data() + 4 * n_; }
    const double* weights()  const { return block_.data() + 5 * n_; }
    const double* iv_bids()  const { return block_.data() + 6 * n_; }
    const double* iv_asks()  const { return block_.data() + 7 * n_; }
    const double* w_prev()   const { return block_.data() + 8 * n_; }

    void alloc(int n) {
        n_ = n;
        block_.resize(N_FIELDS * n);
    }

    int n() const { return n_; }
    double T() const { return sqrtT * sqrtT; }
};

// Per-expiry fitted SANOS model state.
struct ExpiryFit {
    // Model grid
    AVec<double> model_strikes;  // extended strikes including boundaries
    AVec<double> model_vols;     // vol per model strike (constant in "atm" mode)
    AVec<int>    market_ix;      // maps market strikes -> model strike indices (-1 if dropped)

    int N() const { return static_cast<int>(model_strikes.size()); }

    // Kernel matrices (pre-computed, cached)
    DenseMat C_market;   // n_market x N: maps q -> prices at market strikes
    DenseMat U_model;    // N x N: maps q -> prices at model strikes (for monotonicity)
    DenseMat R_cross;    // N x N_prev: maps q_prev -> prices at current model strikes

    // QP data (pre-computed)
    DenseMat H;          // N x N: Hessian = C^T W^2 C + lambda I
    AVec<double> f;      // N: gradient = -C^T W^2 mid

    // Inequality constraints for QP: A_ineq * q >= b_ineq
    // Rows: N non-negativity (q >= 0) + N monotonicity (U*q >= floor)
    AVec<double> A_ineq; // (2N) x N row-major
    AVec<double> b_ineq; // 2N

    // Equality constraints: A_eq * q = b_eq
    // Row 0: sum(q) = 1, Row 1: K^T q = 1
    AVec<double> A_eq;   // 2 x N row-major
    AVec<double> b_eq;   // 2

    // Solution
    AVec<double> q;          // density weights (N)
    AVec<double> fitted;     // fitted prices at market strikes (n_market)
    AVec<double> iv_fitted;  // fitted implied vols at market strikes
    double atm_var  = 0.0;   // ATM variance from market
    double atm_vol  = 0.0;   // ATM vol from market

    // QP solver workspace (reused across solves — zero allocation)
    QPWorkspace qp_ws;

    // Pre-allocated scratch buffers (flyweight pattern — allocated once, reused)
    AVec<double> w2;       // weights squared, length n_market
    AVec<double> w2mid;    // w^2 * mid, length n_market

    // Caching
    double cached_atm_var = -1.0;
    bool kernel_dirty = true;
    bool fit_dirty    = true;

    // Pre-allocate all buffers for a given market size.
    // Called once during setup; hot path never allocates.
    void reserve_buffers(int n_market, int n_model) {
        C_market.reserve(n_market, n_model);
        H.reserve(n_model, n_model);
        f.reserve(n_model);
        A_eq.reserve(2 * n_model);
        b_eq.reserve(2);
        q.reserve(n_model);
        fitted.reserve(n_market);
        iv_fitted.reserve(n_market);
        w2.reserve(n_market);
        w2mid.reserve(n_market);
        model_strikes.reserve(n_model + 20);
    }
};

// Generate model strikes from market strikes, adding fill strikes where gaps > max_dx
// and boundary strikes at min_k, max_k.
// Returns model strikes and index mapping (market_ix: market strike -> model strike index).
void build_model_strikes(
    AVec<double>& model_strikes,
    AVec<int>& market_ix,
    const double* strikes, int n,
    double min_k, double max_k,
    double max_dx, double min_dx);

} // namespace sanos
