#pragma once
#include "types.hpp"
#include "config.hpp"
#include "bs_kernel.hpp"
#include "qp_solver.hpp"
#include <string>

namespace sanos {

// Per-expiry market data (pure/normalized strikes and prices).
struct ExpiryMarket {
    std::string label;
    double sqrtT = 0.0;  // sqrt(time to expiry)

    AVec<double> strikes;   // pure strikes (K/F), sorted ascending, must surround 1.0
    AVec<double> bids;      // pure bid prices
    AVec<double> asks;      // pure ask prices
    AVec<double> mids;      // (bid + ask) / 2
    AVec<double> spreads;   // ask - bid
    AVec<double> weights;   // 1/spread, capped
    AVec<double> iv_bids;   // bid implied vols
    AVec<double> iv_asks;   // ask implied vols

    // Warm IV state for volfi incremental updates
    AVec<double> w_prev;    // previous total variance per strike (for warm restart)

    int n() const { return static_cast<int>(strikes.size()); }
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

    // QP solver workspace (reused)
    QPWorkspace qp_ws;

    // Flags
    bool kernel_dirty = true;  // need to recompute kernel matrices
    bool fit_dirty    = true;  // need to re-solve QP
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
