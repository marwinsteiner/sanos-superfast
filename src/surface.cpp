#include "sanos/surface.hpp"
#include "sanos/bs_kernel.hpp"
#include "sanos/qp_solver.hpp"
#include "sanos/volfi_compat.hpp"
#if !defined(SANOS_PURE_MSVC) && (defined(__AVX2__) || defined(__AVX__))
#include "sanos/simd_bs.hpp"
#endif
#include <cmath>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <cassert>

namespace sanos {

// --- Model strike grid generation ---

void build_model_strikes(
    AVec<double>& model_strikes,
    AVec<int>& market_ix,
    const double* strikes, int n,
    double min_k, double max_k,
    double max_dx, double min_dx)
{
    model_strikes.clear();
    market_ix.assign(n, -1);

    double outer_max_dx = std::min(0.25, max_dx * 10.0);

    // Left boundary region
    double left = strikes[0] - max_dx;
    if (left > min_k) {
        int n_left = std::max(2, 1 + static_cast<int>(std::floor((left - min_k) / outer_max_dx)));
        for (int i = 0; i < n_left; ++i) {
            model_strikes.push_back(min_k + (left - min_k) * i / (n_left - 1));
        }
    } else {
        model_strikes.push_back(min_k);
    }

    // First market strike
    market_ix[0] = static_cast<int>(model_strikes.size());
    model_strikes.push_back(strikes[0]);

    // Interior market strikes with fill
    for (int r = 1; r < n; ++r) {
        double dx = strikes[r] - model_strikes.back();
        if (dx <= min_dx) {
            market_ix[r] = -1;  // dropped
            continue;
        }
        if (dx > max_dx) {
            int n_fill = std::max(1, static_cast<int>(dx / max_dx)) + 1;
            double prev = model_strikes.back();
            for (int f = 1; f < n_fill; ++f) {
                model_strikes.push_back(prev + dx * f / n_fill);
            }
        }
        market_ix[r] = static_cast<int>(model_strikes.size());
        model_strikes.push_back(strikes[r]);
    }

    // Right boundary region
    double right = strikes[n - 1] + max_dx;
    if (right < max_k) {
        int n_right = std::max(2, 1 + static_cast<int>(std::floor((max_k - right) / outer_max_dx)));
        for (int i = 1; i < n_right; ++i) {
            model_strikes.push_back(right + (max_k - right) * i / (n_right - 1));
        }
    } else {
        model_strikes.push_back(max_k);
    }
}

// --- Surface implementation ---

Surface::Surface(SurfaceConfig cfg) : cfg_(cfg) {}

void Surface::clear() {
    markets_.clear();
    fits_.clear();
}

void Surface::add_expiry(
    const std::string& label,
    double sqrtT,
    const double* strikes, int n,
    const double* bids,
    const double* asks)
{
    ExpiryMarket m;
    m.label = label;
    m.sqrtT = sqrtT;
    m.strikes.assign(strikes, strikes + n);
    m.bids.assign(bids, bids + n);
    m.asks.assign(asks, asks + n);
    m.mids.resize(n);
    m.spreads.resize(n);
    m.weights.resize(n);
    m.iv_bids.resize(n);
    m.iv_asks.resize(n);
    m.w_prev.assign(n, 0.0);

    for (int i = 0; i < n; ++i) {
        double intr = std::max(1.0 - strikes[i], 0.0);
        m.bids[i] = std::max(m.bids[i], intr);
        m.asks[i] = std::min(m.asks[i], 1.0);
        m.spreads[i] = m.asks[i] - m.bids[i];
        m.mids[i] = 0.5 * (m.bids[i] + m.asks[i]);
        double inv_spread = 1.0 / std::max(m.spreads[i], 1e-12);
        m.weights[i] = std::min(inv_spread, cfg_.max_inv_spread);
    }

    // Normalize weights
    double wsum = 0.0;
    for (int i = 0; i < n; ++i) wsum += m.weights[i];
    if (wsum > 0.0) {
        for (int i = 0; i < n; ++i) m.weights[i] /= wsum;
    }

    markets_.push_back(std::move(m));
    fits_.emplace_back();
}

void Surface::set_market(
    int n_expiries,
    const std::string* labels,
    const double* sqrtTs,
    const int* n_strikes,
    const double* const* strikes,
    const double* const* bids,
    const double* const* asks)
{
    clear();
    for (int j = 0; j < n_expiries; ++j) {
        add_expiry(labels[j], sqrtTs[j], strikes[j], n_strikes[j], bids[j], asks[j]);
    }
}

void Surface::compute_iv(int j) {
    auto& m = markets_[j];
    int n = m.n();

    for (int i = 0; i < n; ++i) {
        double K = m.strikes[i];
        iv::Status st;

        double bid_vol = iv::implied_volatility(1.0, K, m.bids[i], m.T(), true, &st);
        m.iv_bids[i] = st.ok ? bid_vol : 0.0;

        double ask_vol = iv::implied_volatility(1.0, K, m.asks[i], m.T(), true, &st);
        m.iv_asks[i] = st.ok ? ask_vol : 0.0;

        // Store total variance for warm restart
        m.w_prev[i] = (m.iv_bids[i] + m.iv_asks[i]) * 0.5;
        m.w_prev[i] = m.w_prev[i] * m.w_prev[i] * m.T();
    }
}

void Surface::setup_expiry(int j) {
    auto& m = markets_[j];
    auto& f = fits_[j];

    // Build model strike grid
    build_model_strikes(
        f.model_strikes, f.market_ix,
        m.strikes.data(), m.n(),
        cfg_.min_k, cfg_.max_k,
        cfg_.max_dx, cfg_.min_dx);

    int N = f.N();

    // Compute ATM variance from market
    // Linear interpolation of bid IV at K=1
    double atm_iv = 0.0;
    for (int i = 0; i < m.n() - 1; ++i) {
        if (m.strikes[i] <= 1.0 && m.strikes[i + 1] > 1.0) {
            double t = (1.0 - m.strikes[i]) / (m.strikes[i + 1] - m.strikes[i]);
            atm_iv = m.iv_bids[i] * (1.0 - t) + m.iv_bids[i + 1] * t;
            break;
        }
    }
    f.atm_vol = atm_iv;
    f.atm_var = atm_iv * atm_iv * m.T();

    // Set model vols (ATM mode): vol_fac * sqrt(atm_var) / sqrtT = vol_fac * atm_iv
    double model_vol = cfg_.vol_fac * atm_iv;
    f.model_vols.assign(N, model_vol);

    // Allocate solution
    f.q.assign(N, 0.0);
    f.fitted.assign(m.n(), 0.0);
    f.iv_fitted.assign(m.n(), 0.0);

    f.kernel_dirty = true;
    f.fit_dirty = true;
}

void Surface::build_kernel(int j) {
    auto& m = markets_[j];
    auto& f = fits_[j];
    int N = f.N();
    int n_mkt = m.n();

    // Variance for kernel: eta * atm_var = vol_fac^2 * atm_iv^2 * T
    double variance = cfg_.eta * f.atm_var;

    // C_market: maps q -> prices at market strikes
    // If we have a direct index mapping, we could use it, but for generality
    // we compute the full kernel for market strikes
    fill_kernel_matrix(f.C_market, m.strikes.data(), n_mkt,
                       f.model_strikes.data(), N, variance);

    // Build Hessian H = C^T W^2 C + lambda I
    compute_hessian(f.H, f.C_market, m.weights.data(),
                    cfg_.smoothness_penalty / N, N);

    // Equality constraints: sum(q)=1, K^T q=1
    f.A_eq.assign(2 * N, 0.0);
    f.b_eq.assign(2, 0.0);
    for (int i = 0; i < N; ++i) {
        f.A_eq[i] = 1.0;
        f.A_eq[N + i] = f.model_strikes[i];
    }
    f.b_eq[0] = 1.0;
    f.b_eq[1] = 1.0;

    // No hard inequality constraints — non-negativity and normalization
    // are handled directly by simplex projection in the solver.
    // Term-structure monotonicity is enforced as a soft post-hoc check.
    f.A_ineq.clear();
    f.b_ineq.clear();

    f.kernel_dirty = false;
}

void Surface::build_qp(int j) {
    auto& m = markets_[j];
    auto& f = fits_[j];

    // Update linear term: f = -C^T W^2 mid
    f.f.resize(f.N());
    compute_gradient(f.f.data(), f.C_market, m.weights.data(),
                     m.mids.data(), m.n(), f.N());

    // If using penalty mode, adjust objective for bid/ask violations
    // The QP objective is: 0.5 q^T H q + f^T q
    // This gives a least-squares fit to mid weighted by spreads.
    // Bid/ask penalties are handled post-hoc or via constraint mode.

    f.fit_dirty = true;
}

void Surface::solve_expiry(int j) {
    auto& f = fits_[j];
    int N = f.N();
    int m_ineq = static_cast<int>(f.b_ineq.size());
    int m_eq = static_cast<int>(f.b_eq.size());

    QPProblem prob;
    prob.n = N;
    prob.m_ineq = m_ineq;
    prob.m_eq = m_eq;
    // H is stored column-major in DenseMat but QP expects row-major.
    // Since H is symmetric, column-major == row-major.
    prob.H = f.H.data.data();
    prob.f = f.f.data();
    prob.A_ineq = m_ineq > 0 ? f.A_ineq.data() : nullptr;
    prob.b_ineq = m_ineq > 0 ? f.b_ineq.data() : nullptr;
    prob.A_eq = f.A_eq.data();
    prob.b_eq = f.b_eq.data();

    // Warm start from previous solution if available
    const double* warm = nullptr;
    bool has_prev = false;
    for (int i = 0; i < N; ++i) {
        if (f.q[i] != 0.0) { has_prev = true; break; }
    }
    if (has_prev) warm = f.q.data();

    QPResult res = qp_solve(
        f.q.data(), prob, f.qp_ws,
        cfg_.qp_tol, cfg_.max_qp_iters, warm);

    // Ensure non-negativity and normalization
    double qsum = 0.0;
    for (int i = 0; i < N; ++i) {
        f.q[i] = std::max(f.q[i], 0.0);
        qsum += f.q[i];
    }
    if (qsum > 0.0) {
        for (int i = 0; i < N; ++i) f.q[i] /= qsum;
    }

    // Compute fitted prices at market strikes: fitted = C_market * q
    mat_vec(f.fitted.data(), f.C_market, f.q.data());

    // Compute fitted IVs
    auto& m = markets_[j];
    for (int i = 0; i < m.n(); ++i) {
        iv::Status st;
        double v = iv::implied_volatility(1.0, m.strikes[i], f.fitted[i], m.T(), true, &st);
        f.iv_fitted[i] = st.ok ? v : 0.0;
    }

    f.fit_dirty = false;
}

void Surface::eval_model(int j, const double* strikes, int n, double* prices) const {
    const auto& f = fits_[j];
    double variance = cfg_.eta * f.atm_var;
    int N = f.N();

    if (variance <= 0.0) {
        // Linear mode
        for (int l = 0; l < n; ++l) {
            double price = 0.0;
            for (int i = 0; i < N; ++i)
                price += f.q[i] * std::max(f.model_strikes[i] - strikes[l], 0.0);
            prices[l] = price;
        }
        return;
    }

    double sqrt_v = std::sqrt(variance);
    double half_v = 0.5 * variance;
    double inv_sqrt_v = 1.0 / sqrt_v;

#ifdef SANOS_HAS_SIMD_BS
    // SIMD path: vectorize over model strikes (4 at a time)
    for (int l = 0; l < n; ++l) {
        double K = strikes[l];
        __m256d v_K = _mm256_set1_pd(K);
        __m256d v_half_v = _mm256_set1_pd(half_v);
        __m256d v_inv_sqv = _mm256_set1_pd(inv_sqrt_v);
        __m256d v_sqv = _mm256_set1_pd(sqrt_v);
        __m256d v_price = _mm256_setzero_pd();

        int i = 0;
        for (; i + 4 <= N; i += 4) {
            __m256d v_ki = _mm256_loadu_pd(f.model_strikes.data() + i);
            __m256d v_qi = _mm256_loadu_pd(f.q.data() + i);
            __m256d v_ratio = _mm256_div_pd(v_K, v_ki);
            __m256d v_log_k = simd::fast_log_avx2(v_ratio);
            __m256d v_d1 = _mm256_mul_pd(
                _mm256_sub_pd(v_half_v, v_log_k),
                v_inv_sqv);
            __m256d v_d2 = _mm256_sub_pd(v_d1, v_sqv);
            __m256d v_phi1 = simd::phicdf_4(v_d1);
            __m256d v_phi2 = simd::phicdf_4(v_d2);
            __m256d v_call = _mm256_mul_pd(v_ki,
                _mm256_sub_pd(v_phi1, _mm256_mul_pd(v_ratio, v_phi2)));
            v_price = _mm256_fmadd_pd(v_qi, v_call, v_price);
        }
        double price = simd::avx2_reduce_sum(v_price);
        // Scalar tail
        for (; i < N; ++i) {
            double ratio = K / f.model_strikes[i];
            double log_k = std::log(ratio);
            double d1 = (-log_k + half_v) * inv_sqrt_v;
            double d2 = d1 - sqrt_v;
            double phi1 = 0.5 * std::erfc(-d1 * 0.7071067811865475244);
            double phi2 = 0.5 * std::erfc(-d2 * 0.7071067811865475244);
            price += f.q[i] * f.model_strikes[i] * (phi1 - ratio * phi2);
        }
        prices[l] = price;
    }
#else
    for (int l = 0; l < n; ++l) {
        double K = strikes[l];
        double price = 0.0;
        for (int i = 0; i < N; ++i) {
            double ratio = K / f.model_strikes[i];
            if (ratio <= 0.0) { price += f.q[i] * f.model_strikes[i]; continue; }
            double log_k = std::log(ratio);
            double d1 = (-log_k + half_v) * inv_sqrt_v;
            double d2 = d1 - sqrt_v;
            double phi1 = 0.5 * std::erfc(-d1 * 0.7071067811865475244);
            double phi2 = 0.5 * std::erfc(-d2 * 0.7071067811865475244);
            price += f.q[i] * f.model_strikes[i] * (phi1 - ratio * phi2);
        }
        prices[l] = price;
    }
#endif
}

double Surface::calibrate() {
    auto t0 = std::chrono::high_resolution_clock::now();

    int M = n_expiries();

    for (int j = 0; j < M; ++j) compute_iv(j);
    for (int j = 0; j < M; ++j) setup_expiry(j);
    for (int j = 0; j < M; ++j) {
        build_kernel(j);
        build_qp(j);
        solve_expiry(j);
    }
    update_time_interp();

    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

double Surface::tick_update(int expiry_idx, int strike_idx, double new_bid, double new_ask) {
    auto t0 = std::chrono::high_resolution_clock::now();

    auto& m = markets_[expiry_idx];

    // Update quote
    double intr = std::max(1.0 - m.strikes[strike_idx], 0.0);
    m.bids[strike_idx] = std::max(new_bid, intr);
    m.asks[strike_idx] = std::min(new_ask, 1.0);
    m.spreads[strike_idx] = m.asks[strike_idx] - m.bids[strike_idx];
    m.mids[strike_idx] = 0.5 * (m.bids[strike_idx] + m.asks[strike_idx]);

    // Update weight
    double inv_s = 1.0 / std::max(m.spreads[strike_idx], 1e-12);
    m.weights[strike_idx] = std::min(inv_s, cfg_.max_inv_spread);

    // Renormalize weights
    double wsum = 0.0;
    for (int i = 0; i < m.n(); ++i) wsum += m.weights[i];
    if (wsum > 0.0) {
        for (int i = 0; i < m.n(); ++i) m.weights[i] /= wsum;
    }

    // Update IV for changed strike using volfi warm restart
    double h = std::abs(std::log(m.strikes[strike_idx]));
    double mid_price = m.mids[strike_idx];
    if (h > 0.0 && mid_price > 0.0 && mid_price < 1.0) {
        double w_new = iv::implied_variance_warm(h, mid_price, m.w_prev[strike_idx]);
        m.w_prev[strike_idx] = w_new;
    }

    // Rebuild QP linear term (Hessian unchanged if kernel unchanged)
    auto& f = fits_[expiry_idx];
    build_qp(expiry_idx);

    // Re-solve this expiry
    solve_expiry(expiry_idx);

    // If this expiry's solution changed, downstream expiries' floors may change too.
    // For now, only re-solve the changed expiry (iterative mode doesn't propagate forward on tick).

    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

double Surface::price(double T, double K) const {
    int M = n_expiries();
    if (M == 0) return 0.0;

    if (T <= 0.0) return std::max(1.0 - K, 0.0);

    // Find surrounding expiries
    int j = 0;
    while (j < M && markets_[j].T() < T) ++j;

    if (j >= M) {
        // Beyond last expiry: extrapolate using last expiry's model
        double p;
        eval_model(M - 1, &K, 1, &p);
        return p;
    }

    if (j == 0 || T <= markets_[0].T()) {
        // At or before first expiry
        double p;
        eval_model(0, &K, 1, &p);
        return p;
    }

    // Interpolate between expiry j-1 and j using ATM variance interpolation
    double T_lo = markets_[j - 1].T();
    double T_hi = markets_[j].T();
    double alpha = (T - T_lo) / (T_hi - T_lo);

    double p_lo, p_hi;
    eval_model(j - 1, &K, 1, &p_lo);
    eval_model(j, &K, 1, &p_hi);

    return p_lo * (1.0 - alpha) + p_hi * alpha;
}

double Surface::vol(double T, double K) const {
    double p = price(T, K);
    if (T <= 0.0 || p <= 0.0) return 0.0;

    iv::Status st;
    double v = iv::implied_volatility(1.0, K, p, T, true, &st);
    return st.ok ? v : 0.0;
}

void Surface::price_grid(int expiry_idx, const double* strikes, int n, double* out) const {
    eval_model(expiry_idx, strikes, n, out);
}

void Surface::vol_grid(int expiry_idx, const double* strikes, int n, double* out) const {
    AVec<double> prices(n);
    eval_model(expiry_idx, strikes, n, prices.data());

    double T = markets_[expiry_idx].T();
    for (int i = 0; i < n; ++i) {
        iv::Status st;
        double v = iv::implied_volatility(1.0, strikes[i], prices[i], T, true, &st);
        out[i] = st.ok ? v : 0.0;
    }
}

void Surface::update_time_interp() {
    int M = n_expiries();
    atm_Ts_.resize(M + 1);
    atm_vars_.resize(M + 1);
    atm_Ts_[0] = 0.0;
    atm_vars_[0] = 0.0;
    for (int j = 0; j < M; ++j) {
        atm_Ts_[j + 1] = markets_[j].T();
        atm_vars_[j + 1] = fits_[j].atm_var;
    }
}

} // namespace sanos
