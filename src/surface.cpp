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

// --- Model strike grid generation (wing-aware) ---

void build_model_strikes(
    AVec<double>& model_strikes,
    AVec<int>& market_ix,
    const double* strikes, int n,
    double min_k, double max_k,
    double max_dx, double min_dx)
{
    model_strikes.clear();
    market_ix.assign(n, -1);

    double outer_max_dx = std::min(0.5, max_dx * 10.0);

    // Left boundary: single point
    model_strikes.push_back(min_k);

    // First market strike
    market_ix[0] = static_cast<int>(model_strikes.size());
    model_strikes.push_back(strikes[0]);

    // Interior market strikes with adaptive fill
    for (int r = 1; r < n; ++r) {
        double dx = strikes[r] - model_strikes.back();
        if (dx <= min_dx) { market_ix[r] = -1; continue; }

        // Adaptive: use wider spacing in wings
        double local_dx = max_dx;
        double dist_from_atm = std::min(std::abs(strikes[r] - 1.0),
                                         std::abs(model_strikes.back() - 1.0));
        if (dist_from_atm > 0.15) local_dx = std::max(max_dx, 0.5);

        if (dx > local_dx) {
            int n_fill = std::max(1, static_cast<int>(dx / local_dx)) + 1;
            double prev = model_strikes.back();
            for (int f = 1; f < n_fill; ++f)
                model_strikes.push_back(prev + dx * f / n_fill);
        }
        market_ix[r] = static_cast<int>(model_strikes.size());
        model_strikes.push_back(strikes[r]);
    }

    // Right boundary: single point
    model_strikes.push_back(max_k);
}

// --- Surface implementation ---

Surface::Surface(SurfaceConfig cfg) : cfg_(cfg) {}

void Surface::ensure_pool() {
    if (!pool_) {
        int nt = cfg_.n_threads;
        if (nt <= 0) nt = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
        if (nt > 1) pool_ = std::make_unique<ThreadPool>(nt);
    }
}

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
    m.alloc(n);
    std::memcpy(m.strikes(), strikes, n * sizeof(double));
    std::memcpy(m.bids(), bids, n * sizeof(double));
    std::memcpy(m.asks(), asks, n * sizeof(double));
    std::memset(m.w_prev(), 0, n * sizeof(double));

    for (int i = 0; i < n; ++i) {
        double intr = std::max(1.0 - m.strikes()[i], 0.0);
        m.bids()[i] = std::max(m.bids()[i], intr);
        m.asks()[i] = std::min(m.asks()[i], 1.0);
        m.spreads()[i] = m.asks()[i] - m.bids()[i];
        m.mids()[i] = 0.5 * (m.bids()[i] + m.asks()[i]);
        double inv_spread = 1.0 / std::max(m.spreads()[i], 1e-12);
        m.weights()[i] = std::min(inv_spread, cfg_.max_inv_spread);
    }

    double wsum = 0.0;
    for (int i = 0; i < n; ++i) wsum += m.weights()[i];
    if (wsum > 0.0)
        for (int i = 0; i < n; ++i) m.weights()[i] /= wsum;

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
    for (int j = 0; j < n_expiries; ++j)
        add_expiry(labels[j], sqrtTs[j], strikes[j], n_strikes[j], bids[j], asks[j]);
}

// Fast ATM vol: only compute IV for the two strikes bracketing K=1
static double fast_atm_vol(const ExpiryMarket& m) {
    for (int i = 0; i < m.n() - 1; ++i) {
        if (m.strikes()[i] <= 1.0 && m.strikes()[i + 1] > 1.0) {
            double t = (1.0 - m.strikes()[i]) / (m.strikes()[i + 1] - m.strikes()[i]);
            double mid_lo = m.mids()[i], mid_hi = m.mids()[i + 1];
            double mid_atm = mid_lo * (1.0 - t) + mid_hi * t;

            // Single IV call for ATM mid price
            iv::Status st;
            double vol = iv::implied_volatility(1.0, 1.0, mid_atm, m.T(), true, &st);
            return st.ok ? vol : 0.15;  // fallback
        }
    }
    return 0.15;
}

void Surface::compute_iv(int j) {
    auto& m = markets_[j];
    int n = m.n();

    for (int i = 0; i < n; ++i) {
        double K = m.strikes()[i];
        iv::Status st;

        double bid_vol = iv::implied_volatility(1.0, K, m.bids()[i], m.T(), true, &st);
        m.iv_bids()[i] = st.ok ? bid_vol : 0.0;

        double ask_vol = iv::implied_volatility(1.0, K, m.asks()[i], m.T(), true, &st);
        m.iv_asks()[i] = st.ok ? ask_vol : 0.0;

        m.w_prev()[i] = (m.iv_bids()[i] + m.iv_asks()[i]) * 0.5;
        m.w_prev()[i] = m.w_prev()[i] * m.w_prev()[i] * m.T();
    }
}

void Surface::setup_expiry(int j) {
    auto& m = markets_[j];
    auto& f = fits_[j];

    build_model_strikes(
        f.model_strikes, f.market_ix,
        m.strikes(), m.n(),
        cfg_.min_k, cfg_.max_k,
        cfg_.max_dx, cfg_.min_dx);

    int N = f.N();

    // Fast ATM vol (single IV call, not full strip)
    f.atm_vol = fast_atm_vol(m);
    f.atm_var = f.atm_vol * f.atm_vol * m.T();

    double model_vol = cfg_.vol_fac * f.atm_vol;
    f.model_vols.assign(N, model_vol);

    f.q.resize(N);
    f.fitted.resize(m.n());
    f.iv_fitted.resize(m.n());

    f.kernel_dirty = true;
    f.fit_dirty = true;
}

void Surface::build_kernel(int j) {
    auto& m = markets_[j];
    auto& f = fits_[j];

    // Check cache: skip if ATM var hasn't changed significantly
    if (!f.kernel_dirty && f.cached_atm_var > 0.0) {
        double rel_change = std::abs(f.atm_var - f.cached_atm_var) / f.cached_atm_var;
        if (rel_change < cfg_.kernel_cache_tol) return;
    }

    int N = f.N();
    int n_mkt = m.n();
    double variance = cfg_.eta * f.atm_var;

    fill_kernel_matrix(f.C_market, m.strikes(), n_mkt,
                       f.model_strikes.data(), N, variance);

    // Pre-compute w^2 into persistent buffer (avoids allocation in hot path)
    f.w2.resize(n_mkt);
    for (int i = 0; i < n_mkt; ++i) f.w2[i] = m.weights()[i] * m.weights()[i];

    compute_hessian(f.H, f.C_market, f.w2.data(),
                    cfg_.smoothness_penalty / N, N);

    f.A_eq.assign(2 * N, 0.0);
    f.b_eq.assign(2, 0.0);
    for (int i = 0; i < N; ++i) {
        f.A_eq[i] = 1.0;
        f.A_eq[N + i] = f.model_strikes[i];
    }
    f.b_eq[0] = 1.0;
    f.b_eq[1] = 1.0;

    f.A_ineq.clear();
    f.b_ineq.clear();

    f.cached_atm_var = f.atm_var;
    f.kernel_dirty = false;

    // Invalidate Cholesky cache in QP workspace since H changed
    f.qp_ws.n = 0;
}

void Surface::build_qp(int j) {
    auto& f = fits_[j];
    auto& m = markets_[j];
    int n_mkt = m.n();

    // Pre-compute w^2 * mid into persistent buffer (zero allocation)
    f.w2mid.resize(n_mkt);
    for (int i = 0; i < n_mkt; ++i) f.w2mid[i] = f.w2[i] * m.mids()[i];

    f.f.resize(f.N());
    compute_gradient(f.f.data(), f.C_market, f.w2mid.data(), n_mkt, f.N());
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
    prob.H = f.H.data.data();
    prob.f = f.f.data();
    prob.A_ineq = m_ineq > 0 ? f.A_ineq.data() : nullptr;
    prob.b_ineq = m_ineq > 0 ? f.b_ineq.data() : nullptr;
    prob.A_eq = f.A_eq.data();
    prob.b_eq = f.b_eq.data();

    const double* warm = nullptr;
    for (int i = 0; i < N; ++i)
        if (f.q[i] != 0.0) { warm = f.q.data(); break; }

    qp_solve(f.q.data(), prob, f.qp_ws, cfg_.qp_tol, cfg_.max_qp_iters, warm);

    double qsum = 0.0;
    for (int i = 0; i < N; ++i) {
        f.q[i] = std::max(f.q[i], 0.0);
        qsum += f.q[i];
    }
    if (qsum > 0.0)
        for (int i = 0; i < N; ++i) f.q[i] /= qsum;

    mat_vec(f.fitted.data(), f.C_market, f.q.data());

    // Skip fitted IV computation here — compute on demand in vol_grid()
    f.fit_dirty = false;
}

// Process one expiry end-to-end (for parallel dispatch)
void Surface::calibrate_expiry(int j) {
    setup_expiry(j);
    build_kernel(j);
    build_qp(j);
    solve_expiry(j);
}

void Surface::eval_model(int j, const double* strikes, int n, double* prices) const {
    const auto& f = fits_[j];
    double variance = cfg_.eta * f.atm_var;
    int N = f.N();

    if (variance <= 0.0) {
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
                _mm256_sub_pd(v_half_v, v_log_k), v_inv_sqv);
            __m256d v_d2 = _mm256_sub_pd(v_d1, v_sqv);
            __m256d v_phi1 = simd::phicdf_4(v_d1);
            __m256d v_phi2 = simd::phicdf_4(v_d2);
            __m256d v_call = _mm256_mul_pd(v_ki,
                _mm256_sub_pd(v_phi1, _mm256_mul_pd(v_ratio, v_phi2)));
            v_price = _mm256_fmadd_pd(v_qi, v_call, v_price);
        }
        double price = simd::avx2_reduce_sum(v_price);
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

    // Build priority order: shortest DTE first so near-term expiries
    // are available for hedging as soon as possible.
    // The thread pool's work-stealing picks tasks in index order,
    // so sorting by urgency means the most important expiries finish first.
    AVec<int> order(M);
    for (int j = 0; j < M; ++j) order[j] = j;
    std::sort(order.begin(), order.end(), [this](int a, int b) {
        return markets_[a].T() < markets_[b].T();
    });

    ensure_pool();

    if (pool_ && M > 1) {
        pool_->parallel_for(M, [this, &order](int idx) {
            calibrate_expiry(order[idx]);
        });
    } else {
        for (int idx = 0; idx < M; ++idx) calibrate_expiry(order[idx]);
    }

    update_time_interp();

    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

double Surface::tick_update(int expiry_idx, int strike_idx, double new_bid, double new_ask) {
    auto t0 = std::chrono::high_resolution_clock::now();

    auto& m = markets_[expiry_idx];

    double intr = std::max(1.0 - m.strikes()[strike_idx], 0.0);
    m.bids()[strike_idx] = std::max(new_bid, intr);
    m.asks()[strike_idx] = std::min(new_ask, 1.0);
    m.spreads()[strike_idx] = m.asks()[strike_idx] - m.bids()[strike_idx];
    m.mids()[strike_idx] = 0.5 * (m.bids()[strike_idx] + m.asks()[strike_idx]);

    double inv_s = 1.0 / std::max(m.spreads()[strike_idx], 1e-12);
    m.weights()[strike_idx] = std::min(inv_s, cfg_.max_inv_spread);

    double wsum = 0.0;
    for (int i = 0; i < m.n(); ++i) wsum += m.weights()[i];
    if (wsum > 0.0)
        for (int i = 0; i < m.n(); ++i) m.weights()[i] /= wsum;

    // Update w^2 for the changed weight (single element, no full recompute)
    auto& f = fits_[expiry_idx];
    f.w2[strike_idx] = m.weights()[strike_idx] * m.weights()[strike_idx];

    build_qp(expiry_idx);
    solve_expiry(expiry_idx);

    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

double Surface::batch_update(const TickUpdate* ticks, int n_ticks) {
    auto t0 = std::chrono::high_resolution_clock::now();
    int M = n_expiries();

    // Apply all quote updates, track which expiries are dirty
    AVec<char> dirty(M, 0);
    for (int t = 0; t < n_ticks; ++t) {
        int ej = ticks[t].expiry_idx;
        int si = ticks[t].strike_idx;
        if (ej < 0 || ej >= M) continue;
        auto& m = markets_[ej];
        if (si < 0 || si >= m.n()) continue;

        double intr = std::max(1.0 - m.strikes()[si], 0.0);
        m.bids()[si] = std::max(ticks[t].new_bid, intr);
        m.asks()[si] = std::min(ticks[t].new_ask, 1.0);
        m.spreads()[si] = m.asks()[si] - m.bids()[si];
        m.mids()[si] = 0.5 * (m.bids()[si] + m.asks()[si]);

        double inv_s = 1.0 / std::max(m.spreads()[si], 1e-12);
        m.weights()[si] = std::min(inv_s, cfg_.max_inv_spread);
        dirty[ej] = 1;
    }

    // Renormalize weights and update w2 only for dirty expiries
    for (int j = 0; j < M; ++j) {
        if (!dirty[j]) continue;
        auto& m = markets_[j];
        auto& f = fits_[j];
        double wsum = 0.0;
        for (int i = 0; i < m.n(); ++i) wsum += m.weights()[i];
        if (wsum > 0.0)
            for (int i = 0; i < m.n(); ++i) m.weights()[i] /= wsum;
        for (int i = 0; i < m.n(); ++i) f.w2[i] = m.weights()[i] * m.weights()[i];
    }

    // Re-solve only dirty expiries
    for (int j = 0; j < M; ++j) {
        if (!dirty[j]) continue;
        build_qp(j);
        solve_expiry(j);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

double Surface::price(double T, double K) const {
    int M = n_expiries();
    if (M == 0) return 0.0;
    if (T <= 0.0) return std::max(1.0 - K, 0.0);

    int j = 0;
    while (j < M && markets_[j].T() < T) ++j;

    if (j >= M) { double p; eval_model(M - 1, &K, 1, &p); return p; }
    if (j == 0 || T <= markets_[0].T()) { double p; eval_model(0, &K, 1, &p); return p; }

    double T_lo = markets_[j - 1].T(), T_hi = markets_[j].T();
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
