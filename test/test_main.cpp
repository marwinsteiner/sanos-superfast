#include "sanos/surface.hpp"
#include "sanos/bs_kernel.hpp"
#include "sanos/qp_solver.hpp"
#include <cmath>
#include <cstdio>
#include <cassert>
#include <vector>
#include <cstring>

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        printf("FAIL: %s (line %d): %s\n", msg, __LINE__, #cond); \
        tests_failed++; \
    } else { \
        tests_passed++; \
    } \
} while(0)

#define CHECK_NEAR(a, b, tol, msg) do { \
    double _a = (a), _b = (b), _t = (tol); \
    if (std::abs(_a - _b) > _t) { \
        printf("FAIL: %s (line %d): %.10g != %.10g (tol %.2g)\n", msg, __LINE__, _a, _b, _t); \
        tests_failed++; \
    } else { \
        tests_passed++; \
    } \
} while(0)

void test_bs_call_price() {
    printf("--- test_bs_call_price ---\n");

    // ATM call with variance=0 should be 0 (max(1-1,0)=0)
    CHECK_NEAR(sanos::bs_call_price(1.0, 0.0), 0.0, 1e-12, "ATM linear");

    // Deep ITM: ratio=0.5, var=0 -> max(1-0.5,0)=0.5
    CHECK_NEAR(sanos::bs_call_price(0.5, 0.0), 0.5, 1e-12, "ITM linear");

    // Deep OTM: ratio=2.0, var=0 -> 0
    CHECK_NEAR(sanos::bs_call_price(2.0, 0.0), 0.0, 1e-12, "OTM linear");

    // ATM with var=0.04 (vol=20%, T=1)
    // BS ATM call = 2*Phi(0.1) - 1 ≈ 0.0797
    double atm = sanos::bs_call_price(1.0, 0.04);
    CHECK(atm > 0.07 && atm < 0.09, "ATM BS range");

    // Call is decreasing in strike ratio
    double c1 = sanos::bs_call_price(0.9, 0.04);
    double c2 = sanos::bs_call_price(1.0, 0.04);
    double c3 = sanos::bs_call_price(1.1, 0.04);
    CHECK(c1 > c2, "call decreasing 1");
    CHECK(c2 > c3, "call decreasing 2");
    CHECK(c3 > 0.0, "OTM call positive");
}

void test_kernel_matrix() {
    printf("--- test_kernel_matrix ---\n");

    double model[] = {0.5, 0.8, 1.0, 1.2, 1.5};
    double eval[]  = {0.7, 0.9, 1.0, 1.1, 1.3};
    int N = 5, n = 5;

    sanos::DenseMat C;
    sanos::fill_kernel_matrix(C, eval, n, model, N, 0.04);

    // All entries should be non-negative
    for (int i = 0; i < n * N; ++i) {
        CHECK(C.data[i] >= -1e-15, "kernel non-negative");
    }

    // For a uniform density q = (1/sum_K) * K, C*q should give valid prices
    // Just check dimensions
    CHECK(C.rows == n, "kernel rows");
    CHECK(C.cols == N, "kernel cols");
}

void test_qp_simple() {
    printf("--- test_qp_simple ---\n");

    // Simple QP: min 0.5*(x1^2 + x2^2) - x1 - x2
    // s.t. x1 + x2 <= 1, x1 >= 0, x2 >= 0
    // Solution: x1 = x2 = 0.5
    int n = 2;
    double H[] = {1.0, 0.0, 0.0, 1.0};
    double f[] = {-1.0, -1.0};

    // Inequality: x1 >= 0, x2 >= 0, x1+x2 >= ... no, we want x1+x2 <= 1
    // Rewrite as -(x1+x2) >= -1
    // Plus x1 >= 0, x2 >= 0
    int m_ineq = 3;
    double A_ineq[] = {
        1.0, 0.0,   // x1 >= 0
        0.0, 1.0,   // x2 >= 0
       -1.0,-1.0    // -x1-x2 >= -1
    };
    double b_ineq[] = {0.0, 0.0, -1.0};

    sanos::QPProblem prob;
    prob.n = n;
    prob.m_ineq = m_ineq;
    prob.m_eq = 0;
    prob.H = H;
    prob.f = f;
    prob.A_ineq = A_ineq;
    prob.b_ineq = b_ineq;

    sanos::QPWorkspace ws;
    double x[2];
    auto res = sanos::qp_solve(x, prob, ws);

    CHECK(res.status == sanos::QPStatus::Optimal, "qp status optimal");
    CHECK_NEAR(x[0], 0.5, 1e-6, "qp x1");
    CHECK_NEAR(x[1], 0.5, 1e-6, "qp x2");
}

void test_qp_with_equality() {
    printf("--- test_qp_with_equality ---\n");

    // min 0.5*(x1^2 + x2^2 + x3^2) - x2
    // s.t. x1 + x2 + x3 = 1 (sum to 1)
    //      x1 >= 0, x2 >= 0, x3 >= 0
    // Lagrangian: x_i = lambda + delta_i2 -> solution via KKT
    // x1=0, x3=0, x2=1 or x1=x3=(1-x2)/2
    // Actually: unconstrained on equality: x=(0,1,0) projected -> but x=lambda*(1,1,1)+(0,1,0)
    // With sum=1 and non-neg: x1=x3=0, x2=1? No...
    // min (x1^2+x2^2+x3^2)/2 - x2, x1+x2+x3=1, xi>=0
    // KKT: x1=mu, x2=mu+1, x3=mu, sum: 3mu+1=1 -> mu=0, x=(0,1,0)
    int n = 3;
    double H[] = {1,0,0, 0,1,0, 0,0,1};
    double f[] = {0, -1, 0};

    double A_ineq[] = {1,0,0, 0,1,0, 0,0,1};
    double b_ineq[] = {0, 0, 0};

    double A_eq[] = {1, 1, 1};
    double b_eq[] = {1};

    sanos::QPProblem prob;
    prob.n = n;
    prob.m_ineq = 3;
    prob.m_eq = 1;
    prob.H = H;
    prob.f = f;
    prob.A_ineq = A_ineq;
    prob.b_ineq = b_ineq;
    prob.A_eq = A_eq;
    prob.b_eq = b_eq;

    sanos::QPWorkspace ws;
    double x[3];
    auto res = sanos::qp_solve(x, prob, ws);

    CHECK(res.status == sanos::QPStatus::Optimal, "qp_eq status optimal");
    CHECK_NEAR(x[0], 0.0, 1e-4, "qp_eq x1");
    CHECK_NEAR(x[1], 1.0, 1e-4, "qp_eq x2");
    CHECK_NEAR(x[2], 0.0, 1e-4, "qp_eq x3");
    CHECK_NEAR(x[0] + x[1] + x[2], 1.0, 1e-8, "qp_eq sum");
}

void test_model_strikes() {
    printf("--- test_model_strikes ---\n");

    double strikes[] = {0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15};
    int n = 7;

    sanos::AVec<double> model;
    sanos::AVec<int> ix;
    sanos::build_model_strikes(model, ix, strikes, n, 0.1, 3.0, 0.05, 1e-6);

    CHECK(model.size() > 7, "model strikes expanded");
    CHECK(model.front() <= 0.1, "model min_k");
    CHECK(model.back() >= 3.0, "model max_k");

    // Check monotonicity
    for (size_t i = 1; i < model.size(); ++i) {
        CHECK(model[i] > model[i - 1], "model strikes monotone");
    }

    // Check market strikes are in model strikes
    for (int i = 0; i < n; ++i) {
        if (ix[i] >= 0) {
            CHECK_NEAR(model[ix[i]], strikes[i], 1e-12, "market strike in model");
        }
    }
}

void test_surface_basic() {
    printf("--- test_surface_basic ---\n");

    // Create a simple 1-expiry surface
    sanos::Surface surf;

    // 30 DTE, ~50 strikes
    double sqrtT = std::sqrt(30.0 / 365.0);
    double vol = 0.20;
    double T = 30.0 / 365.0;
    int ns = 30;

    std::vector<double> strikes(ns), bids(ns), asks(ns);
    for (int i = 0; i < ns; ++i) {
        double K = 0.85 + 0.3 * i / (ns - 1);
        strikes[i] = K;

        double log_m = std::log(K);
        double total_var = vol * vol * T;
        double sqrt_v = std::sqrt(total_var);
        double d1 = (-log_m + 0.5 * total_var) / sqrt_v;
        double d2 = d1 - sqrt_v;
        double mid = 0.5 * std::erfc(-d1 * 0.7071067811865475)
                   - K * 0.5 * std::erfc(-d2 * 0.7071067811865475);

        double spread = std::max(0.001, 0.01 * std::exp(-0.5 * d1 * d1));
        bids[i] = std::max(mid - spread, std::max(1.0 - K, 0.0) + 1e-6);
        asks[i] = std::min(mid + spread, 1.0 - 1e-6);
        asks[i] = std::max(asks[i], bids[i] + 1e-6);
    }

    surf.add_expiry("30DTE", sqrtT, strikes.data(), ns, bids.data(), asks.data());
    double cal_us = surf.calibrate();

    printf("  Calibration time: %.1f us\n", cal_us);
    CHECK(cal_us < 50000, "calibration under 50ms");  // generous for first run

    // Check that fitted prices are reasonable
    auto& fit = surf.fit(0);
    auto& mkt = surf.market(0);
    int n_outside = 0;
    for (int i = 0; i < ns; ++i) {
        if (fit.fitted[i] < mkt.bids[i] - 0.01 || fit.fitted[i] > mkt.asks[i] + 0.01) {
            n_outside++;
        }
    }
    printf("  Strikes outside bid/ask (with 1%% tolerance): %d/%d\n", n_outside, ns);
    CHECK(n_outside < ns / 2, "most fits within bid/ask");

    // Check price query works
    double p = surf.price(T, 1.0);
    CHECK(p > 0.0 && p < 0.5, "ATM price reasonable");

    // Check vol query works
    double v = surf.vol(T, 1.0);
    CHECK(v > 0.1 && v < 0.4, "ATM vol reasonable");
    printf("  ATM vol query: %.4f (expected ~%.4f)\n", v, vol);
}

void test_tick_update() {
    printf("--- test_tick_update ---\n");

    sanos::Surface surf;
    double sqrtT = std::sqrt(30.0 / 365.0);
    double vol = 0.20;
    double T = 30.0 / 365.0;
    int ns = 20;

    std::vector<double> strikes(ns), bids(ns), asks(ns);
    for (int i = 0; i < ns; ++i) {
        double K = 0.90 + 0.2 * i / (ns - 1);
        strikes[i] = K;
        double log_m = std::log(K);
        double total_var = vol * vol * T;
        double sqrt_v = std::sqrt(total_var);
        double d1 = (-log_m + 0.5 * total_var) / sqrt_v;
        double d2 = d1 - sqrt_v;
        double mid = 0.5 * std::erfc(-d1 * 0.7071067811865475)
                   - K * 0.5 * std::erfc(-d2 * 0.7071067811865475);
        double spread = 0.005;
        bids[i] = std::max(mid - spread, std::max(1.0 - K, 0.0) + 1e-6);
        asks[i] = std::min(mid + spread, 1.0 - 1e-6);
        asks[i] = std::max(asks[i], bids[i] + 1e-6);
    }

    surf.add_expiry("30DTE", sqrtT, strikes.data(), ns, bids.data(), asks.data());
    surf.calibrate();

    // Simulate a tick update on the ATM strike
    int atm_idx = ns / 2;
    double new_bid = bids[atm_idx] + 0.001;
    double new_ask = asks[atm_idx] + 0.001;

    double us = surf.tick_update(0, atm_idx, new_bid, new_ask);
    printf("  Tick update time: %.1f us\n", us);
    CHECK(us < 5000, "tick update under 5ms");
}

int main() {
    printf("sanos-superfast tests\n");
    printf("=====================\n\n");

    test_bs_call_price();
    test_kernel_matrix();
    test_qp_simple();
    test_qp_with_equality();
    test_model_strikes();
    test_surface_basic();
    test_tick_update();

    printf("\n=====================\n");
    printf("Passed: %d, Failed: %d\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
