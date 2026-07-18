#pragma once

// MSVC compatibility shims for volfi's GCC/Clang builtins
#ifdef _MSC_VER
    #ifndef __builtin_expect
        #define __builtin_expect(expr, val) (expr)
    #endif
    #ifndef __attribute__
        #define __attribute__(x)
    #endif
    #ifndef __builtin_unreachable
        #define __builtin_unreachable() __assume(0)
    #endif
#endif

// volfi's annulus engine uses __attribute__((noinline)) which MSVC doesn't support.
// On MSVC, we use the simpler paper_volfi.hpp (v0.1 kernel) for IV inversion,
// which is still machine-precision and fast (~200ns/quote scalar).
// On GCC/Clang, we use the full annulus engine with SIMD batch drivers.

#ifdef _MSC_VER
    #include <volfi/volfi.hpp>
#else
    #include <volfi/volfi_annulus_all.hpp>
#endif

namespace sanos {
namespace iv {

struct Status {
    bool ok = false;
};

// Portable implied volatility: forward=F, strike=K, price=undiscounted option price, T=time
inline double implied_volatility(double F, double K, double price, double T, bool is_call, Status* st = nullptr) {
    if (T <= 0.0 || price <= 0.0 || F <= 0.0 || K <= 0.0) {
        if (st) st->ok = false;
        return 0.0;
    }

    // Normalize: pure strike k = K/F, pure price c = price/F
    double k = K / F;
    double c = price / F;

    // Put-call parity: if put, convert to call price: c_call = c_put + 1 - k
    if (!is_call) {
        c = c + 1.0 - k;
    }

    // Intrinsic check
    double intrinsic = (1.0 - k > 0.0) ? 1.0 - k : 0.0;
    if (c <= intrinsic) {
        if (st) st->ok = false;
        return 0.0;
    }
    if (c >= 1.0) {
        if (st) st->ok = false;
        return 0.0;
    }

    // Convert to OTM call: h = |log(k)|, c_otm
    double log_k = std::log(k);
    double h = std::abs(log_k);
    double c_otm;

    if (log_k <= 0.0) {
        // ITM call -> use put-call parity: c_otm_put = c_call - (1-k) = c - 1 + k
        c_otm = c - 1.0 + k;
    } else {
        c_otm = c;
    }

    if (c_otm <= 0.0 || c_otm >= 1.0) {
        if (st) st->ok = false;
        return 0.0;
    }

#ifdef _MSC_VER
    double w;
    if (h < 1e-14) {
        // ATM case: use dedicated ATM solver
        w = volfi::implied_variance_atm(c_otm);
    } else {
        if (!volfi::valid_otm(h, c_otm)) {
            if (st) st->ok = false;
            return 0.0;
        }
        w = volfi::implied_variance_otm(h, c_otm);
    }
    if (std::isnan(w) || w <= 0.0) {
        if (st) st->ok = false;
        return 0.0;
    }
    if (st) st->ok = true;
    return std::sqrt(w / T);
#else
    double w;
    if (h < 1e-14) {
        w = volfi::implied_variance_atm(c_otm);
    } else {
        volfi_annulus::iv_status vst;
        w = volfi_annulus::implied_variance_otm_checked(h, c_otm, &vst);
        if (vst != volfi_annulus::iv_status::ok) {
            if (st) st->ok = false;
            return 0.0;
        }
    }
    if (std::isnan(w) || w <= 0.0) {
        if (st) st->ok = false;
        return 0.0;
    }
    if (st) st->ok = true;
    return std::sqrt(w / T);
#endif
}

// Warm implied variance update (from previous total variance)
inline double implied_variance_warm(double h, double c_otm, double w_prev) {
    if (h <= 0.0 || c_otm <= 0.0 || c_otm >= 1.0) return w_prev;

#ifdef _MSC_VER
    // No warm path in v0.1 — just re-solve
    if (!volfi::valid_otm(h, c_otm)) return w_prev;
    double w = volfi::implied_variance_otm(h, c_otm);
    return std::isnan(w) ? w_prev : w;
#else
    return volfi_annulus::implied_variance_warm(h, c_otm, w_prev);
#endif
}

} // namespace iv
} // namespace sanos
