#pragma once

// Detect compiler: Clang (even on Windows MSVC ABI) supports GCC builtins.
// Only pure MSVC (cl.exe without Clang) needs shims.
#if defined(_MSC_VER) && !defined(__clang__)
    #define SANOS_PURE_MSVC 1
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

// On pure MSVC: use volfi v0.1 paper solver (portable, no GCC builtins).
// On Clang/GCC: use full annulus engine with SIMD batch drivers.
#ifdef SANOS_PURE_MSVC
    #include <volfi/volfi.hpp>
#else
    #include <volfi/volfi_annulus_all.hpp>
#endif

#include <cmath>

namespace sanos {
namespace iv {

struct Status {
    bool ok = false;
};

inline double implied_volatility(double F, double K, double price, double T, bool is_call, Status* st = nullptr) {
    if (T <= 0.0 || price <= 0.0 || F <= 0.0 || K <= 0.0) {
        if (st) st->ok = false;
        return 0.0;
    }

    double k = K / F;
    double c = price / F;

    if (!is_call) c = c + 1.0 - k;

    double intrinsic = (1.0 - k > 0.0) ? 1.0 - k : 0.0;
    if (c <= intrinsic || c >= 1.0) {
        if (st) st->ok = false;
        return 0.0;
    }

    double log_k = std::log(k);
    double h = std::abs(log_k);
    double c_otm = (log_k <= 0.0) ? (c - 1.0 + k) : c;

    if (c_otm <= 0.0 || c_otm >= 1.0) {
        if (st) st->ok = false;
        return 0.0;
    }

    double w;
    if (h < 1e-14) {
        w = volfi::implied_variance_atm(c_otm);
    } else {
#ifdef SANOS_PURE_MSVC
        if (!volfi::valid_otm(h, c_otm)) {
            if (st) st->ok = false;
            return 0.0;
        }
        w = volfi::implied_variance_otm(h, c_otm);
#else
        volfi_annulus::iv_status vst;
        w = volfi_annulus::implied_variance_otm_checked(h, c_otm, &vst);
        if (vst != volfi_annulus::iv_status::ok) {
            if (st) st->ok = false;
            return 0.0;
        }
#endif
    }

    if (std::isnan(w) || w <= 0.0) {
        if (st) st->ok = false;
        return 0.0;
    }
    if (st) st->ok = true;
    return std::sqrt(w / T);
}

inline double implied_variance_warm(double h, double c_otm, double w_prev) {
    if (h <= 0.0 || c_otm <= 0.0 || c_otm >= 1.0) return w_prev;

#ifdef SANOS_PURE_MSVC
    if (!volfi::valid_otm(h, c_otm)) return w_prev;
    double w = volfi::implied_variance_otm(h, c_otm);
    return std::isnan(w) ? w_prev : w;
#else
    return volfi_annulus::implied_variance_warm(h, c_otm, w_prev);
#endif
}

} // namespace iv
} // namespace sanos
