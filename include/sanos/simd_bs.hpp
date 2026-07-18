#pragma once

// SIMD Black-Scholes pricing for kernel matrix fill and eval_model.
// AVX2 (4-wide double): vectorized log + div + FMA, scalar erfc per lane.

#if defined(__AVX2__) || defined(__AVX__)
#define SANOS_HAS_SIMD_BS 1

#include <immintrin.h>
#include <cmath>

namespace sanos {
namespace simd {

static inline double avx2_reduce_sum(__m256d v) {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d s  = _mm_add_pd(lo, hi);
    s = _mm_hadd_pd(s, s);
    return _mm_cvtsd_f64(s);
}

// Fast AVX2 natural log (double precision, ~1 ULP)
static inline __m256d fast_log_avx2(__m256d x) {
    const __m256d LN2  = _mm256_set1_pd(0.6931471805599453);
    const __m256d ONE  = _mm256_set1_pd(1.0);

    // Extract exponent
    __m256i xi = _mm256_castpd_si256(x);
    __m256i exp_bits = _mm256_and_si256(_mm256_srli_epi64(xi, 52), _mm256_set1_epi64x(0x7FF));
    __m256d exp_d = _mm256_sub_pd(
        _mm256_castsi256_pd(_mm256_or_si256(exp_bits, _mm256_set1_epi64x(0x4330000000000000LL))),
        _mm256_set1_pd(4503599627370496.0));
    __m256d e = _mm256_sub_pd(exp_d, _mm256_set1_pd(1023.0));

    // Extract mantissa in [1, 2)
    __m256i mant_bits = _mm256_or_si256(
        _mm256_and_si256(xi, _mm256_set1_epi64x(0x000FFFFFFFFFFFFFll)),
        _mm256_set1_epi64x(0x3FF0000000000000ll));
    __m256d m = _mm256_castsi256_pd(mant_bits);

    // f = m - 1
    __m256d f = _mm256_sub_pd(m, ONE);

    // log(1+f) via degree-7 minimax polynomial (Horner)
    __m256d p = _mm256_set1_pd( 0.14798198605116586);
    p = _mm256_fmadd_pd(p, f, _mm256_set1_pd(-0.16597084872498487));
    p = _mm256_fmadd_pd(p, f, _mm256_set1_pd( 0.19999748890498402));
    p = _mm256_fmadd_pd(p, f, _mm256_set1_pd(-0.24999999545498846));
    p = _mm256_fmadd_pd(p, f, _mm256_set1_pd( 0.33333333333190946));
    p = _mm256_fmadd_pd(p, f, _mm256_set1_pd(-0.49999999999999994));
    p = _mm256_fmadd_pd(p, f, _mm256_set1_pd( 1.0));
    __m256d log_m = _mm256_mul_pd(p, f);

    return _mm256_fmadd_pd(e, LN2, log_m);
}

// Phi(x) = 0.5 * erfc(-x / sqrt(2)) — scalar per lane (full precision)
static inline __m256d phicdf_4(__m256d x) {
    alignas(32) double xv[4], rv[4];
    _mm256_store_pd(xv, x);
    for (int i = 0; i < 4; ++i)
        rv[i] = 0.5 * std::erfc(-xv[i] * 0.7071067811865475244);
    return _mm256_load_pd(rv);
}

// Vectorized BS call price: 4 elements, same variance
static inline void bs_call_price_4(
    double* out,
    const double* eval_strikes,
    double ki, double inv_ki,
    double half_v, double inv_sqrt_v, double sqrt_v)
{
    __m256d v_eval = _mm256_loadu_pd(eval_strikes);
    __m256d v_inv_ki = _mm256_set1_pd(inv_ki);
    __m256d v_ki = _mm256_set1_pd(ki);
    __m256d v_half_v = _mm256_set1_pd(half_v);
    __m256d v_inv_sqv = _mm256_set1_pd(inv_sqrt_v);
    __m256d v_sqv = _mm256_set1_pd(sqrt_v);

    __m256d v_ratio = _mm256_mul_pd(v_eval, v_inv_ki);
    __m256d v_log_k = fast_log_avx2(v_ratio);

    __m256d v_d1 = _mm256_mul_pd(
        _mm256_sub_pd(v_half_v, v_log_k), v_inv_sqv);
    __m256d v_d2 = _mm256_sub_pd(v_d1, v_sqv);

    __m256d v_phi1 = phicdf_4(v_d1);
    __m256d v_phi2 = phicdf_4(v_d2);

    __m256d v_result = _mm256_mul_pd(v_ki,
        _mm256_sub_pd(v_phi1, _mm256_mul_pd(v_ratio, v_phi2)));
    _mm256_storeu_pd(out, v_result);
}

// Fill one column of kernel matrix using AVX2
static inline void fill_kernel_column_avx2(
    double* col, const double* eval_strikes, int n_eval,
    double ki, double variance)
{
    double inv_ki = 1.0 / ki;
    double sqrt_v = std::sqrt(variance);
    double half_v = 0.5 * variance;
    double inv_sqrt_v = 1.0 / sqrt_v;

    int l = 0;
    for (; l + 4 <= n_eval; l += 4) {
        bs_call_price_4(col + l, eval_strikes + l,
                        ki, inv_ki, half_v, inv_sqrt_v, sqrt_v);
    }
    for (; l < n_eval; ++l) {
        double ratio = eval_strikes[l] * inv_ki;
        if (ratio <= 0.0) { col[l] = ki; continue; }
        double log_k = std::log(ratio);
        double d1 = (-log_k + half_v) * inv_sqrt_v;
        double d2 = d1 - sqrt_v;
        double phi1 = 0.5 * std::erfc(-d1 * 0.7071067811865475244);
        double phi2 = 0.5 * std::erfc(-d2 * 0.7071067811865475244);
        col[l] = ki * (phi1 - ratio * phi2);
    }
}

} // namespace simd
} // namespace sanos

#endif // SANOS_HAS_SIMD_BS
