#include "sanos/bs_kernel.hpp"
#include "sanos/volfi_compat.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>

#ifdef _MSC_VER
#include <immintrin.h>
#else
#include <x86intrin.h>
#endif

// SIMD BS pricing (Clang/GCC only — uses volfi's AVX2 log/erfc)
#if !defined(SANOS_PURE_MSVC) && (defined(__AVX2__) || defined(__AVX__))
#include "sanos/simd_bs.hpp"
#endif

namespace sanos {

static inline double phi_cdf(double x) {
    return 0.5 * std::erfc(-x * 0.7071067811865475244);
}

double bs_call_price(double forward_ratio, double total_variance) {
    if (total_variance <= 0.0) {
        return std::max(1.0 - forward_ratio, 0.0);
    }
    if (forward_ratio <= 0.0) return 1.0;

    double sqrt_v = std::sqrt(total_variance);
    double log_k  = std::log(forward_ratio);
    double d1     = (-log_k + 0.5 * total_variance) / sqrt_v;
    double d2     = d1 - sqrt_v;
    return phi_cdf(d1) - forward_ratio * phi_cdf(d2);
}

void fill_kernel_matrix(
    DenseMat& out,
    const double* eval_strikes,   int n_eval,
    const double* model_strikes,  int n_model,
    double variance)
{
    out.resize(n_eval, n_model);

    // Pre-hoist sqrt(variance) — avoids n_eval * n_model redundant sqrts
    double sqrt_v = 0.0, half_v = 0.0, inv_sqrt_v = 0.0;
    if (variance > 0.0) {
        sqrt_v = std::sqrt(variance);
        half_v = 0.5 * variance;
        inv_sqrt_v = 1.0 / sqrt_v;
    }

    if (variance <= 0.0) {
        for (int i = 0; i < n_model; ++i) {
            double* col = out.col_ptr(i);
            double ki = model_strikes[i];
            for (int l = 0; l < n_eval; ++l) {
                col[l] = std::max(ki - eval_strikes[l], 0.0);
            }
        }
    } else {
#ifdef SANOS_HAS_SIMD_BS
        // SIMD path: 4-wide AVX2 BS pricing per column
        for (int i = 0; i < n_model; ++i) {
            simd::fill_kernel_column_avx2(
                out.col_ptr(i), eval_strikes, n_eval,
                model_strikes[i], variance);
        }
#else
        for (int i = 0; i < n_model; ++i) {
            double* col = out.col_ptr(i);
            double ki = model_strikes[i];
            double inv_ki = 1.0 / ki;
            for (int l = 0; l < n_eval; ++l) {
                double ratio = eval_strikes[l] * inv_ki;
                if (ratio <= 0.0) { col[l] = ki; continue; }
                double log_k = std::log(ratio);
                double d1 = (-log_k + half_v) * inv_sqrt_v;
                double d2 = d1 - sqrt_v;
                col[l] = ki * (phi_cdf(d1) - ratio * phi_cdf(d2));
            }
        }
#endif
    }
}

void fill_cross_kernel(
    DenseMat& out,
    const double* cur_model_strikes,  int n_cur,
    const double* prev_model_strikes, int n_prev,
    double prev_variance)
{
    out.resize(n_cur, n_prev);
    double sqrt_v = 0.0, half_v = 0.0, inv_sqrt_v = 0.0;
    if (prev_variance > 0.0) {
        sqrt_v = std::sqrt(prev_variance);
        half_v = 0.5 * prev_variance;
        inv_sqrt_v = 1.0 / sqrt_v;
    }

    if (prev_variance <= 0.0) {
        for (int i = 0; i < n_prev; ++i) {
            double* col = out.col_ptr(i);
            double ki = prev_model_strikes[i];
            for (int l = 0; l < n_cur; ++l) {
                col[l] = std::max(ki - cur_model_strikes[l], 0.0);
            }
        }
    } else {
#ifdef SANOS_HAS_SIMD_BS
        for (int i = 0; i < n_prev; ++i) {
            simd::fill_kernel_column_avx2(
                out.col_ptr(i), cur_model_strikes, n_cur,
                prev_model_strikes[i], prev_variance);
        }
#else
        for (int i = 0; i < n_prev; ++i) {
            double* col = out.col_ptr(i);
            double ki = prev_model_strikes[i];
            double inv_ki = 1.0 / ki;
            for (int l = 0; l < n_cur; ++l) {
                double ratio = cur_model_strikes[l] * inv_ki;
                if (ratio <= 0.0) { col[l] = ki; continue; }
                double log_k = std::log(ratio);
                double d1 = (-log_k + half_v) * inv_sqrt_v;
                double d2 = d1 - sqrt_v;
                col[l] = ki * (phi_cdf(d1) - ratio * phi_cdf(d2));
            }
        }
#endif
    }
}

// --- AVX2 helpers ---

#if defined(__AVX2__) || defined(__AVX__)
#define SANOS_HAS_AVX2 1
#endif

// MSVC with /arch:AVX2 defines __AVX2__. If not, fall back to scalar.
#ifdef SANOS_HAS_AVX2

static inline __m256d avx2_hsum_partial(__m256d v) {
    // Returns the horizontal sum in all lanes (approximately)
    __m256d s = _mm256_hadd_pd(v, v); // [a+b, a+b, c+d, c+d]
    return s;
}

static inline double avx2_hsum(__m256d v) {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d s  = _mm_add_pd(lo, hi);
    s = _mm_hadd_pd(s, s);
    return _mm_cvtsd_f64(s);
}

void mat_vec(double* y, const DenseMat& A, const double* x) {
    int m = A.rows, n = A.cols;
    std::memset(y, 0, m * sizeof(double));
    for (int j = 0; j < n; ++j) {
        const double* col = A.col_ptr(j);
        __m256d vxj = _mm256_set1_pd(x[j]);
        int i = 0;
        for (; i + 4 <= m; i += 4) {
            __m256d vy = _mm256_loadu_pd(y + i);
            __m256d vc = _mm256_loadu_pd(col + i);
            vy = _mm256_fmadd_pd(vc, vxj, vy);
            _mm256_storeu_pd(y + i, vy);
        }
        double xj = x[j];
        for (; i < m; ++i) y[i] += col[i] * xj;
    }
}

void mat_t_vec(double* y, const DenseMat& A, const double* x) {
    int m = A.rows, n = A.cols;
    for (int j = 0; j < n; ++j) {
        const double* col = A.col_ptr(j);
        __m256d vsum = _mm256_setzero_pd();
        int i = 0;
        for (; i + 4 <= m; i += 4) {
            __m256d vc = _mm256_loadu_pd(col + i);
            __m256d vx = _mm256_loadu_pd(x + i);
            vsum = _mm256_fmadd_pd(vc, vx, vsum);
        }
        double sum = avx2_hsum(vsum);
        for (; i < m; ++i) sum += col[i] * x[i];
        y[j] = sum;
    }
}

void compute_hessian(
    DenseMat& H,
    const DenseMat& C,
    const double* w2,
    double lambda,
    int n)
{
    int m = C.rows;
    H.resize(n, n);

    for (int j = 0; j < n; ++j) {
        const double* cj = C.col_ptr(j);
        // Prefetch next column pair to hide L2 latency
        if (j + 1 < n) _mm_prefetch(reinterpret_cast<const char*>(C.col_ptr(j + 1)), _MM_HINT_T0);
        for (int k = j; k < n; ++k) {
            const double* ck = C.col_ptr(k);
            if (k + 1 < n) _mm_prefetch(reinterpret_cast<const char*>(C.col_ptr(k + 1)), _MM_HINT_T1);
            __m256d vsum = _mm256_setzero_pd();
            int i = 0;
            for (; i + 4 <= m; i += 4) {
                __m256d vcj = _mm256_loadu_pd(cj + i);
                __m256d vck = _mm256_loadu_pd(ck + i);
                __m256d vw2 = _mm256_loadu_pd(w2 + i);
                vsum = _mm256_fmadd_pd(_mm256_mul_pd(vcj, vw2), vck, vsum);
            }
            double sum = avx2_hsum(vsum);
            for (; i < m; ++i) sum += cj[i] * w2[i] * ck[i];
            if (j == k) sum += lambda;
            H(j, k) = sum;
            H(k, j) = sum;
        }
    }
}

void compute_gradient(
    double* f,
    const DenseMat& C,
    const double* w2mid,
    int m, int n)
{
    for (int j = 0; j < n; ++j) {
        const double* cj = C.col_ptr(j);
        __m256d vsum = _mm256_setzero_pd();
        int i = 0;
        for (; i + 4 <= m; i += 4) {
            __m256d vc  = _mm256_loadu_pd(cj + i);
            __m256d vwm = _mm256_loadu_pd(w2mid + i);
            vsum = _mm256_fmadd_pd(vc, vwm, vsum);
        }
        double sum = avx2_hsum(vsum);
        for (; i < m; ++i) sum += cj[i] * w2mid[i];
        f[j] = -sum;
    }
}

#else // No AVX2 — scalar fallback

void mat_vec(double* y, const DenseMat& A, const double* x) {
    int m = A.rows, n = A.cols;
    std::memset(y, 0, m * sizeof(double));
    for (int j = 0; j < n; ++j) {
        const double* col = A.col_ptr(j);
        double xj = x[j];
        for (int i = 0; i < m; ++i) y[i] += col[i] * xj;
    }
}

void mat_t_vec(double* y, const DenseMat& A, const double* x) {
    int m = A.rows, n = A.cols;
    for (int j = 0; j < n; ++j) {
        const double* col = A.col_ptr(j);
        double sum = 0.0;
        for (int i = 0; i < m; ++i) sum += col[i] * x[i];
        y[j] = sum;
    }
}

void compute_hessian(
    DenseMat& H,
    const DenseMat& C,
    const double* w2,
    double lambda,
    int n)
{
    int m = C.rows;
    H.resize(n, n);
    for (int j = 0; j < n; ++j) {
        const double* cj = C.col_ptr(j);
        for (int k = j; k < n; ++k) {
            const double* ck = C.col_ptr(k);
            double sum = 0.0;
            for (int i = 0; i < m; ++i) sum += cj[i] * w2[i] * ck[i];
            if (j == k) sum += lambda;
            H(j, k) = sum;
            H(k, j) = sum;
        }
    }
}

void compute_gradient(
    double* f,
    const DenseMat& C,
    const double* w2mid,
    int m, int n)
{
    for (int j = 0; j < n; ++j) {
        const double* cj = C.col_ptr(j);
        double sum = 0.0;
        for (int i = 0; i < m; ++i) sum += cj[i] * w2mid[i];
        f[j] = -sum;
    }
}

#endif // SANOS_HAS_AVX2

} // namespace sanos
