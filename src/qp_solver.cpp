#include "sanos/qp_solver.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>

#if defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#define QP_HAS_AVX2 1

static inline double avx2_hsum(__m256d v) {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d s  = _mm_add_pd(lo, hi);
    s = _mm_hadd_pd(s, s);
    return _mm_cvtsd_f64(s);
}
#endif

namespace sanos {

void QPWorkspace::resize(int n_, int m_eq) {
    if (n == n_) return;
    n = n_;
    H_r.resize(n * n);
    L_r.resize(n * n);
    f_r.resize(n);
    A_r.resize(m_eq * n);
    Hinv_At.resize(n * m_eq);
    q.resize(n);
    t1.resize(n);
    t2.resize(n);
    t3.resize(n);
    free_idx.resize(n);
    free_mask.resize(n);
}

// --- Cholesky with AVX2 inner loops ---

static bool cholesky_factor(double* L, const double* A, int n) {
    std::memcpy(L, A, n * n * sizeof(double));
    for (int j = 0; j < n; ++j) {
        // Diagonal: L[j,j] = sqrt(A[j,j] - sum_k L[j,k]^2)
        const double* Lj = L + j * n;
        double sum = L[j * n + j];
#ifdef QP_HAS_AVX2
        {
            __m256d vsum = _mm256_setzero_pd();
            int k = 0;
            for (; k + 4 <= j; k += 4) {
                __m256d v = _mm256_loadu_pd(Lj + k);
                vsum = _mm256_fmadd_pd(v, v, vsum);
            }
            sum -= avx2_hsum(vsum);
            for (; k < j; ++k) sum -= Lj[k] * Lj[k];
        }
#else
        for (int k = 0; k < j; ++k) sum -= Lj[k] * Lj[k];
#endif
        if (sum <= 1e-30) return false;
        L[j * n + j] = std::sqrt(sum);
        double inv = 1.0 / L[j * n + j];

        // Column: L[i,j] = (A[i,j] - sum_k L[i,k]*L[j,k]) / L[j,j]
        for (int i = j + 1; i < n; ++i) {
            double* Li = L + i * n;
            double s = Li[j];
#ifdef QP_HAS_AVX2
            {
                __m256d vs = _mm256_setzero_pd();
                int k = 0;
                for (; k + 4 <= j; k += 4) {
                    __m256d vi = _mm256_loadu_pd(Li + k);
                    __m256d vj = _mm256_loadu_pd(Lj + k);
                    vs = _mm256_fmadd_pd(vi, vj, vs);
                }
                s -= avx2_hsum(vs);
                for (; k < j; ++k) s -= Li[k] * Lj[k];
            }
#else
            for (int k = 0; k < j; ++k) s -= Li[k] * Lj[k];
#endif
            Li[j] = s * inv;
        }
        // Zero upper triangle
        for (int i = 0; i < j; ++i) L[i * n + j] = 0.0;
    }
    return true;
}

// --- Triangular solves with AVX2 ---

static void solve_L(double* y, const double* L, const double* b, int n) {
    for (int i = 0; i < n; ++i) {
        const double* Li = L + i * n;
        double sum = b[i];
#ifdef QP_HAS_AVX2
        {
            __m256d vs = _mm256_setzero_pd();
            int k = 0;
            for (; k + 4 <= i; k += 4) {
                __m256d vl = _mm256_loadu_pd(Li + k);
                __m256d vy = _mm256_loadu_pd(y + k);
                vs = _mm256_fmadd_pd(vl, vy, vs);
            }
            sum -= avx2_hsum(vs);
            for (; k < i; ++k) sum -= Li[k] * y[k];
        }
#else
        for (int k = 0; k < i; ++k) sum -= Li[k] * y[k];
#endif
        y[i] = sum / Li[i];
    }
}

static void solve_Lt(double* y, const double* L, const double* b, int n) {
    for (int i = n - 1; i >= 0; --i) {
        double sum = b[i];
#ifdef QP_HAS_AVX2
        {
            __m256d vs = _mm256_setzero_pd();
            int k = i + 1;
            for (; k + 4 <= n; k += 4) {
                __m256d vl = _mm256_set_pd(L[(k+3)*n+i], L[(k+2)*n+i], L[(k+1)*n+i], L[k*n+i]);
                __m256d vy = _mm256_loadu_pd(y + k);
                vs = _mm256_fmadd_pd(vl, vy, vs);
            }
            sum -= avx2_hsum(vs);
            for (; k < n; ++k) sum -= L[k * n + i] * y[k];
        }
#else
        for (int k = i + 1; k < n; ++k) sum -= L[k * n + i] * y[k];
#endif
        y[i] = sum / L[i * n + i];
    }
}

static void chol_solve(double* x, const double* L, const double* b, double* tmp, int n) {
    solve_L(tmp, L, b, n);
    solve_Lt(x, L, tmp, n);
}

// --- KKT solve via Schur complement ---

static void solve_kkt(
    double* x_out, int n,
    const double* H, const double* f,
    const double* A_eq, const double* b_eq, int m_eq,
    const int* free_idx, int n_free,
    double* H_r, double* L_r, double* f_r, double* A_r,
    double* Hinv_At, double* t1, double* t2, double* t3)
{
    if (n_free == 0) {
        std::memset(x_out, 0, n * sizeof(double));
        return;
    }

    // Build reduced system from free indices
    for (int i = 0; i < n_free; ++i) {
        f_r[i] = f[free_idx[i]];
        for (int j = 0; j < n_free; ++j)
            H_r[i * n_free + j] = H[free_idx[i] * n + free_idx[j]];
    }
    for (int e = 0; e < m_eq; ++e)
        for (int j = 0; j < n_free; ++j)
            A_r[e * n_free + j] = A_eq[e * n + free_idx[j]];

    if (!cholesky_factor(L_r, H_r, n_free)) {
        for (int i = 0; i < n_free; ++i) H_r[i * n_free + i] += 1e-6;
        cholesky_factor(L_r, H_r, n_free);
    }

    if (m_eq == 0) {
        for (int i = 0; i < n_free; ++i) t1[i] = -f_r[i];
        chol_solve(t2, L_r, t1, t3, n_free);
        std::memset(x_out, 0, n * sizeof(double));
        for (int i = 0; i < n_free; ++i) x_out[free_idx[i]] = t2[i];
        return;
    }

    // Schur complement
    for (int e = 0; e < m_eq; ++e)
        chol_solve(Hinv_At + e * n_free, L_r, A_r + e * n_free, t3, n_free);

    double S[4] = {};
    for (int e1 = 0; e1 < m_eq; ++e1)
        for (int e2 = 0; e2 < m_eq; ++e2) {
            double sum = 0.0;
#ifdef QP_HAS_AVX2
            __m256d vs = _mm256_setzero_pd();
            int j = 0;
            for (; j + 4 <= n_free; j += 4) {
                __m256d va = _mm256_loadu_pd(A_r + e1 * n_free + j);
                __m256d vh = _mm256_loadu_pd(Hinv_At + e2 * n_free + j);
                vs = _mm256_fmadd_pd(va, vh, vs);
            }
            sum = avx2_hsum(vs);
            for (; j < n_free; ++j)
                sum += A_r[e1 * n_free + j] * Hinv_At[e2 * n_free + j];
#else
            for (int j = 0; j < n_free; ++j)
                sum += A_r[e1 * n_free + j] * Hinv_At[e2 * n_free + j];
#endif
            S[e1 * m_eq + e2] = sum;
        }

    chol_solve(t2, L_r, f_r, t3, n_free);
    double g[2] = {};
    for (int e = 0; e < m_eq; ++e) {
        double sum = 0.0;
        for (int j = 0; j < n_free; ++j) sum += A_r[e * n_free + j] * t2[j];
        g[e] = -sum - b_eq[e];
    }

    double lambda[2] = {};
    if (m_eq == 1) {
        lambda[0] = (std::abs(S[0]) > 1e-30) ? g[0] / S[0] : 0.0;
    } else {
        double det = S[0] * S[3] - S[1] * S[2];
        if (std::abs(det) > 1e-30) {
            lambda[0] = (S[3] * g[0] - S[1] * g[1]) / det;
            lambda[1] = (-S[2] * g[0] + S[0] * g[1]) / det;
        }
    }

    for (int i = 0; i < n_free; ++i) {
        double rhs = -f_r[i];
        for (int e = 0; e < m_eq; ++e)
            rhs -= A_r[e * n_free + i] * lambda[e];
        t1[i] = rhs;
    }
    chol_solve(t2, L_r, t1, t3, n_free);

    std::memset(x_out, 0, n * sizeof(double));
    for (int i = 0; i < n_free; ++i) x_out[free_idx[i]] = t2[i];
}

// --- Main QP solve ---

QPResult qp_solve(
    double* x_out,
    const QPProblem& prob,
    QPWorkspace& ws,
    double tol,
    int max_iters,
    const double* warm_x)
{
    const int n = prob.n;
    const int m_eq = prob.m_eq;
    ws.resize(n, m_eq);

    QPResult result;
    double* q = ws.q.data();
    char* fmask = ws.free_mask.data();
    int* fidx = ws.free_idx.data();

    if (warm_x) {
        for (int i = 0; i < n; ++i)
            fmask[i] = (warm_x[i] > 1e-12) ? 1 : 0;
    } else {
        std::memset(fmask, 1, n);
    }

    for (int iter = 0; iter < max_iters; ++iter) {
        result.iters = iter + 1;

        int nf = 0;
        for (int i = 0; i < n; ++i)
            if (fmask[i]) fidx[nf++] = i;

        solve_kkt(q, n, prob.H, prob.f, prob.A_eq, prob.b_eq, m_eq,
                  fidx, nf,
                  ws.H_r.data(), ws.L_r.data(), ws.f_r.data(), ws.A_r.data(),
                  ws.Hinv_At.data(), ws.t1.data(), ws.t2.data(), ws.t3.data());

        // Batch-add all negative variables to active set
        int n_neg = 0;
        for (int i = 0; i < n; ++i) {
            if (fmask[i] && q[i] < -tol) {
                fmask[i] = 0;
                q[i] = 0.0;
                n_neg++;
            }
        }
        if (n_neg > 0) continue;

        // Check dual feasibility: gradient at active variables
        int release = -1;
        double worst_grad = 0.0;
        for (int i = 0; i < n; ++i) {
            if (!fmask[i]) {
                double gi = prob.f[i];
#ifdef QP_HAS_AVX2
                __m256d vs = _mm256_setzero_pd();
                int j = 0;
                for (; j + 4 <= n; j += 4) {
                    __m256d vh = _mm256_loadu_pd(prob.H + i * n + j);
                    __m256d vq = _mm256_loadu_pd(q + j);
                    vs = _mm256_fmadd_pd(vh, vq, vs);
                }
                gi += avx2_hsum(vs);
                for (; j < n; ++j) gi += prob.H[i * n + j] * q[j];
#else
                for (int j = 0; j < n; ++j) gi += prob.H[i * n + j] * q[j];
#endif
                if (gi < -tol && gi < worst_grad) {
                    worst_grad = gi;
                    release = i;
                }
            }
        }

        if (release >= 0) {
            fmask[release] = 1;
            continue;
        }

        result.status = QPStatus::Optimal;
        break;
    }

    if (result.iters >= max_iters)
        result.status = QPStatus::MaxIters;

    double qsum = 0.0;
    for (int i = 0; i < n; ++i) {
        q[i] = std::max(q[i], 0.0);
        qsum += q[i];
    }
    if (qsum > 1e-15)
        for (int i = 0; i < n; ++i) q[i] /= qsum;

    double obj = 0.0;
    for (int i = 0; i < n; ++i) {
        obj += prob.f[i] * q[i];
        for (int j = 0; j < n; ++j)
            obj += 0.5 * q[i] * prob.H[i * n + j] * q[j];
    }
    result.obj = obj;

    std::memcpy(x_out, q, n * sizeof(double));
    return result;
}

} // namespace sanos
