#include "sanos/qp_solver.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

namespace sanos {

void QPWorkspace::resize(int n_, int m_total) {
    if (n == n_) return;
    n = n_;
    L.resize(n * n);
    z.resize(n);
    u.resize(n);
    q.resize(n);
    rhs.resize(n);
    tmp.resize(std::max(n, std::max(m_total, 1)));
    tmp2.resize(std::max(n, std::max(m_total, 1)));
    factored = false;
    warm = false;
    cached_rho = -1.0;
}

// Cholesky of symmetric PD matrix A (n×n row-major) into L (lower triangular)
static bool cholesky_factor(double* L, const double* A, int n) {
    std::memcpy(L, A, n * n * sizeof(double));
    for (int j = 0; j < n; ++j) {
        double sum = L[j * n + j];
        for (int k = 0; k < j; ++k)
            sum -= L[j * n + k] * L[j * n + k];
        if (sum <= 1e-30) return false;
        L[j * n + j] = std::sqrt(sum);
        double inv = 1.0 / L[j * n + j];
        for (int i = j + 1; i < n; ++i) {
            double s = L[i * n + j];
            for (int k = 0; k < j; ++k)
                s -= L[i * n + k] * L[j * n + k];
            L[i * n + j] = s * inv;
        }
        for (int i = 0; i < j; ++i)
            L[i * n + j] = 0.0;
    }
    return true;
}

static void solve_L(double* y, const double* L, const double* b, int n) {
    for (int i = 0; i < n; ++i) {
        double sum = b[i];
        for (int k = 0; k < i; ++k) sum -= L[i * n + k] * y[k];
        y[i] = sum / L[i * n + i];
    }
}

static void solve_Lt(double* y, const double* L, const double* b, int n) {
    for (int i = n - 1; i >= 0; --i) {
        double sum = b[i];
        for (int k = i + 1; k < n; ++k) sum -= L[k * n + i] * y[k];
        y[i] = sum / L[i * n + i];
    }
}

static void chol_solve(double* x, const double* L, const double* b, double* tmp, int n) {
    solve_L(tmp, L, b, n);
    solve_Lt(x, L, tmp, n);
}

// Solve the equality-constrained QP using Schur complement:
//   min 0.5 x^T H x + f^T x  s.t. A_eq x = b_eq
//
// KKT: H x + f + A^T lambda = 0, A x = b
// Via Schur complement:
//   S = A H^{-1} A^T  (m_eq × m_eq, solve using pre-factored H)
//   g = -A H^{-1} f - b
//   S lambda = g
//   x = H^{-1}(-f - A^T lambda)
//
// free_mask: if non-null, only include variables where free_mask[i] is true.
// Fixed variables are set to 0.
static void solve_eq_constrained(
    double* x_out, int n,
    const double* H, const double* f,
    const double* A_eq, const double* b_eq, int m_eq,
    const char* free_mask,  // null = all free
    double* L_buf, double* tmp1, double* tmp2, double* tmp3)
{
    // Count free variables
    int n_free = 0;
    std::vector<int> free_idx;
    if (free_mask) {
        for (int i = 0; i < n; ++i) if (free_mask[i]) free_idx.push_back(i);
        n_free = static_cast<int>(free_idx.size());
    } else {
        n_free = n;
        free_idx.resize(n);
        for (int i = 0; i < n; ++i) free_idx[i] = i;
    }

    if (n_free == 0) {
        std::memset(x_out, 0, n * sizeof(double));
        return;
    }

    // Build reduced H_r (n_free × n_free) and f_r (n_free)
    std::vector<double> H_r(n_free * n_free);
    std::vector<double> f_r(n_free);

    for (int i = 0; i < n_free; ++i) {
        f_r[i] = f[free_idx[i]];
        for (int j = 0; j < n_free; ++j)
            H_r[i * n_free + j] = H[free_idx[i] * n + free_idx[j]];
    }

    // Reduced A_eq (m_eq × n_free)
    std::vector<double> A_r(m_eq * n_free);
    for (int e = 0; e < m_eq; ++e)
        for (int j = 0; j < n_free; ++j)
            A_r[e * n_free + j] = A_eq[e * n + free_idx[j]];

    // Factorize reduced H
    std::vector<double> L_r(n_free * n_free);
    if (!cholesky_factor(L_r.data(), H_r.data(), n_free)) {
        for (int i = 0; i < n_free; ++i) H_r[i * n_free + i] += 1e-6;
        cholesky_factor(L_r.data(), H_r.data(), n_free);
    }

    std::vector<double> t1(n_free), t2(n_free), t3(n_free);

    if (m_eq == 0) {
        // No equality constraints: x = H^{-1}(-f)
        for (int i = 0; i < n_free; ++i) t1[i] = -f_r[i];
        chol_solve(t2.data(), L_r.data(), t1.data(), t3.data(), n_free);
        std::memset(x_out, 0, n * sizeof(double));
        for (int i = 0; i < n_free; ++i) x_out[free_idx[i]] = t2[i];
        return;
    }

    // Schur complement: S = A_r H_r^{-1} A_r^T  (m_eq × m_eq)
    // For m_eq = 2, S is 2x2.
    std::vector<double> Hinv_At(n_free * m_eq); // H^{-1} A^T, stored as m_eq columns of length n_free
    for (int e = 0; e < m_eq; ++e) {
        // Column e: solve H_r v = A_r[e,:]
        chol_solve(Hinv_At.data() + e * n_free, L_r.data(),
                   A_r.data() + e * n_free, t3.data(), n_free);
    }

    // S[e1, e2] = A_r[e1,:] * Hinv_At[:,e2]
    std::vector<double> S(m_eq * m_eq);
    for (int e1 = 0; e1 < m_eq; ++e1)
        for (int e2 = 0; e2 < m_eq; ++e2) {
            double sum = 0.0;
            for (int j = 0; j < n_free; ++j)
                sum += A_r[e1 * n_free + j] * Hinv_At[e2 * n_free + j];
            S[e1 * m_eq + e2] = sum;
        }

    // g = -A_r H_r^{-1} f_r - b_eq
    // First: H_r^{-1} f_r
    for (int i = 0; i < n_free; ++i) t1[i] = f_r[i];
    chol_solve(t2.data(), L_r.data(), t1.data(), t3.data(), n_free);

    std::vector<double> g(m_eq);
    for (int e = 0; e < m_eq; ++e) {
        double sum = 0.0;
        for (int j = 0; j < n_free; ++j) sum += A_r[e * n_free + j] * t2[j];
        g[e] = -sum - b_eq[e];
    }

    // Solve S lambda = g
    std::vector<double> lambda(m_eq);
    if (m_eq == 1) {
        lambda[0] = g[0] / S[0];
    } else if (m_eq == 2) {
        double det = S[0] * S[3] - S[1] * S[2];
        if (std::abs(det) < 1e-30) det = 1e-30;
        lambda[0] = (S[3] * g[0] - S[1] * g[1]) / det;
        lambda[1] = (-S[2] * g[0] + S[0] * g[1]) / det;
    } else {
        // General case: solve S lambda = g via Cholesky
        std::vector<double> L_s(m_eq * m_eq), ts(m_eq);
        cholesky_factor(L_s.data(), S.data(), m_eq);
        chol_solve(lambda.data(), L_s.data(), g.data(), ts.data(), m_eq);
    }

    // x = H^{-1}(-f - A^T lambda)
    for (int i = 0; i < n_free; ++i) {
        double rhs = -f_r[i];
        for (int e = 0; e < m_eq; ++e)
            rhs -= A_r[e * n_free + i] * lambda[e];
        t1[i] = rhs;
    }
    chol_solve(t2.data(), L_r.data(), t1.data(), t3.data(), n_free);

    std::memset(x_out, 0, n * sizeof(double));
    for (int i = 0; i < n_free; ++i) x_out[free_idx[i]] = t2[i];
}

QPResult qp_solve(
    double* x_out,
    const QPProblem& prob,
    QPWorkspace& ws,
    double tol,
    int max_iters,
    const double* warm_x)
{
    const int n = prob.n;
    ws.resize(n, prob.m_eq + prob.m_ineq);

    QPResult result;
    double* q = ws.q.data();

    // Active-set method with Schur complement KKT solve.
    // Active set = indices where q_i is fixed to 0 (binding non-negativity).
    std::vector<char> free_mask(n, true);

    // Initialize active set from warm start hint
    if (warm_x) {
        for (int i = 0; i < n; ++i)
            free_mask[i] = (warm_x[i] > 1e-12);
    }

    for (int iter = 0; iter < max_iters; ++iter) {
        result.iters = iter + 1;

        // Solve equality-constrained QP on free variables
        solve_eq_constrained(q, n, prob.H, prob.f,
                             prob.A_eq, prob.b_eq, prob.m_eq,
                             free_mask.data(),
                             ws.L.data(), ws.tmp.data(), ws.tmp2.data(), ws.rhs.data());

        // Check for negative components and add to active set
        bool changed = false;
        int most_neg_idx = -1;
        double most_neg = 0.0;
        for (int i = 0; i < n; ++i) {
            if (free_mask[i] && q[i] < -tol) {
                if (q[i] < most_neg) {
                    most_neg = q[i];
                    most_neg_idx = i;
                }
            }
        }

        if (most_neg_idx >= 0) {
            // Add most negative to active set
            free_mask[most_neg_idx] = false;
            q[most_neg_idx] = 0.0;
            changed = true;
        }

        if (!changed) {
            // All free variables non-negative. Check dual feasibility:
            // For active variables, the gradient (reduced cost) should be >= 0
            // grad_i = (Hq + f)_i + sum_e lambda_e * A_eq[e,i]
            // We need to compute the gradient at the current q.
            bool dual_ok = true;
            int worst_dual_idx = -1;
            double worst_dual = 0.0;

            for (int i = 0; i < n; ++i) {
                if (!free_mask[i]) {
                    // Compute gradient component
                    double gi = prob.f[i];
                    for (int j = 0; j < n; ++j) gi += prob.H[i * n + j] * q[j];
                    // For non-negativity constraint q_i >= 0, the dual variable is -gi
                    // It should be >= 0, so gi should be <= 0... wait.
                    // Actually: the KKT condition is H q + f + A^T lambda - mu = 0
                    // where mu >= 0 are multipliers for q >= 0.
                    // For active constraint q_i = 0: mu_i = (H q + f + A^T lambda)_i
                    // We want mu_i >= 0.
                    // Without computing lambda explicitly, we can check:
                    // If releasing q_i from 0 would decrease the objective, then mu_i < 0.
                    // This happens when gi < 0 (gradient points into the feasible region).
                    if (gi < -tol && gi < worst_dual) {
                        worst_dual = gi;
                        worst_dual_idx = i;
                    }
                }
            }

            if (worst_dual_idx >= 0) {
                // Release this variable from active set
                free_mask[worst_dual_idx] = true;
                changed = true;
            }

            if (!changed) {
                result.status = QPStatus::Optimal;
                break;
            }
        }

        if (iter == max_iters - 1)
            result.status = QPStatus::MaxIters;
    }

    // Ensure non-negativity (numerical cleanup)
    double qsum = 0.0;
    for (int i = 0; i < n; ++i) {
        q[i] = std::max(q[i], 0.0);
        qsum += q[i];
    }
    if (qsum > 1e-15) {
        for (int i = 0; i < n; ++i) q[i] /= qsum;
    }

    // Compute objective
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
