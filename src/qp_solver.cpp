#include "sanos/qp_solver.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>

namespace sanos {

void QPWorkspace::resize(int n_, int m_total) {
    n = n_;
    L.resize(n * n);
    x.resize(n);
    grad.resize(n);
    dx.resize(n);
    tmp.resize(std::max(n, m_total));
    tmp2.resize(std::max(n, m_total));
    lambda.resize(m_total);
    active_set.resize(m_total);
    n_active = 0;
}

// Cholesky decomposition of A (n x n, row-major) into L (lower triangular).
// Returns false if not positive definite.
static bool cholesky(double* L, const double* A, int n) {
    std::memcpy(L, A, n * n * sizeof(double));
    for (int j = 0; j < n; ++j) {
        double sum = L[j * n + j];
        for (int k = 0; k < j; ++k) {
            sum -= L[j * n + k] * L[j * n + k];
        }
        if (sum <= 0.0) return false;
        L[j * n + j] = std::sqrt(sum);
        double inv_ljj = 1.0 / L[j * n + j];
        for (int i = j + 1; i < n; ++i) {
            double s = L[i * n + j];
            for (int k = 0; k < j; ++k) {
                s -= L[i * n + k] * L[j * n + k];
            }
            L[i * n + j] = s * inv_ljj;
        }
        // Zero upper triangle
        for (int i = 0; i < j; ++i) {
            L[i * n + j] = 0.0;
        }
    }
    return true;
}

// Solve L * y = b where L is lower triangular
static void solve_lower(double* y, const double* L, const double* b, int n) {
    for (int i = 0; i < n; ++i) {
        double sum = b[i];
        for (int k = 0; k < i; ++k) {
            sum -= L[i * n + k] * y[k];
        }
        y[i] = sum / L[i * n + i];
    }
}

// Solve L^T * y = b where L is lower triangular
static void solve_upper(double* y, const double* L, const double* b, int n) {
    for (int i = n - 1; i >= 0; --i) {
        double sum = b[i];
        for (int k = i + 1; k < n; ++k) {
            sum -= L[k * n + i] * y[k];
        }
        y[i] = sum / L[i * n + i];
    }
}

// Solve H * x = b using pre-computed Cholesky L: L L^T x = b
static void solve_chol(double* x, const double* L, const double* b, double* tmp, int n) {
    solve_lower(tmp, L, b, n);
    solve_upper(x, L, tmp, n);
}

// Compute gradient: grad = H*x + f
static void compute_grad(double* grad, const double* H, const double* x, const double* f, int n) {
    for (int i = 0; i < n; ++i) {
        double sum = f[i];
        for (int j = 0; j < n; ++j) {
            sum += H[i * n + j] * x[j];
        }
        grad[i] = sum;
    }
}

// Evaluate constraint: a^T x - b (positive = satisfied for >= constraint)
static double eval_constraint(const double* a, const double* x, double b, int n) {
    double sum = 0.0;
    for (int j = 0; j < n; ++j) sum += a[j] * x[j];
    return sum - b;
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
    const int m_eq = prob.m_eq;
    const int m_ineq = prob.m_ineq;

    ws.resize(n, m_eq + m_ineq);

    QPResult result;

    // Step 1: Cholesky factorize H
    if (!cholesky(ws.L.data(), prob.H, n)) {
        result.status = QPStatus::NumericalError;
        return result;
    }

    // Step 2: Find unconstrained optimum: x* = -H^{-1} f
    // Then add equality constraints via null-space projection
    double* x = ws.x.data();
    double* grad = ws.grad.data();
    double* dx = ws.dx.data();

    if (warm_x) {
        std::memcpy(x, warm_x, n * sizeof(double));
    } else {
        // Unconstrained minimum: H x = -f
        for (int i = 0; i < n; ++i) ws.tmp[i] = -prob.f[i];
        solve_chol(x, ws.L.data(), ws.tmp.data(), ws.tmp2.data(), n);
    }

    // Step 3: Project onto equality constraints using iterative correction
    // For each equality constraint a^T x = b, we do a rank-1 correction.
    for (int pass = 0; pass < 3; ++pass) {
        for (int e = 0; e < m_eq; ++e) {
            const double* ae = prob.A_eq + e * n;
            double violation = eval_constraint(ae, x, prob.b_eq[e], n);
            if (std::abs(violation) < tol * 0.01) continue;

            // Compute d = H^{-1} a_e
            solve_chol(dx, ws.L.data(), ae, ws.tmp2.data(), n);

            // Step size: alpha = -violation / (a_e^T d)
            double denom = 0.0;
            for (int j = 0; j < n; ++j) denom += ae[j] * dx[j];
            if (std::abs(denom) < 1e-15) continue;

            double alpha = -violation / denom;
            for (int j = 0; j < n; ++j) x[j] += alpha * dx[j];
        }
    }

    // Step 4: Active-set iterations for inequality constraints
    ws.n_active = 0;
    AVec<bool> in_active(m_ineq, false);

    // Check which inequality constraints are violated and add the most violated
    for (int iter = 0; iter < max_iters; ++iter) {
        result.iters = iter + 1;
        compute_grad(grad, prob.H, x, prob.f, n);

        // Check KKT conditions:
        // 1. Find most violated inequality constraint
        int worst_ineq = -1;
        double worst_violation = -tol;

        for (int i = 0; i < m_ineq; ++i) {
            if (in_active[i]) continue;
            const double* ai = prob.A_ineq + i * n;
            double vi = eval_constraint(ai, x, prob.b_ineq[i], n);
            if (vi < worst_violation) {
                worst_violation = vi;
                worst_ineq = i;
            }
        }

        // 2. Check dual feasibility of active constraints
        // For active inequality constraints, compute Lagrange multipliers
        // lambda = (A_active H^{-1} A_active^T)^{-1} (A_active H^{-1} grad)
        // Simplified: check gradient projected onto active constraint normals
        int drop_idx = -1;
        double worst_lambda = 0.0;

        if (ws.n_active > 0 && worst_ineq < 0) {
            // Compute multipliers for active inequalities
            for (int a = 0; a < ws.n_active; ++a) {
                int ci = ws.active_set[a];
                const double* ai = prob.A_ineq + ci * n;

                // Approximate multiplier: lambda_a = a_i^T grad / (a_i^T H^{-1} a_i)
                solve_chol(dx, ws.L.data(), ai, ws.tmp2.data(), n);
                double atHia = 0.0;
                for (int j = 0; j < n; ++j) atHia += ai[j] * dx[j];

                double atg = 0.0;
                for (int j = 0; j < n; ++j) atg += ai[j] * grad[j];

                double lam = (std::abs(atHia) > 1e-15) ? atg / atHia : 0.0;

                // For >= constraints, multiplier should be >= 0
                if (lam < worst_lambda) {
                    worst_lambda = lam;
                    drop_idx = a;
                }
            }
        }

        // If all constraints satisfied and all multipliers non-negative, we're optimal
        if (worst_ineq < 0 && (ws.n_active == 0 || worst_lambda >= -tol)) {
            result.status = QPStatus::Optimal;
            break;
        }

        if (worst_ineq >= 0) {
            // Add most violated constraint to active set
            const double* ai = prob.A_ineq + worst_ineq * n;

            // Compute step direction: move x to satisfy this constraint
            // Direction: d = H^{-1} a_i
            solve_chol(dx, ws.L.data(), ai, ws.tmp2.data(), n);

            double violation = eval_constraint(ai, x, prob.b_ineq[worst_ineq], n);
            double denom = 0.0;
            for (int j = 0; j < n; ++j) denom += ai[j] * dx[j];

            if (std::abs(denom) < 1e-15) {
                // Constraint normal in null space of H — skip
                continue;
            }

            double alpha = -violation / denom;

            // Clamp step to not violate other constraints
            double max_alpha = alpha;
            for (int i = 0; i < m_ineq; ++i) {
                if (i == worst_ineq || in_active[i]) continue;
                const double* aj = prob.A_ineq + i * n;
                double ad = 0.0;
                for (int j = 0; j < n; ++j) ad += aj[j] * dx[j];
                if (ad < -1e-15) {
                    double slack = eval_constraint(aj, x, prob.b_ineq[i], n);
                    double step = -slack / ad;
                    if (step < max_alpha) max_alpha = step;
                }
            }

            // Also check equality constraint preservation
            for (int j = 0; j < n; ++j) x[j] += max_alpha * dx[j];

            // Re-project onto equality constraints
            for (int e = 0; e < m_eq; ++e) {
                const double* ae = prob.A_eq + e * n;
                double v = eval_constraint(ae, x, prob.b_eq[e], n);
                if (std::abs(v) < tol * 0.01) continue;
                solve_chol(dx, ws.L.data(), ae, ws.tmp2.data(), n);
                double d2 = 0.0;
                for (int j = 0; j < n; ++j) d2 += ae[j] * dx[j];
                if (std::abs(d2) < 1e-15) continue;
                double a2 = -v / d2;
                for (int j = 0; j < n; ++j) x[j] += a2 * dx[j];
            }

            // Add to active set
            ws.active_set[ws.n_active++] = worst_ineq;
            in_active[worst_ineq] = true;

        } else if (drop_idx >= 0) {
            // Drop a constraint with negative multiplier
            int ci = ws.active_set[drop_idx];
            in_active[ci] = false;

            // Remove from active set
            for (int a = drop_idx; a < ws.n_active - 1; ++a) {
                ws.active_set[a] = ws.active_set[a + 1];
            }
            ws.n_active--;

            // Re-optimize: take a step in the gradient direction projected to feasible space
            compute_grad(grad, prob.H, x, prob.f, n);

            // Steepest descent step in null space of active/equality constraints
            // Simple: just solve H dx = -grad, then clamp
            for (int j = 0; j < n; ++j) ws.tmp[j] = -grad[j];
            solve_chol(dx, ws.L.data(), ws.tmp.data(), ws.tmp2.data(), n);

            // Find max step that keeps all active constraints satisfied
            double step = 1.0;
            for (int i = 0; i < m_ineq; ++i) {
                const double* aj = prob.A_ineq + i * n;
                double ad = 0.0;
                for (int j = 0; j < n; ++j) ad += aj[j] * dx[j];
                if (ad < -1e-15) {
                    double slack = eval_constraint(aj, x, prob.b_ineq[i], n);
                    double s = -slack / ad;
                    if (s < step && s > 0.0) step = s;
                }
            }

            for (int j = 0; j < n; ++j) x[j] += step * dx[j];

            // Re-project equalities
            for (int e = 0; e < m_eq; ++e) {
                const double* ae = prob.A_eq + e * n;
                double v = eval_constraint(ae, x, prob.b_eq[e], n);
                if (std::abs(v) < tol * 0.01) continue;
                solve_chol(dx, ws.L.data(), ae, ws.tmp2.data(), n);
                double d2 = 0.0;
                for (int j = 0; j < n; ++j) d2 += ae[j] * dx[j];
                if (std::abs(d2) < 1e-15) continue;
                double a2 = -v / d2;
                for (int j = 0; j < n; ++j) x[j] += a2 * dx[j];
            }
        }

        if (iter == max_iters - 1) {
            result.status = QPStatus::MaxIters;
        }
    }

    // Compute objective
    double obj = 0.0;
    for (int i = 0; i < n; ++i) {
        obj += prob.f[i] * x[i];
        for (int j = 0; j < n; ++j) {
            obj += 0.5 * x[i] * prob.H[i * n + j] * x[j];
        }
    }
    result.obj = obj;

    std::memcpy(x_out, x, n * sizeof(double));
    return result;
}

} // namespace sanos
