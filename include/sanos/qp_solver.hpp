#pragma once
#include "types.hpp"

namespace sanos {

// ADMM-based QP solver for SANOS density estimation.
//
// Solves:  min  0.5 * x^T H x + f^T x
//          s.t. A_ineq * x >= b_ineq    (m_ineq inequality constraints)
//               A_eq   * x  = b_eq      (m_eq   equality constraints)
//
// Uses ADMM splitting: min g(q) + I_C(z) s.t. q = z
// where g(q) = 0.5 q^T H q + f^T q and C is the constraint set.
//
// q-update: q = (H + rho*I)^{-1} (rho*(z - u) - f)   [cached Cholesky]
// z-update: z = project_C(q + u)
// u-update: u += q - z

enum class QPStatus {
    Optimal,
    MaxIters,
    NumericalError
};

struct QPResult {
    QPStatus status = QPStatus::Optimal;
    int      iters  = 0;
    double   obj    = 0.0;
};

struct QPProblem {
    int n       = 0;
    int m_ineq  = 0;
    int m_eq    = 0;

    const double* H      = nullptr;  // n x n (row-major, symmetric)
    const double* f      = nullptr;  // n
    const double* A_ineq = nullptr;  // m_ineq x n (row-major)
    const double* b_ineq = nullptr;  // m_ineq
    const double* A_eq   = nullptr;  // m_eq x n (row-major)
    const double* b_eq   = nullptr;  // m_eq
};

struct QPWorkspace {
    // Cholesky of (H + rho*I)
    AVec<double> L;
    bool factored = false;
    double cached_rho = -1.0;

    // ADMM state (warm-started across solves)
    AVec<double> z;     // consensus variable
    AVec<double> u;     // scaled dual variable
    AVec<double> q;     // primal
    AVec<double> rhs;   // scratch for linear solve
    AVec<double> tmp;   // scratch
    AVec<double> tmp2;  // scratch
    int n = 0;
    bool warm = false;

    void resize(int n_, int m_total);
};

QPResult qp_solve(
    double* x_out,
    const QPProblem& prob,
    QPWorkspace& ws,
    double tol = 1e-8,
    int max_iters = 200,
    const double* warm_x = nullptr);

} // namespace sanos
