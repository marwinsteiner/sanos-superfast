#pragma once
#include "types.hpp"

namespace sanos {

// Active-set QP solver with Schur complement KKT solve.
//
// Solves:  min  0.5 * x^T H x + f^T x
//          s.t. A_eq * x = b_eq       (m_eq equality constraints)
//               x >= 0                (non-negativity, handled by active set)
//
// H must be symmetric positive definite (n x n, row-major).
// For SANOS: m_eq = 2 (sum=1, K^T q=1), plus non-negativity q >= 0.
// Inequality constraints (floor) are not supported in this fast solver.

enum class QPStatus { Optimal, MaxIters, NumericalError };

struct QPResult {
    QPStatus status = QPStatus::Optimal;
    int      iters  = 0;
    double   obj    = 0.0;
};

struct QPProblem {
    int n       = 0;
    int m_ineq  = 0;  // unused in fast solver (kept for API compat)
    int m_eq    = 0;

    const double* H      = nullptr;
    const double* f      = nullptr;
    const double* A_ineq = nullptr;  // unused
    const double* b_ineq = nullptr;  // unused
    const double* A_eq   = nullptr;  // m_eq x n row-major
    const double* b_eq   = nullptr;  // m_eq
};

struct QPWorkspace {
    // Pre-allocated buffers (sized for max N)
    AVec<double> H_r;         // reduced Hessian (n_free x n_free)
    AVec<double> L_r;         // Cholesky of reduced Hessian
    AVec<double> f_r;         // reduced gradient
    AVec<double> A_r;         // reduced equality constraints
    AVec<double> Hinv_At;     // H^{-1} A^T columns
    AVec<double> q;           // current solution
    AVec<double> t1, t2, t3;  // scratch
    AVec<int>    free_idx;    // indices of free variables
    AVec<char>   free_mask;   // 1=free, 0=fixed
    int n = 0;

    void resize(int n_, int m_eq);
};

QPResult qp_solve(
    double* x_out,
    const QPProblem& prob,
    QPWorkspace& ws,
    double tol = 1e-8,
    int max_iters = 200,
    const double* warm_x = nullptr);

} // namespace sanos
