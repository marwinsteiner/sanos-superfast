#pragma once
#include "types.hpp"

namespace sanos {

// Warm-started active-set QP solver (Goldfarb-Idnani style).
//
// Solves:  min  0.5 * x^T H x + f^T x
//          s.t. A_ineq * x >= b_ineq    (m_ineq inequality constraints)
//               A_eq   * x  = b_eq      (m_eq   equality constraints)
//
// H must be symmetric positive definite (n x n).
// A_ineq is m_ineq x n, A_eq is m_eq x n.
//
// Warm start: provide a feasible x0 and the previous active set.
// If x0 is nullptr, starts from unconstrained optimum projected to feasibility.

enum class QPStatus {
    Optimal,
    MaxIters,
    Infeasible,
    NumericalError
};

struct QPResult {
    QPStatus status = QPStatus::Optimal;
    int      iters  = 0;
    double   obj    = 0.0;
};

struct QPProblem {
    // Dimensions
    int n       = 0;  // variables
    int m_ineq  = 0;  // inequality constraints
    int m_eq    = 0;  // equality constraints

    // Problem data (all point to externally owned memory)
    const double* H      = nullptr;  // n x n symmetric PD (row-major)
    const double* f      = nullptr;  // n
    const double* A_ineq = nullptr;  // m_ineq x n (row-major)
    const double* b_ineq = nullptr;  // m_ineq
    const double* A_eq   = nullptr;  // m_eq x n (row-major)
    const double* b_eq   = nullptr;  // m_eq
};

// Solver workspace — pre-allocate once, reuse across solves.
struct QPWorkspace {
    AVec<double> L;          // Cholesky factor of working Hessian (n x n)
    AVec<double> x;          // current solution
    AVec<double> grad;       // gradient at x
    AVec<double> dx;         // step direction
    AVec<double> tmp;        // scratch
    AVec<double> tmp2;       // scratch
    AVec<double> lambda;     // dual variables for active constraints
    AVec<int>    active_set; // indices of active inequality constraints
    int          n_active = 0;
    int          n = 0;

    void resize(int n, int m_total);
};

// Solve the QP. x_out must have space for n doubles.
// If warm_x is not null, it's used as starting point (must be feasible).
QPResult qp_solve(
    double* x_out,
    const QPProblem& prob,
    QPWorkspace& ws,
    double tol = 1e-10,
    int max_iters = 200,
    const double* warm_x = nullptr);

} // namespace sanos
