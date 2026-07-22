#pragma once
#include "types.hpp"
#include <cmath>

namespace sanos {

double bs_call_price(double forward_ratio, double total_variance);

void fill_kernel_matrix(
    DenseMat& out,
    const double* eval_strikes,   int n_eval,
    const double* model_strikes,  int n_model,
    double variance);

void fill_cross_kernel(
    DenseMat& out,
    const double* cur_model_strikes,  int n_cur,
    const double* prev_model_strikes, int n_prev,
    double prev_variance);

void mat_vec(double* y, const DenseMat& A, const double* x);
void mat_t_vec(double* y, const DenseMat& A, const double* x);

// Compute H = C^T diag(w2) C + lambda*I.
// w2 is pre-squared weights (length m). Caller owns the buffer.
void compute_hessian(
    DenseMat& H,
    const DenseMat& C,
    const double* w2,       // pre-computed weights^2, length m
    double lambda,
    int n);

// Compute f = -C^T diag(w2) mid, using pre-computed w2*mid buffer.
// w2mid is pre-computed w^2 * mid (length m). Caller owns the buffer.
void compute_gradient(
    double* f,
    const DenseMat& C,
    const double* w2mid,    // pre-computed w^2 * mid, length m
    int m, int n);

} // namespace sanos
