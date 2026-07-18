#pragma once
#include "types.hpp"
#include <cmath>

namespace sanos {

// Black-Scholes call price for normalized (pure) strikes.
// Call(anchor, strike, variance) = anchor * BS_call(strike/anchor, variance)
// where BS_call(k, v) = Phi(d1) - k*Phi(d2), d1 = (-ln(k) + v/2)/sqrt(v), d2 = d1 - sqrt(v)
//
// For the SANOS kernel: C[l,i] = K_i * BS_call(k_l / K_i, eta * V)
// where K_i are model strikes, k_l are evaluation strikes, V is ATM variance.

double bs_call_price(double forward_ratio, double total_variance);

// Fill kernel matrix C[n_eval x n_model] where:
//   C[l,i] = model_strikes[i] * bs_call_price(eval_strikes[l] / model_strikes[i], variance)
// If variance == 0 (linear mode): C[l,i] = max(model_strikes[i] - eval_strikes[l], 0)
void fill_kernel_matrix(
    DenseMat& out,
    const double* eval_strikes,   int n_eval,
    const double* model_strikes,  int n_model,
    double variance);

// Fill kernel matrix for cross-expiry monotonicity constraint:
//   R[l,i] = prev_model_strikes[i] * bs_call_price(cur_model_strikes[l] / prev_model_strikes[i], prev_variance)
void fill_cross_kernel(
    DenseMat& out,
    const double* cur_model_strikes,  int n_cur,
    const double* prev_model_strikes, int n_prev,
    double prev_variance);

// Matrix-vector product: y = A * x (A is m x n, x is n, y is m)
void mat_vec(double* y, const DenseMat& A, const double* x);

// Matrix-transpose-vector product: y = A^T * x (A is m x n, x is m, y is n)
void mat_t_vec(double* y, const DenseMat& A, const double* x);

// Compute H = C^T W^2 C + lambda * I  (n x n symmetric, stored as DenseMat)
// W is diagonal weight vector of length m, C is m x n
void compute_hessian(
    DenseMat& H,
    const DenseMat& C,
    const double* weights,  // length m
    double lambda,          // regularizer
    int n);

// Compute f = -C^T W^2 mid  (length n)
void compute_gradient(
    double* f,
    const DenseMat& C,
    const double* weights,
    const double* mid,
    int m, int n);

} // namespace sanos
