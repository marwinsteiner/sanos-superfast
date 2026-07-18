#include "sanos/bs_kernel.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>

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
    if (variance <= 0.0) {
        // Linear mode: C[l,i] = max(model_strikes[i] - eval_strikes[l], 0)
        for (int i = 0; i < n_model; ++i) {
            double* col = out.col_ptr(i);
            double ki = model_strikes[i];
            for (int l = 0; l < n_eval; ++l) {
                col[l] = std::max(ki - eval_strikes[l], 0.0);
            }
        }
    } else {
        for (int i = 0; i < n_model; ++i) {
            double* col = out.col_ptr(i);
            double ki = model_strikes[i];
            for (int l = 0; l < n_eval; ++l) {
                double ratio = eval_strikes[l] / ki;
                col[l] = ki * bs_call_price(ratio, variance);
            }
        }
    }
}

void fill_cross_kernel(
    DenseMat& out,
    const double* cur_model_strikes,  int n_cur,
    const double* prev_model_strikes, int n_prev,
    double prev_variance)
{
    out.resize(n_cur, n_prev);
    if (prev_variance <= 0.0) {
        for (int i = 0; i < n_prev; ++i) {
            double* col = out.col_ptr(i);
            double ki = prev_model_strikes[i];
            for (int l = 0; l < n_cur; ++l) {
                col[l] = std::max(ki - cur_model_strikes[l], 0.0);
            }
        }
    } else {
        for (int i = 0; i < n_prev; ++i) {
            double* col = out.col_ptr(i);
            double ki = prev_model_strikes[i];
            for (int l = 0; l < n_cur; ++l) {
                double ratio = cur_model_strikes[l] / ki;
                col[l] = ki * bs_call_price(ratio, prev_variance);
            }
        }
    }
}

void mat_vec(double* y, const DenseMat& A, const double* x) {
    int m = A.rows, n = A.cols;
    std::memset(y, 0, m * sizeof(double));
    for (int j = 0; j < n; ++j) {
        const double* col = A.col_ptr(j);
        double xj = x[j];
        for (int i = 0; i < m; ++i) {
            y[i] += col[i] * xj;
        }
    }
}

void mat_t_vec(double* y, const DenseMat& A, const double* x) {
    int m = A.rows, n = A.cols;
    for (int j = 0; j < n; ++j) {
        const double* col = A.col_ptr(j);
        double sum = 0.0;
        for (int i = 0; i < m; ++i) {
            sum += col[i] * x[i];
        }
        y[j] = sum;
    }
}

void compute_hessian(
    DenseMat& H,
    const DenseMat& C,
    const double* weights,
    double lambda,
    int n)
{
    int m = C.rows;
    H.resize(n, n);

    // H = C^T diag(w^2) C + lambda I
    // H[j,k] = sum_i  C[i,j] * w[i]^2 * C[i,k]
    for (int j = 0; j < n; ++j) {
        const double* cj = C.col_ptr(j);
        for (int k = j; k < n; ++k) {
            const double* ck = C.col_ptr(k);
            double sum = 0.0;
            for (int i = 0; i < m; ++i) {
                double wi = weights[i];
                sum += cj[i] * wi * wi * ck[i];
            }
            if (j == k) sum += lambda;
            H(j, k) = sum;
            H(k, j) = sum;
        }
    }
}

void compute_gradient(
    double* f,
    const DenseMat& C,
    const double* weights,
    const double* mid,
    int m, int n)
{
    // f = -C^T W^2 mid
    for (int j = 0; j < n; ++j) {
        const double* cj = C.col_ptr(j);
        double sum = 0.0;
        for (int i = 0; i < m; ++i) {
            double wi = weights[i];
            sum += cj[i] * wi * wi * mid[i];
        }
        f[j] = -sum;
    }
}

} // namespace sanos
