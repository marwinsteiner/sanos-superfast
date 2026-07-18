#pragma once
#include "config.hpp"
#include "expiry_data.hpp"
#include "thread_pool.hpp"
#include <vector>
#include <memory>

namespace sanos {

class Surface {
public:
    explicit Surface(SurfaceConfig cfg = {});

    void set_market(
        int n_expiries,
        const std::string* labels,
        const double* sqrtTs,
        const int* n_strikes,
        const double* const* strikes,
        const double* const* bids,
        const double* const* asks);

    void clear();
    void add_expiry(
        const std::string& label,
        double sqrtT,
        const double* strikes, int n,
        const double* bids,
        const double* asks);

    double calibrate();
    double tick_update(int expiry_idx, int strike_idx, double new_bid, double new_ask);

    double price(double T, double K) const;
    double vol(double T, double K) const;
    void price_grid(int expiry_idx, const double* strikes, int n, double* out) const;
    void vol_grid(int expiry_idx, const double* strikes, int n, double* out) const;

    int n_expiries() const { return static_cast<int>(markets_.size()); }
    const ExpiryMarket& market(int i) const { return markets_[i]; }
    const ExpiryFit& fit(int i) const { return fits_[i]; }
    const SurfaceConfig& config() const { return cfg_; }

private:
    SurfaceConfig cfg_;
    std::vector<ExpiryMarket> markets_;
    std::vector<ExpiryFit> fits_;
    std::unique_ptr<ThreadPool> pool_;

    AVec<double> atm_Ts_;
    AVec<double> atm_vars_;

    void ensure_pool();
    void setup_expiry(int j);
    void build_kernel(int j);
    void build_qp(int j);
    void solve_expiry(int j);
    void calibrate_expiry(int j);
    void compute_iv(int j);
    void update_time_interp();
    void eval_model(int j, const double* strikes, int n, double* prices) const;
};

} // namespace sanos
