#pragma once
#include "config.hpp"
#include "expiry_data.hpp"
#include <vector>

namespace sanos {

// Main IV surface: calibrate from quotes, query (T, K) -> (price, vol).
// Supports full recalibration and incremental tick updates.
class Surface {
public:
    explicit Surface(SurfaceConfig cfg = {});

    // --- Setup ---

    // Set market data for all expiries. Triggers full recalibration.
    // strikes/bids/asks are pure (divided by forward*DF).
    // sqrtT is sqrt(time to expiry in years).
    void set_market(
        int n_expiries,
        const std::string* labels,
        const double* sqrtTs,
        const int* n_strikes,        // per expiry
        const double* const* strikes,// per expiry
        const double* const* bids,
        const double* const* asks);

    // Simplified: add expiries one by one, then call calibrate().
    void clear();
    void add_expiry(
        const std::string& label,
        double sqrtT,
        const double* strikes, int n,
        const double* bids,
        const double* asks);

    // --- Calibration ---

    // Full calibration of all expiries. Returns total solve time in microseconds.
    double calibrate();

    // Incremental update: one strike on one expiry changed.
    // Returns solve time in microseconds.
    double tick_update(int expiry_idx, int strike_idx, double new_bid, double new_ask);

    // --- Query ---

    // Evaluate fitted call price at (T, K) using time interpolation.
    double price(double T, double K) const;

    // Evaluate fitted implied vol at (T, K).
    double vol(double T, double K) const;

    // Evaluate on a grid of strikes for a single expiry index.
    void price_grid(int expiry_idx, const double* strikes, int n, double* out) const;
    void vol_grid(int expiry_idx, const double* strikes, int n, double* out) const;

    // --- Accessors ---

    int n_expiries() const { return static_cast<int>(markets_.size()); }
    const ExpiryMarket& market(int i) const { return markets_[i]; }
    const ExpiryFit& fit(int i) const { return fits_[i]; }
    const SurfaceConfig& config() const { return cfg_; }

private:
    SurfaceConfig cfg_;
    std::vector<ExpiryMarket> markets_;
    std::vector<ExpiryFit> fits_;

    // PCHIP time interpolation data
    AVec<double> atm_Ts_;     // 0, T1, T2, ..., TM
    AVec<double> atm_vars_;   // 0, V1, V2, ..., VM

    void setup_expiry(int j);
    void build_kernel(int j);
    void build_qp(int j);
    void solve_expiry(int j);
    void calibrate_expiry(int j);
    void compute_iv(int j);
    void update_time_interp();

    // Evaluate SANOS model at arbitrary strikes for expiry j
    void eval_model(int j, const double* strikes, int n, double* prices) const;
};

} // namespace sanos
