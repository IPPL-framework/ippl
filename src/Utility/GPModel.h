//
// Class GPModel
//
//   Anisotropic-RBF (ARD) Gaussian-process regressor with hyperparameter
//   selection via gradient ascent on the log-marginal-likelihood.
//
//   Designed for low-dimensional Bayesian optimization of discrete kernel
//   configurations: a small number of inputs (<200), 1-10 dimensions, and
//   per-observation noise variance derived from benchmark run-to-run
//   variance (heteroscedastic).
//
//   Inputs are normalized to [0, 1]^D using a user-supplied bounding box.
//   Targets are standardized (zero mean, unit variance) inside fit(); both
//   predict() and the acquisition functions operate in the original target
//   units.
//
//   The kernel is the squared-exponential (Gaussian) RBF with a separate
//   length-scale per input dimension:
//
//     k(x, x') = sigma_f^2 * exp( -0.5 * sum_d (x_d - x'_d)^2 / l_d^2 )
//
//   The covariance matrix is K + diag(sigma_n^2) + diag(y_var_i / y_std^2),
//   so per-observation noise estimates are folded into the diagonal in the
//   normalized-target space.

#ifndef IPPL_GP_MODEL_H
#define IPPL_GP_MODEL_H

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace ippl {
    namespace detail {

        /*!
         * @class GPModel
         * @brief ARD Gaussian-process regressor with gradient-ascent hyperparameter fit.
         *
         * Used by the auto-tune sweep for low-dim Bayesian optimization over
         * tile / team / oversubscription configurations.
         *
         * @tparam D Number of input dimensions (small: 1..10).
         */
        template <int D>
        class GPModel {
        public:
            using Point = std::array<double, D>;

            GPModel() {
                lo_.fill(0.0);
                hi_.fill(1.0);
                log_l_.fill(std::log(0.3));
                log_sigma_f_ = std::log(1.0);
                log_sigma_n_ = std::log(0.1);
            }

            void set_bounds(const Point& lo, const Point& hi) {
                lo_ = lo;
                hi_ = hi;
            }

            void clear() {
                X_.clear();
                y_.clear();
                y_var_.clear();
            }

            std::size_t size() const { return X_.size(); }

            // Add an observation (x, y) with optional per-observation noise variance.
            // y_var = sample variance of y from repeated measurements; pass 0 to fall
            // back to a single global noise hyperparameter.
            void add(const Point& x, double y, double y_var = 0.0) {
                X_.push_back(x);
                y_.push_back(y);
                y_var_.push_back(std::max(y_var, 0.0));
            }

            // Fit hyperparameters (log_l per dim, log_sigma_f, log_sigma_n) by
            // gradient ascent on the log-marginal-likelihood with finite-difference
            // gradients and multi-start initialization.
            void fit(int max_iter = 100, int n_restarts = 3) {
                const std::size_t n = y_.size();
                if (n == 0) {
                    return;
                }

                // Standardize targets.
                y_mean_ = std::accumulate(y_.begin(), y_.end(), 0.0) / static_cast<double>(n);
                double var = 0.0;
                for (double v : y_) {
                    var += (v - y_mean_) * (v - y_mean_);
                }
                y_std_ = std::sqrt(var / static_cast<double>(n) + 1e-12);
                y_norm_.resize(n);
                for (std::size_t i = 0; i < n; ++i) {
                    y_norm_[i] = (y_[i] - y_mean_) / y_std_;
                }

                // Cache normalized inputs.
                Xn_.resize(n);
                for (std::size_t i = 0; i < n; ++i) {
                    Xn_[i] = normalize(X_[i]);
                }

                const int H = D + 2;
                std::vector<double> theta(H), best_theta(H);
                for (int d = 0; d < D; ++d) {
                    theta[d] = log_l_[d];
                }
                theta[D]     = log_sigma_f_;
                theta[D + 1] = log_sigma_n_;
                double best_lml = compute_lml(theta);
                best_theta      = theta;

                std::mt19937 rng(987654321u);
                std::uniform_real_distribution<double> u(-2.0, 1.0);

                for (int restart = 0; restart < n_restarts; ++restart) {
                    std::vector<double> th(H);
                    for (int d = 0; d < D; ++d) {
                        th[d] = u(rng);
                    }
                    th[D]     = u(rng);
                    th[D + 1] = u(rng);
                    optimize_lml(th, max_iter);
                    const double lml = compute_lml(th);
                    if (lml > best_lml) {
                        best_lml   = lml;
                        best_theta = th;
                    }
                }

                for (int d = 0; d < D; ++d) {
                    log_l_[d] = best_theta[d];
                }
                log_sigma_f_ = best_theta[D];
                log_sigma_n_ = best_theta[D + 1];

                rebuild_factor();
            }

            // Predict mean and variance at x in original target units.
            // If fit() has not been called yet (Xn_/L_/alpha_ are empty), this
            // returns the prior (y_mean_, y_std_^2). The GP state used for
            // prediction is the one fixed at the last fit; observations added
            // since the last fit are ignored until refit.
            std::pair<double, double> predict(const Point& x) const {
                const std::size_t n = Xn_.size();
                if (n == 0) {
                    return {y_mean_, y_std_ * y_std_};
                }

                const Point xn      = normalize(x);
                const double sigma_f = std::exp(log_sigma_f_);
                const double sf2     = sigma_f * sigma_f;

                std::vector<double> k(n);
                for (std::size_t i = 0; i < n; ++i) {
                    k[i] = sf2 * kernel_norm(Xn_[i], xn);
                }

                double mu_norm = 0.0;
                for (std::size_t i = 0; i < n; ++i) {
                    mu_norm += k[i] * alpha_[i];
                }

                // var = k(x,x) - k^T K^{-1} k = k(x,x) - sum_i v_i^2 with L v = k.
                std::vector<double> v(n);
                for (std::size_t i = 0; i < n; ++i) {
                    double s = k[i];
                    for (std::size_t j = 0; j < i; ++j) {
                        s -= L_[i][j] * v[j];
                    }
                    v[i] = s / L_[i][i];
                }
                const double k_xx = sf2;
                double var        = k_xx;
                for (std::size_t i = 0; i < n; ++i) {
                    var -= v[i] * v[i];
                }
                var = std::max(var, 1e-10);

                return {mu_norm * y_std_ + y_mean_, var * y_std_ * y_std_};
            }

            // Expected improvement over the running best f_best (in target units).
            double expected_improvement(const Point& x, double f_best,
                                        double xi = 0.01) const {
                auto pred         = predict(x);
                const double mu    = pred.first;
                const double sigma = std::sqrt(std::max(pred.second, 0.0));
                if (sigma < 1e-10) {
                    return std::max(0.0, mu - f_best);
                }
                const double imp = mu - f_best - xi * std::max(y_std_, 1.0);
                const double Z   = imp / sigma;
                const double Phi = 0.5 * (1.0 + std::erf(Z / std::sqrt(2.0)));
                const double phi = std::exp(-0.5 * Z * Z) / std::sqrt(2.0 * M_PI);
                return std::max(0.0, imp * Phi + sigma * phi);
            }

            // Upper-confidence-bound acquisition with exploration coefficient beta.
            double upper_confidence_bound(const Point& x, double beta = 2.0) const {
                auto pred = predict(x);
                return pred.first + beta * std::sqrt(std::max(pred.second, 0.0));
            }

            // Per-dimension lengthscales in the *normalized* input space.
            Point lengthscales() const {
                Point l;
                for (int d = 0; d < D; ++d) {
                    l[d] = std::exp(log_l_[d]);
                }
                return l;
            }

            double signal_std() const { return std::exp(log_sigma_f_); }
            double noise_std() const { return std::exp(log_sigma_n_); }

        private:
            Point normalize(const Point& x) const {
                Point n;
                for (int d = 0; d < D; ++d) {
                    const double range = std::max(hi_[d] - lo_[d], 1e-12);
                    n[d]               = (x[d] - lo_[d]) / range;
                }
                return n;
            }

            // Squared-exponential kernel value (without sigma_f^2 prefactor) using
            // the *current* log-lengthscales in log_l_.
            double kernel_norm(const Point& a, const Point& b) const {
                double s = 0.0;
                for (int d = 0; d < D; ++d) {
                    const double diff = a[d] - b[d];
                    const double l    = std::exp(log_l_[d]);
                    s += diff * diff / (l * l);
                }
                return std::exp(-0.5 * s);
            }

            // Same but parameterized by an arbitrary log-lengthscale vector.
            double kernel_norm_l(const Point& a, const Point& b,
                                 const std::vector<double>& log_l) const {
                double s = 0.0;
                for (int d = 0; d < D; ++d) {
                    const double diff = a[d] - b[d];
                    const double l    = std::exp(log_l[d]);
                    s += diff * diff / (l * l);
                }
                return std::exp(-0.5 * s);
            }

            // Log-marginal-likelihood at hyperparameters theta = (log_l..., log_sf, log_sn).
            // Returns -inf-equivalent (-1e300) if the Cholesky breaks down.
            double compute_lml(const std::vector<double>& theta) const {
                const std::size_t n = X_.size();
                if (n == 0) {
                    return -1e300;
                }
                const double sigma_n = std::exp(theta[D + 1]);
                const double sigma_f = std::exp(theta[D]);
                const double sf2     = sigma_f * sigma_f;
                const double sn2     = sigma_n * sigma_n;
                std::vector<double> log_l(theta.begin(), theta.begin() + D);

                std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));
                for (std::size_t i = 0; i < n; ++i) {
                    for (std::size_t j = 0; j <= i; ++j) {
                        double k_ij = sf2 * kernel_norm_l(Xn_[i], Xn_[j], log_l);
                        if (i == j) {
                            // Per-obs noise (heteroscedastic) plus floor.
                            k_ij += sn2 + y_var_[i] / (y_std_ * y_std_) + 1e-8;
                        }
                        double s = k_ij;
                        for (std::size_t kk = 0; kk < j; ++kk) {
                            s -= L[i][kk] * L[j][kk];
                        }
                        if (i == j) {
                            if (s <= 0.0) {
                                return -1e300;
                            }
                            L[i][j] = std::sqrt(s);
                        } else {
                            L[i][j] = s / L[j][j];
                        }
                    }
                }

                std::vector<double> v(n);
                for (std::size_t i = 0; i < n; ++i) {
                    double s = y_norm_[i];
                    for (std::size_t j = 0; j < i; ++j) {
                        s -= L[i][j] * v[j];
                    }
                    v[i] = s / L[i][i];
                }
                std::vector<double> alpha(n, 0.0);
                for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
                    double s = v[i];
                    for (std::size_t j = static_cast<std::size_t>(i + 1); j < n; ++j) {
                        s -= L[j][i] * alpha[j];
                    }
                    alpha[i] = s / L[i][i];
                }

                double fit = 0.0, ldet = 0.0;
                for (std::size_t i = 0; i < n; ++i) {
                    fit += y_norm_[i] * alpha[i];
                    ldet += std::log(std::max(L[i][i], 1e-300));
                }
                return -0.5 * fit - ldet
                       - 0.5 * static_cast<double>(n) * std::log(2.0 * M_PI);
            }

            // Backtracking gradient ascent with finite-difference gradients.
            void optimize_lml(std::vector<double>& theta, int max_iter) {
                const int H          = D + 2;
                const double fd_eps  = 1e-3;
                const double max_step = 0.5;
                const double lo_clamp = -5.0;
                const double hi_clamp = 3.0;
                double step           = 0.1;

                double lml = compute_lml(theta);
                for (int it = 0; it < max_iter; ++it) {
                    std::vector<double> grad(H, 0.0);
                    for (int h = 0; h < H; ++h) {
                        std::vector<double> tp = theta, tm = theta;
                        tp[h] += fd_eps;
                        tm[h] -= fd_eps;
                        const double fp = compute_lml(tp);
                        const double fm = compute_lml(tm);
                        if (fp <= -1e290 || fm <= -1e290) {
                            grad[h] = 0.0;
                            continue;
                        }
                        grad[h] = (fp - fm) / (2.0 * fd_eps);
                    }
                    double gnorm = 0.0;
                    for (double g : grad) {
                        gnorm += g * g;
                    }
                    gnorm = std::sqrt(gnorm);
                    if (gnorm < 1e-6) {
                        break;
                    }

                    bool improved = false;
                    double s      = step;
                    for (int bt = 0; bt < 10; ++bt) {
                        std::vector<double> th2(H);
                        for (int h = 0; h < H; ++h) {
                            th2[h] = std::clamp(theta[h] + s * grad[h] / gnorm,
                                                lo_clamp, hi_clamp);
                        }
                        const double lml2 = compute_lml(th2);
                        if (lml2 > lml + 1e-6) {
                            theta    = th2;
                            lml      = lml2;
                            step     = std::min(s * 1.2, max_step);
                            improved = true;
                            break;
                        }
                        s *= 0.5;
                    }
                    if (!improved) {
                        step *= 0.5;
                        if (step < 1e-4) {
                            break;
                        }
                    }
                }
            }

            // Build Cholesky and alpha for the final hyperparameter set.
            void rebuild_factor() {
                const std::size_t n = X_.size();
                if (n == 0) {
                    return;
                }
                const double sigma_f = std::exp(log_sigma_f_);
                const double sigma_n = std::exp(log_sigma_n_);
                const double sf2     = sigma_f * sigma_f;
                const double sn2     = sigma_n * sigma_n;

                L_.assign(n, std::vector<double>(n, 0.0));
                for (std::size_t i = 0; i < n; ++i) {
                    for (std::size_t j = 0; j <= i; ++j) {
                        double k_ij = sf2 * kernel_norm(Xn_[i], Xn_[j]);
                        if (i == j) {
                            k_ij += sn2 + y_var_[i] / (y_std_ * y_std_) + 1e-8;
                        }
                        double s = k_ij;
                        for (std::size_t kk = 0; kk < j; ++kk) {
                            s -= L_[i][kk] * L_[j][kk];
                        }
                        if (i == j) {
                            L_[i][j] = (s > 1e-16) ? std::sqrt(s) : 1e-8;
                        } else {
                            L_[i][j] = s / L_[j][j];
                        }
                    }
                }

                std::vector<double> v(n);
                for (std::size_t i = 0; i < n; ++i) {
                    double s = y_norm_[i];
                    for (std::size_t j = 0; j < i; ++j) {
                        s -= L_[i][j] * v[j];
                    }
                    v[i] = s / L_[i][i];
                }
                alpha_.assign(n, 0.0);
                for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
                    double s = v[i];
                    for (std::size_t j = static_cast<std::size_t>(i + 1); j < n; ++j) {
                        s -= L_[j][i] * alpha_[j];
                    }
                    alpha_[i] = s / L_[i][i];
                }
            }

            std::vector<Point> X_;
            std::vector<double> y_;
            std::vector<double> y_var_;

            Point lo_{}, hi_{};

            std::array<double, D> log_l_{};
            double log_sigma_f_ = 0.0;
            double log_sigma_n_ = 0.0;

            double y_mean_ = 0.0;
            double y_std_  = 1.0;
            std::vector<double> y_norm_;

            std::vector<Point> Xn_;
            std::vector<std::vector<double>> L_;
            std::vector<double> alpha_;
        };

    }  // namespace detail
}  // namespace ippl

#endif  // IPPL_GP_MODEL_H
