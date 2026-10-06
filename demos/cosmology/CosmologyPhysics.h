/**
 * @brief Scientific implementation and contracts for CosmologyPhysics.h.
 *
 * @file CosmologyPhysics.h
 * @ingroup cosmology_core
 * @see cosmology_model cosmology_numerics cosmology_contracts
 */
#ifndef IPPL_COSMOLOGY_PHYSICS_H
#define IPPL_COSMOLOGY_PHYSICS_H

#include "CosmologyConfig.h"

#include <Kokkos_MathematicalConstants.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <utility>
#include <vector>

namespace cosmology {
namespace detail {

// Adaptive Simpson quadrature is host-only. Its tolerance is relative to the first estimate.
template <class Function>
/**
 * @brief Refine an adaptive Simpson interval and apply the Richardson correction.
 *
 * @tparam Function Host-callable scalar function.
 * Uses |S_left+S_right-S_parent|<=15*tolerance and a difference/15 correction.
 * @throws std::runtime_error If the refinement depth is exhausted.
 *
 * @param function Host scalar integrand.
 * @param lower Lower integration coordinate.
 * @param upper Upper integration coordinate.
 * @param fLower Cached function(lower).
 * @param fMid Cached midpoint integrand.
 * @param fUpper Cached function(upper).
 * @param whole Parent Simpson estimate.
 * @param tolerance Absolute error budget for this interval.
 * @param remaining Remaining bisection depth.
 * @return Corrected integral estimate.
 */
double refineIntegral(const Function& function, double lower, double upper, double fLower,
                      double fMid, double fUpper, double whole, double tolerance, int remaining) {
    const double middle = 0.5 * (lower + upper);
    const double fLeft = function(0.5 * (lower + middle));
    const double fRight = function(0.5 * (middle + upper));
    const double left = (middle - lower) * (fLower + 4.0 * fLeft + fMid) / 6.0;
    const double right = (upper - middle) * (fMid + 4.0 * fRight + fUpper) / 6.0;
    const double difference = left + right - whole;
    if (std::abs(difference) <= 15.0 * tolerance)
        return left + right + difference / 15.0;
    if (remaining == 0) throw std::runtime_error("Cosmology quadrature did not converge");
    return refineIntegral(function, lower, middle, fLower, fLeft, fMid, left,
                          tolerance * 0.5, remaining - 1)
           + refineIntegral(function, middle, upper, fMid, fRight, fUpper, right,
                            tolerance * 0.5, remaining - 1);
}

template <class Function>
/**
 * @brief Integrate a host scalar function with signed adaptive Simpson quadrature.
 *
 * @tparam Function Host-callable scalar function.
 * Initial tolerance is 1e-11*max(abs(first Simpson estimate),1e-100); maximum depth is 24.
 * @throws std::runtime_error If quadrature does not converge.
 *
 * @param function Host scalar integrand.
 * @param lower First endpoint in the integrand coordinate.
 * @param upper Second endpoint; may be equal to or less than lower.
 * @return Signed integral; zero for identical endpoints.
 */
double integrate(const Function& function, double lower, double upper) {
    if (lower == upper) return 0.0;
    if (upper < lower) return -integrate(function, upper, lower);
    const double fLower = function(lower);
    const double fMid = function(0.5 * (lower + upper));
    const double fUpper = function(upper);
    const double whole = (upper - lower) * (fLower + 4.0 * fMid + fUpper) / 6.0;
    return refineIntegral(function, lower, upper, fLower, fMid, fUpper, whole,
                          1.0e-11 * std::max(std::abs(whole), 1.0e-100), 24);
}

/**
 * @brief Validate a scale factor before host background calculations.
 *
 * @throws std::invalid_argument If a is nonfinite or not positive.
 *
 * @param a Finite positive dimensionless scale factor.
 */
inline void positiveScaleFactor(double a) {
    if (!std::isfinite(a) || a <= 0.0)
        throw std::invalid_argument("Scale factor must be finite and positive");
}

}  // namespace detail

/** Flat, radiation-free LCDM; D is the growing mode normalized to D(1)=1. */
/**
 * @brief Host-only flat, radiation-free Lambda-CDM expansion and normalized growth.
 *
 * @see cosmology_model
 * Growth, kick and drift quadratures use the same E(a). No approximation of the growth rate is fitted.
 */
class Background {
public:
    /**
     * @brief Construct and normalize the supported growing-mode background.
     *
     * Sets the normalization G(1) once. Host-only.
     * @throws std::invalid_argument For unsupported configuration.
     * @throws std::runtime_error On failed growth quadrature.
     *
     * @param config Validated flat matter-plus-Lambda parameters; validation is repeated here.
     */
    explicit Background(const Config& config) : omegaMatter_m(config.omegaMatter) {
        config.validate();
        growthNormalization_m = unnormalizedGrowth(1.0);
    }

    /**
     * @brief Evaluate the dimensionless Hubble expansion rate.
     *
     * @throws std::invalid_argument For nonpositive/nonfinite a.
     * @see cosmology_model
     *
     * @param a Finite positive dimensionless scale factor.
     * @return E(a)=sqrt(Omega_m/a^3+1-Omega_m).
     */
    double E(double a) const {
        detail::positiveScaleFactor(a);
        return std::sqrt(omegaMatter_m / (a * a * a) + 1.0 - omegaMatter_m);
    }

    /**
     * @brief Evaluate the normalized growing mode.
     *
     * The integral representation is described in @ref cosmology_model and @cite heath1977.
     * @throws std::invalid_argument For invalid a.
     * @throws std::runtime_error On quadrature failure.
     *
     * @param a Finite positive dimensionless scale factor.
     * @return Dimensionless D(a)=G(a)/G(1), with D(1)=1.
     */
    double D(double a) const { return unnormalizedGrowth(a) / growthNormalization_m; }

    /**
     * @brief Evaluate the logarithmic growing-mode rate.
     *
     * Uses the derivative of the same growth integral as D(), not an Omega_m(a)^gamma approximation.
     * @throws std::invalid_argument For invalid a.
     * @throws std::runtime_error On quadrature failure.
     *
     * @param a Finite positive dimensionless scale factor.
     * @return Dimensionless f(a)=d ln D/d ln a.
     */
    double f(double a) const {
        const double expansion = E(a);
        const double integral = growthIntegral(a);
        const double omegaAtA = omegaMatter_m / (a * a * a * expansion * expansion);
        return -1.5 * omegaAtA + 1.0 / (a * a * expansion * expansion * expansion * integral);
    }

    // phi0 obeys laplacian(phi0)=3 Omega_m delta/2; force=-grad(phi0).
    // For p=a^2 dx/d(H0 t), p += force*kick and x += p*drift.
    /**
     * @brief Integrate the canonical momentum kick factor.
     *
     * Multiply F0 in Mpc/h by this dimensionless factor to update p.
     * Computed in log(a) on the host; no device kernel evaluates the quadrature.
     * @throws std::invalid_argument For invalid endpoints.
     * @throws std::runtime_error On quadrature failure.
     *
     * @param aLower Finite positive lower scale factor.
     * @param aUpper Finite positive upper scale factor; reversing endpoints reverses the sign.
     * @return Signed integral of da/(a^2 E(a)).
     */
    double kick(double aLower, double aUpper) const {
        detail::positiveScaleFactor(aLower);
        detail::positiveScaleFactor(aUpper);
        return detail::integrate([&](double logA) {
            const double a = std::exp(logA);
            return 1.0 / (a * E(a));
        }, std::log(aLower), std::log(aUpper));
    }

    /**
     * @brief Integrate the canonical position drift factor.
     *
     * Multiply canonical momentum in Mpc/h to update comoving position.
     * Computed in log(a); equal endpoints return zero.
     * @throws std::invalid_argument For invalid endpoints.
     * @throws std::runtime_error On quadrature failure.
     *
     * @param aLower Finite positive lower scale factor.
     * @param aUpper Finite positive upper scale factor; reversing endpoints reverses the sign.
     * @return Signed integral of da/(a^3 E(a)).
     */
    double drift(double aLower, double aUpper) const {
        detail::positiveScaleFactor(aLower);
        detail::positiveScaleFactor(aUpper);
        return detail::integrate([&](double logA) {
            const double a = std::exp(logA);
            return 1.0 / (a * a * E(a));
        }, std::log(aLower), std::log(aUpper));
    }

private:
    // Substitution a'=a*s^2 removes the matter-era endpoint's fractional power.
    /**
     * @brief Evaluate the smooth-endpoint pressure-free growth integral.
     *
     * Substitution aPrime=a*s^2 removes the matter-era fractional endpoint power; see @ref cosmology_model.
     *
     * @param a Finite positive dimensionless scale factor.
     * @return I(a)=integral_0^a da/(a^3 E(a)^3).
     */
    double growthIntegral(double a) const {
        detail::positiveScaleFactor(a);
        return 2.0 * std::pow(a, 2.5) * detail::integrate([&](double s) {
            const double s2 = s * s;
            const double denominator = omegaMatter_m + (1.0 - omegaMatter_m) * a * a * a
                                                                  * s2 * s2 * s2;
            return s2 * s2 / std::pow(denominator, 1.5);
        }, 0.0, 1.0);
    }

    /**
     * @brief Evaluate the unnormalized growing solution G(a).
     *
     * Host-only; normalized by G(1) when returned through D().
     *
     * @param a Finite positive dimensionless scale factor.
     * @return G(a)=(5 Omega_m/2) E(a) I(a).
     */
    double unnormalizedGrowth(double a) const {
        return 2.5 * omegaMatter_m * E(a) * growthIntegral(a);
    }

    double omegaMatter_m; ///< Flat-background matter fraction at a=1.
    double growthNormalization_m; ///< Unnormalized growth G(1) used to impose D(1)=1.
};

/** Zarija-compatible BBKS or weighted CMBFAST T(k), with sigma8 normalization.
 *  k is in h/Mpc, P(k) in (Mpc/h)^3; P(k) is extrapolated to z=0.
 *  As in Zarija, sigma8 integration ends at k=10 for BBKS or at the last table row.
 */
/**
 * @brief Host z=0 BBKS or weighted-transfer power with a fixed finite-sigma8 contract.
 *
 * @see cosmology_model
 * P(k) has units (Mpc/h)^3; all k arguments are h/Mpc. Transfer weighting is not hydrodynamic baryon evolution.
 */
class PowerSpectrum {
public:
    /**
     * @brief Construct a z=0 spectrum with the finite sigma8 normalization.
     *
     * @throws std::invalid_argument For invalid configuration.
     * @throws std::runtime_error For invalid table/variance or insufficient k coverage.
     * @see cosmology_model
     *
     * @param config Host configuration specifying BBKS/table transfer and positive sigma8.
     */
    explicit PowerSpectrum(const Config& config)
        : spectralIndex_m(config.spectralIndex), shape_m(config.omegaMatter * config.hubble),
          transferFunction_m(config.transferFunction) {
        config.validate();
        if (transferFunction_m == 0) readTransferTable(config);
        const double variance = rawVariance(8.0);
        if (!std::isfinite(variance) || variance <= 0.0)
            throw std::runtime_error("Invalid unnormalized sigma8 variance");
        normalization_m = config.sigma8 * config.sigma8 / variance;
    }

    /**
     * @brief Evaluate the dimensionless BBKS or weighted-tabulated transfer.
     *
     * BBKS uses q=k/(Omega_m*h) @cite bbks1986. Table interpolation is linear in k.
     * @throws std::invalid_argument For negative/nonfinite k.
     * @throws std::out_of_range If a table does not cover k.
     *
     * @param k Finite nonnegative wavenumber in h/Mpc.
     * @return Dimensionless T(k), normalized to one at low k.
     */
    double transfer(double k) const {
        if (!std::isfinite(k) || k < 0.0)
            throw std::invalid_argument("Transfer-function k must be finite and nonnegative");
        if (transferFunction_m == 4) {
            if (k == 0.0) return 1.0;
            const double q = k / shape_m;
            return std::log1p(2.34 * q) / (2.34 * q)
                   * std::pow(1.0 + 3.89 * q + std::pow(16.1 * q, 2)
                                  + std::pow(5.46 * q, 3) + std::pow(6.71 * q, 4), -0.25);
        }
        // The first row is explicitly normalized too (fixes a legacy first-row bug).
        if (k <= tableK_m.front()) return 1.0;
        if (k > tableK_m.back())
            throw std::out_of_range("Transfer-function table does not cover requested k");
        const auto above = std::lower_bound(tableK_m.begin() + 1, tableK_m.end(), k);
        const auto index = static_cast<std::size_t>(above - tableK_m.begin());
        const double fraction = (k - tableK_m[index - 1]) / (tableK_m[index] - tableK_m[index - 1]);
        return tableTransfer_m[index - 1]
               + fraction * (tableTransfer_m[index] - tableTransfer_m[index - 1]);
    }

    /**
     * @brief Evaluate the normalized z=0 density power.
     *
     * Transfer-domain errors propagate from transfer().
     *
     * @param k Finite nonnegative wavenumber in h/Mpc.
     * @return P0(k)=A*k^n_s*T(k)^2 in (Mpc/h)^3.
     */
    double operator()(double k) const {
        const double transferValue = transfer(k);
        return normalization_m * std::pow(k, spectralIndex_m) * transferValue * transferValue;
    }
    /**
     * @brief Scale z=0 power by the normalized linear growing mode.
     *
     * This is a linear prediction, not a nonlinear evolved spectrum.
     *
     * @param k Wavenumber in h/Mpc.
     * @param a Finite positive scale factor.
     * @param background Consistent flat-Lambda background used for growth.
     * @return D(a)^2*P0(k) in (Mpc/h)^3.
     */
    double atScaleFactor(double k, double a, const Background& background) const {
        const double growth = background.D(a);
        return (*this)(k) * growth * growth;
    }
    /**
     * @brief Return the fixed spectrum amplitude A.
     *
     * The amplitude is set by the finite-cutoff sigma8 integral, not fitted to one realization.
     * @return Normalization multiplying k^n_s*T(k)^2.
     */
    double normalization() const { return normalization_m; }
    /**
     * @brief Calculate the finite-cutoff top-hat RMS at radius R.
     *
     * Uses the same kmax and numerical quadrature as the normalization.
     * @throws std::invalid_argument For invalid radius.
     *
     * @param radius Finite positive smoothing radius in comoving Mpc/h.
     * @return Dimensionless sigma_R.
     */
    double sigmaR(double radius) const { return std::sqrt(normalization_m * rawVariance(radius)); }
    /**
     * @brief Return the upper wavenumber of the normalization contract.
     *
     * This is not an infinite-upper-limit sigma8 claim.
     * @return kmax in h/Mpc: 10 for BBKS, final input k for tables.
     */
    double normalizationKMax() const { return normalizationKMax_m; }

    // Samples are host data ready to copy once into a Kokkos view for distributed IC kernels.
    /**
     * @brief Sample P0(k) on a host logarithmic k grid.
     *
     * @throws std::invalid_argument For invalid bounds/count.
     * Samples are host data; callers explicitly copy them to execution memory if needed.
     *
     * @param kMin Positive lower k in h/Mpc.
     * @param kMax Upper k in h/Mpc, strictly greater than kMin.
     * @param count At least two samples; both endpoints included.
     * @return Pairs (k,P0(k)) in h/Mpc and (Mpc/h)^3.
     */
    std::vector<std::pair<double, double>> tabulate(double kMin, double kMax,
                                                   std::size_t count = 4096) const {
        if (!(kMin > 0.0 && kMax > kMin) || count < 2)
            throw std::invalid_argument("Invalid spectrum tabulation bounds/count");
        std::vector<std::pair<double, double>> samples(count);
        for (std::size_t index = 0; index < count; ++index) {
            const double k = index + 1 == count ? kMax
                : kMin * std::exp(std::log(kMax / kMin) * double(index) / double(count - 1));
            samples[index] = {k, (*this)(k)};
        }
        return samples;
    }

private:
    /**
     * @brief Evaluate the Fourier-space spherical top-hat window stably.
     *
     * Uses the even series through x^6 for abs(x)<1e-2.
     *
     * @param x Dimensionless argument k*R.
     * @return W(x)=3*(sin(x)-x*cos(x))/x^3, with W(0)=1.
     */
    static double topHat(double x) {
        if (std::abs(x) < 1.0e-2) {
            const double x2 = x * x;
            return 1.0 - x2 / 10.0 + x2 * x2 / 280.0 - x2 * x2 * x2 / 15120.0;
        }
        return 3.0 * (std::sin(x) - x * std::cos(x)) / (x * x * x);
    }

    /**
     * @brief Compute the unnormalized finite-cutoff smoothed variance.
     *
     * Uses 65536 Simpson intervals in ln(k), an analytic k<1e-8 tail, and normalizationKMax().
     *
     * @param radius Positive finite R in comoving Mpc/h.
     * @return Integral k^2*k^n_s*T(k)^2*W(kR)^2/(2*pi^2), before A.
     */
    double rawVariance(double radius) const {
        if (!std::isfinite(radius) || radius <= 0.0)
            throw std::invalid_argument("Sigma radius must be finite and positive");
        // Fixed log-k grid resolves the top-hat oscillations at R=8 and Zarija's cutoff.
        // The negligible k<1e-8 contribution is analytic because T,W tend to one.
        constexpr int Intervals = 65536;
        constexpr double KMin = 1.0e-8;
        const double logMin = std::log(KMin);
        const double logMax = std::log(normalizationKMax_m);
        const double step = (logMax - logMin) / Intervals;
        double sum = 0.0;
        for (int index = 0; index <= Intervals; ++index) {
            const double k = index == Intervals ? normalizationKMax_m
                                                : std::exp(logMin + double(index) * step);
            const double t = transfer(k);
            const double window = topHat(k * radius);
            const double integrand = std::pow(k, 3.0 + spectralIndex_m) * t * t * window * window;
            const int weight = index == 0 || index == Intervals ? 1 : (index % 2 == 0 ? 2 : 4);
            sum += weight * integrand;
        }
        const double belowCutoff = std::pow(KMin, 3.0 + spectralIndex_m) / (3.0 + spectralIndex_m);
        constexpr double Pi = Kokkos::numbers::pi_v<double>;
        return (sum * step / 3.0 + belowCutoff) / (2.0 * Pi * Pi);
    }

    /**
     * @brief Load and normalize an increasing total-matter transfer table.
     *
     * Requires at least two positive increasing k rows; weights CDM/baryon columns and normalizes every row.
     * @throws std::runtime_error For file/row errors or insufficient 3D mesh-corner coverage.
     *
     * @param config Configuration with transferFile and consistent matter/baryon densities.
     */
    void readTransferTable(const Config& config) {
        std::ifstream input(config.transferFile);
        if (!input) throw std::runtime_error("Cannot open transfer file: " + config.transferFile);
        std::string line;
        int lineNumber = 0;
        while (std::getline(input, line)) {
            ++lineNumber;
            line = line.substr(0, line.find('#'));
            line = line.substr(0, line.find("//"));
            if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
            std::istringstream row(line);
            double k = 0.0, cdm = 0.0, baryon = 0.0;
            if (!(row >> k >> cdm >> baryon) || !std::isfinite(k) || !std::isfinite(cdm)
                || !std::isfinite(baryon) || k <= 0.0 || (!tableK_m.empty() && k <= tableK_m.back()))
                throw std::runtime_error(config.transferFile + ":" + std::to_string(lineNumber)
                                         + ": expected increasing k, T_cdm, T_baryon");
            const double weighted = (config.omegaBaryon * baryon
                                      + (config.omegaMatter - config.omegaBaryon) * cdm)
                                     / config.omegaMatter;
            if (!(weighted > 0.0))
                throw std::runtime_error("CDM transfer table must contain positive total-matter T");
            tableK_m.push_back(k);
            tableTransfer_m.push_back(weighted);
        }
        if (tableK_m.size() < 2)
            throw std::runtime_error("Transfer table needs at least two rows");
        const double first = tableTransfer_m.front();
        for (double& value : tableTransfer_m) value /= first;
        normalizationKMax_m = tableK_m.back();
        constexpr double Pi = Kokkos::numbers::pi_v<double>;
        const double maximumMeshK = std::sqrt(3.0) * Pi * config.nGrid / config.boxSize;
        if (maximumMeshK > normalizationKMax_m)
            throw std::runtime_error("Transfer table must cover the 3D mesh corner wave number");
    }

    double spectralIndex_m; ///< Primordial tilt n_s.
    double shape_m; ///< BBKS shape Omega_m*h; k/shape is dimensionless in the selected k convention.
    int transferFunction_m; ///< Selected BBKS/table mode.
    double normalization_m = 1.0; ///< Finite-sigma8 spectrum amplitude A.
    double normalizationKMax_m = 10.0; ///< Upper normalization wavenumber in h/Mpc.
    std::vector<double> tableK_m; ///< Strictly increasing host transfer-table k samples in h/Mpc.
    std::vector<double> tableTransfer_m; ///< Dimensionless host total-matter transfer values normalized by their first entry.
};

}  // namespace cosmology

#endif
