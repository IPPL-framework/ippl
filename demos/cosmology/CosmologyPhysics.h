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

inline void positiveScaleFactor(double a) {
    if (!std::isfinite(a) || a <= 0.0)
        throw std::invalid_argument("Scale factor must be finite and positive");
}

}  // namespace detail

/** Flat, radiation-free LCDM; D is the growing mode normalized to D(1)=1. */
class Background {
public:
    explicit Background(const Config& config) : omegaMatter_m(config.omegaMatter) {
        config.validate();
        growthNormalization_m = unnormalizedGrowth(1.0);
    }

    double E(double a) const {
        detail::positiveScaleFactor(a);
        return std::sqrt(omegaMatter_m / (a * a * a) + 1.0 - omegaMatter_m);
    }

    double D(double a) const { return unnormalizedGrowth(a) / growthNormalization_m; }

    double f(double a) const {
        const double expansion = E(a);
        const double integral = growthIntegral(a);
        const double omegaAtA = omegaMatter_m / (a * a * a * expansion * expansion);
        return -1.5 * omegaAtA + 1.0 / (a * a * expansion * expansion * expansion * integral);
    }

    // phi0 obeys laplacian(phi0)=3 Omega_m delta/2; force=-grad(phi0).
    // For p=a^2 dx/d(H0 t), p += force*kick and x += p*drift.
    double kick(double aLower, double aUpper) const {
        detail::positiveScaleFactor(aLower);
        detail::positiveScaleFactor(aUpper);
        return detail::integrate([&](double logA) {
            const double a = std::exp(logA);
            return 1.0 / (a * E(a));
        }, std::log(aLower), std::log(aUpper));
    }

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
    double growthIntegral(double a) const {
        detail::positiveScaleFactor(a);
        return 2.0 * std::pow(a, 2.5) * detail::integrate([&](double s) {
            const double s2 = s * s;
            const double denominator = omegaMatter_m + (1.0 - omegaMatter_m) * a * a * a
                                                                  * s2 * s2 * s2;
            return s2 * s2 / std::pow(denominator, 1.5);
        }, 0.0, 1.0);
    }

    double unnormalizedGrowth(double a) const {
        return 2.5 * omegaMatter_m * E(a) * growthIntegral(a);
    }

    double omegaMatter_m;
    double growthNormalization_m;
};

/** Zarija-compatible BBKS or weighted CMBFAST T(k), with sigma8 normalization.
 *  k is in h/Mpc, P(k) in (Mpc/h)^3; P(k) is extrapolated to z=0.
 *  As in Zarija, sigma8 integration ends at k=10 for BBKS or at the last table row.
 */
class PowerSpectrum {
public:
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

    double operator()(double k) const {
        const double transferValue = transfer(k);
        return normalization_m * std::pow(k, spectralIndex_m) * transferValue * transferValue;
    }
    double atScaleFactor(double k, double a, const Background& background) const {
        const double growth = background.D(a);
        return (*this)(k) * growth * growth;
    }
    double normalization() const { return normalization_m; }
    double sigmaR(double radius) const { return std::sqrt(normalization_m * rawVariance(radius)); }
    double normalizationKMax() const { return normalizationKMax_m; }

    // Samples are host data ready to copy once into a Kokkos view for distributed IC kernels.
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
    static double topHat(double x) {
        if (std::abs(x) < 1.0e-2) {
            const double x2 = x * x;
            return 1.0 - x2 / 10.0 + x2 * x2 / 280.0 - x2 * x2 * x2 / 15120.0;
        }
        return 3.0 * (std::sin(x) - x * std::cos(x)) / (x * x * x);
    }

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

    double spectralIndex_m;
    double shape_m;
    int transferFunction_m;
    double normalization_m = 1.0;
    double normalizationKMax_m = 10.0;
    std::vector<double> tableK_m;
    std::vector<double> tableTransfer_m;
};

}  // namespace cosmology

#endif
