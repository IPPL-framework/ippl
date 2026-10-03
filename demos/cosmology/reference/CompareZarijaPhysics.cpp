// Public-API comparison with the original, unmodified Zarija cosmology code.
// Compile Cosmology.cpp and MT_Random.cpp with DOUBLE_REAL and USENAMESPACE.
#include "../CosmologyPhysics.h"
#include "Cosmology.h"
#include "DataBase.h"

#include <mpi.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace {
// Preset from the algorithms' declared accuracies, before running comparisons:
// Zarija's midpoint normalization uses EPS=1e-4, and its RK solver EPS=1e-6.
constexpr double TransferTolerance = 2.0e-12;
constexpr double PowerTolerance = 5.0e-4;
constexpr double GrowthTolerance = 2.0e-5;

struct Report {
    std::ofstream stream_m;
    int failures_m = 0;

    explicit Report(const char* path) : stream_m(path) {
        if (!stream_m) throw std::runtime_error("Cannot open comparison CSV");
        stream_m << std::setprecision(17)
                 << "quantity,k2,argument,ippl,zarija,relative_error,tolerance,gated,passed,note\n";
    }
    void value(const std::string& quantity, int k2, double argument, double ippl,
               double zarija, double tolerance, const std::string& note = "", bool gated = true) {
        const double relative = std::abs(ippl - zarija)
                                / std::max(std::abs(ippl), 1.0e-300);
        const bool passed = std::isfinite(relative) && relative <= tolerance;
        if (gated && !passed) ++failures_m;
        stream_m << quantity << ',' << k2 << ',' << argument << ',' << ippl << ',' << zarija
                 << ',' << relative << ',' << tolerance << ',' << gated << ',' << passed
                 << ',' << note << '\n';
    }
};
}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int result = 0;
    try {
        if (argc != 3)
            throw std::invalid_argument("Usage: CompareZarijaPhysics input.par output.csv");
        const auto config = cosmology::Config::fromFile(argv[1]);
        if (config.transferFunction != 0 && config.transferFunction != 4)
            throw std::invalid_argument("Reference comparison supports TFFlag=0 and 4");
        initializer::GlobalStuff parameters{};
        parameters.ngrid = config.nGrid;
        parameters.box_size = config.boxSize;
        parameters.dim = 3;
        parameters.seed = config.seed;
        parameters.z_in = config.zInitial;
        parameters.Omega_m = config.omegaMatter;
        parameters.Omega_bar = config.omegaBaryon;
        parameters.Omega_nu = 0.0;
        parameters.Omega_r = 0.0;
        parameters.Hubble = config.hubble;
        parameters.Sigma_8 = config.sigma8;
        parameters.n_s = config.spectralIndex;
        parameters.w_de = -1.0;
        parameters.f_NL = 0.0;
        parameters.TFFlag = config.transferFunction;
        parameters.N_nu = 3;
        parameters.nu_pairs = 0;

        initializer::CosmoClass original;
        original.SetParameters(parameters, config.transferFile.c_str());
        const cosmology::Background background(config);
        const cosmology::PowerSpectrum spectrum(config);
        Report report(argv[2]);

        const double originalSigmaRaw = original.Sigma_r(8.0, 1.0);
        const double originalNormalization = std::pow(config.sigma8 / originalSigmaRaw, 2);
        const double ipplSigmaRaw = config.sigma8 / std::sqrt(spectrum.normalization());
        report.value("sigma8_raw", -1, 8.0, ipplSigmaRaw, originalSigmaRaw,
                     PowerTolerance / 2.0, "unnormalized P(k)=k^ns T(k)^2");
        report.value("power_normalization", -1, 8.0, spectrum.normalization(),
                     originalNormalization, PowerTolerance, "A=sigma8^2/sigma8_raw^2");
        report.value("sigma8_at_ippl_normalization", -1, 8.0, config.sigma8,
                     original.Sigma_r(8.0, spectrum.normalization()), PowerTolerance / 2.0);

        // Full radial lookup domain used by the IPPL Gaussian initializer.
        constexpr double Pi = Kokkos::numbers::pi_v<double>;
        const double fundamental = 2.0 * Pi / config.boxSize;
        const int maximumK2 = 3 * (config.nGrid / 2) * (config.nGrid / 2);
        for (int k2 = 1; k2 <= maximumK2; ++k2) {
            const double k = fundamental * std::sqrt(double(k2));
            const double originalTransfer = original.TransferFunction(k);
            report.value("transfer_grid", k2, k, spectrum.transfer(k), originalTransfer,
                         TransferTolerance);
            const double originalPower = originalNormalization
                * std::pow(k, config.spectralIndex) * originalTransfer * originalTransfer;
            report.value("power_grid", k2, k, spectrum(k), originalPower, PowerTolerance);
        }
        for (double z : {0.0, 9.0, 49.0, 200.0}) {
            initializer::real originalGrowth, originalDot;
            original.GrowthFactor(z, &originalGrowth, &originalDot);
            const double a = 1.0 / (1.0 + z);
            const double ipplGrowth = background.D(a);
            const double ipplDot = background.E(a) * background.f(a) * ipplGrowth;
            report.value("growth_D", -1, z, ipplGrowth, originalGrowth, GrowthTolerance,
                         "D(1)=1; reference finite-start zero derivative");
            report.value("growth_Ddot", -1, z, ipplDot, originalDot, GrowthTolerance,
                         "derivative with respect to H0*t");
            report.value("growth_f", -1, z, background.f(a),
                         originalDot / (background.E(a) * originalGrowth), GrowthTolerance);
        }
        if (config.transferFunction == 4) {
            report.value("transfer_DC_deliberate_difference", -1, 0.0,
                         spectrum.transfer(0.0), original.TransferFunction(0.0),
                         TransferTolerance, "IPPL takes continuous limit T(0)=1; P(0)=0", false);
        } else {
            std::ifstream table(config.transferFile);
            double firstK, secondK, firstCdm, firstBaryon, unused;
            table >> firstK >> firstCdm >> firstBaryon >> unused >> unused >> unused >> unused;
            table >> secondK >> unused >> unused >> unused >> unused >> unused >> unused;
            report.value("transfer_second_table_row", -1, secondK,
                         spectrum.transfer(secondK), original.TransferFunction(secondK),
                         TransferTolerance);
            for (double k : {0.0, firstK, 0.5 * (firstK + secondK)}) {
                report.value("transfer_first_interval_deliberate_difference", -1, k,
                             spectrum.transfer(k), original.TransferFunction(k),
                             TransferTolerance, "original leaves first row unnormalized and extrapolates", false);
            }
            // The original extrapolation below row 1 and its first interval
            // form one straight line T(k)=b+c*k for 0<=k<=secondK.
            // Integrate its contribution analytically with W(8k)=1.
            // The relative error from that replacement is bounded by
            // (8*secondK)^2/5 at these tiny wave numbers (about 2.3e-9 here).
            const double lowTransfer = original.TransferFunction(0.0);
            const double change = original.TransferFunction(secondK) - lowTransfer;
            const double exponent = 3.0 + config.spectralIndex;
            const double firstIntervalVariance = std::pow(secondK, exponent) / (2.0 * Pi * Pi)
                * (lowTransfer * lowTransfer / exponent
                   + 2.0 * lowTransfer * change / (exponent + 1.0)
                   + change * change / (exponent + 2.0));
            const double regularFirstInterval = std::pow(secondK, exponent)
                                               / (exponent * 2.0 * Pi * Pi);
            report.value("first_interval_variance_estimate", -1, secondK,
                         regularFirstInterval, firstIntervalVariance,
                         TransferTolerance, "analytic first-bin integral W=1; not an agreement gate", false);
        }
        std::cout << "Reference comparison: " << report.failures_m
                  << " failed agreement gates; full data in " << argv[2] << '\n';
        result = report.failures_m == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        result = 2;
    }
    MPI_Finalize();
    return result;
}
