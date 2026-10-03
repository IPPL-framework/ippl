#include "CosmologyPhysics.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {

void close(double actual, double expected, double tolerance, const std::string& message) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        std::ostringstream error;
        error << std::setprecision(17) << message << ": " << actual << " != " << expected
              << " (absolute tolerance " << tolerance << ")";
        throw std::runtime_error(error.str());
    }
}

template <class Function>
void rejects(const Function& function, const std::string& message) {
    bool failed = false;
    try { function(); }
    catch (const std::exception&) { failed = true; }
    if (!failed) throw std::runtime_error("Expected rejection: " + message);
}

// Independent growing-mode ODE in ln(a), integrated by fixed-step RK4.
// y=[D,dD/dln(a)]; initialized in the matter era, then normalized at a=1.
std::array<double, 2> odeGrowth(double omegaMatter, double a) {
    constexpr double InitialA = 1.0e-6;
    constexpr int Steps = 16000;
    const double step = std::log(a / InitialA) / Steps;
    std::array<double, 2> y{InitialA, InitialA};
    const auto derivative = [omegaMatter](double logA, const std::array<double, 2>& state) {
        const double a3 = std::exp(3.0 * logA);
        const double omegaAtA = omegaMatter / (omegaMatter + (1.0 - omegaMatter) * a3);
        return std::array<double, 2>{state[1], 1.5 * omegaAtA * state[0]
                                                 - (2.0 - 1.5 * omegaAtA) * state[1]};
    };
    for (int index = 0; index < Steps; ++index) {
        const double logA = std::log(InitialA) + index * step;
        const auto k1 = derivative(logA, y);
        const auto k2 = derivative(logA + step / 2.0,
                                  {y[0] + step * k1[0] / 2.0, y[1] + step * k1[1] / 2.0});
        const auto k3 = derivative(logA + step / 2.0,
                                  {y[0] + step * k2[0] / 2.0, y[1] + step * k2[1] / 2.0});
        const auto k4 = derivative(logA + step, {y[0] + step * k3[0], y[1] + step * k3[1]});
        for (int component = 0; component < 2; ++component)
            y[component] += step * (k1[component] + 2.0 * k2[component]
                                    + 2.0 * k3[component] + k4[component]) / 6.0;
    }
    return y;
}

// Independent linear-k midpoint integration of P(k), contrasting the log-k Simpson algorithm.
double midpointSigma8(const cosmology::PowerSpectrum& spectrum) {
    constexpr int Intervals = 200000;
    constexpr double Pi = Kokkos::numbers::pi_v<double>;
    const double step = spectrum.normalizationKMax() / Intervals;
    double variance = 0.0;
    for (int index = 0; index < Intervals; ++index) {
        const double k = (index + 0.5) * step;
        const double x = 8.0 * k;
        const double window = 3.0 * (std::sin(x) - x * std::cos(x)) / (x * x * x);
        variance += k * k * spectrum(k) * window * window;
    }
    return std::sqrt(variance * step / (2.0 * Pi * Pi));
}

struct TemporaryDirectory {
    std::filesystem::path path_m;
    TemporaryDirectory() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path_m = std::filesystem::temp_directory_path()
                 / ("ippl-cosmology-physics-" + std::to_string(stamp));
        std::filesystem::create_directory(path_m);
    }
    ~TemporaryDirectory() { std::filesystem::remove_all(path_m); }
};

}  // namespace

int main() {
    try {
        cosmology::Config config;
        config.validate();
        cosmology::Background background(config);
        close(background.D(1.0), 1.0, 1.0e-14, "growth normalization");
        const double odeToday = odeGrowth(config.omegaMatter, 1.0)[0];
        for (const double a : {0.001, 0.02, 0.1, 0.5, 1.0}) {
            const auto ode = odeGrowth(config.omegaMatter, a);
            close(background.D(a), ode[0] / odeToday, 2.0e-10, "LCDM D versus independent ODE");
            close(background.f(a), ode[1] / ode[0], 2.0e-10, "LCDM f versus independent ODE");
        }
        config.omegaMatter = 1.0;
        cosmology::Background eds(config);
        for (const double a : {0.001, 0.02, 0.1, 0.5, 1.0}) {
            close(eds.E(a), std::pow(a, -1.5), 1.0e-10, "Einstein-de Sitter E");
            close(eds.D(a), a, 1.0e-12, "Einstein-de Sitter D=a");
            close(eds.f(a), 1.0, 1.0e-12, "Einstein-de Sitter f=1");
        }
        close(eds.kick(0.02, 0.1), 2.0 * (std::sqrt(0.1) - std::sqrt(0.02)),
              1.0e-12, "analytic EdS kick");
        close(eds.drift(0.02, 0.1), 2.0 * (1.0 / std::sqrt(0.02) - 1.0 / std::sqrt(0.1)),
              1.0e-11, "analytic EdS drift");
        close(background.kick(0.02, 0.1), -background.kick(0.1, 0.02), 1.0e-13,
              "reversed kick");
        close(background.drift(0.02, 0.02), 0.0, 0.0, "zero drift");
        rejects([&] { background.D(0.0); }, "invalid scale factor");

        config = cosmology::Config();
        cosmology::PowerSpectrum spectrum(config);
        close(spectrum.transfer(0.0), 1.0, 0.0, "BBKS zero-k transfer limit");
        close(spectrum(0.0), 0.0, 0.0, "zero DC power");
        close(spectrum.sigmaR(8.0), config.sigma8, 1.0e-13, "sigma8 normalization");
        close(midpointSigma8(spectrum), config.sigma8, 2.0e-7,
              "sigma8 independent linear-k quadrature");
        close(spectrum.atScaleFactor(0.1, 0.1, background) / spectrum(0.1),
              std::pow(background.D(0.1), 2), 1.0e-14, "power grows as D squared");
        rejects([&] { spectrum(-1.0); }, "negative wave number");

        TemporaryDirectory temporary;
        const auto inputPath = temporary.path_m / "input.par";
        const auto transferPath = temporary.path_m / "synthetic.tf";
        {
            std::ofstream transfer(transferPath);
            // Omega_bar/Omega_m=1/6 gives normalized T=1, 1/2, 1/4 at these knots.
            transfer << "0.00001 12 6 0 0 0 0\n0.1 6 3 0 0 0 0\n10 3 1.5 0 0 0 0\n";
            std::ofstream input(inputPath);
            input << "  # parser test\nnp = 16\nnt=20\nseed=73452342811 // exceeds 32 bits\n"
                  << "TFFlag=0\ntransfer_file=synthetic.tf\nwrite_particles=false\n"
                  << "Omega_nu=0\nOmega_r=0\nf_NL=0\nw_de=-1\n";
        }
        auto parsed = cosmology::Config::fromFile(inputPath.string());
        if (parsed.seed != UINT64_C(73452342811) || parsed.nGrid != 16 || parsed.writeParticles)
            throw std::runtime_error("Parsed values do not match input");
        if (parsed.transferFile != transferPath.string())
            throw std::runtime_error("Transfer file did not resolve relative to config");
        cosmology::PowerSpectrum table(parsed);
        close(table.transfer(0.00001), 1.0, 0.0, "CMBFAST first-row normalization");
        close(table.transfer(0.1), 0.5, 1.0e-15, "CMBFAST weighted normalization");
        close(table.transfer(5.05), 0.375, 1.0e-15, "CMBFAST linear interpolation");
        close(midpointSigma8(table), parsed.sigma8, 3.0e-7,
              "table sigma8 independent quadrature");
        rejects([&] { table.transfer(11.0); }, "table extrapolation");
        for (const std::string badInput : {
                 "Omega_nu=0.01", "Omega_r=0.0001", "f_NL=1", "w_de=-0.9", "np=15", "np=100000",
                 "seed=-1", "seed=18446744073709551616", "np=16garbage", "np=16\nnp=32",
                 "TFFlag=5", "ic_mode=sine\nmode_x=0", "unknown_parameter=1"}) {
            { std::ofstream input(inputPath); input << badInput << '\n'; }
            rejects([&] { cosmology::Config::fromFile(inputPath.string()); }, badInput);
        }
        std::cout << "Cosmology config, LCDM background, quadrature, and spectrum tests passed\n";
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
