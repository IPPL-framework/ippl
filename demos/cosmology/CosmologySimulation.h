#ifndef IPPL_COSMOLOGY_SIMULATION_H
#define IPPL_COSMOLOGY_SIMULATION_H

#include "Ippl.h"
#include "CosmologyConfig.h"
#include "CosmologyPhysics.h"
#include <Kokkos_MathematicalConstants.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

namespace cosmology {
constexpr unsigned Dim = 3;
using Vector = ippl::Vector<double, Dim>;
using Mesh = ippl::UniformCartesian<double, Dim>;
using Layout = ippl::FieldLayout<Dim>;
using ParticleLayout = ippl::ParticleSpatialLayout<double, Dim, Mesh>;
using RealField = ippl::Field<double, Dim, Mesh, Mesh::DefaultCentering>;
using Complex = Kokkos::complex<double>;
using ComplexField = ippl::Field<Complex, Dim, Mesh, Mesh::DefaultCentering>;
using VectorField = ippl::Field<Vector, Dim, Mesh, Mesh::DefaultCentering>;
using FFT = ippl::FFT<ippl::CCTransform, ComplexField>;
using Index = typename ippl::RangePolicy<Dim>::index_array_type;

class Particles : public ippl::ParticleBase<ParticleLayout> {
public:
    ippl::ParticleAttrib<double> mass;
    ippl::ParticleAttrib<Vector> momentum, force, lagrangian;
    ippl::ParticleAttrib<std::uint64_t> globalId;

    explicit Particles(ParticleLayout& layout) : ippl::ParticleBase<ParticleLayout>(layout) {
        addAttribute(mass);
        addAttribute(momentum);
        addAttribute(force);
        addAttribute(lagrangian);
        addAttribute(globalId);
        // Simulation::updateParticles supplies robust periodic wrapping itself.
        setParticleBC(ippl::BC::NO);
    }
};

// SplitMix64 finalizer: a fixed mapping of (seed, canonical global Fourier index).
// There is no shared RNG pool, rank seed, or dependence on kernel scheduling.
KOKKOS_INLINE_FUNCTION std::uint64_t mixBits(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

KOKKOS_INLINE_FUNCTION double uniformOpen(std::uint64_t x) {
    // 52 random bits; neither endpoint is reachable, including after rounding.
    return (static_cast<double>(mixBits(x) >> 12) + 0.5) / 4503599627370496.0;
}

class Simulation {
    Config config_m;
    Background background_m;
    ippl::NDIndex<Dim> domain_m;
    std::unique_ptr<Mesh> mesh_m;
    std::unique_ptr<Layout> layout_m;
    std::unique_ptr<ParticleLayout> particleLayout_m;
    std::unique_ptr<Particles> particles_m;
    RealField density_m;
    VectorField force_m;
    ComplexField modes_m, scratch_m;
    std::unique_ptr<FFT> fft_m;
    std::ofstream diagnostics_m;
    double initialModeAmplitude_m = 0.0;
    double massError_m = 0.0;
    double maxImaginary_m = 0.0;
    // Only the imported-evolution diagnostic sets this; normal ICs remain Nmesh^3.
    std::uint64_t expectedParticleCount_m = 0;

    std::uint64_t totalParticles() const {
        if (expectedParticleCount_m != 0) return expectedParticleCount_m;
        const auto n = static_cast<std::uint64_t>(config_m.nGrid);
        return n * n * n;
    }

    static double globalSum(double value) {
        double result;
        MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_SUM, ippl::Comm->getCommunicator());
        return result;
    }

    static double globalMax(double value) {
        double result;
        MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_MAX, ippl::Comm->getCommunicator());
        return result;
    }

    void checkParticleCount() const {
        std::uint64_t local = particles_m->getLocalNum(), total = 0;
        MPI_Allreduce(&local, &total, 1, MPI_UINT64_T, MPI_SUM,
                      ippl::Comm->getCommunicator());
        if (total != totalParticles()) throw std::runtime_error("Global particle count changed");
    }

    void updateParticles() {
        // The legacy IPPL periodic BC assumes small boundary overshoots. Reduce
        // arbitrary finite displacements into the box before ownership lookup.
        auto r = particles_m->R.getView();
        const double box = config_m.boxSize;
        Kokkos::parallel_for("Cosmology periodic positions", particles_m->getLocalNum(),
            KOKKOS_LAMBDA(const std::size_t l) {
                for (int d = 0; d < 3; ++d) {
                    r(l)[d] -= box * Kokkos::floor(r(l)[d] / box);
                    // A tiny negative input can round the subtraction to box.
                    if (r(l)[d] >= box) r(l)[d] = 0.0;
                }
            });
        particles_m->update();
        checkParticleCount();
    }

    void createLattice() {
        auto dom = layout_m->getLocalNDIndex();
        const std::size_t nx = dom[0].length(), ny = dom[1].length();
        const std::size_t count = nx * ny * dom[2].length();
        particles_m->create(count);
        particles_m->mass = 1.0;
        particles_m->momentum = 0.0;
        particles_m->force = 0.0;
        auto r = particles_m->R.getView();
        auto q = particles_m->lagrangian.getView();
        auto id = particles_m->globalId.getView();
        const int n = config_m.nGrid;
        const double h = config_m.boxSize / n;
        Kokkos::parallel_for("Cosmology lattice", count, KOKKOS_LAMBDA(const std::size_t l) {
            const std::uint64_t i = l % nx + dom[0].first();
            const std::uint64_t j = (l / nx) % ny + dom[1].first();
            const std::uint64_t k = l / (nx * ny) + dom[2].first();
            q(l) = Vector((i + 0.5) * h, (j + 0.5) * h, (k + 0.5) * h);
            r(l) = q(l);
            id(l) = i + static_cast<std::uint64_t>(n) * (j + static_cast<std::uint64_t>(n) * k);
        });
    }

    void initializeModes() {
        auto view = modes_m.getView();
        const auto dom = layout_m->getLocalNDIndex();
        const int ng = modes_m.getNghost(), n = config_m.nGrid;
        const double pi = Kokkos::numbers::pi_v<double>;
        const double volume = std::pow(config_m.boxSize, 3);
        const auto seed = config_m.seed;
        const double initialGrowth = background_m.D(config_m.aInitial());
        const double amplitude = config_m.amplitude / initialGrowth;
        const ippl::Vector<int, 3> mode(config_m.mode[0], config_m.mode[1], config_m.mode[2]);
        const bool gaussian = config_m.icMode == "gaussian";
        const bool sine = config_m.icMode == "sine";
        // Isotropic P(k) only depends on the squared integer wave number. This
        // compact host table avoids host copies of distributed 3D fields.
        const int maxK2 = 3 * (n / 2) * (n / 2);
        Kokkos::View<double*> power("Cosmology radial P(k)", maxK2 + 1);
        auto hostPower = Kokkos::create_mirror_view(power);
        hostPower(0) = 0.0;
        if (gaussian) {
            PowerSpectrum spectrum(config_m);
            const double fundamental = 2 * pi / config_m.boxSize;
            for (int k2 = 1; k2 <= maxK2; ++k2)
                hostPower(k2) = spectrum(fundamental * std::sqrt(double(k2)));
        } else {
            for (int k2 = 1; k2 <= maxK2; ++k2) hostPower(k2) = 0.0;
        }
        Kokkos::deep_copy(power, hostPower);
        ippl::parallel_for("Cosmology Gaussian modes", ippl::getRangePolicy(view, ng),
            KOKKOS_LAMBDA(const Index& idx) {
                int g[3], k[3], neg[3];
                int k2 = 0;
                bool nyquist = false;
                for (int d = 0; d < 3; ++d) {
                    g[d] = idx[d] - ng + dom[d].first();
                    k[d] = g[d] <= n / 2 ? g[d] : g[d] - n;
                    neg[d] = (n - g[d]) % n;
                    k2 += k[d] * k[d];
                    nyquist = nyquist || (g[d] == n / 2);
                }
                Complex delta(0.0, 0.0);
                if (k2 != 0 && !nyquist) {
                    if (sine) {
                        bool positive = true, negative = true;
                        for (int d = 0; d < 3; ++d) {
                            positive = positive && k[d] == mode[d];
                            negative = negative && k[d] == -mode[d];
                        }
                        if (positive || negative) delta = Complex(amplitude / 2, 0);
                    } else if (gaussian) {
                        const std::uint64_t key = (std::uint64_t(g[0]) * n + g[1]) * n + g[2];
                        const std::uint64_t other = (std::uint64_t(neg[0]) * n + neg[1]) * n + neg[2];
                        const std::uint64_t canonical = key < other ? key : other;
                        const double u1 = uniformOpen(mixBits(seed) ^ (2 * canonical));
                        const double u2 = uniformOpen(mixBits(seed) ^ (2 * canonical + 1));
                        const double radius = Kokkos::sqrt(-Kokkos::log(u1) * power(k2) / volume);
                        delta = Complex(radius * Kokkos::cos(2 * pi * u2),
                                        radius * Kokkos::sin(2 * pi * u2) * (key <= other ? 1 : -1));
                    }
                    // FFT samples are cell centered; translate physical Fourier phases.
                    const double phase = pi * (k[0] + k[1] + k[2]) / n;
                    delta *= Complex(Kokkos::cos(phase), Kokkos::sin(phase));
                }
                ippl::apply(view, idx) = delta;
            });
    }

    // Compute inverse FFT of i*k_component/k² times modes. The full k² is
    // retained on Nyquist planes; only the differentiated Nyquist component
    // is zero, as required for a real discrete derivative.
    void inverseGradient(int component, double factor) {
        const auto dom = layout_m->getLocalNDIndex();
        const int ng = modes_m.getNghost(), n = config_m.nGrid;
        const double fundamental = 2 * Kokkos::numbers::pi_v<double> / config_m.boxSize;
        auto src = modes_m.getView();
        auto dst = scratch_m.getView();
        ippl::parallel_for("Cosmology spectral force", ippl::getRangePolicy(dst, ng),
            KOKKOS_LAMBDA(const Index& idx) {
                double wave[3], k2 = 0;
                int g[3];
                for (int d = 0; d < 3; ++d) {
                    g[d] = idx[d] - ng + dom[d].first();
                    wave[d] = fundamental * (g[d] <= n / 2 ? g[d] : g[d] - n);
                    k2 += wave[d] * wave[d];
                }
                const double scale = (k2 == 0 || g[component] == n / 2) ? 0 :
                    factor * wave[component] / k2;
                ippl::apply(dst, idx) = Complex(0, scale) * ippl::apply(src, idx);
            });
        fft_m->transform(ippl::BACKWARD, scratch_m);
        double imag = 0;
        ippl::parallel_reduce("Cosmology reality", ippl::getRangePolicy(dst, ng),
            KOKKOS_LAMBDA(const Index& idx, double& value) {
                value = Kokkos::max(value, Kokkos::abs(ippl::apply(dst, idx).imag()));
            }, Kokkos::Max<double>(imag));
        maxImaginary_m = std::max(maxImaginary_m, globalMax(imag));
    }

    void displaceParticles() {
        const auto dom = layout_m->getLocalNDIndex();
        const std::size_t nx = dom[0].length(), ny = dom[1].length();
        const int ng = scratch_m.getNghost();
        const double a = config_m.aInitial(), growth = background_m.D(a);
        const double momentumFactor = a * a * background_m.E(a) * background_m.f(a) * growth;
        auto r = particles_m->R.getView(), p = particles_m->momentum.getView();
        const auto count = particles_m->getLocalNum();
        for (int d = 0; d < 3; ++d) {
            inverseGradient(d, 1.0);
            const auto displacement = scratch_m.getView();
            Kokkos::parallel_for("Cosmology 1LPT particles", count,
                KOKKOS_LAMBDA(const std::size_t l) {
                    const auto i = l % nx + ng, j = (l / nx) % ny + ng, k = l / (nx * ny) + ng;
                    const double psi = displacement(i, j, k).real();
                    r(l)[d] += growth * psi;
                    p(l)[d] = momentumFactor * psi;
                });
        }
        updateParticles();
    }

    void densityToModes() {
        const auto real = density_m.getView();
        auto complex = modes_m.getView();
        const int ng = density_m.getNghost();
        ippl::parallel_for("Cosmology density transform", ippl::getRangePolicy(real, ng),
            KOKKOS_LAMBDA(const Index& idx) {
                ippl::apply(complex, idx) = Complex(ippl::apply(real, idx), 0);
            });
        fft_m->transform(ippl::FORWARD, modes_m);
    }

    void solveFromDensity() {
        densityToModes();
        auto force = force_m.getView();
        const int ng = force_m.getNghost();
        for (int d = 0; d < 3; ++d) {
            inverseGradient(d, 1.5 * config_m.omegaMatter);
            auto component = scratch_m.getView();
            ippl::parallel_for("Cosmology force component", ippl::getRangePolicy(force, ng),
                KOKKOS_LAMBDA(const Index& idx) {
                    ippl::apply(force, idx)[d] = ippl::apply(component, idx).real();
                });
        }
    }

    void solveForce() {
        density_m = 0.0;
        scatter(particles_m->mass, density_m, particles_m->R);
        massError_m = std::abs(density_m.sum() / double(totalParticles()) - 1.0);
        if (!std::isfinite(massError_m) || massError_m > 1.e-10)
            throw std::runtime_error("CIC mass conservation failed");
        if (totalParticles() == config_m.particleCount()) {
            density_m = density_m - 1.0; // preserve the production one-particle-per-cell path
        } else {
            // Imported particles may be held fixed while the force mesh changes.
            // Unit masses give mean cell mass Nparticles/Ncells; solve for delta.
            density_m = density_m * (double(config_m.particleCount()) / double(totalParticles())) - 1.0;
        }
        solveFromDensity();
        gather(particles_m->force, force_m, particles_m->R);
    }

    void kick(double integral) {
        particles_m->momentum = particles_m->momentum + particles_m->force * integral;
    }

    void drift(double integral) {
        particles_m->R = particles_m->R + particles_m->momentum * integral;
        updateParticles();
    }

    void advanceStep(double a0, double a1) {
        const double half = std::sqrt(a0 * a1);
        kick(background_m.kick(a0, half));
        drift(background_m.drift(a0, a1));
        solveForce();
        kick(background_m.kick(half, a1));
    }

    void diagnose(int step, double a) {
        const auto r = particles_m->R.getView(), p = particles_m->momentum.getView();
        const auto force = particles_m->force.getView(), q = particles_m->lagrangian.getView();
        const auto count = particles_m->getLocalNum();
        const double box = config_m.boxSize;
        const Vector k(2 * Kokkos::numbers::pi_v<double> * config_m.mode[0] / box,
                       2 * Kokkos::numbers::pi_v<double> * config_m.mode[1] / box,
                       2 * Kokkos::numbers::pi_v<double> * config_m.mode[2] / box);
        double real = 0, imag = 0, displ2 = 0, p2 = 0, force2 = 0;
        Kokkos::parallel_reduce("Cosmology diagnostics", count,
            KOKKOS_LAMBDA(const std::size_t l, double& re, double& im,
                          double& d2, double& mom2, double& f2) {
                double phase = 0;
                for (int d = 0; d < 3; ++d) {
                    phase += k[d] * r(l)[d];
                    double delta = r(l)[d] - q(l)[d];
                    delta -= box * Kokkos::floor(delta / box + 0.5);
                    d2 += delta * delta;
                    mom2 += p(l)[d] * p(l)[d];
                    f2 += force(l)[d] * force(l)[d];
                }
                re += Kokkos::cos(phase);
                im -= Kokkos::sin(phase);
            }, real, imag, displ2, p2, force2);
        const double total = double(totalParticles());
        real = globalSum(real) / total;
        imag = globalSum(imag) / total;
        displ2 = globalSum(displ2) / total;
        p2 = globalSum(p2) / total;
        force2 = globalSum(force2) / total;
        double density2 = 0;
        auto rho = density_m.getView();
        ippl::parallel_reduce("Cosmology density rms", ippl::getRangePolicy(rho, density_m.getNghost()),
            KOKKOS_LAMBDA(const Index& idx, double& sum) {
                const double value = ippl::apply(rho, idx);
                sum += value * value;
            }, density2);
        density2 = globalSum(density2) / total;
        const double amplitude = 2 * std::hypot(real, imag);
        if (step == 0) initialModeAmplitude_m = amplitude;
        const double reference = config_m.icMode == "sine" ? config_m.amplitude : initialModeAmplitude_m;
        const double expected = reference * background_m.D(a) / background_m.D(config_m.aInitial());
        if (!std::isfinite(displ2 + p2 + force2 + density2 + amplitude))
            throw std::runtime_error("Nonfinite particle/field diagnostic");
        if (ippl::Comm->rank() == 0) {
            diagnostics_m << step << ',' << a << ',' << background_m.D(a) << ',' << background_m.f(a)
                << ',' << totalParticles() << ',' << massError_m << ',' << std::sqrt(density2)
                << ',' << real << ',' << imag << ',' << amplitude << ',' << expected
                << ',' << std::sqrt(displ2) << ',' << std::sqrt(p2) << ',' << std::sqrt(force2) << '\n';
            diagnostics_m.flush();
            std::cout << "step=" << step << " a=" << a << " delta_rms=" << std::sqrt(density2)
                      << " mode=" << amplitude << " linear=" << expected << '\n';
        }
    }

    void snapshot(const std::string& name) {
        if (!config_m.writeParticles) return;
        auto r = particles_m->R.getHostMirror(), p = particles_m->momentum.getHostMirror();
        auto id = particles_m->globalId.getHostMirror();
        Kokkos::deep_copy(r, particles_m->R.getView());
        Kokkos::deep_copy(p, particles_m->momentum.getView());
        Kokkos::deep_copy(id, particles_m->globalId.getView());
        const auto path = std::filesystem::path(config_m.output) /
            ("particles_" + name + "_rank" + std::to_string(ippl::Comm->rank()) + ".csv");
        std::ofstream out(path);
        if (!out) throw std::runtime_error("Cannot open snapshot " + path.string());
        out << std::setprecision(17) << "id,x,y,z,px,py,pz\n";
        for (std::size_t i = 0; i < particles_m->getLocalNum(); ++i) {
            out << id(i);
            for (int d = 0; d < 3; ++d) out << ',' << r(i)[d];
            for (int d = 0; d < 3; ++d) out << ',' << p(i)[d];
            out << '\n';
        }
        if (!out) throw std::runtime_error("Snapshot write failed");
    }

public:
    // Diagnostic-only entry point, defined by tests/CompareCosmologyForce.cpp.
    // Imports validated equal-mass particles and exercises the production CIC/FFT/gather path.
    void compareFrozenForce(const std::string& inputCsv, const std::string& outputDirectory);

    // Imported phase-space diagnostic, defined by tests/CompareCosmologyEvolution.cpp.
    // Uses the same production KDK/force path, with optional fixed-particle mesh refinement.
    void compareImportedEvolution(int particleGrid, const std::string& inputCsv,
                                  const std::string& outputDirectory, double aInitial,
                                  double aFinal, int checkpoints);

    explicit Simulation(const Config& config) : config_m(config), background_m(config) {
        config_m.validate();
        for (int d = 0; d < 3; ++d) domain_m[d] = ippl::Index(config_m.nGrid);
        std::array<bool, 3> parallel = {true, true, true};
        layout_m = std::make_unique<Layout>(ippl::Comm->getCommunicator(), domain_m, parallel, true);
        mesh_m = std::make_unique<Mesh>(domain_m, Vector(config_m.boxSize / config_m.nGrid), Vector(0.0));
        particleLayout_m = std::make_unique<ParticleLayout>(*layout_m, *mesh_m);
        particles_m = std::make_unique<Particles>(*particleLayout_m);
        density_m.initialize(*mesh_m, *layout_m);
        force_m.initialize(*mesh_m, *layout_m);
        modes_m.initialize(*mesh_m, *layout_m);
        scratch_m.initialize(*mesh_m, *layout_m);
        ippl::ParameterList params;
        params.add("use_heffte_defaults", true);
        fft_m = std::make_unique<FFT>(*layout_m, params);
    }

    // Independent manufactured-density test for the actual spectral force path.
    void testSpectralForce() {
        const auto dom = layout_m->getLocalNDIndex();
        const int ng = density_m.getNghost(), n = config_m.nGrid;
        const double pi = Kokkos::numbers::pi_v<double>;
        const double k0 = 2 * pi / config_m.boxSize;
        auto rho = density_m.getView();
        ippl::parallel_for("Cosmology manufactured density", ippl::getRangePolicy(rho, ng),
            KOKKOS_LAMBDA(const Index& idx) {
                double phase = 0;
                for (int d = 0; d < 3; ++d)
                    phase += 2 * pi * (idx[d] - ng + dom[d].first() + 0.5) / n;
                ippl::apply(rho, idx) = Kokkos::cos(phase);
            });
        solveFromDensity();
        const double forceScale = -0.5 * config_m.omegaMatter / k0;
        auto force = force_m.getView();
        double error = 0;
        ippl::parallel_reduce("Cosmology force error", ippl::getRangePolicy(force, ng),
            KOKKOS_LAMBDA(const Index& idx, double& result) {
                double phase = 0;
                for (int d = 0; d < 3; ++d)
                    phase += 2 * pi * (idx[d] - ng + dom[d].first() + 0.5) / n;
                for (int d = 0; d < 3; ++d)
                    result = Kokkos::max(result, Kokkos::abs(ippl::apply(force, idx)[d] / forceScale
                                                            - Kokkos::sin(phase)));
            }, Kokkos::Max<double>(error));
        error = globalMax(error);
        if (error > 1.e-11 || maxImaginary_m > 1.e-11)
            throw std::runtime_error("Manufactured spectral force test failed");
        if (ippl::Comm->rank() == 0)
            std::cout << "spectral force PASS error=" << error << " imaginary=" << maxImaginary_m << '\n';
    }

    // Force a domain and periodic-boundary crossing independently of the small
    // displacements used by linear-growth tests. Verify every migrated attribute.
    void testMigration() {
        createLattice();
        const double box = config_m.boxSize;
        const Vector shift(2.37 * box, -1.61 * box, 0.53 * box);
        const auto count = particles_m->getLocalNum();
        auto r = particles_m->R.getView(), p = particles_m->momentum.getView();
        auto f = particles_m->force.getView();
        auto ids = particles_m->globalId.getView();
        Kokkos::parallel_for("Cosmology migration setup", count,
            KOKKOS_LAMBDA(const std::size_t l) {
                r(l) += shift;
                if (ids(l) == 0) r(l) = Vector(-box, box, 0.0);
                for (int d = 0; d < 3; ++d) {
                    p(l)[d] = double(ids(l)) + 0.25 * d;
                    f(l)[d] = -p(l)[d];
                }
            });
        updateParticles();
        r = particles_m->R.getView();
        p = particles_m->momentum.getView();
        f = particles_m->force.getView();
        ids = particles_m->globalId.getView();
        auto q = particles_m->lagrangian.getView();
        auto mass = particles_m->mass.getView();
        const auto n = std::uint64_t(config_m.nGrid);
        const auto localDomain = layout_m->getLocalNDIndex();
        double error = 0.0;
        Kokkos::parallel_reduce("Cosmology migration check", particles_m->getLocalNum(),
            KOKKOS_LAMBDA(const std::size_t l, double& maximum) {
                const std::uint64_t index[3] = {ids(l) % n, (ids(l) / n) % n, ids(l) / (n * n)};
                maximum = Kokkos::max(maximum, Kokkos::abs(mass(l) - 1.0));
                for (int d = 0; d < 3; ++d) {
                    const double lower = localDomain[d].first() * (box / n);
                    const double upper = (localDomain[d].last() + 1) * (box / n);
                    if (!(r(l)[d] >= lower && r(l)[d] <= upper
                          && r(l)[d] >= 0.0 && r(l)[d] < box)) maximum = 1.0;
                    const double original = (index[d] + 0.5) * (box / n);
                    double expected = original + shift[d];
                    expected -= box * Kokkos::floor(expected / box);
                    if (ids(l) == 0) expected = 0.0;
                    maximum = Kokkos::max(maximum, Kokkos::abs((r(l)[d] - expected) / box));
                    maximum = Kokkos::max(maximum, Kokkos::abs((q(l)[d] - original) / box));
                    maximum = Kokkos::max(maximum, Kokkos::abs(p(l)[d] - (double(ids(l)) + 0.25 * d)));
                    maximum = Kokkos::max(maximum, Kokkos::abs(f(l)[d] + p(l)[d]));
                }
            }, Kokkos::Max<double>(error));
        auto hostIds = particles_m->globalId.getHostMirror();
        Kokkos::deep_copy(hostIds, ids);
        std::vector<int> localCounts(totalParticles(), 0), globalCounts(totalParticles(), 0);
        for (std::size_t i = 0; i < particles_m->getLocalNum(); ++i) {
            if (hostIds(i) >= totalParticles()) throw std::runtime_error("Invalid migrated particle ID");
            ++localCounts[hostIds(i)];
        }
        MPI_Allreduce(localCounts.data(), globalCounts.data(), int(totalParticles()), MPI_INT,
                      MPI_SUM, ippl::Comm->getCommunicator());
        error = globalMax(error);
        const bool unique = std::all_of(globalCounts.begin(), globalCounts.end(),
                                        [](int value) { return value == 1; });
        if (error > 1.e-12 || !unique)
            throw std::runtime_error("Particle migration test failed: error=" + std::to_string(error)
                                     + "; unique IDs=" + std::to_string(unique));
        if (ippl::Comm->rank() == 0) std::cout << "particle migration PASS error=" << error << '\n';
    }

    void run() {
        const auto start = MPI_Wtime();
        // Refuse accidental overwrite of an earlier run, including stale rank files.
        if (ippl::Comm->rank() == 0) {
            const std::filesystem::path output(config_m.output);
            if (std::filesystem::exists(output) && !std::filesystem::is_empty(output))
                throw std::runtime_error("Output directory must be new or empty: " + config_m.output);
            std::filesystem::create_directories(output);
            diagnostics_m.open(output / "diagnostics.csv");
            if (!diagnostics_m) throw std::runtime_error("Cannot create diagnostics.csv");
            diagnostics_m << std::setprecision(17)
                << "step,a,D,f,particles,mass_error,delta_rms,mode_re,mode_im,mode_amplitude,expected_amplitude,displacement_rms,momentum_rms,force_rms\n";
        }
        ippl::Comm->barrier();
        createLattice();
        initializeModes();
        displaceParticles();
        if (maxImaginary_m > 1.e-10)
            throw std::runtime_error("Initial displacements are not real");
        solveForce();
        diagnose(0, config_m.aInitial());
        snapshot("initial");
        const double logStep = std::log(config_m.aFinal() / config_m.aInitial()) / config_m.nSteps;
        for (int step = 0; step < config_m.nSteps; ++step) {
            const double a0 = config_m.aInitial() * std::exp(step * logStep);
            const double a1 = config_m.aInitial() * std::exp((step + 1) * logStep);
            advanceStep(a0, a1);
            if ((step + 1) % config_m.diagnosticsEvery == 0 || step + 1 == config_m.nSteps)
                diagnose(step + 1, a1);
        }
        snapshot("final");
        const double seconds = globalMax(MPI_Wtime() - start);
        if (ippl::Comm->rank() == 0) {
            std::ofstream metadata(std::filesystem::path(config_m.output) / "metadata.txt");
            metadata << std::setprecision(17)
                << "model=flat Gaussian LCDM; 1LPT IC; periodic CIC particle-mesh KDK\n"
                << "position_unit=Mpc/h\np=a^2 dx/d(H0 t)\npeculiar_velocity_km_s=100*p/a\n"
                << "fft=forward 1/N^3; inverse unnormalized\n"
                << "rng=SplitMix64 canonical Fourier pair; Box-Muller\n"
                << "nyquist=IC Nyquist planes zero; force differentiated Nyquist component zero\n"
                << "execution_space=" << Kokkos::DefaultExecutionSpace::name()
                << "\nmemory_space=" << Kokkos::DefaultExecutionSpace::memory_space::name() << '\n'
                << "ranks=" << ippl::Comm->size() << "\nthreads=" << Kokkos::DefaultExecutionSpace().concurrency()
                << "\nnp=" << config_m.nGrid << "\nnt=" << config_m.nSteps
                << "\nbox_size=" << config_m.boxSize << "\nseed=" << config_m.seed
                << "\nz_in=" << config_m.zInitial << "\nz_fi=" << config_m.zFinal
                << "\nhubble=" << config_m.hubble << "\nOmega_m=" << config_m.omegaMatter
                << "\nOmega_bar=" << config_m.omegaBaryon << "\nSigma_8=" << config_m.sigma8
                << "\nn_s=" << config_m.spectralIndex << "\nTFFlag=" << config_m.transferFunction
                << "\ntransfer_file=" << config_m.transferFile
                << "\nic_mode=" << config_m.icMode << "\namplitude=" << config_m.amplitude
                << "\nmode=" << config_m.mode[0] << ',' << config_m.mode[1] << ',' << config_m.mode[2]
                << "\nmax_inverse_imaginary=" << maxImaginary_m << "\nwall_seconds=" << seconds << '\n';
            if (!metadata) throw std::runtime_error("Cannot write metadata");
            std::cout << "Completed " << config_m.nSteps << " steps on " << ippl::Comm->size()
                      << " ranks in " << seconds << " s; output " << config_m.output << '\n';
        }
    }
};
} // namespace cosmology
#endif
