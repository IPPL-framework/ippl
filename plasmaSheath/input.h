#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <algorithm>
#include <limits>

#ifndef PLASMA_INPUT_H_
#define PLASMA_INPUT_H_

constexpr double pi       = 3.14159265358979323846;
constexpr double infinity = std::numeric_limits<double>::infinity();

namespace params {
    // -- PHYSICAL PARAMETERS --
    // normalization:
    // q is in units of e
    // m is in units of m_i
    // v is in units of v_th_i = sqrt(T_i / m_i)
    // T is in units of T_e
    // n is in units of n_e(x=MPE)
    // x is in units of L_ref, usually the Debye length sqrt(eps T_e / (e² n_e))
    // phi is in units of T_e/e

    // ion charge
    constexpr double Z_i = 1.0;
    // electron charge
    constexpr double Z_e = -1.0;
    static_assert(Z_e == -1.0);

    // electron density at MPE, by definition of normalization
    constexpr double n_e0 = 1.0;
    static_assert(n_e0 == 1.0);
    // ion density at MPE, such that Z n_i - n_e = 0
    constexpr double n_i0 = 1.0 / Z_i;

    // ion mass, by definition of normalization
    constexpr double m_i = 1.0;
    static_assert(m_i == 1.0);
    constexpr double m_e = 1.0 / 1836.0;  // electron mass

    // ion-electron temperature ratio, τ = T_i / T_e
    constexpr double tau = 1.0;
    // perp-parallel temperature anisotropy, ν = v_th_perp_i / v_th_par_i
    constexpr double nu = 1.0;

    // Debye length (setting this to 1.0 is equivalent to setting L_ref = λ_D)
    constexpr double D_D = 1.0;
    // ion thermal gyroradius ρ_th_i, in units of L_ref.
    // To have B = 0, set D_C = ∞
    // constexpr double D_C = 10.0;
    constexpr double D_C = infinity;

    // magnetic field incidence angle
    // set this to 90deg for Debye sheath simulations, so that vpar = -vx !
    constexpr double alpha = pi / 2;  // 10 * pi / 180.0;

    // wall bias. note that phi(x=MPE) = 0
    constexpr double phi0 = -2.37;

    // toggles between adiabatic electrons and kinetic electron
    constexpr bool kinetic_electrons = false;

    // derived quantities from the physical parameters
    // in normalized units, v_th_i = 1.0   and v_th_e = √(T_i/T_e) √m_e/m_i = √τ √~m_e
    //                      ρ_th_i = D_C   and ρ_th_e = D_C √~m_e / √τ
    //                      Ω_ci = 1/D_C   and Ω_ce = 1/(Z ~m_e D_C)
    // ion thermal velocity, by definition of normalization
    constexpr double v_th_i = 1.0;
    static_assert(v_th_i == 1.0);
    const double v_th_e = Kokkos::sqrt(tau / m_e);  // can't use constexpr since sqrt not constexpr
    constexpr double rho_th_i = D_C;
    const double rho_th_e     = D_C * v_th_e;  // can't use constexpr since v_th_e not constexpr
    constexpr double Omega_ci = 1.0 / D_C;
    constexpr double Omega_ce = 1.0 / (Z_i * m_e * D_C);

    // -- SIMULATION PARAMETERS --
    // length of the simulation domain, in units of L_ref
    constexpr double L = 100.0;
    // resolution of the smallest length scale min(ρ_th_e, λ_D, ρ_th_i). should be < 1.0
    constexpr double f_x = 0.1;
    // resolution of the smallest time scale 2π/Ω_ce. should be < 1.0
    constexpr double f_t = 0.1;
    // β_max = v_max Δx/Δt, should be < 1.0
    constexpr double CFL_max = 0.5;

    // velocity at which to truncate the ion distribution function
    constexpr double v_trunc_i = 6.0 * v_th_i;
    // velocity at which to truncate the electron distribution function
    const double v_trunc_e =
        6.0 * v_th_e;  // can't use constexpr since v_th_e not constexpr
                       // rough estimate of the velocity of the ions as the impact the wall
    constexpr double f_ion_speedup = 10.0;

    // postprocessing of simulation parameters
    // resolution such that dx << smallest length scale
    // can't use constexpr since rho_th_e, min(), ceil() not marked as constexpr
    const double dx0 = f_x * std::min({kinetic_electrons ? rho_th_e : infinity, rho_th_i, D_D});
    const unsigned int nx = Kokkos::ceil(L / dx0);
    // the actual dx
    const double dx = L / (double)nx;
    // maximum velocity that we expect to encounter
    const double v_max = std::max({v_trunc_e, f_ion_speedup* v_trunc_i});
    const double dt    = std::min({
        std::isfinite(D_C)
               ? (f_t * 2.0 * pi / std::max({Omega_ci, kinetic_electrons ? Omega_ce : 0.0}))
               : infinity,      // resolution such that dt << smallest time scale
           dx / v_max* CFL_max  // time step constraint due to the CFL condition
       });

    // -- OUTPUT PARAMS --
    // dump once every 1000 timesteps
    constexpr unsigned int dump_interval = 1000;
}  // namespace params
#endif
