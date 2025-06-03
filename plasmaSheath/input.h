#include <algorithm>
#include <cmath>

constexpr double pi = 3.14159265358979323846;

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

	constexpr double Z_i = 1.0;  // ion charge
	constexpr double Z_e = -1.0;  // electron charge
	static_assert(Z_e == -1.0);
	
	constexpr double n_e0 = 1.0;  // electron density at MPE, by definition of normalization
	static_assert(n_e0 == 1.0);
	constexpr double n_i0 = 1.0/Z_i;  // ion density at MPE, such that Z n_i - n_e = 0

	constexpr double m_i = 1.0;  // ion mass, by definition of normalization
	static_assert(m_i == 1.0);
	constexpr double m_e = 1.0/1836.0;  // electron mass

	constexpr double tau = 1.0;  // ion-electron temperature ratio, τ = T_i / T_e
	constexpr double nu = 1.0;  // perp-parallel temperature anisotropy, ν = v_th_perp_i / v_th_par_i

	constexpr double D_D = 1.0;  // Debye length (setting this to 1.0 is equivalent to setting L_ref = λ_D)
	constexpr double D_C = 10.0;  // ion thermal gyroradius ρ_th_i, in units of L_ref. Set this to ∞ (Kokkos::numbers::infinity_v<double>) to effectively set B = 0

	constexpr double alpha = 10*pi/180.0;  // magnetic field incidence angle

	constexpr double phi0 = -2.37;  // wall bias. note that phi(x=MPE) = 0

	constexpr bool kinetic_electrons = true;  // TODO: figure this out

	// derived quantities from the physical parameters
	// in normalized units, v_th_i = 1.0   and v_th_e = √(T_i/T_e) √m_e/m_i = √τ √~m_e
	//                      ρ_th_i = D_C   and ρ_th_e = D_C √~m_e / √τ
	//                      Ω_ci = 1/D_C   and Ω_ce = 1/(Z ~m_e D_C)
	constexpr double v_th_i = 1.0;  // ion thermal velocity, by definition of normalization
	constexpr double v_th_e = std::sqrt(tau / m_e);
	constexpr double rho_th_i = 1.0;
	constexpr double rho_th_e = D_C * std::sqrt(m_e / tau);
	constexpr double Omega_ci = 1.0/D_C;
	constexpr double Omega_ce = 1.0/(Z_i * m_e * D_C);
	static_assert(v_th_i == 1.0);
	static_assert(rho_th_i == 1.0);

	// -- SIMULATION PARAMETERS --
	constexpr double L = 100.0;  // length of the simulation domain, in units of L_ref
	constexpr double f_x = 0.1;  // resolution of the smallest length scale min(ρ_th_e, λ_D, ρ_th_i). should be < 1.0
	constexpr double f_t = 0.1;  // resolution of the smallest time scale 2π/Ω_ce. should be < 1.0
	constexpr double CFL_max = 0.5;  // β_max = v_max Δx/Δt, should be < 1.0
	
	constexpr double v_trunc_i = 6.0 * v_th_i;  // velocity at which to truncate the ion distribution function
	constexpr double v_trunc_e = 6.0 * v_th_e;  // velocity at which to truncate the electron distribution function
	constexpr double f_ion_speedup = 10.0;  // rough estimate of the velocity of the ions as the impact the wall

	// postprocessing of simulation parameters
	constexpr double dx0 = f_x * std::min({rho_th_e, rho_th_i, D_D});  // resolution such that dx << smallest length scale
	constexpr unsigned int nx = std::ceil(L / dx0);
	constexpr double dx = L / (double) nx;  // the actual dx
	constexpr double v_max = std::max({v_trunc_e, f_ion_speedup * v_trunc_i});  // maximum velocity that we expect to encounter
	constexpr double dt = std::min({
		f_t * 2.0*pi / std::max({Omega_ci, Omega_ce}),  // resolution such that dt << smallest time scale
		dx/v_max*CFL_max  // time step constraint due to the CFL condition
	});
}
