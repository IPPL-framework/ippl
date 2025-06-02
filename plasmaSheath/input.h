// Input file for physical parameters
// in plasma sheath simulation

const double pi = Kokkos::numbers::pi_v<T>;

// there are two species: electrons and ions
double q_i = 1.0; // ion charge
double q_e = -1.0; // electron charge
T m_i = 1; // ion mass
T m_e = 1/1836; // electron mass

// ion distribution: (normal distribution)
double v_thi = 1; // thermal velocity of ions
double n_i = 1;
double v_max = 5; // max velocity
double f_max = Kokkos::sqrt(2.0); // max f(v) for rejection sampling
double K = n_i * Kokkos::pow(2 / pi, 1.5) / v_thi; // normalization factor

// electron distribution: Maxwellian (normal distribution)
Vector_t<double, 3> v0 = {0.0, 0.0, 0.0}; // avg velocity
double T_e = 1; // temperature
double n_e = n_i; // n_e
double prefactor_e = n_e * m_e / (2 * pi * T_e);
double stdeviation_e = Kokkos::sqrt(T_e/m_e); 

