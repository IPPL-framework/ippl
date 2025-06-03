// Input file for physical parameters
// in plasma sheath simulation

const double pi = Kokkos::numbers::pi_v<T>;

// there are two species: electrons and ions
double q_i = 1.0; // ion charge
double q_e = -1.0; // electron charge
double m_i = 1.0; // ion mass
double m_e = 1.0/1836.0; // electron mass

// ion distribution: (normal distribution)
double v_thi = 1.0; // thermal velocity of ions
double n_i = 1.0;
double v_max = 5.0; // max velocity
double K = n_i * Kokkos::pow(2.0 / pi, 1.5) / v_thi; // normalization factor (doubts...?)
double stdeviation_i = v_thi; // sigma for normal distribution

// electron distribution: Maxwellian (normal distribution)
double T_e = 1.0; // temperature
double n_e = n_i; // n_e
double stdeviation_e = Kokkos::sqrt(T_e/m_e); // sigma for normal distribution
Vector_t<double, 3> v0 = {0.0, 0.0, 0.0}; // mu for normal distribution; use this also for ions
