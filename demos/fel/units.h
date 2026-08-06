#ifndef UNITS_HPP
#define UNITS_HPP
constexpr double sqrt_4pi = 3.54490770181103205459;
constexpr double alpha_scaling_factor = 1e30;
constexpr double unit_length_in_meters = 1.616255e-35 * alpha_scaling_factor;
constexpr double unit_charge_in_electron_charges = 11.70623710394218618969 / sqrt_4pi;
constexpr double unit_time_in_seconds = 5.391247e-44 * alpha_scaling_factor;
constexpr double electron_mass_in_kg = 9.1093837015e-31;
constexpr double unit_mass_in_kg = 2.176434e-8 / alpha_scaling_factor;
constexpr double unit_energy_in_joule = unit_mass_in_kg * unit_length_in_meters * unit_length_in_meters / (unit_time_in_seconds * unit_time_in_seconds);
constexpr double kg_in_unit_masses = 1.0 / unit_mass_in_kg;
constexpr double meter_in_unit_lengths = 1.0 / unit_length_in_meters;
constexpr double electron_charge_in_coulombs = 1.602176634e-19;
constexpr double coulomb_in_electron_charges = 1.0 / electron_charge_in_coulombs;

constexpr double electron_charge_in_unit_charges = 1.0 / unit_charge_in_electron_charges;
constexpr double second_in_unit_times = 1.0 / unit_time_in_seconds;
constexpr double electron_mass_in_unit_masses = electron_mass_in_kg * kg_in_unit_masses;
constexpr double unit_force_in_newtons = unit_mass_in_kg * unit_length_in_meters / (unit_time_in_seconds * unit_time_in_seconds);

constexpr double coulomb_in_unit_charges = coulomb_in_electron_charges * electron_charge_in_unit_charges;
constexpr double unit_voltage_in_volts = unit_energy_in_joule * coulomb_in_unit_charges;
constexpr double unit_charges_in_coulomb = 1.0 / coulomb_in_unit_charges;
constexpr double unit_current_in_amperes = unit_charges_in_coulomb / unit_time_in_seconds;
constexpr double ampere_in_unit_currents = 1.0 / unit_current_in_amperes;
constexpr double unit_current_length_in_ampere_meters = unit_current_in_amperes * unit_length_in_meters;
constexpr double unit_magnetic_fluxdensity_in_tesla = unit_voltage_in_volts * unit_time_in_seconds / (unit_length_in_meters * unit_length_in_meters);
constexpr double unit_electric_fieldstrength_in_voltpermeters = (unit_voltage_in_volts / unit_length_in_meters);
constexpr double voltpermeter_in_unit_fieldstrengths = 1.0 / unit_electric_fieldstrength_in_voltpermeters;
constexpr double unit_powerdensity_in_watt_per_square_meter = 1.389e122 / (alpha_scaling_factor * alpha_scaling_factor * alpha_scaling_factor * alpha_scaling_factor);
constexpr double volts_in_unit_voltages = 1.0 / unit_voltage_in_volts;
constexpr double epsilon0_in_si = unit_current_in_amperes * unit_time_in_seconds / (unit_voltage_in_volts * unit_length_in_meters);
constexpr double mu0_in_si = unit_force_in_newtons / (unit_current_in_amperes * unit_current_in_amperes);
constexpr double G = unit_length_in_meters * unit_length_in_meters * unit_length_in_meters / (unit_mass_in_kg * unit_time_in_seconds * unit_time_in_seconds);
constexpr double verification_gravity = unit_mass_in_kg * unit_mass_in_kg / (unit_length_in_meters * unit_length_in_meters) * G;
constexpr double verification_coulomb = (unit_charges_in_coulomb * unit_charges_in_coulomb / (unit_length_in_meters * unit_length_in_meters) * (1.0 / (epsilon0_in_si))) / unit_force_in_newtons;
#endif