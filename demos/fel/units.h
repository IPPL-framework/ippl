/** @file units.h
 * @brief Compile-time conversion factors between SI and the FEL's internal units.
 * @ingroup fel_config
 *
 * The base length and time are tabulated Planck values multiplied by the common
 * scaling factor 1e30; the base mass is the tabulated Planck mass divided by that
 * factor. Thus these are scaled internal units, not unscaled Planck units. Charge
 * uses the rationalized convention given by unit_charge_in_electron_charges.
 *
 * With base units @f$L_0,T_0,M_0,Q_0@f$, convert a physical length by
 * @f$x_{\rm internal}=x_{\rm SI}/L_0@f$ and recover SI by multiplication.
 * The same rule applies to every unit_*_in_* factor below. Reciprocal constants
 * such as meter_in_unit_lengths convert one SI unit into internal units.
 * The derived velocity scale @f$L_0/T_0@f$ approximates the speed of light;
 * the solver equations use dimensionless light speed one. Constants are stored
 * as doubles with the precision of their tabulated decimal inputs.
 *
 * Names referring to electron charge denote the positive elementary-charge
 * magnitude. The sign of a configured bunch charge is supplied by the input.
 */
#ifndef UNITS_HPP
#define UNITS_HPP

/** @addtogroup fel_config
 * @{ */

/** @brief Square root of four pi, used to rationalize the charge unit; dimensionless. */
constexpr double sqrt_4pi = 3.54490770181103205459;
/** @brief Dimensionless internal-unit scaling factor, 1e30; not the fine-structure constant. */
constexpr double alpha_scaling_factor = 1e30;
/** @brief One internal length unit @f$L_0@f$ in metres (approximately 16.16 micrometres). */
constexpr double unit_length_in_meters = 1.616255e-35 * alpha_scaling_factor;
/** @brief One positive internal charge unit @f$Q_0@f$ in elementary-charge magnitudes,
 * approximately 3.302. */
constexpr double unit_charge_in_electron_charges = 11.70623710394218618969 / sqrt_4pi;
/** @brief One internal time unit @f$T_0@f$ in seconds (approximately 53.91 femtoseconds). */
constexpr double unit_time_in_seconds = 5.391247e-44 * alpha_scaling_factor;
/** @brief Tabulated electron rest mass in kilograms. */
constexpr double electron_mass_in_kg = 9.1093837015e-31;
/** @brief One internal mass unit @f$M_0@f$ in kilograms; Planck mass divided by the scaling factor.
 */
constexpr double unit_mass_in_kg = 2.176434e-8 / alpha_scaling_factor;
/** @brief One internal energy unit @f$M_0 L_0^2/T_0^2@f$ in joules. */
constexpr double unit_energy_in_joule = unit_mass_in_kg * unit_length_in_meters
                                        * unit_length_in_meters
                                        / (unit_time_in_seconds * unit_time_in_seconds);
/** @brief One kilogram expressed in internal mass units, @f$1/M_0@f$. */
constexpr double kg_in_unit_masses = 1.0 / unit_mass_in_kg;
/** @brief One metre expressed in internal length units, @f$1/L_0@f$. */
constexpr double meter_in_unit_lengths = 1.0 / unit_length_in_meters;
/** @brief Positive elementary-charge magnitude in coulombs; this constant carries no electron minus
 * sign. */
constexpr double electron_charge_in_coulombs = 1.602176634e-19;
/** @brief One coulomb expressed as a number of positive elementary-charge magnitudes. */
constexpr double coulomb_in_electron_charges = 1.0 / electron_charge_in_coulombs;

/** @brief One positive elementary charge expressed in internal charge units. */
constexpr double electron_charge_in_unit_charges = 1.0 / unit_charge_in_electron_charges;
/** @brief One second expressed in internal time units, @f$1/T_0@f$. */
constexpr double second_in_unit_times = 1.0 / unit_time_in_seconds;
/** @brief One electron rest mass expressed in internal mass units. */
constexpr double electron_mass_in_unit_masses = electron_mass_in_kg * kg_in_unit_masses;
/** @brief One internal force unit @f$M_0 L_0/T_0^2@f$ in newtons. */
constexpr double unit_force_in_newtons =
    unit_mass_in_kg * unit_length_in_meters / (unit_time_in_seconds * unit_time_in_seconds);

/** @brief One coulomb expressed in internal charge units, @f$1/Q_0@f$. */
constexpr double coulomb_in_unit_charges =
    coulomb_in_electron_charges * electron_charge_in_unit_charges;
/** @brief One internal voltage unit @f$V_0=M_0 L_0^2/(T_0^2 Q_0)@f$ in volts. */
constexpr double unit_voltage_in_volts = unit_energy_in_joule * coulomb_in_unit_charges;
/** @brief One internal charge unit @f$Q_0@f$ in coulombs. */
constexpr double unit_charges_in_coulomb = 1.0 / coulomb_in_unit_charges;
/** @brief One internal current unit @f$I_0=Q_0/T_0@f$ in amperes. */
constexpr double unit_current_in_amperes = unit_charges_in_coulomb / unit_time_in_seconds;
/** @brief One ampere expressed in internal current units, @f$1/I_0@f$. */
constexpr double ampere_in_unit_currents = 1.0 / unit_current_in_amperes;
/** @brief One internal current-times-length unit @f$I_0 L_0@f$ in ampere metres. */
constexpr double unit_current_length_in_ampere_meters =
    unit_current_in_amperes * unit_length_in_meters;
/** @brief One internal magnetic-field unit @f$V_0 T_0/L_0^2@f$ in tesla. */
constexpr double unit_magnetic_fluxdensity_in_tesla =
    unit_voltage_in_volts * unit_time_in_seconds / (unit_length_in_meters * unit_length_in_meters);
/** @brief One internal electric-field unit @f$V_0/L_0@f$ in volts per metre. */
constexpr double unit_electric_fieldstrength_in_voltpermeters =
    (unit_voltage_in_volts / unit_length_in_meters);
/** @brief One volt per metre expressed in internal electric-field units. */
constexpr double voltpermeter_in_unit_fieldstrengths =
    1.0 / unit_electric_fieldstrength_in_voltpermeters;
/** @brief Legacy power-density conversion in watts per square metre, using the stored 1.389e122
 * prefactor and inverse fourth power of the scaling factor. */
constexpr double unit_powerdensity_in_watt_per_square_meter =
    1.389e122
    / (alpha_scaling_factor * alpha_scaling_factor * alpha_scaling_factor * alpha_scaling_factor);
/** @brief One volt expressed in internal voltage units, @f$1/V_0@f$. */
constexpr double volts_in_unit_voltages = 1.0 / unit_voltage_in_volts;
/** @brief SI permittivity corresponding to internal permittivity one, @f$Q_0/(V_0 L_0)@f$, in
 * farads per metre. */
constexpr double epsilon0_in_si = unit_current_in_amperes * unit_time_in_seconds
                                  / (unit_voltage_in_volts * unit_length_in_meters);
/** @brief SI permeability corresponding to internal permeability one, @f$(M_0L_0/T_0^2)/I_0^2@f$,
 * in newtons per ampere squared. */
constexpr double mu0_in_si =
    unit_force_in_newtons / (unit_current_in_amperes * unit_current_in_amperes);
/** @brief Gravitational-coupling unit @f$L_0^3/(M_0 T_0^2)@f$ in SI units; after scaling this is
 * not the physical Newtonian gravitational constant. */
constexpr double G = unit_length_in_meters * unit_length_in_meters * unit_length_in_meters
                     / (unit_mass_in_kg * unit_time_in_seconds * unit_time_in_seconds);
/** @brief Dimensional consistency check @f$M_0^2 G/L_0^2@f$ in newtons, equal algebraically to the
 * internal force unit. */
constexpr double verification_gravity =
    unit_mass_in_kg * unit_mass_in_kg / (unit_length_in_meters * unit_length_in_meters) * G;
/** @brief Dimensionless check @f$[Q_0^2/(\epsilon_0 L_0^2)]/(M_0L_0/T_0^2)@f$; the geometrical
 * Coulomb-law factor 1/(4 pi) is absent from this expression. */
constexpr double verification_coulomb =
    (unit_charges_in_coulomb * unit_charges_in_coulomb
     / (unit_length_in_meters * unit_length_in_meters) * (1.0 / (epsilon0_in_si)))
    / unit_force_in_newtons;

/** @} */

#endif
