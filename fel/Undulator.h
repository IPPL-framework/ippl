#ifndef UNDULATOR_H
#define UNDULATOR_H
#include <Kokkos_Core.hpp>

#include <cmath>

#include "Types/Vector.h"

#include "units.h"
#include "LorentzTransform.h"
namespace ippl {
    template <typename scalar>
    struct undulator_parameters {
        scalar lambda_u;  // MITHRA: lambda_u
        scalar K;         // Undulator parameter
        scalar length;
        scalar B_magnitude;
        undulator_parameters(scalar K_undulator_parameter, scalar lambda_u, scalar _length)
            : lambda_u(lambda_u)
            , K(K_undulator_parameter)
            , length(_length) {
            B_magnitude = (2 * M_PI * electron_mass_in_unit_masses * K)
                          / (electron_charge_in_unit_charges * lambda_u);
        }
    };
    /**
     * @brief Struct representing an undulator.
     *
     * @tparam scalar Type of the scalar values (e.g., float, double).
     */
    template <typename scalar>
    struct Undulator {
        undulator_parameters<scalar> uparams;  ///< Parameters of the undulator.
        scalar distance_to_entry;              ///< Distance to the entry of the undulator.
        scalar k_u;                            ///< Wavenumber of the undulator.

        /**
         * @brief Constructor to initialize undulator parameters and calculate k_u.
         *
         * @param p Parameters of the undulator.
         * @param dte Distance to the entry of the undulator.
         */
        KOKKOS_FUNCTION Undulator(const undulator_parameters<scalar>& p, scalar dte)
            : uparams(p)
            , distance_to_entry(dte)
            , k_u(2 * M_PI / p.lambda_u) {}

        /**
         * @brief Overloaded operator() to compute magnetic field components.
         *
         * @param position_in_lab_frame Position vector in the lab frame.
         * @return Kokkos::pair<ippl::Vector<scalar, 3>, ippl::Vector<scalar, 3>>
         *         Pair containing magnetic field and its derivative.
         */
        KOKKOS_INLINE_FUNCTION Kokkos::pair<ippl::Vector<scalar, 3>, ippl::Vector<scalar, 3>>
        operator()(const ippl::Vector<scalar, 3>& position_in_lab_frame) const noexcept {
            using Kokkos::cos;
            using Kokkos::cosh;
            using Kokkos::exp;
            using Kokkos::sin;
            using Kokkos::sinh;

            Kokkos::pair<ippl::Vector<scalar, 3>, ippl::Vector<scalar, 3>>
                ret;             // Return pair containing magnetic field and its derivative.
            ret.first.fill(0);   // Initialize magnetic field vector.
            ret.second.fill(0);  // Initialize derivative vector.

            // If the position is before the undulator entry.
            if (position_in_lab_frame[2] < distance_to_entry) {
                scalar z_in_undulator = position_in_lab_frame[2] - distance_to_entry;
                assert(z_in_undulator < 0);  // Ensure we are in the correct region.
                scalar scal = exp(-((k_u * z_in_undulator) * (k_u * z_in_undulator)
                                    * 0.5));  // Gaussian decay factor.

                ret.second[0] = 0;  // No x-component.
                ret.second[1] = uparams.B_magnitude * cosh(k_u * position_in_lab_frame[1])
                                * z_in_undulator * k_u * scal;  // y-component.
                ret.second[2] = uparams.B_magnitude * sinh(k_u * position_in_lab_frame[1])
                                * scal;  // z-component.
            }
            // If the position is within the undulator.
            else if (position_in_lab_frame[2] > distance_to_entry
                     && position_in_lab_frame[2] < distance_to_entry + uparams.length) {
                scalar z_in_undulator = position_in_lab_frame[2] - distance_to_entry;
                assert(z_in_undulator >= 0);  // Ensure we are in the correct region.

                ret.second[0] = 0;  // No x-component.
                ret.second[1] = uparams.B_magnitude * cosh(k_u * position_in_lab_frame[1])
                                * sin(k_u * z_in_undulator);  // y-component.
                ret.second[2] = uparams.B_magnitude * sinh(k_u * position_in_lab_frame[1])
                                * cos(k_u * z_in_undulator);  // z-component.
            }
            return ret;
        }
    };
}  // namespace ippl
#endif  // UNDULATOR_H