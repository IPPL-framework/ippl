/** @file Undulator.h
 * @brief Prescribed planar-undulator magnetic field in laboratory coordinates.
 * @ingroup fel_physics
 */
#ifndef UNDULATOR_H
#define UNDULATOR_H
#include <Kokkos_Core.hpp>

#include <cmath>

#include "Types/Vector.h"

#include "units.h"
#include "LorentzTransform.h"
namespace ippl {
    /** @brief Host-created parameters in FEL's internal c=1 unit system.
     * @ingroup fel_physics
     * @tparam scalar Floating-point type used for lengths and field amplitude.
     *
     * The amplitude follows @f$K=eB_0\lambda_u/(2\pi m_e)@f$, where e and
     * m_e are the positive electron-charge magnitude and electron mass expressed
     * in the internal units from units.h. The constructor does not convert SI
     * lengths or validate the supplied period and length.
     */
    template <typename scalar>
    struct undulator_parameters {
        scalar lambda_u;  ///< Laboratory undulator period, in internal length units.
        scalar K;         ///< Dimensionless deflection parameter; its sign sets the field sign.
        scalar length;    ///< Length of the sinusoidal section, in internal length units.
        scalar B_magnitude;  ///< B0 in internal magnetic-field units, not tesla.
        /** @brief Precompute the laboratory magnetic amplitude on the host.
         * @param K_undulator_parameter Dimensionless undulator deflection parameter.
         * @param lambda_u Positive laboratory period in internal length units.
         * @param _length Nonnegative sinusoidal-section length in internal length units.
         * @pre Finite parameters and lambda_u>0; no input checks are performed here.
         */
        undulator_parameters(scalar K_undulator_parameter, scalar lambda_u, scalar _length)
            : lambda_u(lambda_u)
            , K(K_undulator_parameter)
            , length(_length) {
            B_magnitude = (2 * M_PI * electron_mass_in_unit_masses * K)
                          / (electron_charge_in_unit_charges * lambda_u);
        }
    };
    /**
     * @brief Static planar-undulator field with prescribed Gaussian entrance/exit fringes.
     * @ingroup fel_physics
     *
     * @tparam scalar Type of the scalar values (e.g., float, double).
     *
     * This value object samples an external laboratory field; it does not evolve
     * fields or model a material. All lengths and fields use units.h scales.
     * The laboratory electric field is zero. The field is independent of x and
     * has hyperbolic dependence on y, with no finite transverse aperture.
     * Constructor and evaluation are host/device callable; store the object by
     * value in the particle-push kernel. FreeElectronLaserManager first converts
     * particle positions to the lab frame, samples this object, then boosts its
     * E/B pair into the moving frame.
     *
     * @note The fringe profile is prescribed, not a magnetostatic boundary-value
     * solution. Its phase is fixed independently at each end; it matches the
     * sinusoidal exit values when length is an integer number of periods.
     * No such length constraint is checked by this class.
     */
    template <typename scalar>
    struct Undulator {
        undulator_parameters<scalar> uparams;  ///< Parameters of the undulator.
        scalar distance_to_entry;              ///< Laboratory z coordinate of entry, internal length.
        scalar k_u;                            ///< Wavenumber 2*pi/lambda_u, inverse internal length.

        /**
         * @brief Constructor to initialize undulator parameters and calculate k_u.
         *
         * @param p Host-prepared internal-unit undulator parameters.
         * @param dte Laboratory z coordinate of entry in internal length units.
         * @pre p.lambda_u is finite and positive; this class performs no validation.
         */
        KOKKOS_FUNCTION Undulator(const undulator_parameters<scalar>& p, scalar dte)
            : uparams(p)
            , distance_to_entry(dte)
            , k_u(2 * M_PI / p.lambda_u) {}

        /**
         * @brief Evaluate the laboratory (E,B) pair without allocations.
         *
         * Inside the sinusoidal section, with @f$\zeta=z-z_{\rm entry}@f$,
         * @f[
         * E_x=E_y=E_z=B_x=0,\qquad
         * B_y=B_0\cosh(k_u y)\sin(k_u\zeta),\qquad
         * B_z=B_0\sinh(k_u y)\cos(k_u\zeta).
         * @f]
         * Before entry use @f$\delta=z-z_{\rm entry}<0@f$; at/after exit use
         * @f$\delta=z-z_{\rm entry}-L\ge0@f$. Both fringes are implemented as
         * @f[
         * f=\exp[-(k_u\delta)^2/2],\qquad
         * B_y=B_0\cosh(k_u y)k_u\delta f,\qquad
         * B_z=B_0\sinh(k_u y)f.
         * @f]
         * @param position_in_lab_frame Laboratory (x,y,z) in internal length units.
         * @return Pair first=E (zero), second=B, in internal field units.
         * No derivative is returned.
         * @note The current strict entrance comparisons leave the field zero at
         * exactly z=distance_to_entry. At nonzero y this differs from the limiting
         * longitudinal field of the adjacent branches. Large |k_u*y| can overflow
         * the hyperbolic functions; the intended use is near the undulator axis.
         */
        KOKKOS_INLINE_FUNCTION Kokkos::pair<ippl::Vector<scalar, 3>, ippl::Vector<scalar, 3>>
        operator()(const ippl::Vector<scalar, 3>& position_in_lab_frame) const noexcept {
            using Kokkos::cos;
            using Kokkos::cosh;
            using Kokkos::exp;
            using Kokkos::sin;
            using Kokkos::sinh;

            Kokkos::pair<ippl::Vector<scalar, 3>, ippl::Vector<scalar, 3>>
                ret;             // First is laboratory E; second is laboratory B.
            ret.first  = scalar(0);  // No laboratory electric field.
            ret.second = scalar(0);  // Magnetic field defaults to zero.

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
            // If the position is past the undulator exit, ramp the field down with
            // the same Gaussian-damped linear profile as the entrance fringe (mirror
            // of MITHRA's staticUndulator exit branch, beam.cc). Without this the
            // field terminates abruptly at the end of the undulator, leaving each
            // electron with its residual transverse wiggle momentum (~K). That
            // uncompensated transverse kick ejects the beam from the domain.
            else if (position_in_lab_frame[2] >= distance_to_entry + uparams.length) {
                scalar z_past_exit =
                    position_in_lab_frame[2] - (distance_to_entry + uparams.length);
                assert(z_past_exit >= 0);  // Ensure we are in the correct region.
                scalar scal = exp(-((k_u * z_past_exit) * (k_u * z_past_exit)
                                    * 0.5));  // Gaussian decay factor.

                ret.second[0] = 0;  // No x-component.
                ret.second[1] = uparams.B_magnitude * cosh(k_u * position_in_lab_frame[1])
                                * z_past_exit * k_u * scal;  // y-component.
                ret.second[2] = uparams.B_magnitude * sinh(k_u * position_in_lab_frame[1])
                                * scal;  // z-component.
            }
            return ret;
        }
    };
}  // namespace ippl
#endif  // UNDULATOR_H
