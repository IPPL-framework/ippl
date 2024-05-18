#ifndef UNDULATOR_H
#define UNDULATOR_H
#include <Kokkos_Core.hpp>

#include <cmath>

#include "Types/Vector.h"

#include "units.h"
namespace ippl {
    /**
     * @brief A template struct representing a uniaxial Lorentz frame.
     *
     * The UniaxialLorentzframe struct represents a Lorentz transformation along a specific axis
     * (default is the z-axis, axis 2). It includes methods for transforming vectors and
     * electromagnetic fields between frames, as well as for computing gamma factors and beta
     * velocities. The struct uses Kokkos for portability and performance.
     *
     * @tparam T The scalar type used for computations (e.g., float, double).
     * @tparam axis The axis along which the Lorentz transformation is applied (default is 2, the
     * z-axis).
     */
    template <typename T, unsigned axis = 2>
    struct UniaxialLorentzframe {
        /// Speed of light, set to 1 for natural units.
        constexpr static T c = 1.0;
        /// Alias for the scalar type used in the struct.
        using scalar = T;
        /// Alias for a 3-dimensional vector of the scalar type.
        using Vector3 = ippl::Vector<T, 3>;

        /// Beta velocity component along the specified axis.
        scalar beta_m;
        /// Product of gamma factor and beta velocity.
        scalar gammaBeta_m;
        /// Gamma factor for the Lorentz transformation.
        scalar gamma_m;

        /**
         * @brief Create a UniaxialLorentzframe from a given gamma factor.
         *
         * This static member function constructs a UniaxialLorentzframe object using a given
         * gamma factor. It computes the corresponding beta velocity and gamma*beta product.
         *
         * @param gamma The gamma factor for the Lorentz transformation.
         * @return A UniaxialLorentzframe object with computed beta and gamma*beta.
         */
        KOKKOS_INLINE_FUNCTION static UniaxialLorentzframe from_gamma(const scalar gamma) {
            UniaxialLorentzframe ret;
            ret.gamma_m      = gamma;
            scalar beta      = Kokkos::sqrt(1 - double(1) / (gamma * gamma));
            scalar gammabeta = gamma * beta;
            ret.beta_m       = beta;
            ret.gammaBeta_m  = gammabeta;
            return ret;
        }

        /**
         * @brief Get a UniaxialLorentzframe with negative beta velocity.
         *
         * This member function returns a new UniaxialLorentzframe object with the same gamma
         * factor but with a negative beta velocity, effectively representing the inverse
         * Lorentz transformation.
         *
         * @return A UniaxialLorentzframe object with negative beta velocity.
         */
        KOKKOS_INLINE_FUNCTION UniaxialLorentzframe<T, axis> negative() const noexcept {
            UniaxialLorentzframe ret;
            ret.beta_m      = -beta_m;
            ret.gammaBeta_m = -gammaBeta_m;
            ret.gamma_m     = gamma_m;
            return ret;
        }

        /// Default constructor.
        KOKKOS_INLINE_FUNCTION UniaxialLorentzframe() = default;

        /**
         * @brief Construct a UniaxialLorentzframe from a gamma*beta value.
         *
         * This constructor initializes a UniaxialLorentzframe object using a given gamma*beta
         * value. It computes the corresponding beta velocity and gamma factor.
         *
         * @param gammaBeta The product of the gamma factor and beta velocity.
         */
        KOKKOS_INLINE_FUNCTION UniaxialLorentzframe(const scalar gammaBeta) {
            using Kokkos::sqrt;
            gammaBeta_m = gammaBeta;
            beta_m      = gammaBeta / sqrt(1 + gammaBeta * gammaBeta);
            gamma_m     = sqrt(1 + gammaBeta * gammaBeta);
        }

        /**
         * @brief Transform a spatial vector from the primed frame to the unprimed frame.
         *
         * This member function transforms a given spatial vector from the primed frame to the
         * unprimed frame using the Lorentz transformation along the specified axis and the given
         * time.
         *
         * @param arg The spatial vector to be transformed.
         * @param time The time component for the transformation.
         */
        KOKKOS_INLINE_FUNCTION void primedToUnprimed(Vector3& arg, scalar time) const noexcept {
            arg[axis] = gamma_m * (arg[axis] + beta_m * time);
        }

        /**
         * @brief Transform electric and magnetic fields from the unprimed to the primed frame.
         *
         * This member function transforms a pair of electric and magnetic field vectors from the
         * unprimed frame to the primed frame using the Lorentz transformation along the specified
         * axis.
         *
         * @param unprimedEB A pair of vectors representing the electric and magnetic fields in the
         * unprimed frame.
         * @return A pair of vectors representing the electric and magnetic fields in the primed
         * frame.
         */
        KOKKOS_INLINE_FUNCTION Kokkos::pair<ippl::Vector<T, 3>, ippl::Vector<T, 3>> transform_EB(
            const Kokkos::pair<ippl::Vector<T, 3>, ippl::Vector<T, 3>>& unprimedEB) const noexcept {
            Kokkos::pair<ippl::Vector<T, 3>, ippl::Vector<T, 3>> ret;
            ippl::Vector<scalar, 3> betavec{0, 0, beta_m};
            ippl::Vector<scalar, 3> vnorm{axis == 0, axis == 1, axis == 2};
            ret.first =
                ippl::Vector<T, 3>(unprimedEB.first + betavec.cross(unprimedEB.second)) * gamma_m
                - (vnorm * (gamma_m - 1) * (unprimedEB.first.dot(vnorm)));
            ret.second =
                ippl::Vector<T, 3>(unprimedEB.second - betavec.cross(unprimedEB.first)) * gamma_m
                - (vnorm * (gamma_m - 1) * (unprimedEB.second.dot(vnorm)));
            ret.first[axis] -= (gamma_m - 1) * unprimedEB.first[axis];
            ret.second[axis] -= (gamma_m - 1) * unprimedEB.second[axis];
            return ret;
        }

        /**
         * @brief Transform electric and magnetic fields from the primed to the unprimed frame.
         *
         * This member function transforms a pair of electric and magnetic field vectors from the
         * primed frame to the unprimed frame using the inverse Lorentz transformation along the
         * specified axis.
         *
         * @param primedEB A pair of vectors representing the electric and magnetic fields in the
         * primed frame.
         * @return A pair of vectors representing the electric and magnetic fields in the unprimed
         * frame.
         */
        KOKKOS_INLINE_FUNCTION Kokkos::pair<ippl::Vector<T, 3>, ippl::Vector<T, 3>>
        inverse_transform_EB(
            const Kokkos::pair<ippl::Vector<T, 3>, ippl::Vector<T, 3>>& primedEB) const noexcept {
            return negative().transform_EB(primedEB);
        }
    };
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