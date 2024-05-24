#ifndef LORENTZ_TRANSFORM_H
#define LORENTZ_TRANSFORM_H
#include <Kokkos_Core.hpp>

#include "Types/Matrix.h"
#include "Types/Vector.h"
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
    /**
     * @brief Represents a Lorentz frame with associated transformations and operations.
     * 
     * This template struct models a Lorentz frame used in special relativity to handle
     * transformations between reference frames moving at a constant velocity relative to each other.
     * The LorentzFrame supports various operations such as converting between frames and transforming
     * vectors and electromagnetic fields.
     * 
     * @tparam T The type for scalar values, typically a floating-point type.
     */
    template <typename T>
    struct LorentzFrame {
        /// Speed of light in natural units (c = 1)
        constexpr static T c = 1.0;
    
        /// Alias for the scalar type.
        using scalar = T;
    
        /// Alias for a 3D vector of type T.
        using Vector3 = ippl::Vector<T, 3>;
    
        /// Velocity vector (beta) in the frame, normalized by the speed of light.
        ippl::Vector<T, 3> beta_m;
    
        /// Product of gamma and beta vectors.
        ippl::Vector<T, 3> gammaBeta_m;
    
        /// Lorentz factor gamma.
        T gamma_m;
    
        /**
         * @brief Constructs a LorentzFrame from a given gammaBeta vector.
         * 
         * This constructor initializes the Lorentz frame using a vector that represents
         * the product of the Lorentz factor gamma and the velocity vector beta.
         * The velocity vector beta is then computed by normalizing this gammaBeta vector,
         * and the Lorentz factor gamma is computed from the magnitude of gammaBeta.
         * 
         * @param gammaBeta The gammaBeta vector.
         */
        KOKKOS_INLINE_FUNCTION LorentzFrame(const ippl::Vector<T, 3>& gammaBeta) {
            beta_m      = gammaBeta / sqrt(1 + gammaBeta.dot(gammaBeta));
            gamma_m     = sqrt(1 + gammaBeta.dot(gammaBeta));
            gammaBeta_m = gammaBeta;
        }
        /**
         * @brief Constructs a uniaxial Lorentz frame given a gamma value and an axis.
         * 
         * This static member function creates a Lorentz frame that has motion along one of the principal
         * axes (x, y, or z) based on the provided gamma value. The beta vector is calculated accordingly
         * and aligned with the specified axis.
         * 
         * @tparam axis The axis along which the frame is moving ('x', 'y', or 'z').
         * @param gamma The Lorentz factor gamma, which must be greater than or equal to 1.
         * @return LorentzFrame A Lorentz frame with motion along the specified axis.
         */
        template <char axis>
        static LorentzFrame uniaxialGamma(T gamma) {
            static_assert(axis == 'x' || axis == 'y' || axis == 'z', "Only xyz axis suproted");
            assert(gamma >= 1.0 && "Gamma must be >= 1");
            using Kokkos::sqrt;

            T beta = gamma == 1 ? T(0) : sqrt(gamma * gamma - 1) / gamma;
            Vector3 arg{0, 0, 0};
            arg[axis - 'x'] = gamma * beta;
            return LorentzFrame<T>(arg);
        }

        /**
         * @brief Constructs the Lorentz transformation matrix for converting from unprimed to primed frame.
         * 
         * This function computes the Lorentz transformation matrix that converts vectors
         * from the unprimed reference frame to the primed reference frame. The transformation
         * accounts for the relative velocity between the frames using the stored beta and gamma values.
         * 
         * If the magnitude of the beta vector is very small (less than 1e-10), the function returns
         * an identity matrix, indicating no significant transformation.
         * 
         * @return matrix<T, 4, 4> The transformation matrix.
         */
        KOKKOS_INLINE_FUNCTION matrix<T, 4, 4> unprimedToPrimed() const noexcept {
            T betaMagsq = beta_m.dot(beta_m);
            using Kokkos::abs;
            if (abs(betaMagsq) < 1e-10) {
                return matrix<T, 4, 4>(T(1));
            }
            ippl::Vector<T, 3> betaSquared = beta_m * beta_m;

            matrix<T, 4, 4> ret;

            ret.data[0] = ippl::Vector<T, 4>{gamma_m, -gammaBeta_m[0], -gammaBeta_m[1], -gammaBeta_m[2]};
            ret.data[1] = ippl::Vector<T, 4>{-gammaBeta_m[0], 1 + (gamma_m - 1) * betaSquared[0] / betaMagsq,
                                   (gamma_m - 1) * beta_m[0] * beta_m[1] / betaMagsq,
                                   (gamma_m - 1) * beta_m[0] * beta_m[2] / betaMagsq};
            ret.data[2] = ippl::Vector<T, 4>{-gammaBeta_m[1],
                                             (gamma_m - 1) * beta_m[0] * beta_m[1] / betaMagsq,
                                             1 + (gamma_m - 1) * betaSquared[1] / betaMagsq,
                                             (gamma_m - 1) * beta_m[1] * beta_m[2] / betaMagsq};
            ret.data[3] = ippl::Vector<T, 4>{-gammaBeta_m[2],
                                             (gamma_m - 1) * beta_m[0] * beta_m[2] / betaMagsq,
                                             (gamma_m - 1) * beta_m[1] * beta_m[2] / betaMagsq,
                                             1 + (gamma_m - 1) * betaSquared[2] / betaMagsq};
            return ret;
        }
        /**
         * @brief Constructs the Lorentz transformation matrix for converting from primed to unprimed frame.
         * 
         * This function computes the inverse of the Lorentz transformation matrix obtained
         * from unprimedToPrimed(), allowing conversion of vectors from the primed reference frame
         * back to the unprimed reference frame. It utilizes the matrix inversion function to achieve this.
         * 
         * @return matrix<T, 4, 4> The inverse transformation matrix. This could also be done by taking the negative velocity
         */
        KOKKOS_INLINE_FUNCTION matrix<T, 4, 4> primedToUnprimed() const noexcept {
            return unprimedToPrimed().inverse();
        }

        KOKKOS_INLINE_FUNCTION Vector3 transformV(const Vector3& unprimedV) const noexcept {
            T factor    = T(1.0) / (1.0 - unprimedV.dot(beta_m));
            Vector3 ret = unprimedV * scalar(1.0 / gamma_m);
            ret -= beta_m;
            ret += beta_m * (unprimedV.dot(beta_m) * (gamma_m / (gamma_m + 1)));
            return ret * factor;
        }

        KOKKOS_INLINE_FUNCTION Vector3 transformGammabeta(const Vector3& gammabeta) const noexcept {
            using Kokkos::sqrt;
            T gamma      = sqrt(T(1) + gammabeta.dot(gammabeta));
            Vector3 beta = gammabeta;
            beta /= gamma;
            Vector3 betatrf = transformV(beta);
            betatrf *= sqrt(1 - betatrf.dot(betatrf));
            return betatrf;
        }
        /**
         * @brief Transforms electric and magnetic fields from the unprimed to the primed frame.
         * 
         * This function applies the Lorentz transformation to a pair of 3D vectors representing
         * the electric field (E) and magnetic field (B) in the unprimed reference frame,
         * converting them to the corresponding fields in the primed reference frame. The transformation
         * accounts for relativistic effects on the fields due to the relative motion of the frames.
         * 
         * @param unprimedEB A pair of vectors representing the electric and magnetic fields in the unprimed frame.
         * @return Kokkos::pair<ippl::Vector<T, 3>, ippl::Vector<T, 3>> A pair of transformed vectors representing the electric and magnetic fields in the primed frame.
         */
        KOKKOS_INLINE_FUNCTION Kokkos::pair<ippl::Vector<T, 3>, ippl::Vector<T, 3>> transform_EB(
            const Kokkos::pair<ippl::Vector<T, 3>, ippl::Vector<T, 3>>& unprimedEB) const noexcept {
            using Kokkos::sqrt;
            Kokkos::pair<ippl::Vector<T, 3>, ippl::Vector<T, 3>> ret;
            Vector3 vnorm = beta_m * (T(1.0) / sqrt(beta_m.dot(beta_m)));

            ret.first =
                (ippl::Vector<T, 3>(unprimedEB.first + beta_m.cross(unprimedEB.second)) * gamma_m)
                - (vnorm * (gamma_m - 1) * (unprimedEB.first.dot(vnorm)));
            ret.second =
                (ippl::Vector<T, 3>(unprimedEB.second - beta_m.cross(unprimedEB.first)) * gamma_m)
                - (vnorm * (gamma_m - 1) * (unprimedEB.second.dot(vnorm)));
            return ret;
        }
        KOKKOS_INLINE_FUNCTION Kokkos::pair<ippl::Vector<T, 3>, ippl::Vector<T, 3>>
        inverse_transform_EB(
            const Kokkos::pair<ippl::Vector<T, 3>, ippl::Vector<T, 3>>& primedEB) const noexcept {
            ippl::Vector<T, 3> mgb(gammaBeta_m * scalar(-1.0));
            return LorentzFrame<T>(mgb).transform_EB(primedEB);
        }
    };

}  // namespace ippl
#endif
