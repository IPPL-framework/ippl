/** @file LorentzTransform.h
 * @brief Position and electromagnetic-field boosts in FEL's internal c=1 units.
 * @ingroup fel_physics
 */
#ifndef LORENTZ_TRANSFORM_H
#define LORENTZ_TRANSFORM_H
#include <Kokkos_Core.hpp>

#include "Types/Vector.h"
namespace ippl {
    /**
     * @brief Value-type boost from the laboratory frame to a moving frame.
     * @ingroup fel_physics
     *
     * Unprimed quantities are laboratory-frame values; primed quantities belong
     * to a frame moving at @f$+\beta c@f$ along the boost axis. Time and lengths
     * use the internal scales in units.h, with @f$c=1@f$. Electric and magnetic
     * fields must use the compatible internal normalization, not raw SI values.
     * Methods are callable on host and device and allocate no dynamic storage.
     *
     * @warning Position transformations use the axis template argument, but the
     * cross-product velocity in transform_EB() is currently fixed to z. Therefore
     * the complete field-transform API is supported only for axis=2, as used by
     * FreeElectronLaserManager. No run-time axis or gamma validation is performed.
     *
     * @tparam T The scalar type used for computations (e.g., float, double).
     * @tparam axis Cartesian boost axis, 0=x, 1=y, 2=z; use 2 for field transforms.
     */
    template <typename T, unsigned axis = 2>
    struct UniaxialLorentzframe {
        /// Speed of light, set to 1 for natural units.
        constexpr static T c = 1.0;
        /// Alias for the scalar type used in the struct.
        using scalar = T;
        /// Alias for a 3-dimensional vector of the scalar type.
        using Vector3 = ippl::Vector<T, 3>;

        /// Signed dimensionless frame velocity beta=v/c.
        scalar beta_m;
        /// Signed product gamma*beta for the frame, not a particle momentum.
        scalar gammaBeta_m;
        /// Dimensionless frame Lorentz factor gamma=1/sqrt(1-beta^2).
        scalar gamma_m;

        /**
         * @brief Construct a boost with nonnegative velocity from its Lorentz factor.
         *
         * Uses @f$\beta=\sqrt{1-\gamma^{-2}}@f$. Use negative() to select the
         * opposite direction. A factor of one gives the identity boost.
         *
         * @param gamma Finite, dimensionless Lorentz factor, at least one.
         * @return A UniaxialLorentzframe object with computed beta and gamma*beta.
         * @pre gamma satisfies the physical range; this routine does not check it.
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
         * @brief Reverse the frame velocity while preserving gamma.
         *
         * Negates beta and gamma*beta, producing the inverse boost even if the
         * original velocity is already negative.
         *
         * @return A frame object with opposite signed velocity and unchanged gamma.
         */
        KOKKOS_INLINE_FUNCTION UniaxialLorentzframe<T, axis> negative() const noexcept {
            UniaxialLorentzframe ret;
            ret.beta_m      = -beta_m;
            ret.gammaBeta_m = -gammaBeta_m;
            ret.gamma_m     = gamma_m;
            return ret;
        }

        /// Default construction leaves the frame scalars uninitialized; assign them before use.
        KOKKOS_INLINE_FUNCTION UniaxialLorentzframe() = default;

        /**
         * @brief Construct a UniaxialLorentzframe from a gamma*beta value.
         *
         * Computes @f$\gamma=\sqrt{1+(\gamma\beta)^2}@f$ and preserves the
         * supplied sign in beta. This constructor performs no range checking.
         *
         * @param gammaBeta Finite dimensionless signed product gamma*beta.
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
         * Replaces the axial coordinate with
         * @f$x_a=\gamma(x'_a+\beta t')@f$ and leaves transverse coordinates
         * unchanged. The corresponding laboratory time is not returned. This
         * is sufficient to sample the static laboratory undulator at a particle's
         * moving-frame position and time; it is not a full four-vector interface.
         *
         * @param[in,out] arg Primed position on entry, laboratory position on return,
         * in internal length units.
         * @param time Primed time in internal time units, with c=1.
         */
        KOKKOS_INLINE_FUNCTION void primedToUnprimed(Vector3& arg, scalar time) const noexcept {
            arg[axis] = gamma_m * (arg[axis] + beta_m * time);
        }

        /**
         * @brief Transform electric and magnetic fields from the unprimed to the primed frame.
         *
         * For the supported z boost, with @f${\bf b}=\beta\hat{\bf z}@f$,
         * @f[
         * {\bf E}'=\gamma({\bf E}+{\bf b}\times{\bf B})
         * -(\gamma-1)\hat{\bf z}E_z,\qquad
         * {\bf B}'=\gamma({\bf B}-{\bf b}\times{\bf E})
         * -(\gamma-1)\hat{\bf z}B_z.
         * @f]
         * The parallel components remain invariant. All fields use internal
         * c=1 normalization; SI E and B cannot be passed directly.
         *
         * @param unprimedEB Laboratory fields, first=E and second=B.
         * @return Moving-frame fields, first=E' and second=B'.
         * @pre axis=2; see the class limitation for other template axes.
         */
        KOKKOS_INLINE_FUNCTION Kokkos::pair<ippl::Vector<T, 3>, ippl::Vector<T, 3>> transform_EB(
            const Kokkos::pair<ippl::Vector<T, 3>, ippl::Vector<T, 3>>& unprimedEB) const noexcept {
            Kokkos::pair<ippl::Vector<T, 3>, ippl::Vector<T, 3>> ret;
            ippl::Vector<scalar, 3> betavec{0, 0, beta_m};
            ippl::Vector<scalar, 3> vnorm{axis == 0, axis == 1, axis == 2};
            ret.first =
                ippl::Vector<T, 3>(unprimedEB.first + cross(betavec, unprimedEB.second)) * gamma_m
                - (vnorm * (gamma_m - 1) * (unprimedEB.first.dot(vnorm)));
            ret.second =
                ippl::Vector<T, 3>(unprimedEB.second - cross(betavec, unprimedEB.first)) * gamma_m
                - (vnorm * (gamma_m - 1) * (unprimedEB.second.dot(vnorm)));
            // NOTE: the two assignments above are already the complete boost: the
            // -vnorm*(gamma-1)*(field.vnorm) term leaves the boost-axis (parallel)
            // component invariant (gamma*B_z - (gamma-1)*B_z = B_z). The original
            // code additionally subtracted (gamma-1)*field[axis] here, which
            // double-corrects the parallel component to (2-gamma)*field[axis] --
            // a large, wrong-sign spurious longitudinal field that blows the beam
            // up once the undulator (which has B_z != 0 off-axis) turns on. That
            // erroneous correction is removed.
            return ret;
        }

        /**
         * @brief Transform electric and magnetic fields from the primed to the unprimed frame.
         *
         * Evaluates transform_EB() with the frame velocity reversed. The same
         * c=1 field normalization and z-axis limitation apply.
         *
         * @param primedEB Moving-frame fields, first=E' and second=B'.
         * @return Laboratory fields, first=E and second=B.
         * @pre axis=2 and initialized frame parameters.
         */
        KOKKOS_INLINE_FUNCTION Kokkos::pair<ippl::Vector<T, 3>, ippl::Vector<T, 3>>
        inverse_transform_EB(
            const Kokkos::pair<ippl::Vector<T, 3>, ippl::Vector<T, 3>>& primedEB) const noexcept {
            return negative().transform_EB(primedEB);
        }
    };

}  // namespace ippl
#endif
