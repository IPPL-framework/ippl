//   This file contains the abstract base class for
//   field boundary conditions and other child classes
//   which represent specific BCs. At the moment the
//   following field BCs are supported
//
//   1. Periodic BC
//   2. Zero BC
//   3. Specifying a constant BC
//   4. No BC (default option)
//   5. Constant extrapolation BC
//   Only cell-centered field BCs are implemented
//   at the moment.
// Copyright (c) 2021, Sriramkrishnan Muralikrishnan,
// Paul Scherrer Institut, Villigen PSI, Switzerland
// Matthias Frey, University of St Andrews,
// St Andrews, Scotland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//
#ifndef IPPL_FIELD_BC_TYPES_H
#define IPPL_FIELD_BC_TYPES_H

#include "Types/IpplTypes.h"
#include "Types/ViewTypes.h"

#include "Communicate/Archive.h"
#include "FieldLayout/FieldLayout.h"
#include "Index/NDIndex.h"
#include "Meshes/UniformCartesian.h"

namespace ippl {
    template <typename T, unsigned Dim, class Mesh, class Centering>
    class Field;

    /*
     * Enum type to identify different kinds of
     * field boundary conditions. Since ZeroFace is
     * a special case of ConstantFace, both will match
     * a bitwise AND with CONSTANT_FACE
     * (to avoid conflict with particle BC enum, add _FACE)
     */
    enum FieldBC {
        PERIODIC_FACE    = 0b0000,
        CONSTANT_FACE    = 0b0001,
        ZERO_FACE        = 0b0011,
        EXTRAPOLATE_FACE = 0b0100,
        NO_FACE          = 0b1000,
    };

    namespace detail {
        template <typename T, unsigned Dim, class Mesh, class Centering>
        class BCondBase;

        template <typename T, unsigned Dim, class Mesh, class Centering>
        std::ostream& operator<<(std::ostream&, const BCondBase<T, Dim, Mesh, Centering>&);

        template <typename T, unsigned Dim, class Mesh, class Centering>
        class BCondBase {
        public:
            using Field_t  = Field<T, Dim, Mesh, Centering>;
            using Layout_t = FieldLayout<Dim>;

            // Constructor takes:
            // face: the face to apply the boundary condition on.
            // i : what component of T to apply the boundary condition to.
            // The components default to setting all components.
            BCondBase(unsigned int face);

            virtual ~BCondBase() = default;

            virtual FieldBC getBCType() const { return NO_FACE; }

            virtual void findBCNeighbors(Field<T, Dim, Mesh, Centering>& field) = 0;
            virtual void apply(Field<T, Dim, Mesh, Centering>& field)           = 0;
            virtual void write(std::ostream&) const                             = 0;

            // Return face on which BC applies
            unsigned int getFace() const { return face_m; }

            // Returns whether or not this BC changes physical cells.
            bool changesPhysicalCells() const { return changePhysical_m; }

        protected:
            // What face to apply the boundary condition to.
            unsigned int face_m;

            // True if this boundary condition changes physical cells.
            bool changePhysical_m;
        };

    }  // namespace detail

    template <typename T, unsigned Dim, class Mesh, class Centering>
    class ExtrapolateFace : public detail::BCondBase<T, Dim, Mesh, Centering> {
    public:
        // Constructor takes zero, one, or two int's specifying components of
        // multicomponent types like Vector this BC applies to.
        // Zero int's specified means apply to all components; one means apply to
        // component (i), and two means apply to component (i,j),
        using base_type = detail::BCondBase<T, Dim, Mesh, Centering>;
        using Field_t   = typename detail::BCondBase<T, Dim, Mesh, Centering>::Field_t;
        using Layout_t  = typename detail::BCondBase<T, Dim, Mesh, Centering>::Layout_t;

        ExtrapolateFace(unsigned face, T offset, T slope)
            : base_type(face)
            , offset_m(offset)
            , slope_m(slope) {}

        virtual ~ExtrapolateFace() = default;

        virtual FieldBC getBCType() const { return EXTRAPOLATE_FACE; }

        virtual void findBCNeighbors(Field_t& /*field*/) {}
        virtual void apply(Field_t& field);

        virtual void write(std::ostream& out) const;

        const T& getOffset() const { return offset_m; }
        const T& getSlope() const { return slope_m; }

    protected:
        T offset_m;
        T slope_m;
    };

    template <typename T, unsigned Dim, class Mesh, class Centering>
    class NoBcFace : public detail::BCondBase<T, Dim, Mesh, Centering> {
    public:
        using Field_t = typename detail::BCondBase<T, Dim, Mesh, Centering>::Field_t;
        NoBcFace(int face)
            : detail::BCondBase<T, Dim, Mesh, Centering>(face) {}

        virtual void findBCNeighbors(Field_t& /*field*/) {}
        virtual void apply(Field_t& /*field*/) {}

        virtual void write(std::ostream& out) const;
    };

    template <typename T, unsigned Dim, class Mesh, class Centering>
    class ConstantFace : public ExtrapolateFace<T, Dim, Mesh, Centering> {
    public:
        ConstantFace(unsigned int face, T constant)
            : ExtrapolateFace<T, Dim, Mesh, Centering>(face, constant, 0) {}

        virtual FieldBC getBCType() const { return CONSTANT_FACE; }

        virtual void write(std::ostream& out) const;
    };

    template <typename T, unsigned Dim, class Mesh, class Centering>
    class ZeroFace : public ConstantFace<T, Dim, Mesh, Centering> {
    public:
        ZeroFace(unsigned face)
            : ConstantFace<T, Dim, Mesh, Centering>(face, 0.0) {}

        virtual FieldBC getBCType() const { return ZERO_FACE; }

        virtual void write(std::ostream& out) const;
    };

    template <typename T, unsigned Dim, class Mesh, class Centering>
    class PeriodicFace : public detail::BCondBase<T, Dim, Mesh, Centering> {
    public:
        using face_neighbor_type = std::array<std::vector<int>, 2 * Dim>;
        using Field_t            = typename detail::BCondBase<T, Dim, Mesh, Centering>::Field_t;
        using Layout_t           = typename detail::BCondBase<T, Dim, Mesh, Centering>::Layout_t;

        PeriodicFace(unsigned face)
            : detail::BCondBase<T, Dim, Mesh, Centering>(face) {}

        virtual FieldBC getBCType() const { return PERIODIC_FACE; }

        virtual void findBCNeighbors(Field_t& field);
        virtual void apply(Field_t& field);

        virtual void write(std::ostream& out) const;

    private:
        face_neighbor_type faceNeighbors_m;
        detail::FieldBufferData<T> haloData_m;
    };
}  // namespace ippl

#include "Field/BcTypes.hpp"

#endif
