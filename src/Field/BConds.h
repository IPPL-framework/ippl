//   Class BConds
//   This is the container class for the field BCs.
//   It calls the findBCNeighbors and apply in the
//   respective BC classes to apply field BCs
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
#ifndef IPPL_FIELD_BC_H
#define IPPL_FIELD_BC_H

#include <array>
#include <iostream>
#include <memory>

#include "Field/BcTypes.h"

namespace ippl {
    template <typename T, unsigned Dim, class Mesh, class Centering>
    class Field;

    template <typename T, unsigned Dim, class Mesh, class Centering>
    class BConds;

    template <typename T, unsigned Dim, class Mesh, class Centering>
    std::ostream& operator<<(std::ostream&, const BConds<T, Dim, Mesh, Centering>&);

    template <typename T, unsigned Dim, class Mesh, class Centering>
    class BConds {
    public:
        using bc_type        = detail::BCondBase<T, Dim, Mesh, Centering>;
        using container      = std::array<std::shared_ptr<bc_type>, 2 * Dim>;
        using iterator       = typename container::iterator;
        using const_iterator = typename container::const_iterator;

        BConds()  = default;
        ~BConds() = default;

        void findBCNeighbors(Field<T, Dim, Mesh, Centering>& field);
        void apply(Field<T, Dim, Mesh, Centering>& field);

        bool changesPhysicalCells() const;
        virtual void write(std::ostream&) const;

        const std::shared_ptr<bc_type>& operator[](const int& i) const noexcept { return bc_m[i]; }

        std::shared_ptr<bc_type>& operator[](const int& i) noexcept { return bc_m[i]; }

        const_iterator begin() const {
            return bc_m.begin();
        }

        const_iterator end() const {
            return bc_m.end();
        }

    private:
        container bc_m;
    };

    template <typename T, unsigned Dim, class Mesh, class Centering>
    inline std::ostream& operator<<(std::ostream& os, const BConds<T, Dim, Mesh, Centering>& bc) {
        bc.write(os);
        return os;
    }
}  // namespace ippl

#include "Field/BConds.hpp"

#endif
