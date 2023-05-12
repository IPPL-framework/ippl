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
    template <typename Field>
    class BConds {
        constexpr static unsigned Dim = Field::dim;

    public:
        using bc_type        = detail::BCondBase<Field>;
        using container      = std::array<std::shared_ptr<bc_type>, 2 * Dim>;
        using iterator       = typename container::iterator;
        using const_iterator = typename container::const_iterator;

        BConds()  = default;
        ~BConds() = default;

        void findBCNeighbors(Field& field);
        void apply(Field& field);

        bool changesPhysicalCells() const;
        virtual void write(std::ostream&) const;

        const std::shared_ptr<bc_type>& operator[](const int& i) const noexcept { return bc_m[i]; }

        std::shared_ptr<bc_type>& operator[](const int& i) noexcept { return bc_m[i]; }

    private:
        container bc_m;
    };

    template <typename Field>
    inline std::ostream& operator<<(std::ostream& os, const BConds<Field>& bc) {
        bc.write(os);
        return os;
    }
}  // namespace ippl

#include "Field/BConds.hpp"

#endif
