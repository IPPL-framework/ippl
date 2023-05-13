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
namespace ippl {
    template <typename Field, unsigned Dim>
    void BConds<Field, Dim>::write(std::ostream& os) const {
        os << "BConds: (" << std::endl;
        const_iterator it = bc_m.begin();
        for (; it != bc_m.end() - 1; ++it) {
            (*it)->write(os);
            os << "," << std::endl;
        }
        (*it)->write(os);
        os << std::endl << ")";
    }

    template <typename Field, unsigned Dim>
    void BConds<Field, Dim>::findBCNeighbors(Field& field) {
        for (auto& bc : bc_m) {
            bc->findBCNeighbors(field);
        }
        Kokkos::fence();
        Ippl::Comm->barrier();
    }

    template <typename Field, unsigned Dim>
    void BConds<Field, Dim>::apply(Field& field) {
        for (auto& bc : bc_m) {
            bc->apply(field);
        }
        Kokkos::fence();
        Ippl::Comm->barrier();
    }

    template <typename Field, unsigned Dim>
    bool BConds<Field, Dim>::changesPhysicalCells() const {
        for (const auto& bc : bc_m) {
            if (bc->changesPhysicalCells()) {
                return true;
            }
        }
        return false;
    }
}  // namespace ippl
