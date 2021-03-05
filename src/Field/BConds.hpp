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
        template<typename T, unsigned Dim, class Mesh, class Cell>
        void
        BConds<T, Dim, Mesh, Cell>::write(std::ostream& os) const
        {
            os << "BConds: (" << std::endl;
            const_iterator it = bc_m.begin();
            for ( ; it != bc_m.end() - 1; ++it) {
                (*it)->write(os);
                os << "," << std::endl;
            }
            (*it)->write(os);
            os << std::endl << ")";
        }

        template<typename T, unsigned Dim, class Mesh, class Cell>
        void
        BConds<T, Dim, Mesh, Cell>::findBCNeighbors(Field<T, Dim, Mesh, Cell>& field)
        {
            for (iterator it = bc_m.begin(); it != bc_m.end(); ++it) {
                (*it)->findBCNeighbors(field);
            }
            Kokkos::fence();
            Ippl::Comm->barrier();
        }

        template<typename T, unsigned Dim, class Mesh, class Cell>
        void
        BConds<T, Dim, Mesh, Cell>::apply(Field<T, Dim, Mesh, Cell>& field)
        {
            for (iterator it = bc_m.begin(); it != bc_m.end(); ++it) {
                (*it)->apply(field);
            }
            Kokkos::fence();
            Ippl::Comm->barrier();
        }

        template<typename T, unsigned Dim, class Mesh, class Cell>
        bool
        BConds<T, Dim, Mesh, Cell>::changesPhysicalCells() const
        {
            bool doesChange = false;
            for (const_iterator it = bc_m.begin(); it != bc_m.end(); ++it) {
                doesChange |= (*it)->changesPhysicalCells();
            }
            return doesChange;
        }
}
