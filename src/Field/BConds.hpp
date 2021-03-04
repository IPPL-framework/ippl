
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
