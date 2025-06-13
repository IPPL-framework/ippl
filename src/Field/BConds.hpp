//   Class BConds
//   This is the container class for the field BCs.
//   It calls the findBCNeighbors and apply in the
//   respective BC classes to apply field BCs
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
        field.getCommunicator().barrier();
    }

    template <typename Field, unsigned Dim>
    void BConds<Field, Dim>::apply(Field& field) {
        for (auto& bc : bc_m) {
            bc->apply(field);
        }
        Kokkos::fence();
        field.getCommunicator().barrier();
    }

    template <typename Field, unsigned Dim>
    void BConds<Field, Dim>::assignGhostToPhysical(Field& field) {
        for (auto& bc : bc_m) {
            bc->assignGhostToPhysical(field);
        }
        Kokkos::fence();
        field.getCommunicator().barrier();
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
