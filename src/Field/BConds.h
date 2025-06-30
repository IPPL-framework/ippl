//   Class BConds
//   This is the container class for the field BCs.
//   It calls the findBCNeighbors and apply in the
//   respective BC classes to apply field BCs
//
#ifndef IPPL_FIELD_BC_H
#define IPPL_FIELD_BC_H

#include <array>
#include <iostream>
#include <memory>

#include "Field/BcTypes.h"

namespace ippl {
    /*!
     * A container for boundary conditions
     * @tparam Field the type of the field to which the boundary conditions will be applied
     * @tparam Dim the rank of the field (redundant parameter required to avoid a circular
     * dependency loop between Field and BConds)
     */
    template <typename Field, unsigned Dim>
    class BConds {
    public:
        using bc_type        = detail::BCondBase<Field>;
        using container      = std::array<std::shared_ptr<bc_type>, 2 * Dim>;
        using iterator       = typename container::iterator;
        using const_iterator = typename container::const_iterator;

        BConds()  = default;
        ~BConds() = default;

        void findBCNeighbors(Field& field);
        void apply(Field& field);
        void assignGhostToPhysical(Field& field);

        bool changesPhysicalCells() const;
        virtual void write(std::ostream&) const;

        const std::shared_ptr<bc_type>& operator[](const int& i) const noexcept { return bc_m[i]; }

        std::shared_ptr<bc_type>& operator[](const int& i) noexcept { return bc_m[i]; }

    private:
        container bc_m;
    };

    template <typename Field, unsigned Dim>
    inline std::ostream& operator<<(std::ostream& os, const BConds<Field, Dim>& bc) {
        bc.write(os);
        return os;
    }
}  // namespace ippl

#include "Field/BConds.hpp"

#endif
