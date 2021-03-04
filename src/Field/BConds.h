#ifndef IPPL_FIELD_BC_H
#define IPPL_FIELD_BC_H

#include "Field/BcTypes.h"

#include <array>
#include <iostream>
#include <memory>

namespace ippl {
    template<typename T, unsigned Dim, class Mesh, class Cell> class Field;

    template<typename T, unsigned Dim, class Mesh, class Cell> class BConds;

    template<typename T, unsigned Dim, class Mesh, class Cell>
    std::ostream& operator<<(std::ostream&, const BConds<T, Dim, Mesh, Cell>&);


    template<typename T,
             unsigned Dim,
             class Mesh = UniformCartesian<double, Dim>,
             class Cell = typename Mesh::DefaultCentering>
    class BConds
    {
    public:
        using bc_type = detail::BCondBase<T, Dim, Mesh, Cell>;
        using container = std::array<std::shared_ptr<bc_type>, 2 * Dim>;
        using iterator = typename container::iterator;
        using const_iterator = typename container::const_iterator;

        void findBCNeighbors(Field<T, Dim, Mesh, Cell>& field);
        void apply(Field<T, Dim, Mesh, Cell>& field);

        bool changesPhysicalCells() const;
        virtual void write(std::ostream&) const;

        const std::shared_ptr<bc_type>& operator[](const int& i) const noexcept {
            return bc_m[i];
        }

        std::shared_ptr<bc_type>& operator[](const int& i) noexcept {
            return bc_m[i];
        }

    private:
        container bc_m;
    };


    template<typename T, unsigned Dim, class Mesh, class Cell>
    inline std::ostream&
    operator<<(std::ostream& os, const BConds<T, Dim, Mesh, Cell>& bc)
    {
        bc.write(os);
        return os;
    }
}


#include "Field/BConds.hpp"

#endif
