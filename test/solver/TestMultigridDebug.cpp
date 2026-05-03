#include <Kokkos_Core.hpp>
#include "Ippl.h"

#include <iostream>

#include "LinearSolvers/Multigrid.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        constexpr unsigned int dim = 2;

        using Mesh_t      = ippl::UniformCartesian<double, dim>;
        using Centering_t = Mesh_t::DefaultCentering;
        using field_type  = ippl::Field<double, dim, Mesh_t, Centering_t>;
        using layout_type = ippl::FieldLayout<dim>;
        using bc_type     = ippl::BConds<field_type, dim>;

        const int pt = 5;

        ippl::Index I(pt);
        ippl::Index J(pt);
        ippl::NDIndex<dim> domain(I, J);

        std::array<bool, dim> isParallel;
        isParallel.fill(true);

        layout_type layout(MPI_COMM_WORLD, domain, isParallel);

        ippl::Vector<double, dim> hx;
        hx[0] = 1.0 / static_cast<double>(pt);
        hx[1] = 1.0 / static_cast<double>(pt);

        ippl::Vector<double, dim> origin;
        origin[0] = 0.0;
        origin[1] = 0.0;

        Mesh_t mesh(domain, hx, origin);

        bc_type bcs;
        for (unsigned int face = 0; face < 2 * dim; ++face) {
            bcs[face] = std::make_shared<ippl::PeriodicFace<field_type>>(face);
        }

        field_type rhs(mesh, layout);
        rhs.setFieldBC(bcs);

        rhs = 0.0;

        const auto lDom  = layout.getLocalNDIndex();
        const int nghost = rhs.getNghost();

        auto rhs_view = rhs.getView();

        Kokkos::parallel_for(
            "set rhs", rhs.getFieldRangePolicy(), KOKKOS_LAMBDA(const int i, const int j) {
                const int ig = i + lDom[0].first() - nghost;
                const int jg = j + lDom[1].first() - nghost;

                rhs_view(i, j) = 10.0 * ig + jg;
            });

        rhs.fillHalo();

        if (ippl::Comm->rank() == 0) {
            std::cout << "\nInitial RHS field:\n";
        }

        rhs.write(std::cout);

        auto op = [](field_type& u) {
            return -laplace(u);
        };

        ippl::multigrid_preconditioner<field_type, decltype(op)> mg(std::move(op), 1, 1, 0.8);

        mg.setDebugPrint(true);
        mg.init_fields(rhs);

        field_type result = mg(rhs);

        if (ippl::Comm->rank() == 0) {
            std::cout << "\nFinal multigrid result:\n";
        }

        result.write(std::cout);
    }
    ippl::finalize();

    return 0;
}
