// Example program to demonstrate solver design
#include "Ippl.h"

#include <iostream>
#include <string>
#include <typeinfo>

#include "PoissonSolvers/Poisson.h"

constexpr unsigned int dim = 3;
using Mesh_t               = ippl::UniformCartesian<double, dim>;
using Centering_t          = Mesh_t::DefaultCentering;
using field_type           = ippl::Field<double, dim, Mesh_t, Centering_t>;

class TestSolver : public ippl::Poisson<field_type, field_type> {
public:
    void solve() override {
        *rhs_mp = *lhs_mp + *rhs_mp;

        if (params_m.get<int>("output_type") & GRAD) {
            *grad_mp = ippl::grad(*lhs_mp);
        }
    }
};

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        int pt = 4;
        ippl::Index I(pt);
        ippl::NDIndex<dim> owned(I, I, I);

        ippl::e_dim_tag allParallel[dim];  // Specifies SERIAL, PARALLEL dims
        for (unsigned int d = 0; d < dim; d++) {
            allParallel[d] = ippl::SERIAL;
        }

        // all parallel layout, standard domain, normal axis order
        ippl::FieldLayout<dim> layout(owned, allParallel);

        // Unit box
        double dx                        = 1.0 / double(pt);
        ippl::Vector<double, dim> hx     = {dx, dx, dx};
        ippl::Vector<double, dim> origin = {0, 0, 0};
        Mesh_t mesh(owned, hx, origin);

        field_type lhs(mesh, layout), rhs(mesh, layout);

        typedef ippl::Field<ippl::Vector<double, dim>, dim, Mesh_t, Centering_t> vfield_type;
        vfield_type grad(mesh, layout);

        lhs = 1.0;
        rhs = 2.0;

        TestSolver tsolver;

        tsolver.setLhs(lhs);

        tsolver.setRhs(rhs);

        tsolver.setGradient(grad);

        tsolver.solve();

        rhs.write();

        grad.write();
    }
    ippl::finalize();

    return 0;
}
