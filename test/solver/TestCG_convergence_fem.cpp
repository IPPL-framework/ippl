// Tests the conjugate gradient solver for Poisson problems
// by checking the relative error from the exact solution
// Usage:
//      TestCGSolver [size [scaling_type , preconditioner]]
//      ./TestCGSolver 6 j --info 5

#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <cstdlib>
#include <iostream>
#include <string>
#include <typeinfo>

#include "Utility/Inform.h"
#include "Utility/IpplTimings.h"

#include "PoissonSolvers/PoissonCG.h"
#include "PoissonSolvers/FEMPoissonSolver.h"

template <typename T, unsigned Dim>
struct AnalyticSol {
    const T pi = Kokkos::numbers::pi_v<T>;

    KOKKOS_FUNCTION const T operator()(ippl::Vector<T, Dim> x_vec) const {
        T val = 1.0;
        for (unsigned d = 0; d < Dim; d++) {
            //val *= -sin(pi*x_vec[d]);
            val *= sin(sin(pi*x_vec[d]));
        }
        return val;
    }
};


int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        constexpr unsigned int dim = 3;
        using Mesh_t               = ippl::UniformCartesian<double, dim>;
        using Centering_t          = Mesh_t::DefaultCentering;

        Inform m("");
        m << "size, relError, residue, itCount" << endl;

        for (unsigned pt = 1 << 2; pt <= 1 << 8; pt = pt << 1) {
            ippl::Vector <unsigned, dim> I(pt);
            ippl::NDIndex<dim> domain(I);

            std::array<bool, dim> isParallel;  // Specifies SERIAL, PARALLEL dims
            for (unsigned int d = 0; d < dim; d++) {
                isParallel[d] = true;
            }

            // all parallel layout, standard domain, normal axis order
            ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, domain, isParallel);

            // Unit box
            double dx                        = 4.0 / double(pt- 1);
            ippl::Vector<double, dim> hx     = dx;
            ippl::Vector<double, dim> origin = 0.0;
            Mesh_t mesh(domain, hx, origin);

            double pi = Kokkos::numbers::pi_v<double>;

            typedef ippl::Field<double, dim, Mesh_t, Centering_t> field_type;
            field_type rhs(mesh, layout), lhs(mesh, layout), solution(mesh, layout);

            typedef ippl::BConds<field_type, dim> bc_type;

            bc_type bcField;

            for (unsigned int i = 0; i < 2 * dim; ++i) {
                bcField[i] = std::make_shared<ippl::PeriodicFace<field_type>>(i);
                //bcField[i] = std::make_shared<ippl::ZeroFace<field_type>>(i);
            }

            lhs.setFieldBC(bcField);
            rhs.setFieldBC(bcField);

            typename field_type::view_type &viewRHS = rhs.getView(), viewSol = solution.getView();

            const ippl::NDIndex<dim>& lDom = layout.getLocalNDIndex();

            using Kokkos::pow, Kokkos::sin, Kokkos::cos;

            int shift1     = solution.getNghost();
            auto policySol = solution.getFieldRangePolicy();
            Kokkos::parallel_for(
                "Assign solution", policySol, KOKKOS_LAMBDA(const int i, const int j, const int k) {
                    const size_t ig = i + lDom[0].first() - shift1;
                    const size_t jg = j + lDom[1].first() - shift1;
                    const size_t kg = k + lDom[2].first() - shift1;
                    double x        = (ig ) * hx[0];
                    double y        = (jg ) * hx[1];
                    double z        = (kg ) * hx[2];

                    //viewSol(i) = sin(pi*x); 
                    //viewSol(i) = sin(sin(pi*x)); 
                    //viewSol(i, j) = sin(sin(pi*x)) * sin(sin(pi*y)); 
                    viewSol(i, j, k) = sin(sin(pi*x)) * sin(sin(pi*y)) * sin(sin(pi*z));
            });

            const int shift2 = rhs.getNghost();
            auto policyRHS   = rhs.getFieldRangePolicy();

            Kokkos::parallel_for(
                "Assign rhs", policyRHS, KOKKOS_LAMBDA(const int i, const int j, const int k) {
                    const size_t ig = i + lDom[0].first() - shift2;
                    const size_t jg = j + lDom[1].first() - shift2;
                    const size_t kg = k + lDom[2].first() - shift2;
                    double x        = (ig ) * hx[0];
                    double y        = (jg ) * hx[1];
                    double z        = (kg ) * hx[2];

                    viewRHS(i, j, k) =
                        pow(pi, 2)
                        * (cos(sin(pi * z)) * sin(pi * z) * sin(sin(pi * x)) * sin(sin(pi * y))
                           + (cos(sin(pi * y)) * sin(pi * y) * sin(sin(pi * x))
                              + (cos(sin(pi * x)) * sin(pi * x)
                                 + (pow(cos(pi * x), 2) + pow(cos(pi * y), 2) + pow(cos(pi * z), 2))
                                       * sin(sin(pi * x)))
                                    * sin(sin(pi * y)))
                                 * sin(sin(pi * z)));
                });
            /*
            Kokkos::parallel_for(
                "Assign rhs", policyRHS, KOKKOS_LAMBDA(const int i, const int j) {
                    const size_t ig = i + lDom[0].first() - shift2;
                    const size_t jg = j + lDom[1].first() - shift2;
                    double x        = (ig ) * hx[0];
                    double y        = (jg ) * hx[1];

                    viewRHS(i, j) =
                        pow(pi, 2)
                        * (cos(sin(pi * y)) * sin(pi * y) * sin(sin(pi * x))
                              + (cos(sin(pi * x)) * sin(pi * x)
                                 + (pow(cos(pi * x), 2) + pow(cos(pi * y), 2)) * sin(sin(pi * x)))
                                    * sin(sin(pi * y)));
                });
            */
            /*
            Kokkos::parallel_for(
                "Assign rhs", policyRHS, KOKKOS_LAMBDA(const int i) {
                    const size_t ig = i + lDom[0].first() - shift2;
                    double x        = (ig ) * hx[0];

                    viewRHS(i) =
                        pow(pi, 2) * ((cos(sin(pi * x)) * sin(pi * x)) 
                                 + (pow(cos(pi * x), 2) * sin(sin(pi * x))));
                    //viewRHS(i) = pow(pi, 2) * sin(pi*x);
                });
            */
            //std::cout << "lhs before = " << std::endl;
            //lhs.write();

            //std::cout << "rhs before = " << std::endl;
            //rhs.write();

            ippl::FEMPoissonSolver<field_type, field_type> lapsolver(lhs, rhs);

            ippl::ParameterList params;
            params.add("max_iterations", 2000);
            params.add("tolerance", 1e-13);

            lapsolver.mergeParameters(params);

            lhs = 0;
            lapsolver.solve();

            //std::cout << "lhs after = " << std::endl;
            //lhs.write();

            //std::cout << "rhs after = " << std::endl;
            //rhs.write();

            field_type error(mesh, layout);
            // Solver solution - analytical solution
            error           = lhs - solution;
            double relError = norm(error) / norm(solution);

            AnalyticSol<double, dim> analytic;
            double relError_int = lapsolver.getL2Error(analytic);

            // Laplace(solver solution) - rhs
            error          = -laplace(lhs) - rhs;
            double residue = norm(error) / norm(rhs);
            
            residue = lapsolver.getResidue();

            int size    = pt;
            int itCount = lapsolver.getIterationCount();
            m << size << "," << std::setprecision(16) << relError << "," << residue << "," << itCount << ","
              << relError_int << endl;

        }
        IpplTimings::print("timings.dat");
    }
    ippl::finalize();

    return 0;
}
