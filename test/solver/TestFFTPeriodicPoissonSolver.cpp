#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <iostream>
#include <typeinfo>

#include "PoissonSolvers/FFTPeriodicPoissonSolver.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        constexpr unsigned int dim = 3;
        using Mesh_t               = ippl::UniformCartesian<double, dim>;
        using Centering_t          = Mesh_t::DefaultCentering;

        const int npts            = 7;
        std::array<int, npts> pts = {2, 4, 8, 16, 32, 64, 128};

        if (ippl::Comm->size() > 4) {
            if (ippl::Comm->rank() == 0) {
                std::cerr << " Too many MPI ranks please use <= 4 ranks" << std::endl;
            }
        }

        for (int p = 0; p < npts; ++p) {
            int pt = pts[p];
            ippl::Index I(pt);
            ippl::NDIndex<dim> owned(I, I, I);

            std::array<bool, dim> isParallel;
            isParallel.fill(true);

            // all parallel layout, standard domain, normal axis order
            ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

            //[-1, 1] box
            double dx                        = 2.0 / double(pt);
            ippl::Vector<double, dim> hx     = {dx, dx, dx};
            ippl::Vector<double, dim> origin = {-1.0, -1.0, -1.0};
            Mesh_t mesh(owned, hx, origin);

            double pi = Kokkos::numbers::pi_v<double>;

            typedef ippl::Field<double, dim, Mesh_t, Centering_t> Field_t;
            typedef ippl::Vector<double, 3> Vector_t;
            typedef ippl::Field<Vector_t, dim, Mesh_t, Centering_t> VField_t;

            Field_t field;
            field.initialize(mesh, layout);

            typedef ippl::FFTPeriodicPoissonSolver<VField_t, Field_t> Solver_t;

            ippl::ParameterList params;
            params.add("output_type", Solver_t::SOL);
            params.add("use_heffte_defaults", false);
            params.add("use_pencils", true);
            // params.add("use_reorder", false);
            params.add("use_gpu_aware", true);
            params.add("comm", ippl::a2av);
            params.add("r2c_direction", 0);

            Solver_t FFTsolver;

            FFTsolver.mergeParameters(params);

            FFTsolver.setRhs(field);

            const ippl::NDIndex<dim>& lDom   = layout.getLocalNDIndex();
            const int nghost                 = field.getNghost();
            typename Field_t::view_type view = field.getView();

            switch (params.template get<int>("output_type")) {
                case Solver_t::SOL: {
                    Field_t phifield_exact(mesh, layout);

                    auto view_exact = phifield_exact.getView();

                    Kokkos::parallel_for(
                        "Assign rhs", ippl::getRangePolicy(view, nghost),
                        KOKKOS_LAMBDA(const int i, const int j, const int k) {
                            using Kokkos::pow, Kokkos::cos, Kokkos::sin;
                            // local to global index conversion
                            const size_t ig = i + lDom[0].first() - nghost;
                            const size_t jg = j + lDom[1].first() - nghost;
                            const size_t kg = k + lDom[2].first() - nghost;
                            double x        = origin[0] + (ig + 0.5) * hx[0];
                            double y        = origin[1] + (jg + 0.5) * hx[1];
                            double z        = origin[2] + (kg + 0.5) * hx[2];

                            // view(i, j, k) = 3.0 * pow(pi, 2) * sin(pi * x) * sin(pi * y) * sin(pi
                            // * z);
                            view(i, j, k) = pow(pi, 2)
                                            * (cos(sin(pi * z)) * sin(pi * z) * sin(sin(pi * x))
                                                   * sin(sin(pi * y))
                                               + (cos(sin(pi * y)) * sin(pi * y) * sin(sin(pi * x))
                                                  + (cos(sin(pi * x)) * sin(pi * x)
                                                     + (pow(cos(pi * x), 2) + pow(cos(pi * y), 2)
                                                        + pow(cos(pi * z), 2))
                                                           * sin(sin(pi * x)))
                                                        * sin(sin(pi * y)))
                                                     * sin(sin(pi * z)));

                            // view_exact(i, j, k) = sin(pi * x) * sin(pi * y) * sin(pi * z);
                            view_exact(i, j, k) =
                                sin(sin(pi * x)) * sin(sin(pi * y)) * sin(sin(pi * z));
                        });

                    FFTsolver.solve();
                    // Compute the relative error norm
                    field              = field - phifield_exact;
                    field              = pow(field, 2);
                    phifield_exact     = pow(phifield_exact, 2);
                    double error1      = sqrt(field.sum());
                    double error2      = sqrt(phifield_exact.sum());
                    double error_norm2 = error1 / error2;

                    if (ippl::Comm->rank() == 0) {
                        std::cout << "L2 relative error norm: " << error_norm2 << std::endl;
                    }
                    break;
                }

                case Solver_t::GRAD: {
                    VField_t Efield, Efield_exact;

                    Efield.initialize(mesh, layout);

                    Efield_exact.initialize(mesh, layout);

                    auto Eview_exact = Efield_exact.getView();

                    Kokkos::parallel_for(
                        "Assign rhs", ippl::getRangePolicy(view, nghost),
                        KOKKOS_LAMBDA(const int i, const int j, const int k) {
                            using Kokkos::pow, Kokkos::cos, Kokkos::sin;
                            // local to global index conversion
                            const size_t ig = i + lDom[0].first() - nghost;
                            const size_t jg = j + lDom[1].first() - nghost;
                            const size_t kg = k + lDom[2].first() - nghost;
                            double x        = origin[0] + (ig + 0.5) * hx[0];
                            double y        = origin[1] + (jg + 0.5) * hx[1];
                            double z        = origin[2] + (kg + 0.5) * hx[2];

                            view(i, j, k) = pow(pi, 2)
                                            * (cos(sin(pi * z)) * sin(pi * z) * sin(sin(pi * x))
                                                   * sin(sin(pi * y))
                                               + (cos(sin(pi * y)) * sin(pi * y) * sin(sin(pi * x))
                                                  + (cos(sin(pi * x)) * sin(pi * x)
                                                     + (pow(cos(pi * x), 2) + pow(cos(pi * y), 2)
                                                        + pow(cos(pi * z), 2))
                                                           * sin(sin(pi * x)))
                                                        * sin(sin(pi * y)))
                                                     * sin(sin(pi * z)));

                            Eview_exact(i, j, k)[0] = -pi * cos(pi * x) * cos(sin(pi * x))
                                                      * sin(sin(pi * y)) * sin(sin(pi * z));
                            Eview_exact(i, j, k)[1] = -pi * cos(pi * y) * cos(sin(pi * y))
                                                      * sin(sin(pi * x)) * sin(sin(pi * z));
                            Eview_exact(i, j, k)[2] = -pi * cos(pi * z) * cos(sin(pi * z))
                                                      * sin(sin(pi * x)) * sin(sin(pi * y));
                        });

                    FFTsolver.setLhs(Efield);

                    FFTsolver.solve();

                    ippl::Vector<double, 3> errorNr, errorDr, error_norm2;
                    Efield     = Efield - Efield_exact;
                    auto Eview = Efield.getView();

                    // We don't have a vector reduce yet..
                    for (size_t d = 0; d < dim; ++d) {
                        double temp = 0.0;
                        Kokkos::parallel_reduce(
                            "Vector errorNr reduce", ippl::getRangePolicy(view, nghost),
                            KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k,
                                          double& valL) {
                                double myVal = Kokkos::pow(Eview(i, j, k)[d], 2);
                                valL += myVal;
                            },
                            Kokkos::Sum<double>(temp));
                        double globaltemp = 0.0;
                        ippl::Comm->allreduce(temp, globaltemp, 1, std::plus<double>());
                        errorNr[d] = std::sqrt(globaltemp);

                        temp = 0.0;
                        Kokkos::parallel_reduce(
                            "Vector errorDr reduce", ippl::getRangePolicy(view, nghost),
                            KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k,
                                          double& valL) {
                                double myVal = Kokkos::pow(Eview_exact(i, j, k)[d], 2);
                                valL += myVal;
                            },
                            Kokkos::Sum<double>(temp));
                        globaltemp = 0.0;
                        ippl::Comm->allreduce(temp, globaltemp, 1, std::plus<double>());
                        errorDr[d] = std::sqrt(globaltemp);

                        error_norm2[d] = errorNr[d] / errorDr[d];
                    }

                    if (ippl::Comm->rank() == 0) {
                        for (size_t d = 0; d < dim; ++d) {
                            std::cout << "L2 relative error norm Efield[" << d
                                      << "]: " << error_norm2[d] << std::endl;
                        }
                    }
                    break;
                }

                default:
                    std::cout << "Unrecognized option" << std::endl;
            }
        }
    }
    ippl::finalize();

    return 0;
}
