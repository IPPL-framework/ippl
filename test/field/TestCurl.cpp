// Tests the Laplacian on a scalar field
#include "Ippl.h"

#include <iostream>
#include <typeinfo>
#include <array>

int main(int argc, char *argv[]) {

    Ippl ippl(argc,argv);

    constexpr unsigned int dim = 3;

    int pt = std::atoi(argv[1]);
    ippl::Index I(pt);
    ippl::NDIndex<dim> owned(I, I, I);

    ippl::e_dim_tag decomp[dim];    // Specifies SERIAL, PARALLEL dims
    for (unsigned int d=0; d<dim; d++)
        decomp[d] = ippl::PARALLEL;

    // all parallel layout, standard domain, normal axis order
    ippl::FieldLayout<dim> layout(owned,decomp);

    // domain [0,1]^3
    double dx = 1.0 / double(pt);
    ippl::Vector<double, 3> hx = {dx, dx, dx};
    ippl::Vector<double, 3> origin = {0.0, 0.0, 0.0};
    ippl::UniformCartesian<double, 3> mesh(owned, hx, origin);

    typedef ippl::Vector<double, dim> Vector_t;
    typedef ippl::Field<Vector_t, dim> Vfield_t;

    Vfield_t vfield(mesh, layout);
    Vfield_t result(mesh, layout);
    Vfield_t exact (mesh, layout);

    typename Vfield_t::view_type& view = vfield.getView();
    typename Vfield_t::view_type& view_exact = exact.getView();
    typename Vfield_t::view_type& view_result = result.getView();

    const ippl::NDIndex<dim>& lDom = layout.getLocalNDIndex();
    const int nghost = vfield.getNghost();
    using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

    for (unsigned int gd = 0; gd < dim; ++gd) {
        Kokkos::parallel_for("Assign field",
                mdrange_type({0,0,0},
                             {view.extent(0),
                              view.extent(1),
                              view.extent(2)}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {

                //local to global index conversion
                const int ig = i + lDom[0].first() - nghost;
                const int jg = j + lDom[1].first() - nghost;
                const int kg = k + lDom[2].first() - nghost;
            
                double x = (ig + 0.5) * hx[0] + origin[0];
                double y = (jg + 0.5) * hx[1] + origin[1];
                double z = (kg + 0.5) * hx[2] + origin[2];

                view(i, j, k)[gd] = x*y*z;
                if (gd == 0) {
                    view_exact(i, j, k)[gd] = x*z - x*y;
                } else if (gd == 1) {
                    view_exact(i, j, k)[gd] = x*y - y*z;
                } else {
                    view_exact(i, j, k)[gd] = y*z - x*z;
                }
        });
    }

    result = 0.0;
    result = curl(vfield);

    Vector_t errE;
    result = result - exact;

    for (unsigned int gd = 0; gd < dim; ++gd) {

        double temp = 0.0;

        Kokkos::parallel_reduce("Vector errorNr reduce",
                mdrange_type({nghost, nghost, nghost},
                             {view_result.extent(0) - nghost,
                              view_result.extent(1) - nghost,
                              view_result.extent(2) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k, double& valL) {
                double myVal = pow(view_result(i,j,k)[gd], 2);
                valL += myVal;
        }, Kokkos::Sum<double>(temp));
        double globaltemp = 0.0;
        MPI_Allreduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
        double errorNr = std::sqrt(globaltemp);

        temp = 0.0;
        globaltemp = 0.0;

        Kokkos::parallel_reduce("Vector errorDr reduce",
                mdrange_type({nghost, nghost, nghost},
                             {view_exact.extent(0) - nghost,
                              view_exact.extent(1) - nghost,
                              view_exact.extent(2) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k, double& valL) {
                double myVal = pow(view_exact(i,j,k)[gd], 2);
                valL += myVal;
        }, Kokkos::Sum<double>(temp));
        MPI_Allreduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
        double errorDr = std::sqrt(globaltemp);

        errE[gd] = errorNr/errorDr;
    }

    if (Ippl::Comm->rank() == 0) {
        std::cout << "Error: " << errE[0] << ", " << errE[1] << ", " << errE[2] << std::endl;
    }


    return 0;
}
