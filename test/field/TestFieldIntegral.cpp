#include "Ippl.h"

#include <iostream>
#include <typeinfo>

#include <cstdlib>

int main(int argc, char *argv[]) {
    Ippl ippl(argc,argv);
    constexpr unsigned int dim = 3;

    constexpr unsigned int nX = 4, nY = 8, nZ = 16;
    ippl::Index X(nX), Y(nY), Z(nZ);
    ippl::NDIndex<dim> owned(X, Y, Z);

    ippl::e_dim_tag allParallel[dim];    // Specifies SERIAL, PARALLEL dims
    for (unsigned int d=0; d<dim; d++)
        allParallel[d] = ippl::PARALLEL;

    // all parallel layout, standard domain, normal axis order
    ippl::FieldLayout<dim> layout(owned,allParallel);

    ippl::Vector<double, 3> hx = {1. / nX, 1. / nY, 1. / nZ};
    ippl::Vector<double, 3> origin = {0, 0, 0};
    ippl::UniformCartesian<double, 3> mesh(owned, hx, origin);

    typedef ippl::Field<double, dim> field_type;

    field_type field(mesh, layout);

    PAssert_EQ(field.getMeshVolume(), 1.);
    std::cout << "Mesh volume test passed" << std::endl;

    double pi = acos(-1.0);
    const int shift = field.getNghost();
    auto view = field.getView();
    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
        {shift, shift, shift},
        {view.extent(0) - shift, view.extent(1) - shift, view.extent(2) - shift}
    );

    Kokkos::parallel_for("assign field", policy,
        KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
            double x = (i + 0.5) * hx[0] + origin[0];
            double y = (j + 0.5) * hx[1] + origin[1];
            double z = (k + 0.5) * hx[2] + origin[2];

            view(i, j, k) = sin(2 * pi * x) * sin(2 * pi * y) * sin(2 * pi * z);
        }
    );

    PAssert_EQ(field.getVolumeIntegral(), 0.);
    std::cout << "Volume integral test passed" << std::endl;

    return 0;
}
