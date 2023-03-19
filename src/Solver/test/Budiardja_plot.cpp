// This program recreates the convergence test plot from the Budiardja et al. (2010) paper.
// Gravitational potential of a sphere.

#include "Ippl.h"

#include "FFTPoissonSolver.h"

KOKKOS_INLINE_FUNCTION double source(double x, double y, double z, double density = 1.0,
                                     double R = 1.0, double mu = 1.2) {
    double pi = std::acos(-1.0);
    double G  = 6.674e-11;

    double r         = std::sqrt((x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu));
    bool checkInside = (r <= R);

    return double(checkInside) * 4.0 * pi * G * density;
}

KOKKOS_INLINE_FUNCTION double exact_fct(double x, double y, double z, double density = 1.0,
                                        double R = 1.0, double mu = 1.2) {
    double pi = std::acos(-1.0);
    double G  = 6.674e-11;

    double r = std::sqrt((x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu));

    bool checkInside = (r <= R);
    return -(double(checkInside) * (2.0 / 3.0) * pi * G * density * (3 * R * R - r * r))
           - ((1.0 - double(checkInside)) * (4.0 / 3.0) * pi * G * density * R * R * R / r);
}

int main(int argc, char* argv[]) {
    Ippl ippl(argc, argv);

    // number of interations
    const int n = 5;

    using Mesh_t = ippl::UniformCartesian<double, 3>;
    using Centering_t = Mesh_t::DefaultCentering;

    // number of gridpoints to iterate over
    std::array<int, n> N = {48, 144, 288, 384, 576};

    std::cout << "Spacing Error" << std::endl;

    for (int p = 0; p < n; ++p) {
        // domain
        int pt = N[p];
        ippl::Index I(pt);
        ippl::NDIndex<3> owned(I, I, I);

        // specifies decomposition; here all dimensions are parallel
        ippl::e_dim_tag decomp[3];
        for (unsigned int d = 0; d < 3; d++)
            decomp[d] = ippl::PARALLEL;

        // define computational box of side 2.4
        double dx                      = 2.4 / pt;
        ippl::Vector<double, 3> hx     = {dx, dx, dx};
        ippl::Vector<double, 3> origin = {0.0, 0.0, 0.0};
        Mesh_t mesh(owned, hx, origin);

        // all parallel layout, standard domain, normal axis order
        ippl::FieldLayout<3> layout(owned, decomp);

        // define the L (phi) and R (rho) fields
        typedef ippl::Field<double, 3, Mesh_t, Centering_t> field;
        field rho;
        rho.initialize(mesh, layout);

        // define the exact solution field
        field exact;
        exact.initialize(mesh, layout);

        // assign the rho field with its value
        typename field::view_type view_rho = rho.getView();
        const int nghost                   = rho.getNghost();
        const auto& ldom                   = layout.getLocalNDIndex();

        Kokkos::parallel_for(
            "Assign rho field",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                {nghost, nghost, nghost}, {view_rho.extent(0) - nghost, view_rho.extent(1) - nghost,
                                           view_rho.extent(2) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                // go from local to global indices
                const int ig = i + ldom[0].first() - nghost;
                const int jg = j + ldom[1].first() - nghost;
                const int kg = k + ldom[2].first() - nghost;

                // define the physical points (cell-centered)
                double x = (ig + 0.5) * hx[0] + origin[0];
                double y = (jg + 0.5) * hx[1] + origin[1];
                double z = (kg + 0.5) * hx[2] + origin[2];

                view_rho(i, j, k) = source(x, y, z);
            });

        // assign the exact field with its values
        typename field::view_type view_exact = exact.getView();

        Kokkos::parallel_for(
            "Assign exact field",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                {nghost, nghost, nghost},
                {view_exact.extent(0) - nghost, view_exact.extent(1) - nghost,
                 view_exact.extent(2) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                const int ig = i + ldom[0].first() - nghost;
                const int jg = j + ldom[1].first() - nghost;
                const int kg = k + ldom[2].first() - nghost;

                double x = (ig + 0.5) * hx[0] + origin[0];
                double y = (jg + 0.5) * hx[1] + origin[1];
                double z = (kg + 0.5) * hx[2] + origin[2];

                view_exact(i, j, k) = exact_fct(x, y, z);
            });

        // set FFT parameters
        ippl::ParameterList fftParams;
        fftParams.add("use_heffte_defaults", false);
        fftParams.add("use_pencils", true);
        fftParams.add("use_gpu_aware", true);
        fftParams.add("comm", ippl::a2av);
        fftParams.add("r2c_direction", 0);

        // define an FFTPoissonSolver object
        ippl::FFTPoissonSolver<ippl::Vector<double, 3>, double, 3, Mesh_t, Centering_t> FFTsolver(rho,
                                                                                                  fftParams,
                                                                                                  "HOCKNEY");

        // solve the Poisson equation -> rho contains the solution (phi) now
        FFTsolver.solve();

        // compute the L1 error
        rho        = (rho - exact);
        double err = norm(rho, 1) / norm(exact, 1);

        std::cout << dx << " " << err << std::endl;
    }
    return 0;
}
