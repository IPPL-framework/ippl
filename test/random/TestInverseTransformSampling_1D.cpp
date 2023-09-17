#include "Ippl.h"

#include "Random/InverseTransformSampling_1D.h"

/*
double cdf_y(double y, double alpha, double k) {
    return y + (alpha / k) * std::sin(k * y);
}
double pdf_y(double y, double alpha, double k) {
    return  (1.0 + alpha * Kokkos::cos(k * y));
}
double estimate_y(double u) {
    return u; // maybe E[x] is good enough as the first guess
}
*/

/*
static KOKKOS_INLINE_FUNCTION double cdf(double y, const double *p) {
    return y + (p[0] / p[1]) * Kokkos::sin(p[1] * y);
}
static KOKKOS_INLINE_FUNCTION double pdf(double y, const double *p) {
    return  1.0 + p[0] * Kokkos::cos(p[1] * y);
}
static KOKKOS_INLINE_FUNCTION double estimate(double u, const double *p) {
    return (u + p[0])*0.0; // maybe E[x] is good enough as the first guess
}
*/
int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        using Mesh_t = ippl::UniformCartesian<double, 2>;

        ippl::Vector<int, 2> nr   = {std::atoi(argv[1]), std::atoi(argv[2])};
        const unsigned int ntotal = std::atol(argv[3]);

        ippl::NDIndex<2> domain;
        for (unsigned i = 0; i < 2; i++) {
            domain[i] = ippl::Index(nr[i]);
        }

        ippl::e_dim_tag decomp[2];
        for (unsigned d = 0; d < 2; ++d) {
            decomp[d] = ippl::PARALLEL;
        }

        // create mesh and layout objects for this problem domain
        ippl::Vector<double, 2> rmin   = -4.;
        ippl::Vector<double, 2> rmax   = 4.;
        ippl::Vector<double, 2> length = rmax - rmin;

        ippl::Vector<double, 2> hr     = length / nr;
        ippl::Vector<double, 2> origin = rmin;

        const bool isAllPeriodic = true;
        Mesh_t mesh(domain, hr, origin);
        ippl::FieldLayout<2> fl(domain, decomp, isAllPeriodic);

        ippl::detail::RegionLayout<double, 2, Mesh_t> rlayout(fl, mesh);

        using Dist_t = ippl::random::Distribution<double, 2>;
        using view_type = ippl::detail::ViewType<ippl::Vector<double, 1>, 1>::view_type;
        using sampling_t = ippl::random::sample_its<double, Kokkos::Serial, Dist_t>;
        
        // Define a distribution that is normal in dim=0, and harmonic in dim=1
        const double mu = 1.0;
        const double sd = 0.5;
        const double par[2] = {mu, sd};
        
        Dist_t dist(par);
        dist.setNormalDistribution();
        sampling_t sampling(dist, 0, rmax[0], rmin[0], rlayout, ntotal);
        unsigned int nlocal = sampling.getLocalNum();
        view_type position("position", nlocal);
        sampling.sample_ITS(position, 42);
        
        
        const double par2[2] = {-1.0, 1.0};
        Dist_t dist2(par2);
        dist2.setNormalDistribution();
        sampling_t sampling2(dist2, 1, rmax[1], rmin[1], rlayout, ntotal);
        unsigned int nlocal2 = sampling.getLocalNum();
        view_type position2("position2", nlocal2);
        sampling2.sample_ITS(position2, 0);
        
        
        /*
        const double pi = Kokkos::numbers::pi_v<double>;
        const double kw = 2.*pi/(rmax[1]-rmin[1])*4.0;
        const double alpha = 1.0;
        const double par2[2] = {alpha, kw};
        Dist_t dist2(par2);
        dist2.setCdfFunction(cdf);
        dist2.setPdfFunction(pdf);
        dist2.setEstimationFunction(estimate);
        sampling_t sampling2(dist2, 1, rmax[1], rmin[1], rlayout, ntotal);
        unsigned int nlocal2 = sampling.getLocalNum();
        view_type position2("position2", nlocal2);
        sampling2.sample_ITS(position2, 42);
        */
        for (unsigned int i = 0; i < nlocal; ++i) {
            std::cout << ippl::Comm->rank() << " " << position(i)[0] << " " << position2(i)[0]
                      << std::endl;
        }
        
    }
    ippl::finalize();

    return 0;
}
