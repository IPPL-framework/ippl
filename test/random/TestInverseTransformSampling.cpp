#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_Random.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>
#include "Utility/IpplTimings.h"
#include "Ippl.h"
#include "Random/InverseTransformSampling.h"

struct custom_cdf{
       KOKKOS_INLINE_FUNCTION double operator()(double x, unsigned int d, const double *params) const {
           if(d==0){
               return ippl::random::uniform_cdf_func<double>(x);
           }
           else{
               return x + (params[2] / params[3]) * Kokkos::sin(params[3] * x);
           }
       }
};
struct custom_pdf{
       KOKKOS_INLINE_FUNCTION double operator()(double x, unsigned int d, double const *params) const {
           if(d==0){
               return ippl::random::uniform_pdf_func<double>();;
           }
           else{
               return  1.0 + params[2] * Kokkos::cos(params[3] * x);
           }
       }
};
struct custom_estimate{
        KOKKOS_INLINE_FUNCTION double operator()(double u, unsigned int d, double const *params) const {
            if(d==0){
                return ippl::random::uniform_estimate_func<double>(u+params[0]*0);
            }
            else{
                return u;
            }
        }
};

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        const int Dim = 2;
        Inform msg("LandauDamping");
        Inform msg2all("LandauDamping", INFORM_ALL_NODES);

        using Mesh_t = ippl::UniformCartesian<double, Dim>;

        ippl::Vector<int, 2> nr   = {std::atoi(argv[1]), std::atoi(argv[2])};
        const unsigned int ntotal = std::atol(argv[3]);

        ippl::NDIndex<2> domain;
        for (unsigned i = 0; i < Dim; i++) {
            domain[i] = ippl::Index(nr[i]);
        }

        ippl::e_dim_tag decomp[Dim];
        for (unsigned d = 0; d < Dim; ++d) {
            decomp[d] = ippl::PARALLEL;
        }

        ippl::Vector<double, Dim> rmin   = -4.;
        ippl::Vector<double, Dim> rmax   = 4.;
        ippl::Vector<double, Dim> length = rmax - rmin;

        ippl::Vector<double, Dim> hr     = length / nr;
        ippl::Vector<double, Dim> origin = rmin;

        const bool isAllPeriodic = true;

        Mesh_t mesh(domain, hr, origin);

        ippl::FieldLayout<Dim> fl(domain, decomp, isAllPeriodic);

        ippl::detail::RegionLayout<double, Dim, Mesh_t> rlayout(fl, mesh);

        using view_type  = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;
        int seed = 42;
        using size_type = ippl::detail::size_type;
        unsigned int nlocal;
        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(seed + 100 * ippl::Comm->rank()));

        // example of sampling normal in both dimensions
        const double mu = 1.0;
        const double sd = 0.5;
        const double par[4] = {mu, sd, -mu, 0.5*sd};
        using Dist_t = ippl::random::Normal<double, Dim>;
        using sampling_t = ippl::random::sample_its<double, Dim, Kokkos::DefaultExecutionSpace, Dist_t>;

        Dist_t dist(par);
        sampling_t sampling(dist, rmax, rmin, rlayout, ntotal);
        nlocal = sampling.getLocalNum();
        view_type position("position", nlocal);
        sampling.generate(position, rand_pool64);

        // example of sampling normal/uniform in one and harmonic in another with custom functors
        const int DimP = 4;
        double pi    = Kokkos::numbers::pi_v<double>;
        using DistH_t = ippl::random::Distribution<double, Dim, DimP, custom_pdf, custom_cdf, custom_estimate>;
        using samplingH_t = ippl::random::sample_its<double, Dim, Kokkos::DefaultExecutionSpace, DistH_t>;
        const double parH[DimP] = {rmin[0], rmax[0], 0.5, 2.*pi/(rmax[1]-rmin[1])*4.0};

        DistH_t distH(parH);
        samplingH_t samplingH(distH, rmax, rmin, rlayout, ntotal);
        nlocal = samplingH.getLocalNum();
        view_type positionH("positionH", nlocal);
        samplingH.generate(positionH, rand_pool64);

        //for (unsigned int i = 0; i < nlocal; ++i) {
        //     std::cout << positionH(i)[0] << " " << positionH(i)[1] << std::endl;
        //}
        //msg << "End of program" << endl;
    }
    ippl::finalize();
    return 0;
}
