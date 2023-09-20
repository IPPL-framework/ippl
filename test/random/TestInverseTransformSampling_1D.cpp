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

#include "Random/Generator.h"
#include "Random/InverseTransformSampling_1D.h"

/*
template <typename T, unsigned DimP>
class NormalDistribution{
  public:
    static constexpr T pi = Kokkos::numbers::pi_v<T>;
    T par[DimP];
    NormalDistribution(const T *par_) {
            for(unsigned int i=0; i<DimP; i++){ par[i] = par_[i]; }
   }
    struct custom_cdf{
       KOKKOS_INLINE_FUNCTION double operator()(T x, const T *params) const {
          T mean = params[0];
          T stddev = params[1];
          return 0.5 * (1 + Kokkos::erf((x - mean) / (stddev * Kokkos::sqrt(2.0))));
       }
    };
    struct custom_pdf{
       KOKKOS_INLINE_FUNCTION double operator()(T x, T const *params) const {
          T mean = params[0];
          T stddev = params[1];
          return (1.0 / (stddev * Kokkos::sqrt(2 * pi))) * Kokkos::exp(-(x - mean) * (x - mean) / (2 * stddev * stddev));
       }
    };
    struct custom_estimate{
        KOKKOS_INLINE_FUNCTION double operator()(T u, T const *params) const {
          T mean = params[0];
          T stddev = params[1];
          return (Kokkos::sqrt(pi / 2.0) * (2.0 * u - 1.0)) * stddev + mean;
        }
    };
    KOKKOS_INLINE_FUNCTION T cdf(T x) const{
            return cdf_functor(x, par);
    }
    KOKKOS_INLINE_FUNCTION T pdf(T x) const {
            return pdf_functor(x, par);
    }
    KOKKOS_INLINE_FUNCTION T obj_func(T x, T u) const{
            return cdf(x) - u;
    }
    KOKKOS_INLINE_FUNCTION T der_obj_func(T x) const{
            return pdf(x);
    }
    KOKKOS_INLINE_FUNCTION T estimate (T u) const{
            return estimate_functor(u, par);
    }
    custom_cdf cdf_functor;
    custom_pdf pdf_functor;
    custom_estimate estimate_functor;
};


template <typename T, unsigned DimP>
class HarmonicDistribution{
  public:
    static constexpr T pi = Kokkos::numbers::pi_v<T>;
    T par[DimP];
    HarmonicDistribution(const T *par_) {
            for(unsigned int i=0; i<DimP; i++){ par[i] = par_[i]; }
   }
   struct custom_cdf{
       KOKKOS_INLINE_FUNCTION double operator()(T x, const T *params) const {
           return x + (params[0] / params[1]) * Kokkos::sin(params[1] * x);
       }
    };
    struct custom_pdf{
       KOKKOS_INLINE_FUNCTION double operator()(T x, T const *params) const {
           return  1.0 + params[0] * Kokkos::cos(params[1] * x);
       }
    };
    struct custom_estimate{
        KOKKOS_INLINE_FUNCTION double operator()(T u, T const *params) const {
            return u +  params[0]*0.0;
        }
    };
    KOKKOS_INLINE_FUNCTION T cdf(T x) const{
            return cdf_functor(x, par);
    }
    KOKKOS_INLINE_FUNCTION T pdf(T x) const {
            return pdf_functor(x, par);
    }
    KOKKOS_INLINE_FUNCTION T obj_func(T x, T u) const{
            return cdf(x) - u;
    }
    KOKKOS_INLINE_FUNCTION T der_obj_func(T x) const{
            return pdf(x);
    }
    KOKKOS_INLINE_FUNCTION T estimate (T u) const{
            return estimate_functor(u, par);
    }
    custom_cdf cdf_functor;
    custom_pdf pdf_functor;
    custom_estimate estimate_functor;
};
*/

struct custom_cdf{
       KOKKOS_INLINE_FUNCTION double operator()(double x, const double *params) const {
           return x + (params[0] / params[1]) * Kokkos::sin(params[1] * x);
       }
};
struct custom_pdf{
       KOKKOS_INLINE_FUNCTION double operator()(double x, double const *params) const {
           return  1.0 + params[0] * Kokkos::cos(params[1] * x);
       }
};
struct custom_estimate{
        KOKKOS_INLINE_FUNCTION double operator()(double u, double const *params) const {
            return u +  params[0]*0.0;
        }
};

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg("LandauDamping");
        Inform msg2all("LandauDamping", INFORM_ALL_NODES);

        //double pi    = Kokkos::numbers::pi_v<double>;

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

        ippl::Vector<double, 2> rmin   = -4.;
        ippl::Vector<double, 2> rmax   = 4.;
        ippl::Vector<double, 2> length = rmax - rmin;

        ippl::Vector<double, 2> hr     = length / nr;
        ippl::Vector<double, 2> origin = rmin;

        const bool isAllPeriodic = true;

        Mesh_t mesh(domain, hr, origin);

        ippl::FieldLayout<2> fl(domain, decomp, isAllPeriodic);

        ippl::detail::RegionLayout<double, 2, Mesh_t> rlayout(fl, mesh);

        using Dist_t = ippl::random::Normal<double>;
        using view_type  = typename ippl::detail::ViewType<double, 1>::view_type;
        using sampling_t = ippl::random::sample_its<double, Kokkos::DefaultExecutionSpace, Dist_t>;

        int seed = 42;
        using size_type = ippl::detail::size_type;
        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(seed + 100 * ippl::Comm->rank()));

        const double mu = 1.0;
        const double sd = 0.5;
        const double par[2] = {mu, sd};
        Dist_t dist(par);
        dist.estimate(0.0);
        //sampling_t sampling(dist, 0, rmax[0], rmin[0], rlayout, ntotal);
        //unsigned int nlocal = sampling.getLocalNum();
        //view_type position("position", nlocal);
        //sampling.generate(position, rand_pool64);

/*
        using DistH_t = ippl::random::Distribution<double, 2>;
        using samplingH_t = ippl::random::sample_its<double, Kokkos::DefaultExecutionSpace, DistH_t>;
        const double parH[2] = {0.5, 2.*pi/(rmax[1]-rmin[1])*4.0};
        DistH_t distH(parH);
        custom_pdf pdf;
        custom_cdf cdf;
        custom_estimate estimate;
        distH.setCustomFunctions(pdf, cdf, estimate);

        samplingH_t samplingH(distH, 1, rmax[1], rmin[1], rlayout, ntotal);
        nlocal = samplingH.getLocalNum();
        view_type positionH("positionH", nlocal);
        samplingH.generate(positionH, rand_pool64);
*/        

        //for (unsigned int i = 0; i < nlocal; ++i) {
        //    msg << position(i) << " " << positionH(i) << endl;
        //}

        msg << "End of program" << endl;
    }
    ippl::finalize();

    return 0;
}
