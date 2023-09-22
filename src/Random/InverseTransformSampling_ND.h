#ifndef IPPL_INVERSE_TRANSFORM_SAMPLING_H
#define IPPL_INVERSE_TRANSFORM_SAMPLING_H

#include "Types/ViewTypes.h"

#include "Random/Generator.h"
#include "Random/Random.h"

namespace ippl {
  namespace random {
    namespace detail{
        template <typename T>
        struct NewtonRaphson {
            KOKKOS_FUNCTION
            NewtonRaphson() = default;

            KOKKOS_FUNCTION
            ~NewtonRaphson() = default;

            template <class Distribution>
            KOKKOS_INLINE_FUNCTION void solve(Distribution dist, int d, T& x, T& u, T atol = 1.0e-12,
                                              unsigned int max_iter = 20) {
                unsigned int iter = 0;
                while (iter < max_iter && Kokkos::fabs(dist.obj_func(x, d, u)) > atol) {
                    // Find x, such that "cdf(x) - u = 0" for a given sample of u~uniform(0,1)
                    x = x - (dist.obj_func(x, d, u) / dist.der_obj_func(x, d));
                    iter += 1;
                }
            }
        };
    }// name space detial

    template <typename T, unsigned Dim, class DeviceType, class Distribution>
    class sample_its{
    public:
        using view_type = typename ippl::detail::ViewType<Vector<T, Dim>, 1>::view_type;
        Distribution dist;
        const Vector<T, Dim> rmax;
        const Vector<T, Dim> rmin;
        Vector<T, Dim> umin, umax;
        unsigned int ntotal;
        template <class RegionLayout>
        sample_its(Distribution dist_, Vector<T, Dim> rmax_, Vector<T, Dim> rmin_, const RegionLayout& rlayout, unsigned int ntotal_)
        : dist(dist_),  // Initialize the 'dist' member with the provided 'dist_' argument
        rmax(rmax_),
        rmin(rmin_),
        ntotal(ntotal_){
            const typename RegionLayout::host_mirror_type regions = rlayout.gethLocalRegions();
            
            int rank = ippl::Comm->rank();
            Vector<T, Dim> nr_m, dr_m;
            for (unsigned d = 0; d < Dim; ++d) {
                nr_m[d] = dist.cdf(regions(rank)[d].max(), d) - dist.cdf(regions(rank)[d].min(),d);
                dr_m[d]   = dist.cdf(rmax[d],d) - dist.cdf(rmin[d],d);
                umin[d] = dist.cdf(regions(rank)[d].min(),d);
                umax[d] = dist.cdf(regions(rank)[d].max(),d);
            }
            T pnr = std::accumulate(nr_m.begin(), nr_m.end(), 1.0, std::multiplies<T>());
            T pdr = std::accumulate(dr_m.begin(), dr_m.end(), 1.0, std::multiplies<T>());
            
            T factor = pnr / pdr;
            nlocal_m      = factor * ntotal;
            
            unsigned int ngobal = 0;
            MPI_Allreduce(&nlocal_m, &ngobal, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                          ippl::Comm->getCommunicator());
            
            int rest = (int)(ntotal - ngobal);
            
            if (rank < rest) {
                ++nlocal_m;
            }
        }

        template <class GeneratorPool>
        struct fill_random {
            //using view_type  = typename ippl::detail::ViewType<Tt, 1>::view_type;
            using value_type = T; //typename Tt::value_type;
            Distribution dist;
            view_type x;
            GeneratorPool rand_pool;
            Vector<T, Dim> umin_m;
            Vector<T, Dim> umax_m;
            KOKKOS_FUNCTION
            fill_random(Distribution dist_, view_type x_, GeneratorPool rand_pool_, Vector<T, Dim> umin_, Vector<T, Dim> umax_)
            : dist(dist_)
            , x(x_)
            , rand_pool(rand_pool_)
            , umin_m(umin_)
            , umax_m(umax_){}
            KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
                typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();
                
                value_type u = 0.0;
                for (unsigned d = 0; d < Dim; ++d) {
                    u       = rand_gen.drand(umin_m[d], umax_m[d]);
                    
                    // first guess for Newton-Raphson
                    x(i)[d] = dist.estimate(u, d);
                    
                    // solve
                    ippl::random::detail::NewtonRaphson<value_type> solver;
                    solver.solve(dist, d, x(i)[d], u);
                    rand_pool.free_state(rand_gen);
                }
            }
        };
        KOKKOS_INLINE_FUNCTION unsigned int getLocalNum() const { return nlocal_m; }
        void generate(view_type view, Kokkos::Random_XorShift64_Pool<> rand_pool64) {
            Kokkos::parallel_for(nlocal_m, fill_random<Kokkos::Random_XorShift64_Pool<>>(dist, view, rand_pool64, umin, umax));
            Kokkos::fence();
        }
    private:
        unsigned int nlocal_m;
    };

    template <typename T, unsigned Dim, unsigned DimP, typename PDF, typename CDF, typename ESTIMATE>
    class Distribution {
    public:
       T par_m[DimP];

       KOKKOS_INLINE_FUNCTION Distribution(const T *par_) {
            for(unsigned int i=0; i<DimP; i++){
                par_m[i] = par_[i];
            }
       }
       PDF pdf_m;
       CDF cdf_m;
       ESTIMATE estimate_m;
       KOKKOS_INLINE_FUNCTION T pdf(T x, unsigned int d) const{
          return pdf_m(x, d, par_m);
       }
       KOKKOS_INLINE_FUNCTION T cdf(T x, unsigned int d) const{
          return cdf_m(x, d, par_m);
       }
       KOKKOS_INLINE_FUNCTION T estimate(T x, unsigned int d) const{
          return estimate_m(x, d, par_m);
       }
       KOKKOS_INLINE_FUNCTION T obj_func(T x, unsigned int d, T u) const{
            return cdf(x, d) - u;
       }
       KOKKOS_INLINE_FUNCTION T der_obj_func(T x, unsigned int d) const{
            return pdf(x, d);
       }
       KOKKOS_INLINE_FUNCTION T full_pdf(ippl::Vector<T, Dim> x) const{
          T total_pdf = 1.0;
          for(unsigned int d=0; d<Dim; d++){
             total_pdf *= pdf(x[d], d);
          }
          return total_pdf;
       }
    };

  template<typename T>
  KOKKOS_FUNCTION T normal_cdf_func(T x, T mean, T stddev) {
      return 0.5 * (1 + Kokkos::erf((x - mean) / (stddev * Kokkos::sqrt(2.0))));
  }
  template<typename T>
  KOKKOS_FUNCTION T normal_pdf_func(T x, T mean, T stddev) {
      T pi = Kokkos::numbers::pi_v<T>;
      return (1.0 / (stddev * Kokkos::sqrt(2 * pi))) * Kokkos::exp(-(x - mean) * (x - mean) / (2 * stddev * stddev));
  }
  template<typename T>
  KOKKOS_FUNCTION T normal_estimate_func(T u, T mean, T stddev) {
      T pi = Kokkos::numbers::pi_v<T>;
      return (Kokkos::sqrt(pi / 2.0) * (2.0 * u - 1.0)) * stddev + mean;
  }
  
  template<typename T>
  KOKKOS_FUNCTION T uniform_cdf_func(T x){
      return x;
  }
  template<typename T>
  KOKKOS_FUNCTION T uniform_pdf_func(){
      return 1.;
  }
  template<typename T>
  KOKKOS_FUNCTION T uniform_estimate_func(T u){
      return u;
  }
  
  
    template <typename T, unsigned Dim>
    struct normal_cdf{
        KOKKOS_INLINE_FUNCTION double operator()(T x, unsigned int d, const T *params) const {
                T mean = params[d*Dim + 0];
                T stddev = params[d*Dim + 1];
                //return 0.5 * (1 + Kokkos::erf((x - mean) / (stddev * Kokkos::sqrt(2.0))));
                return ippl::random::normal_cdf_func<T>(x, mean, stddev);
        }
    };
    template <typename T, unsigned Dim>
    struct normal_pdf{
        KOKKOS_INLINE_FUNCTION double operator()(T x, unsigned int d, T const *params) const {
                T mean = params[d*Dim + 0];
                T stddev = params[d*Dim + 1];
                //static constexpr T pi = Kokkos::numbers::pi_v<T>;
                //return (1.0 / (stddev * Kokkos::sqrt(2 * pi))) * Kokkos::exp(-(x - mean) * (x - mean) / (2 * stddev * stddev));
                return ippl::random::normal_pdf_func<T>(x, mean, stddev);
        }
    };
    template <typename T, unsigned Dim>
    struct normal_estimate{
        KOKKOS_INLINE_FUNCTION double operator()(T u, unsigned int d,  T const *params) const {
                //static constexpr T pi = Kokkos::numbers::pi_v<T>;
                T mean = params[d*Dim + 0];
                T stddev = params[d*Dim + 1];
                //return (Kokkos::sqrt(pi / 2.0) * (2.0 * u - 1.0)) * stddev + mean;
                return ippl::random::normal_estimate_func<T>(u, mean, stddev);
        }
    };

    template<typename T, unsigned Dim>
    class Normal : public Distribution<T, Dim, 2*Dim, normal_pdf<T, Dim>, normal_cdf<T, Dim>, normal_estimate<T, Dim>>{
    public:
       KOKKOS_INLINE_FUNCTION Normal(const T *par_) : Distribution<T, Dim, 2*Dim, normal_pdf<T, Dim>, normal_cdf<T, Dim>, normal_estimate<T, Dim>>(par_) {}
    };
  }  // namespace random
}  // namespace ippl

#endif
