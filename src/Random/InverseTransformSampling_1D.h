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
            KOKKOS_INLINE_FUNCTION void solve(Distribution dist, T& x, T& u, T atol = 1.0e-12,
                                              unsigned int max_iter = 20) {
                unsigned int iter = 0;
                while (iter < max_iter && Kokkos::fabs(dist.obj_func(x, u)) > atol) {
                    // Find x, such that "cdf(x) - u = 0" for a given sample of u~uniform(0,1)
                    x = x - (dist.obj_func(x, u) / dist.der_obj_func(x));
                    iter += 1;
                }
            }
        };
    }
    
    template <typename T, class DeviceType, class Distribution>
    class sample_its{
    public:
        using view_type = typename ippl::detail::ViewType<Vector<T, 1>, 1>::view_type;
        
    public:
        Distribution dist;
        int d;
        T rmax;
        T rmin;
        T umin, umax;
        unsigned int ntotal;
        
        template <class RegionLayout>
        sample_its(Distribution dist_, int d_, T rmax_, T rmin_, const RegionLayout& rlayout, unsigned int ntotal_)
        : dist(dist_),  // Initialize the 'dist' member with the provided 'dist_' argument
        d(d_),        // Initialize other members as needed
        rmax(rmax_),
        rmin(rmin_),
        ntotal(ntotal_){
            const typename RegionLayout::host_mirror_type regions = rlayout.gethLocalRegions();
            
            int rank = ippl::Comm->rank();
            T nr_m = dist.cdf(regions(rank)[d].max()) - dist.cdf(regions(rank)[d].min());
            T dr_m   = dist.cdf(rmax) - dist.cdf(rmin);
            umin = dist.cdf(regions(rank)[d].min());
            umax = dist.cdf(regions(rank)[d].max());
            
            T pnr = nr_m ;//std::accumulate(nr_m.begin(), nr_m.end(), 1.0, std::multiplies<T>());
            T pdr = dr_m ;//std::accumulate(dr_m.begin(), dr_m.end(), 1.0, std::multiplies<T>());
            
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
        
        struct fill_random {
            Distribution dist;
            view_type x;
            Generator<DeviceType> gen;
            T umin_m;
            T umax_m;
            uniform_real_distribution<DeviceType, T> unif;
            // Initialize all members
            KOKKOS_FUNCTION
            fill_random(Distribution dist_, view_type x_, int seed, T umin_, T umax_)
            : dist(dist_)
            , x(x_)
            , gen(seed)
            , umin_m(umin_)
            , umax_m(umax_)
            , unif(0.0, 1.0){}
            
            KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
                T u = 0.0;
                
                // get uniform random number between umin and umax
                u = (umax_m - umin_m) * unif(gen) + umin_m;
                
                // first guess for Newton-Raphson
                x(i) = dist.estimate(u);
                
                // solve
                ippl::random::detail::NewtonRaphson<T> solver;
                solver.solve(dist, x(i)[0], u);
            }
        };
        unsigned int getLocalNum() const { return nlocal_m; }
        void generate(view_type view, int seed) {
            Kokkos::parallel_for(nlocal_m, fill_random(dist, view, seed, umin, umax));
        }
        
        void sample_ITS(view_type view, int seed){
            generate(view, seed);
        }
    private:
        unsigned int nlocal_m;
    };
    
    // class of one-dimensional pdf
    template <typename T, unsigned DimP>
    class Distribution {
    public:
        using view_type = typename ippl::detail::ViewType<Vector<T, 1>, 1>::view_type;
        using FunctionPtr = T (*)(T, const T*);

        T par[DimP];
        
        Distribution(const T *par_) {
            cdfFunction = nullptr;
            pdfFunction = nullptr;
            estimationFunction = nullptr;
            for(unsigned int i=0; i<DimP; i++){
                par[i] = par_[i];
            }
        }
        
	    void setCdfFunction(FunctionPtr cdfFunc) {
		    cdfFunction = cdfFunc;
	    }
            
	    void setPdfFunction(FunctionPtr pdfFunc) {
		    pdfFunction = pdfFunc;
	    }
            
	    void setEstimationFunction(FunctionPtr estimationFunc) {
		    estimationFunction = estimationFunc;
	    }
        
        void setNormalDistribution() {
                cdfFunction = cdf_normal_wrapper;
                pdfFunction = pdf_normal_wrapper;
                estimationFunction = estimate_normal_wrapper;
        }

        KOKKOS_INLINE_FUNCTION T cdf(T x) const{
            return cdfFunction(x, par);
        }

        KOKKOS_INLINE_FUNCTION T pdf(T x) const {
            return pdfFunction(x, par);
        }

        KOKKOS_INLINE_FUNCTION T obj_func(T x, T u) const{
            return cdfFunction(x, par) - u;
        }
        
        KOKKOS_INLINE_FUNCTION T der_obj_func(T x) const{
            return pdfFunction(x, par);//pdf(x);
        }
        
        KOKKOS_INLINE_FUNCTION T estimate (T u) const{
            return estimationFunction(u, par);
        }
        
    private:
        FunctionPtr cdfFunction;
        FunctionPtr pdfFunction;
        FunctionPtr estimationFunction;
            
        static KOKKOS_INLINE_FUNCTION T cdf_normal_wrapper(T x, const T* params) {
                    T mean = params[0];
                    T stddev = params[1];
            return 0.5 * (1 + Kokkos::erf((x - mean) / (stddev * Kokkos::sqrt(2.0))));
        }

        static KOKKOS_INLINE_FUNCTION T pdf_normal_wrapper(T x, const T* params) {
                T mean = params[0];
                T stddev = params[1];
            return (1.0 / (stddev * Kokkos::sqrt(2 * M_PI))) * Kokkos::exp(-(x - mean) * (x - mean) / (2 * stddev * stddev));
        }

        static KOKKOS_INLINE_FUNCTION T estimate_normal_wrapper(T u, const T* params) {
                T mean = params[0];
                T stddev = params[1];
            return (Kokkos::sqrt(M_PI / 2.0) * (2.0 * u - 1.0)) * stddev + mean;
        }
            
    };
  }  // namespace random
}  // namespace ippl

#endif
