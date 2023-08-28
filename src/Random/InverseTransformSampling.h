#ifndef IPPL_INVERSE_TRANSFORM_SAMPLING_H
#define IPPL_INVERSE_TRANSFORM_SAMPLING_H

#include "Types/ViewTypes.h"

#include "Random/Generator.h"
#include "Random/Random.h"

namespace ippl {

    namespace random {

        namespace detail {

            template <typename T, unsigned Dim>
            struct NewtonRaphson {
                KOKKOS_FUNCTION
                NewtonRaphson() = default;

                KOKKOS_FUNCTION
                ~NewtonRaphson() = default;

                template <class Distribution>
	        KOKKOS_INLINE_FUNCTION void solve(Distribution dist, unsigned dim, T& x, T& u, T atol = 1.0e-12,
		                              unsigned int max_iter = 20) {
			unsigned int iter = 0;
			while (iter < max_iter && Kokkos::fabs(dist.obj_func(dim, x, u)) > atol) {
			    // The goal is to find x, such that "cdf(x) - u = 0" for a given sample of u. Here, dist.function = 2*(cdf(x) - u)
			    x = x - (dist.obj_func(dim, x, u) / dist.der_obj_func(dim, x));
			    iter += 1;
			}
	        }
            };
        }  // namespace detail

	template <typename T, unsigned Dim, class DeviceType>
	class InverseTransformSampling {
	public:
	    using view_type = typename ippl::detail::ViewType<Vector<T, Dim>, 1>::view_type;

	    template <class Distribution, class RegionLayout>
	    InverseTransformSampling(const Vector<T, Dim>& rmin, const Vector<T, Dim>& rmax,
		                     const RegionLayout& rlayout, Distribution dist,
		                     unsigned int ntotal) {
		const typename RegionLayout::host_mirror_type regions = rlayout.gethLocalRegions();

		int rank = ippl::Comm->rank();
		for (unsigned d = 0; d < Dim; ++d) {
		    nr_m[d] =
		        dist.cdf(d, regions(rank)[d].max()) - dist.cdf(d, regions(rank)[d].min());
		    dr_m[d]   = dist.cdf(d, rmax[d]) - dist.cdf(d, rmin[d]);
		    umin_m[d] = dist.cdf(d, regions(rank)[d].min());
		    umax_m[d] = dist.cdf(d, regions(rank)[d].max());
		}

		T pnr = std::accumulate(nr_m.begin(), nr_m.end(), 1.0, std::multiplies<T>());
		T pdr = std::accumulate(dr_m.begin(), dr_m.end(), 1.0, std::multiplies<T>());

		double factor = pnr / pdr;
		nlocal_m      = factor * ntotal;

		unsigned int ngobal = 0;
		MPI_Allreduce(&nlocal_m, &ngobal, 1, MPI_UNSIGNED_LONG, MPI_SUM,
		              ippl::Comm->getCommunicator());

		int rest = (int)(ntotal - ngobal);

		if (rank < rest) {
		    ++nlocal_m;
		}
	    }

	    unsigned int getLocalNum() const { return nlocal_m; }

	    template <class Distribution>
	    void generate(Distribution dist_, view_type view, int seed) {
		Kokkos::parallel_for(nlocal_m, fill_random<Distribution>(dist_, view, seed, umin_m, umax_m));
	    }

	    template <class Distribution>
	    struct fill_random {
		Distribution dist;

		Generator<DeviceType> gen;
		uniform_real_distribution<DeviceType, T> unif;
		Vector<T, Dim> umin, umax;

		// Output View for the random numbers
		view_type x;

		// Initialize all members
		KOKKOS_FUNCTION
		fill_random(Distribution dist_, view_type x_, int seed, Vector<T, Dim> umin_,
		            Vector<T, Dim> umax_)
		    : dist(dist_)
		    , gen(seed)
		    , unif(0.0, 1.0)
		    , umin(umin_)
		    , umax(umax_)
		    , x(x_) {
		    //for (unsigned int d = 0; d < Dim; ++d) {
		    //    dist = dist_;
		    //}
		}

		KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
		    T u = 0.0;

		    for (unsigned d = 0; d < Dim; ++d) {
		        // get uniform random number between umin and umax
		        u = (umax[d] - umin[d]) * unif(gen) + umin[d];

		        // first guess for Newton-Raphson
		        x(i)[d] = dist.estimate(d, u);

		        // solve
		        ippl::random::detail::NewtonRaphson<T, Dim> solver; // Add the dimension as a template parameter
		        solver.solve(dist, d, x(i)[d], u);
		    }
		}
	    };

	private:
	    unsigned int nlocal_m;
	    Vector<T, Dim> nr_m, dr_m, umin_m, umax_m;
	};

        
        // class of uncorollated multi-dimensional pdf
        template <typename T, unsigned Dim>
        class Distribution {
        public:
           // Constructor that takes arrays of user-defined CDF and PDF functions
           Distribution(std::vector<std::function<T(T)>> userCDFs, 
                              std::vector<std::function<T(T)>> userPDFs,
                              std::vector<std::function<T(T)>> userEstimations)
              : cdfFunctions(userCDFs), pdfFunctions(userPDFs) , estimationFunctions(userEstimations) {
              if (cdfFunctions.size() != Dim || pdfFunctions.size() != Dim || estimationFunctions.size() != Dim) {
                  throw std::invalid_argument("Invalid number of CDF or PDF functions provided.");
             }
           }
           // Member function to compute CDF for a specific dimension
           T cdf(unsigned dim, T x) {
              if (dim < Dim) {
                  return cdfFunctions[dim](x);
              } else {
                  throw std::invalid_argument("Invalid dimension specified.");
              }
           }
           // Member function to compute PDF for a specific dimension
           T pdf(unsigned dim, T x) {
              if (dim < Dim) {
                 return pdfFunctions[dim](x);
              } else {
                 throw std::invalid_argument("Invalid dimension specified.");
              }
           }
           // Member function to compute objective function CDF(x)-u for a specific dimension
           T obj_func(unsigned dim, T x, T u) {
             if (dim < Dim) {
                 return cdfFunctions[dim](x) - u;
             } else {
                throw std::invalid_argument("Invalid dimension specified.");
             }
           }
           // Member function to compute derivative of objective function CDF(x)-u for a specific dimension
           T der_obj_func(unsigned dim, T x) {
             if (dim < Dim) {
                 return pdf(dim, x);
             } else {
                throw std::invalid_argument("Invalid dimension specified.");
             }
           }
           T estimate(unsigned dim, T u) const{
             if (dim < Dim) {
                 return estimationFunctions[dim](u);
             } else {
                throw std::invalid_argument("Invalid dimension specified.");
             }
           }

        private:
    
           // Array of function pointers or lambdas for user to define
           std::vector<std::function<T(T)>> cdfFunctions;
           std::vector<std::function<T(T)>> pdfFunctions;
           std::vector<std::function<T(T)>> estimationFunctions;
        };

    }  // namespace random
}  // namespace ippl

#endif
