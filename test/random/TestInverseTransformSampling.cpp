#include "Ippl.h"

#include "Random/InverseTransformSampling.h"    

double cdf_y(double y, double alpha, double k) {
    return y + (alpha / k) * std::sin(k * y);
}
double pdf_y(double y, double alpha, double k) {
    return  (1.0 + alpha * Kokkos::cos(k * y));
}
double estimate_y(double u) {
    return u; // maybe E[x] is good enough as the first guess
}


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

        using InvTransSampl_t = ippl::random::InverseTransformSampling<double, 2, Kokkos::Serial>;
        
        // Define a distribution that is normal in dim=0, and harmonic in dim=1
        double mu = 1.0;
        double sd = 0.5;
        
        double pi    = Kokkos::numbers::pi_v<double>;
        double kw = 2.*pi/(rmax[1]-rmin[1])*4.0;
        double alpha = 0.5;
        
        ippl::random::Distribution<double, 2> dist;

        // For normal distr, user can use the pre-defined distr. from the class Distribution
	// Set a Normal Distribution for dimension 0 with mean mu and standard deviation sd
	dist.setNormalDistribution(0, mu, sd);

	// Set custom CDF and PDF functions for dimension 1
	dist.setCdfFunction(1, [alpha, kw](double y) { return cdf_y(y, alpha, kw);});
	dist.setPdfFunction(1, [alpha, kw](double y) { return pdf_y(y, alpha, kw);});
        dist.setEstimationFunction(1, [](double u) { return estimate_y(u);});

        InvTransSampl_t its(rmin, rmax, rlayout, dist, ntotal);

        using view_type = ippl::detail::ViewType<ippl::Vector<double, 2>, 1>::view_type;

        unsigned int nlocal = its.getLocalNum();

        view_type position("position", nlocal);

        its.generate(dist, position, 42);

        for (unsigned int i = 0; i < nlocal; ++i) {
            std::cout << ippl::Comm->rank() << " " << position(i)[0] << " " << position(i)[1]
                      << std::endl;
        }
    }
    ippl::finalize();

    return 0;
}
