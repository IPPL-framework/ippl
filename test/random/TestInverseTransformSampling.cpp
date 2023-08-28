#include "Ippl.h"

#include "Random/InverseTransformSampling.h"


/*
KOKKOS_INLINE_FUNCTION double cdf_x(double x) {
     return 0.5 * (1.0 + std::erf(x / (sqrt(2.0))));
}

KOKKOS_INLINE_FUNCTION double pdf_x(double x) {
    return (1.0 / sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x);
}

KOKKOS_INLINE_FUNCTION double cdf_y(double y) {
   return 0.5 * (1.0 + std::erf(y / (sqrt(2.0))));
}

KOKKOS_INLINE_FUNCTION double pdf_y(double y) {
   return (1.0 / sqrt(2.0 * M_PI)) * std::exp(-0.5 * y * y);
}

KOKKOS_INLINE_FUNCTION double estimate_x(double u) {
   return (Kokkos::sqrt(M_PI / 2.0) * (2.0 * u - 1.0));
}
KOKKOS_INLINE_FUNCTION double estimate_y(double u) {
   return (Kokkos::sqrt(M_PI / 2.0) * (2.0 * u - 1.0)) ;
}
*/      
 
double cdf_x(double x, double mu, double sd) {
    return 0.5 * (1.0 + std::erf((x - mu) / (sd * sqrt(2.0))));
}

double pdf_x(double x, double mu, double sd) {
    return (1.0 / (sd * sqrt(2.0 * M_PI))) * std::exp(-0.5 * ((x - mu) / sd) * ((x - mu) / sd));
}

double cdf_y(double y, double mu, double sd) {
    return 0.5 * (1.0 + std::erf((y - mu) / (sd * sqrt(2.0))));
}

double pdf_y(double y, double mu, double sd) {
    return (1.0 / (sd * sqrt(2.0 * M_PI))) * std::exp(-0.5 * ((y - mu) / sd) * ((y - mu) / sd));
}
KOKKOS_INLINE_FUNCTION double estimate_x(double u, double mu, double sd) {
    return (Kokkos::sqrt(M_PI / 2.0) * (2.0 * u - 1.0)) * sd + mu;
}

KOKKOS_INLINE_FUNCTION double estimate_y(double u, double mu, double sd) {
    return (Kokkos::sqrt(M_PI / 2.0) * (2.0 * u - 1.0)) * sd + mu;
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

        
        //using normal_t = ippl::random::mpi::normal_distribution<Kokkos::Serial>;
        //normal_t dist[2];

        ippl::Vector<double, 2> mu, sd;
        //mu    = 0.5 * length + origin;
        mu[0] = 1.;//(rmin[0]+rmax[0])/2. + length[0]/8.;
        mu[1] = -1;// (rmin[1]+rmax[1])/2. -  length[1]/8.;
        sd[0] = 0.5;// 0.15 * length[0];
        sd[1] = 0.8;//0.05 * length[1];

        //dist[0] = normal_t(mu[0], sd[0]);
        //dist[1] = normal_t(mu[1], sd[1]);
        
        /*
        // Define the cumulative distribution function (CDF) for the X dimension
        auto cdf_x = [](double x, double mu, double sd) {
            return 0.5 * (1.0 + std::erf((x - mu) / (sd * sqrt(2.0))));
        };

        // Define the probability density function (PDF) for the X dimension
        auto pdf_x = [](double x, double mu, double sd) {
            return (1.0 / (sd * sqrt(2.0 * M_PI))) * std::exp(-0.5 * ((x - mu) / sd) * ((x - mu) / sd));
        };

        // Define the estimation function for the X dimension
        auto estimate_x = [](double u) {
            return (Kokkos::sqrt(M_PI / 2.0) * (2.0 * u - 1.0));
        };

	// Define the CDF for the Y dimension
        auto cdf_y = [](double y, double mu, double sd) {
            return 0.5 * (1.0 + std::erf((y - mu) / (sd * sqrt(2.0))));
        };

        // Define the PDF for the Y dimension
        auto pdf_y = [](double y, double mu, double sd) {
            return (1.0 / (sd * sqrt(2.0 * M_PI))) * std::exp(-0.5 * ((y - mu) / sd) * ((y - mu) / sd));
        };

        // Define the estimation function for the Y dimension
        auto estimate_y = [](double u) {
            return (Kokkos::sqrt(M_PI / 2.0) * (2.0 * u - 1.0));
        };
        */

	// Create a 2D normal distribution using the Distribution class
	//std::vector<std::function<double(double)>> cdfFunctions = {cdf_x, cdf_y};
	//std::vector<std::function<double(double)>> pdfFunctions = {pdf_x, pdf_y};
	// Create a 2D normal distribution using the Distribution class
        std::vector<std::function<double(double)>> cdfFunctions = {
            [mu_x = mu[0], sd_x = sd[0]](double x) { return cdf_x(x, mu_x, sd_x); },
            [mu_y = mu[1], sd_y = sd[1]](double y) { return cdf_y(y, mu_y, sd_y); }
        };
        std::vector<std::function<double(double)>> pdfFunctions = {
            [mu_x = mu[0], sd_x = sd[0]](double x) { return pdf_x(x, mu_x, sd_x); },
            [mu_y = mu[1], sd_y = sd[1]](double y) { return pdf_y(y, mu_y, sd_y); }
        };
        std::vector<std::function<double(double)>> estimationFunctions = {
            [mu_x = mu[0], sd_x = sd[0]](double u) { return estimate_x(u, mu_x, sd_x); },
            [mu_y = mu[1], sd_y = sd[1]](double u) { return estimate_y(u, mu_y, sd_y); }
        };
        
	ippl::random::Distribution<double, 2> dist(cdfFunctions, pdfFunctions, estimationFunctions);
    

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
