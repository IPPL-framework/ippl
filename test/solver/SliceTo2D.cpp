#include "Ippl.h"

#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>
#include <fstream>
#include <map>


#include "Utility/IpplException.h"
#include "Utility/IpplTimings.h"
#include "Solver/FFTPoissonSolver.h"

template <unsigned Dim>
using Mesh_t = ippl::UniformCartesian<double, Dim>;

template <typename T, unsigned Dim>
using PLayout_t = typename ippl::ParticleSpatialLayout<T, Dim, Mesh_t<Dim>>;

template <unsigned Dim>
using Centering_t = typename Mesh_t<Dim>::DefaultCentering;

template <unsigned Dim>
using FieldLayout_t = ippl::FieldLayout<Dim>;

using size_type = ippl::detail::size_type;

template <typename T, unsigned Dim>
using Vector = ippl::Vector<T, Dim>;

template <typename T, unsigned Dim, class... ViewArgs>
using Field = ippl::Field<T, Dim, Mesh_t<Dim>, Centering_t<Dim>, ViewArgs...>;

template <typename T>
using ParticleAttrib = ippl::ParticleAttrib<T>;

template <typename T, unsigned Dim>
using Vector_t = ippl::Vector<T, Dim>;

template <unsigned Dim, class... ViewArgs>
using Field_t = Field<double, Dim, ViewArgs...>;

//template <typename T, unsigned Dim, class... ViewArgs>
//using VField_t = Field<Vector_t<T, Dim>, Dim, ViewArgs...>;
template <typename T>
using VectorField_t = typename ippl::Field<ippl::Vector<T, 2>, 2, Mesh_t<2>, Centering_t<2>>;

template <typename T>
using ScalarField_t = typename ippl::Field<T, 2, Mesh_t<2>, Centering_t<2>>;

template <typename T>
using Solver_t = ippl::FFTPoissonSolver<VectorField_t<T>, ScalarField_t<T>>;

//template <typename T = double, unsigned Dim = 2>
//using FFTSolver_t = ConditionalType<Dim == 2 || Dim == 3,
//                                    ippl::FFTPoissonSolver<VField_t<T, Dim>, Field_t<Dim>>>;


/*
//template <unsigned Dim = 3>
//using Mesh_t = ippl::UniformCartesian<double, Dim>;

//template <typename T, unsigned Dim>
//using playout_type = ippl::ParticleSpatialLayout<T, Dim>;

//using Centering_t = typename Mesh_t::DefaultCentering;


template <typename T, unsigned Dim = 3, class... ViewArgs>
using Field = ippl::Field<T, Dim, Mesh_t, Centering_t, ViewArgs...>;

template <unsigned Dim = 3, class... ViewArgs>
using Field_t = Field<double, Dim, ViewArgs...>;

template <typename T = double, unsigned Dim = 3, class... ViewArgs>
using VField_t = Field<Vector<T, Dim>, Dim, ViewArgs...>;
*/

//    constexpr unsigned sliceCount;
//    std::array<Field2D_t, sliceCount> slices;

//    Solver_t solver;

//    ScalarField_t<T> rho;
//    VectorField_t<T> E;
template <class PLayout>
class Solve25D : public ippl::ParticleBase<PLayout> {
    using Base = ippl::ParticleBase<PLayout>;

public:
    // kokkos array of these types below, but not phi because, I just use that once but I need to keep E_m and rho_m
    //VField_t<double, 2> E_m; // have an array of these, one for each slice Kokkos::Array<Field_t::view_type, slice_count> 
    // define the R (rho) field

    // need to define a Vfield and scalar field for E and rho
    // also define a kokkos array of vfields and scalarviews' views ie
    //Kokkos::Array<Field_t::view_type, slice_count> rhoViews; 

    // std::array<Scarlfield, slices> rhos; ie relace below with <- and similarly for E field
    //ScalarField_t<double> rho_m; // have an array of these, one for each slice Kokkos::Array<Field_t::view_type, slice_count>

    Field<double, 2> phi_m;

    //Vector<int, 3> nr_m;

    ippl::e_dim_tag decomp_m[3];

    Vector_t<double, 3> hr_m;
    Vector_t<double, 3> rmin_m;
    Vector_t<double, 3> rmax_m;

    std::string stype_m;

    double Q_m;

public:
    typedef double value_type;

    ParticleAttrib<double> q;                 // charge
    typename Base::particle_position_type P;  // particle velocity
    typename Base::particle_position_type R;  // particle position
    typename Base::particle_position_type E;  // electric field at particle position
    ParticleAttrib<int> mapping; // record the bin index of the i'th particle

    //constexpr unsigned sliceCount;
    //std::array<Field2D_t, sliceCount> slices;

    //Solver_t solver;

    //ScalarField_t<T> rho;
    //VectorField_t<T> E;
    Solve25D(PLayout& pl)
        : Base(pl) {
        // register the particle attributes
        registerAttributes();
    }

    Solve25D(PLayout& pl, Vector_t<double, 3>  hr, Vector_t<double, 3>  rmin, Vector_t<double, 3>  rmax,
                     ippl::e_dim_tag decomp[3], double Q, std::string solver)
        : Base(pl) 
        , hr_m(hr)
        , rmin_m(rmin)
        , rmax_m(rmax)
        , stype_m(solver)
        , Q_m(Q) {
        registerAttributes();
        for (unsigned int i = 0; i < 3; i++) {
            decomp_m[i] = decomp[i];
        }
    }

    void registerAttributes() {
        // register the particle attributes
        this->addAttribute(q);
        this->addAttribute(P);
        this->addAttribute(R);
        this->addAttribute(E);
        this->addAttribute(mapping);
    }

    ~Solve25D() {}

    template <size_t nSlices>
    void scatterToSlices(int numSlices, std::array<ScalarField_t<double>, nSlices>& rhos, Kokkos::Array<Field_t<2>::view_type, nSlices>& rhoViews) {

        // copy stuff from particleattrib.hpp
        static IpplTimings::TimerRef scatterTimer = IpplTimings::getTimer("scatter");
        IpplTimings::startTimer(scatterTimer);

        using mesh_type = typename Field_t<2>::Mesh_t;
        mesh_type& mesh = rhos[0].get_mesh(); // mesh is the same for all rho

        using vector_type = typename mesh_type::vector_type;
        const vector_type& dx = mesh.getMeshSpacing();

        FieldLayout_t<2>& layout = rhos[0].getLayout();

        const vector_type& origin = mesh.getOrigin();

        const vector_type invdx      = 1.0 / dx;
        const ippl::NDIndex<2>& lDom = layout.getLocalNDIndex();
        const int nghost             = rhos[0].getNghost();

        // loop to find min and max
        double sMin, sMax;
        sMin = sMax = R(0)[2]; // Initialize min and max with the first element

        for (size_t i = 1; i < this->getLocalNum(); i++) {
            if (R(i)[2] < sMin) {
                sMin = R(i)[2];
            } else if (R(i)[2] > sMax) {
                sMax = R(i)[2];
            }
        }
        std::cerr << "sMin = " << sMin << std::endl;
        std::cerr << "sMax = " << sMax << std::endl;
        double binWidth = (sMax - sMin) / numSlices;
        std::cerr << "binWidth = " << binWidth << std::endl;
       
        //Kokkos::Array<double, nSlices - 1> binBoundaries;
        //Kokkos::View<double*> binBoundaries;

        //Vector<Kokkos::View<double**>, numSlices>  binViews;
        //Vector_t binViews;

        //for (int i = 0; i < numSlices; i++){
        //    binBoundaries[i] = sMin + i * binWidth;
        //    std::cerr << "binBoundaries = " <<sMin + i * binWidth  << std::endl;
        //}

       // Kokkos::Array<unsigned int, 10000> mapping;
        //Kokkos::View<unsigned int*> mapping("mapping", this->getLocalNum());
        //Kokkos::View<unsigned int*> mapping;

        Kokkos::parallel_for(
            "Slice and scatter", this->getLocalNum(), KOKKOS_LAMBDA(const size_t i) {

                // mapping to record which particle corresponds to which bin 
                // ie the i'th particle corresponds to the n'th bin
                unsigned int n = (R(i)[2] - sMin) / binWidth; 

                mapping(i) = n;

                vector_type pos2D = {R(i)[0], R(i)[1]}; 

                // interpolation stuff
                vector_type l = (pos2D - origin) * invdx + 0.5;
                Vector<int, 2> index = l;
                Vector<double, 2> whi = l - index;
                Vector<double, 2> wlo = 1.0 - whi;
                Vector<size_t, 2> args = index - lDom.first() + nghost;

                const value_type &val = q(i);
                
                // check templtes arguments for make index sequence
                ippl::detail::scatterToField(std::make_index_sequence<1 << 2>{}, rhoViews[n], wlo, whi, args ,val);
                //ippl::detail::scatterToField(std::make_index_sequence<1 << 2>{}, rhoViews[n], wlo, whi, args ,val);
            });

        // Collect mapping values for writing to a file
        //std::vector<unsigned int> mappingValues(this->getLocalNum());
        //Kokkos::deep_copy(Kokkos::View<unsigned int*>::HostMirror(mappingValues.data(), this->getLocalNum()), mapping);
        // Write mapping values to a file
        std::ofstream outFile("mapping_output.csv");
        if (outFile.is_open()) {
            outFile << "index,n\n";
            for (size_t i = 0; i < this->getLocalNum(); i++) {
                outFile << i << "," << mapping(i) << "\n";
            }
            outFile.close();
        } else {
            std::cerr << "Unable to open file for writing." << std::endl;
        }
    }
};


std::vector<double> generateRandomPointOnCylinder(double centerX, double centerY, double radius, double height) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> angle_distribution(0.0, 2 * Kokkos::numbers::pi_v<double>);
    //std::uniform_real_distribution<double> height_distribution(0.0, height);
    std::normal_distribution<double> height_distribution(height/4, 0.05);
    std::uniform_real_distribution<double> radius_distribution(0.0, radius);

    double theta = angle_distribution(gen);
    double l = height_distribution(gen);
    double r = radius_distribution(gen);

    double x = centerX + r * std::cos(theta);
    double y = centerY + r * std::sin(theta);

    return {x, y, l};
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg(argv[0]);
        Inform msg2all(argv[0], INFORM_ALL_NODES);

        ippl::Vector<int, 3> nr = {std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3])};

        static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("mainTimer");
        IpplTimings::startTimer(mainTimer);
        //auto start                = std::chrono::high_resolution_clock::now();
        const unsigned int totalP = std::atoi(argv[5]);

        msg << "benchmarkUpdate" << endl
            << " Np= " << totalP << " grid = " << nr << endl;
        using bunch_type = Solve25D<PLayout_t<double, 3>>;

        std::unique_ptr<bunch_type> P;

        ippl::NDIndex<3> domain;
        for (unsigned i = 0; i < 2; i++) {
            domain[i] = ippl::Index(nr[i]);
        }

        ippl::e_dim_tag decomp[3];
        for (unsigned d = 0; d < 3; d++) {
            decomp[d] = ippl::PARALLEL;
        }

        // create mesh and layout objects for this problem domain
        Vector_t<double, 3> rmin(0.0);
        Vector_t<double, 3> rmax(1.0);
        double dx       = rmax[0] / double(nr[0]);
        double dy       = rmax[1] / double(nr[1]);
        double dz       = rmax[2] / double(nr[2]);
        Vector_t<double, 3> hr     = {dx, dy, dz};
        Vector_t<double, 3> origin = {rmin[0], rmin[1], rmin[2]};
        //double hr_min   = std::min({dx, dy, dz});
        //const double dt = 1.0;  // size of timestep

        Mesh_t<3> mesh(domain, hr, origin); // this has dim 3 as this is the 3D particle distribution
        FieldLayout_t<3> FL(domain, decomp);
        PLayout_t<double, 3> PL(FL, mesh);

        std::string solver = argv[6];   

        double Q = 1.0;
        P        = std::make_unique<bunch_type>(PL, hr, rmin, rmax, decomp, Q, solver);

        unsigned long int nloc = totalP / ippl::Comm->size();

        static IpplTimings::TimerRef particleCreation = IpplTimings::getTimer("particlesCreation");
        IpplTimings::startTimer(particleCreation);
        P->create(nloc);

        std::mt19937_64 eng[3];
        for (unsigned i = 0; i < 3; i++) {
            eng[i].seed(42 + i * 3);
            eng[i].discard(nloc * ippl::Comm->rank());
        }
        std::uniform_real_distribution<double> unif(0, 1);

        typename bunch_type::particle_position_type::HostMirror R_host = P->R.getHostMirror();

        double radius = 0.05;
        double length = 0.8;
        std::ofstream outfile("random_points_on_cylinder.csv");
        outfile << "x,y,z\n";

        if (!outfile) {
            std::cerr << "Error opening the output file." << std::endl;
            return 1;
        }

        double sum_coord = 0.0;
        double centreX = (rmin[0] + rmax[0]) / 2; // make sure that the test distribution in centred on the mesh origin
        double centreY = (rmin[1] + rmax[1]) / 2;
        for (unsigned long int i = 0; i < nloc; i++) {
            std::vector<double> point = generateRandomPointOnCylinder(centreX, centreY, radius, length);
            for (int d = 0; d < 3; d++) {
                R_host(i)[d] = point[d];
                sum_coord += R_host(i)[d];
            }
            outfile << point[0] << "," << point[1] << "," << point[2] << "\n";
        }
        
        double global_sum_coord = 0.0;
        MPI_Reduce(&sum_coord, &global_sum_coord, 1, MPI_DOUBLE, MPI_SUM, 0,
                   ippl::Comm->getCommunicator());

        if (ippl::Comm->rank() == 0) {
            std::cout << "Sum Coord: " << std::setprecision(16) << global_sum_coord << std::endl;
        }

        Kokkos::deep_copy(P->R.getView(), R_host);
        P->q = P->Q_m;// / totalP;
        IpplTimings::stopTimer(particleCreation);
        P->E = 0.0;

        //bunch_type bunchBuffer(PL);
        //static IpplTimings::TimerRef UpdateTimer = IpplTimings::getTimer("ParticleUpdate");
        //IpplTimings::startTimer(UpdateTimer);
        //PL.update(*P, bunchBuffer);
        //IpplTimings::stopTimer(UpdateTimer);

        msg << "particles created and initial conditions assigned " << endl;

        const size_t numSlices = 10; //std::atoi(argv[4]);

        // create mesh and initialise fields
        ippl::Index I(nr[0]);
        ippl::NDIndex<2> owned(I, I);

        // unit box
        double dxRho                      = 1.0 / nr[0];
        double dyRho                      = 1.0 / nr[1];
        ippl::Vector<double, 2> hxRho     = {dxRho, dyRho};
        ippl::Vector<double, 2> originRho = {0.0, 0.0};
        Mesh_t<2> meshRho(owned, hxRho, originRho);

        // all parallel layout, standard domain, normal axis order
        ippl::FieldLayout<2> layout(owned, decomp);
        
        std::array<ScalarField_t<double>, numSlices+1> rhos;
        Kokkos::Array<ScalarField_t<double>::view_type, numSlices+1> rhoViews;

        std::array<VectorField_t<double>, numSlices+1> Efields;
        Kokkos::Array<VectorField_t<double>::view_type, numSlices+1> EViews;

        for (size_t i = 0; i < numSlices+1; i++) {
            rhos[i] = 0.0; // reset rho field 

            rhos[i].initialize(meshRho, layout);
            rhoViews[i] = rhos[i].getView();

            Efields[i].initialize(meshRho, layout);
            EViews[i] = Efields[i].getView();
        }

        
        // call slice and scatter function
        P->scatterToSlices(numSlices, rhos, rhoViews); 

        // 2d solve 
        ippl::ParameterList params;

        // set the FFT parameters
        params.add("use_heffte_defaults", false);
        params.add("use_pencils", true);
        params.add("use_gpu_aware", true);
        params.add("comm", ippl::a2av);
        params.add("r2c_direction", 0);

        // define an FFTPoissonSolver object
        std::ofstream inputFileRho("input_rho.csv");

        // Write the view data to the output file
        inputFileRho << "i,j,val\n";
        for (size_t i = 0; i < rhos[7].getView().extent(0); i++) {
            for (size_t j = 0; j < rhos[7].getView().extent(1); j++) {
                inputFileRho << i <<"," << j<< "," << rhos[7](i, j) << "\n";
            }
        }
        inputFileRho.close();

        std::string precision = argv[6];

        if (precision == "DOUBLE") {
        
            params.add("algorithm", Solver_t<double>::HOCKNEY);
            // add output type
            params.add("output_type", Solver_t<double>::SOL_AND_GRAD);

            for (size_t i = 0; i < numSlices+1; i++) {
                Solver_t<double> FFTsolver(Efields[i], rhos[i], params);
                // solve the Poisson equation -> rho contains the solution (phi) now   
                FFTsolver.solve();
            }
        }
        
        // else {
            // doesnt work at the moment because the fields are init as double
            // would need to template all those types too.

        //    params.add("algorithm", Solver_t<float>::HOCKNEY);
            // add output type
        //    params.add("output_type", Solver_t<float>::SOL_AND_GRAD);

        //    Solver_t<float> FFTsolver(Efields[0], rhos[0], params);

        //    FFTsolver.solve();
        //}
        std::ofstream outputFileRho("output_rho.csv");

        // Write the view data to the output file
        outputFileRho << "i,j,val\n";
        for (size_t i = 0; i < rhos[7].getView().extent(0); i++) {
            for (size_t j = 0; j < rhos[7].getView().extent(1); j++) {
                outputFileRho << i <<"," << j<< "," << rhos[7](i, j) << "\n";
            }
        }
        // Close the output file
        outputFileRho.close();

        // 1D SPACE CHARGE CALCULATION START 

        // 1d solve before gather!!
        // 1) get num particles per slice from mapping - this is the longitudinal line density
        // 2) calculate the gradient of the line density wrt to the slicing axis
        // 3) compute geometery factor
        // 4) evaluate formula for E-z field

        // use mapping to deduce the number of particles per slice
        std::map<int, int> occurrences;

        // Count occurrences of each value
        //for (int num : P->mapping) {

        auto mappingView = P->mapping.getView();

        for (size_t i = 0; i < totalP; i++) {
            occurrences[mappingView(i)]++;
        }

        std::vector<double> particlesPerSlice;

        for (const auto& entry : occurrences) {
            std::cerr << "Bin index: " << entry.first << ", num particles: " << entry.second << std::endl;
            particlesPerSlice.push_back(entry.second); //  this is the longitudinal line density
        }

        std::vector<double> lineDensityGradient(particlesPerSlice.size());

        for (size_t i = 1; i < particlesPerSlice.size() - 1; ++i) {
            lineDensityGradient[i] = particlesPerSlice[i + 1] - particlesPerSlice[i - 1];

        }

        // Handle edge cases manually
        lineDensityGradient[0] = particlesPerSlice[1] - particlesPerSlice[0];
        lineDensityGradient[particlesPerSlice.size() - 1] = particlesPerSlice[particlesPerSlice.size() - 1] - particlesPerSlice[particlesPerSlice.size() - 2];

        std::ofstream outputFileLamba("output_lambda.csv");
        outputFileLamba << "index,lin_den,gradient\n";
        for (size_t i = 0; i < particlesPerSlice.size(); ++i) {
            outputFileLamba << i << "," << particlesPerSlice[i] << "," << lineDensityGradient[i] << "\n";
        }
        outputFileLamba.close();

        // parameters for 1d calc. eventually g should be selected by user, ie if user selects cyclindrical vacuum chamber then the g for a cyclindrical chmaber is used
        double d = 7e-2; // half gap between parallel plates
        // radius is bunch radius, defined above
        double g = 0.67 + 4 * log(d / (Kokkos::numbers::pi_v<double> * radius)); // geometery factor for parallel plates
        double elemCharge = 1.6e-19;
        std::cerr << "g = " << g << std::endl;
        double epsilon0 = 8.8541878128e-12;

        double gamma = 3/938.272 +1; // will need to be parsed from OPAL

        std::ofstream outputFileEz("output_Ez.csv");

        // Write the view data to the output file
        outputFileEz << "index,val\n";

        for (size_t i = 0; i < particlesPerSlice.size(); i++) {

            double Ez = elemCharge * (- g / (4 * Kokkos::numbers::pi_v<double> * epsilon0 * gamma*gamma)) * lineDensityGradient[i];
            outputFileEz << i << "," << Ez << "\n";
            //update value of E
            P->E(i)[2] = Ez;
        }
        outputFileEz.close();

        // 1D SPACE CHARGE CALCULATION END

        // mayn need to im0plemnt 2d interpolate/gather, whereas 1d is discrete 
        // for each particle, add the 2 E field compoents from the interpolated hockney E feld to the P-> attribute


        // gather 

    }
    ippl::finalize();
    return 0;
}

