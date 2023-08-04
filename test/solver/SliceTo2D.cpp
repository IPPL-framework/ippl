#include "Ippl.h"

#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "Utility/IpplTimings.h"
//template <unsigned Dim = 3>
//using Mesh_t = ippl::UniformCartesian<double, Dim>;
constexpr unsigned Dim = 3;

typedef ippl::ParticleSpatialLayout<double, Dim> PLayout_t;
typedef ippl::UniformCartesian<double, Dim> Mesh_t;
typedef ippl::FieldLayout<Dim> FieldLayout_t;

template <typename T>
using ParticleAttrib = ippl::ParticleAttrib<T>;

//template <typename T, unsigned Dim = 2>
//using Vector_t = ippl::Vector<T, Dim>;
template <typename T, unsigned Dim>
using Vector = ippl::Vector<T, Dim>;

typedef Vector<double, Dim> Vector_t;

template <typename T, unsigned Dim>
using playout_type = ippl::ParticleSpatialLayout<T, Dim>;

//    constexpr unsigned sliceCount;
//    std::array<Field2D_t, sliceCount> slices;

//    Solver_t solver;

//    ScalarField_t<T> rho;
//    VectorField_t<T> E;
template <class PLayout>
class Solve25D : public ippl::ParticleBase<PLayout> {
    using Base = ippl::ParticleBase<PLayout>;

public:
    Vector<int, 3> nr_m;

    ippl::e_dim_tag decomp_m[3];

    Vector_t hr_m;
    Vector_t rmin_m;
    Vector_t rmax_m;

    std::string stype_m;

    double Q_m;

public:
    typedef double value_type;

    ParticleAttrib<double> q;                 // charge
    typename Base::particle_position_type P;  // particle velocity
    typename Base::particle_position_type R;  // particle position
    typename Base::particle_position_type E;  // electric field at particle position

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

    Solve25D(PLayout& pl, Vector_t  hr, Vector_t  rmin, Vector_t  rmax,
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
    }

    ~Solve25D() {}

    template <typename Field>
    void scatterToSlices(Field& f, int numSlices) {

        // copy stuff from particleattrib.hpp
        static IpplTimings::TimerRef scatterTimer = IpplTimings::getTimer("scatter");
        IpplTimings::startTimer(scatterTimer);

        using mesh_type       = typename Field::Mesh_t;
        const mesh_type& mesh = f.get_mesh();

        using vector_type = typename mesh_type::vector_type;

        const vector_type& dx     = mesh.getMeshSpacing();
        const vector_type& origin = mesh.getOrigin();
        const vector_type invdx   = 1.0 / dx;

        const FieldLayout_t& layout = f.getLayout();
        const ippl::NDIndex<3>& lDom   = layout.getLocalNDIndex();
        const int nghost                 = f.getNghost();

        // loop to find min and max
        double sMin, sMax;
        sMin = sMax = R(0)[2]; // Initialize min and max with the first element

        for (size_t i = 1; i <= P->getLocalNum(); ++i) {
            if (R(i)[2] < sMin) {
                sMin = R(i)[2];
            } else if (R(i)[2] > sMax) {
                sMax = R(i)[2];
            }
        }
        
        double binWidth = (sMax - sMin) / numSlices;
        Vector_t binBoundaries;

        Vector_t binViews;

        for (int i = 0; i < numSlices; i++){
            binBoundaries[i] = sMin + i * binWidth;
        }

        Vector_t mapping;

        Kokkos::parallel_for(
            "Slice and scatter", P->getLocalNum(), KOKKOS_LAMBDA(const size_t i, const size_t j) {
                // Alex's suggestion on how to slice and project - not sure how this works
                // how does the unsigned n line determine an integer slice index
                // projection (and get slice index)
                Vector_t pos2D = {R(0), R(1)}; // ie cut off the 3rd axis, frenet - serret s, to tructate the slice to 2D
                //unsigned n = (P(2) - sMin) / binWidth; // stuff to define
               // mapping(idx) = n;

                // R(2) is the s coord of the bunch, bunch will need to be defined with this in mind when the bunch is made/ parsed
                for (int binIndx = 1; binIndx <= numSlices; binIndx++){ // don't need this if you do analytically?? see above 
                    if ((binBoundaries[binIndx - 1] <= R(i)[2]) && (R(i)[2] <= binBoundaries[binIndx])){

                        // fix this
                        //binViews[binIndx - 1](i, j) = R(i)[0], R(i)[1];

                        mapping[i] = binIndx; // map which particle belongs in which longitudinal bin
                    }
                }

                // interpolation stuff
                vector_type l = (pos2D - origin) * invdx + 0.5;
                ippl::Vector<int, Field::dim> index = l;
                ippl::Vector<double, Field::dim> whi = l - index;
                ippl::Vector<double, Field::dim> wlo = 1.0 - whi;
                ippl::Vector<size_t, Field::dim> args = index - lDom.first() + nghost;

                const value_type &val = q(i);

                for (int n = 0; n <= numSlices; n++){
                    ippl::detail::scatterToField(
                        std::make_index_sequence<1 << Field::dim>{}, 
                        binViews[n],
                        wlo, whi, args, val);
                }
            });
    }
};

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg(argv[0]);
        Inform msg2all(argv[0], INFORM_ALL_NODES);

        ippl::Vector<int, 3> nr = {std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3])};

        static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("mainTimer");
        IpplTimings::startTimer(mainTimer);
        //auto start                = std::chrono::high_resolution_clock::now();
        const unsigned int totalP = std::atoi(argv[4]);
        const unsigned int nt     = std::atoi(argv[5]);

        msg << "benchmarkUpdate" << endl
            << "nt " << nt << " Np= " << totalP << " grid = " << nr << endl;
        using bunch_type = Solve25D<PLayout_t>;

        std::unique_ptr<bunch_type> P;

        ippl::NDIndex<3> domain;
        for (unsigned i = 0; i < 3; i++) {
            domain[i] = ippl::Index(nr[i]);
        }

        ippl::e_dim_tag decomp[3];
        for (unsigned d = 0; d < 3; ++d) {
            decomp[d] = ippl::PARALLEL;
        }

        // create mesh and layout objects for this problem domain
        Vector_t rmin(0.0);
        Vector_t rmax(1.0);
        double dx       = rmax[0] / double(nr[0]);
        double dy       = rmax[1] / double(nr[1]);
        double dz       = rmax[2] / double(nr[2]);
        Vector_t hr     = {dx, dy, dz};
        Vector_t origin = {rmin[0], rmin[1], rmin[2]};
        //double hr_min   = std::min({dx, dy, dz});
        //const double dt = 1.0;  // size of timestep

        Mesh_t mesh(domain, hr, origin);
        FieldLayout_t FL(domain, decomp);
        PLayout_t PL(FL, mesh);

        /*
         * In case of periodic BC's define
         * the domain with hr and rmin
         */
        std::string solver = argv[6];

        double Q = 1e6;
        P        = std::make_unique<bunch_type>(PL, hr, rmin, rmax, decomp, Q, solver);

        unsigned long int nloc = totalP / ippl::Comm->size();

        static IpplTimings::TimerRef particleCreation = IpplTimings::getTimer("particlesCreation");
        IpplTimings::startTimer(particleCreation);
        P->create(nloc);

        std::mt19937_64 eng[3];
        for (unsigned i = 0; i < 3; ++i) {
            eng[i].seed(42 + i * 3);
            eng[i].discard(nloc * ippl::Comm->rank());
        }
        std::uniform_real_distribution<double> unif(0, 1);

        typename bunch_type::particle_position_type::HostMirror R_host = P->R.getHostMirror();

        double sum_coord = 0.0;
        for (unsigned long int i = 0; i < nloc; i++) {
            for (int d = 0; d < 3; d++) {
                R_host(i)[d] = unif(eng[d]);
                sum_coord += R_host(i)[d];
            }
        }
        double global_sum_coord = 0.0;
        MPI_Reduce(&sum_coord, &global_sum_coord, 1, MPI_DOUBLE, MPI_SUM, 0,
                   ippl::Comm->getCommunicator());

        if (ippl::Comm->rank() == 0) {
            std::cout << "Sum Coord: " << std::setprecision(16) << global_sum_coord << std::endl;
        }

        Kokkos::deep_copy(P->R.getView(), R_host);
        P->q = P->Q_m / totalP;
        IpplTimings::stopTimer(particleCreation);
        P->E = 0.0;

        bunch_type bunchBuffer(PL);
        static IpplTimings::TimerRef UpdateTimer = IpplTimings::getTimer("ParticleUpdate");
        IpplTimings::startTimer(UpdateTimer);
        PL.update(*P, bunchBuffer);
        IpplTimings::stopTimer(UpdateTimer);

        msg << "particles created and initial conditions assigned " << endl;
    }
    ippl::finalize();
    return 0;
}


    /*
    void scatterToSlices(int numSlices) {
        auto Rview = P->R.getView(); // positions
        auto Pview = P->P.getView(); // velocities
        auto Eview = P->E.getView(); // E field

        //ippl::Vector<T, P->getLocalNum()> r; // these should probably be views
        //ippl::Vector<T, P->getLocalNum()> phi;
        //Kokkos::View<T*> sView; // this is the y comp of the particles (probs) based on how I define it when I parse P to this class

        sMin = sView.min(); // proabably won't work fix later
        sMax = sView.max(); // proabably won't work fix later

        T binWidth = (sMax - sMin) / numSlices;
        ippl::Vector<T, numSlices> binBoundaries;
        // other way around 
        Kokkos::View<ippl::Vector<T>, numSlices> bins;// rename to binView = //  is this the right way to make a view of ippl::Vectors?

        for (i = 0; i < numSlices; i++){
            binBoundaries[i] = sMin + i * binWidth;
        }

        ippl::Vector<Kokkos::view<T**>, P->getLocalNum()> slices;
        Kokkos::parallel_for(
            "Slice and scatter", P->getLocalNum(), KOKKOS_LAMBDA(const size_t i) {
                for (binIndx = 1; binIndx <= numSlices; binIndx++){ // don't need this if you do analytically??
                    if ((binBoundaries[binIndx - 1] <= sView(i)) && (sView(i) <= binBoundaries[binIndx])){
                        bins(binIndx - 1)[i] = sView(i);
                        mapping[i] = {binIndex, i};

                    }
                }
                ippl::detail::scatterToSlices(binView[slice_index]
            , x, ...); // copy rest from particle attrib

        });


        //x = project particle position
        //slice_index = position.s

    }
    

    void run() {
        // init particles
    // define bunch distribution here 
    using bunch_type = ChargedParticles<PLayout_t<T, Dim>, T, Dim>;

    std::unique_ptr<bunch_type> P;

        for (t = 0 to whatever) {
            scatterToSlices();
            for (auto& slice : slices) {
                solve(slice);
            }
            rho = collate from slices

            kick
            drift
            whatever
        }
    }
};
*/

