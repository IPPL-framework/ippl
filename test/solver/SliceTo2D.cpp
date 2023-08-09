#include "Ippl.h"

#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "Utility/IpplTimings.h"

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

template <typename T, unsigned Dim, class... ViewArgs>
using VField_t = Field<Vector_t<T, Dim>, Dim, ViewArgs...>;

template <typename T>
using ScalarField_t = typename ippl::Field<T, 2, Mesh_t<2>, Centering_t<2>>;


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

        for (size_t i = 1; i <= this->getLocalNum(); i++) {
            if (R(i)[2] < sMin) {
                sMin = R(i)[2];
            } else if (R(i)[2] > sMax) {
                sMax = R(i)[2];
            }
        }
        std::cerr << "sMin = " << sMin << std::endl;
        std::cerr << "sMax = " << sMax << std::endl;
        double binWidth = (sMax - sMin) / numSlices;
        Kokkos::View<double*> binBoundaries;

        //Vector<Kokkos::View<double**>, numSlices>  binViews;
        //Vector_t binViews;

        for (int i = 0; i < numSlices; i++){
            binBoundaries[i] = sMin + i * binWidth;
        }

        Kokkos::View<int*> mapping;

        Kokkos::parallel_for(
            "Slice and scatter", this->getLocalNum(), KOKKOS_LAMBDA(const size_t i) {
                // Alex's suggestion on how to slice and project - not sure how this works
                // how does the unsigned n line determine an integer slice index
                // projection (and get slice index)
                 // ie cut off the 3rd axis, frenet - serret s, to tructate the slice to 2D
                //unsigned n = (P(2) - sMin) / binWidth; // stuff to define
               // mapping(idx) = n;

                // R(2) is the s coord of the bunch, bunch will need to be defined with this in mind when the bunch is made/ parsed
                for (int binIndx = 1; binIndx <= numSlices; binIndx++){ // don't need this if you do analytically?? see above 
                    if ((binBoundaries(binIndx - 1) <= R(i)[2]) && (R(i)[2] <= binBoundaries(binIndx))){

                        //binViews[binIndx - 1](i, j) = R(i)[0], R(i)[1];-

                        mapping(i) = binIndx; // map which particle belongs in which longitudinal bin
                    }
                }

                ippl::Vector<double, 2> pos2D = {R(i)[0], R(i)[1]}; 

                // interpolation stuff
                ippl::Vector<double, 2> l = (pos2D - origin) * invdx + 0.5;
                Vector<int, 2> index = l;
                Vector<double, 2> whi = l - index;
                Vector<double, 2> wlo = 1.0 - whi;
                Vector<size_t, 2> args = index - lDom.first() + nghost;

                const value_type &val = q(i);

                for (int n = 0; n < numSlices; n++){
                    ippl::detail::scatterToField(std::make_index_sequence<1 << 2>{}, rhoViews[n], wlo, whi, args ,val);
                }
            });
    }
};

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg(argv[0]);
        Inform msg2all(argv[0], INFORM_ALL_NODES);

        ippl::Vector<int, 2> nr = {std::atoi(argv[1]), std::atoi(argv[2])};

        static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("mainTimer");
        IpplTimings::startTimer(mainTimer);
        //auto start                = std::chrono::high_resolution_clock::now();
        const unsigned int totalP = std::atoi(argv[4]);

        msg << "benchmarkUpdate" << endl
            << " Np= " << totalP << " grid = " << nr << endl;
        using bunch_type = Solve25D<PLayout_t<double, 3>>;

        std::unique_ptr<bunch_type> P;

        ippl::NDIndex<3> domain;
        for (unsigned i = 0; i < 2; i++) {
            domain[i] = ippl::Index(nr[i]);
        }

        ippl::e_dim_tag decomp[3];
        for (unsigned d = 0; d < 3; ++d) {
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

        /*
         * In case of periodic BC's define
         * the domain with hr and rmin
         */
        std::string solver = argv[5];   

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

        const size_t numSlices = 10;//std::atoi(argv[3]);

        // create mesh and initialise fields
        ippl::Index I(nr[0]);
        ippl::NDIndex<2> owned(I, I);

        // unit box
        double dxRho                      = 1.0 / nr[0];
        ippl::Vector<double, 2> hxRho     = {dxRho, dxRho};
        ippl::Vector<double, 2> originRho = {0.0, 0.0};
        Mesh_t<2> meshRho(owned, hxRho, originRho);

        // all parallel layout, standard domain, normal axis order
        ippl::FieldLayout<2> layout(owned, decomp);
        
        std::array<ScalarField_t<double>, numSlices> rhos; // hard coded 10 slices -  repalce later
        Kokkos::Array<Field_t<2>::view_type, numSlices> rhoViews;

        std::array<VField_t<double, 2>, numSlices> Efields;
        Kokkos::Array<VField_t<double, 2>::view_type, numSlices> EViews;

        for (size_t i = 0; i < numSlices; i++) {
            rhos[i] = 0.0; // reset rho field 

            rhos[i].initialize(meshRho, layout);
            rhoViews[i] = rhos[i].getView();

            Efields[i].initialize(meshRho, layout);
            EViews[i] = Efields[i].getView();
        }

        // call slice and scatter function
        P->scatterToSlices(numSlices, rhos, rhoViews); 

        // 2d solve 

        // 1d solve before gather!!

        // for each particle, add the 1d solve component to the P->E abribute
        // for each particle, add the 2 E field compoents from the interpolated hockney E feld to the P-> attribute
        // gather 

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

