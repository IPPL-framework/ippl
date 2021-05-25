#include <random>

#include "Ippl.h"
#include "Utility/IpplTimings.h"

template<class PLayout>
struct Bunch : public ippl::ParticleBase<PLayout>
{

    Bunch(PLayout& playout)
    : ippl::ParticleBase<PLayout>(playout)
    {
        this->addAttribute(expectedRank);
        this->addAttribute(Q);
    }

    ~Bunch(){ }

    typedef ippl::ParticleAttrib<int> rank_type;
    typedef ippl::ParticleAttrib<double> charge_type;
    rank_type expectedRank;
    charge_type Q;

    //void update() {
    //    PLayout& layout = this->getLayout();
    //    layout.update(*this);
    //}
};

int main(int argc, char *argv[]) {
    Ippl ippl(argc, argv);
    Inform msg("PreallocationBuffer");

    static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("mainTimer");
    IpplTimings::startTimer(mainTimer);
    typedef ippl::ParticleSpatialLayout<double, 3> playout_type;
    typedef Bunch<playout_type> bunch_type;

    constexpr unsigned int dim = 3;

    Inform m("TestParticleSpatialLayout: ");
    int pt = 32;
    ippl::Index I(pt);
    ippl::NDIndex<dim> owned(I, I, I);

    ippl::e_dim_tag allParallel[dim];    // Specifies SERIAL, PARALLEL dims
    for (unsigned int d=0; d<dim; d++)
        allParallel[d] = ippl::PARALLEL;

    ippl::FieldLayout<dim> layout(owned,allParallel);

    double dx = 1.0 / double(pt);
    ippl::Vector<double, 3> hx = {dx, dx, dx};
    ippl::Vector<double, 3> origin = {0, 0, 0};
    typedef ippl::UniformCartesian<double, 3> Mesh_t;
    Mesh_t mesh(owned, hx, origin);

    playout_type pl(layout, mesh);

    bunch_type bunch(pl);

    using BC = ippl::BC;

    bunch_type::bc_container_type bcs = {
        BC::PERIODIC,
        BC::PERIODIC,
        BC::PERIODIC,
        BC::PERIODIC,
        BC::PERIODIC,
        BC::PERIODIC
    };

    bunch.setParticleBC(bcs);

    int nRanks = Ippl::Comm->size();
    unsigned int nParticles = 6400000;//std::pow(32, 3);

    if (nParticles % nRanks > 0) {
        if (Ippl::Comm->rank() == 0) {
            std::cerr << nParticles << " not a multiple of " << nRanks << std::endl;
        }
        return 0;
    }

    bunch.create(nParticles / nRanks);

#ifdef KOKKOS_ENABLE_CUDA
    int id = -1;
    auto err = cudaGetDevice(&id);
    if (err != cudaSuccess) printf("kernel cuda error: %d, %s\n", (int)err, cudaGetErrorString(err));
    std::cout << "Rank " << Ippl::Comm->rank() << " has device " << id << "\n";
#endif

    std::mt19937_64 eng(Ippl::Comm->rank());
    std::uniform_real_distribution<double> unif0(0.1, 0.4);
    std::uniform_real_distribution<double> unif1(0.6, 0.9);

    typename bunch_type::particle_position_type::HostMirror R_host = bunch.R.getHostMirror();
    for (size_t i = 0; i < 100000; ++i) {
        ippl::Vector<double, dim> r = {unif0(eng), unif0(eng), unif0(eng)};
        R_host(i) = r;
    }
    for (size_t i = 100000; i < 200000; ++i) {
        ippl::Vector<double, dim> r = {unif0(eng), unif0(eng), unif1(eng)};
        R_host(i) = r;
    }
    for (size_t i = 200000; i < 300000; ++i) {
        ippl::Vector<double, dim> r = {unif0(eng), unif1(eng), unif0(eng)};
        R_host(i) = r;
    }
    for (size_t i = 300000; i < 400000; ++i) {
        ippl::Vector<double, dim> r = {unif0(eng), unif1(eng), unif1(eng)};
        R_host(i) = r;
    }

//////////////////right half of the domain///////////////////////////////////
    for (size_t i = 400000; i < 500000; ++i) {
        ippl::Vector<double, dim> r = {unif1(eng), unif0(eng), unif0(eng)};
        R_host(i) = r;
    }
    for (size_t i = 500000; i < 600000; ++i) {
        ippl::Vector<double, dim> r = {unif1(eng), unif0(eng), unif1(eng)};
        R_host(i) = r;
    }
    for (size_t i = 600000; i < 700000; ++i) {
        ippl::Vector<double, dim> r = {unif1(eng), unif1(eng), unif0(eng)};
        R_host(i) = r;
    }
    for (size_t i = 700000; i < 800000; ++i) {
        ippl::Vector<double, dim> r = {unif1(eng), unif1(eng), unif1(eng)};
        R_host(i) = r;
    }

    bunch.Q = 1.0;
    Ippl::Comm->barrier();
    //Kokkos::deep_copy(bunch.R.getView(), R_host);

    typedef ippl::detail::RegionLayout<double, 3, Mesh_t> RegionLayout_t;
    RegionLayout_t RLayout = pl.getRegionLayout();

    std::cout << RLayout << std::endl;

    auto& positions = bunch.R.getView();
    typename RegionLayout_t::view_type Regions = RLayout.getdLocalRegions();
    using size_type = typename RegionLayout_t::view_type::size_type;
    using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
    typedef ippl::ParticleAttrib<int> ER_t;
    ER_t::view_type ER = bunch.expectedRank.getView();

    Kokkos::parallel_for("Expected Rank",
            mdrange_type({0, 0},
                         {ER.extent(0), Regions.extent(0)}),
            KOKKOS_LAMBDA(const size_t i, const size_type j) {
                bool x_bool = false;
                bool y_bool = false;
                bool z_bool = false;
                if((positions(i)[0] >= Regions(j)[0].min()) &&
                   (positions(i)[0] <= Regions(j)[0].max())) {
                    x_bool = true;
                }
                if((positions(i)[1] >= Regions(j)[1].min()) &&
                   (positions(i)[1] <= Regions(j)[1].max())) {
                    y_bool = true;
                }
                if((positions(i)[2] >= Regions(j)[2].min()) &&
                   (positions(i)[2] <= Regions(j)[2].max())) {
                    z_bool = true;
                }
                if(x_bool && y_bool && z_bool){
                    ER(i) = j;
                }
        });
    Kokkos::fence();

    typename bunch_type::particle_index_type::HostMirror ID_host = bunch.ID.getHostMirror();
    Kokkos::deep_copy(ID_host, bunch.ID.getView());

    ER_t::view_type::host_mirror_type ER_host = bunch.expectedRank.getHostMirror();
    Kokkos::deep_copy(ER_host, bunch.expectedRank.getView());
//     typedef ippl::ParticleAttrib<double> Q_t;

    if (Ippl::Comm->rank() == 0) {
        std::cout << "Before update:" << std::endl;
    }


    //std::cout << layout << std::endl;
    bunch_type bunchBuffer(pl);
    bunchBuffer.create(100000);

    int nsteps = 300;

    for (int nt=0; nt < nsteps; ++nt) {

        Kokkos::deep_copy(bunch.R.getView(), R_host);
        static IpplTimings::TimerRef UpdateTimer = IpplTimings::getTimer("Update");
        IpplTimings::startTimer(UpdateTimer);
        //bunch.update();
        pl.update(bunch, bunchBuffer);
        //pl.update(bunch);
        IpplTimings::stopTimer(UpdateTimer);
        msg << "Update: " << nt+1 << endl;
        //Kokkos::resize(R_host, bunch.R.size());
        //Kokkos::deep_copy(R_host, bunch.R.getView());
        Ippl::Comm->barrier();

    }



    //Kokkos::resize(ID_host, bunch.ID.size());
    //Kokkos::deep_copy(ID_host, bunch.ID.getView());
    //
    //
    //Kokkos::resize(ER_host, bunch.expectedRank.size());
    //Kokkos::deep_copy(ER_host, bunch.expectedRank.getView());

    //if (Ippl::Comm->rank() == 0) {
    //    std::cout << "After update:" << std::endl;
    //}

    //for (size_t i = 0; i < bunch.getLocalNum(); ++i) {
    //    if(Ippl::Comm->rank() != ER_host(i)) {
    //        std::cout << "Particle with ID: " << ID_host(i) << " "
    //                  << "has wrong rank!" << std::endl;
    //        }
    //    }
    //Ippl::Comm->barrier();

    unsigned int Total_particles = 0;
    unsigned int local_particles = bunch.getLocalNum();

    MPI_Reduce(&local_particles, &Total_particles, 1,
                MPI_UNSIGNED, MPI_SUM, 0, Ippl::getComm());
    if (Ippl::Comm->rank() == 0) {
        std::cout << "All expected ranks correct!!" << std::endl;

        std::cout << "Total particles before: " << nParticles
                  << " " << "after: " << Total_particles << std::endl;
    }
    IpplTimings::stopTimer(mainTimer);
    IpplTimings::print();
    IpplTimings::print(std::string("timing.dat"));
    return 0;
}
