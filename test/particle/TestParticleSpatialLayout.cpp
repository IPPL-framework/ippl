#include <random>

#include "Ippl.h"

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

    void update() {
        PLayout& layout = this->getLayout();
        layout.update(*this);
//         ippl::ParticleBase<PLayout>::update<Bunch<PLayout> >();
    }
};

int main(int argc, char *argv[]) {
    Ippl ippl(argc, argv);

    typedef ippl::ParticleSpatialLayout<double, 3> playout_type;
    typedef Bunch<playout_type> bunch_type;

    constexpr unsigned int dim = 3;

    int pt = 8;
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
    int nParticles = 8;

    if (nParticles % nRanks > 0) {
        if (Ippl::Comm->rank() == 0) {
            std::cerr << nParticles << " not a multiple of " << nRanks << std::endl;
        }
        return 0;
    }

    bunch.create(nParticles / nRanks);

    std::mt19937_64 eng(Ippl::Comm->rank());
    std::uniform_real_distribution<double> unif(0, 1);

    typename bunch_type::particle_position_type::HostMirror R_host = bunch.R.getHostMirror();
    for (size_t i = 0; i < bunch.getLocalNum(); ++i) {
        ippl::Vector<double, dim> r = {unif(eng), unif(eng), unif(eng)};
        R_host(i) = r;
    }

    bunch.Q = 1.0;
    Ippl::Comm->barrier();
    Kokkos::deep_copy(bunch.R.getView(), R_host);

    typedef ippl::detail::RegionLayout<double, 3, Mesh_t> RegionLayout_t;
    RegionLayout_t RLayout = pl.getRegionLayout();

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
    typedef ippl::ParticleAttrib<double> Q_t;
    Q_t::view_type::host_mirror_type Q_host = bunch.Q.getHostMirror();

    Kokkos::deep_copy(Q_host, bunch.Q.getView());
    if (Ippl::Comm->rank() == 0) {
        std::cout << "Before update:" << std::endl;
    }

    for (int rank = 0; rank < Ippl::Comm->size(); ++rank) {
        if (Ippl::Comm->rank() == rank) {
            std::cout << "------------" << std::endl
                      << "Rank " << rank << std::endl;
            for (size_t i = 0; i < bunch.getLocalNum(); ++i) {
                std::cout << ID_host(i) << " " << R_host(i) << " " << Q_host(i) << " " << ER_host(i) << std::endl;
            }
        }
        Ippl::Comm->barrier();
    }

    std::cout << layout << std::endl;
    std::cout << RLayout << std::endl;

    bunch.update();

    Ippl::Comm->barrier();

    Kokkos::resize(R_host, bunch.R.size());
    Kokkos::deep_copy(R_host, bunch.R.getView());

    Kokkos::resize(ID_host, bunch.ID.size());
    Kokkos::deep_copy(ID_host, bunch.ID.getView());
    
    Kokkos::resize(Q_host, bunch.Q.size());
    Kokkos::deep_copy(Q_host, bunch.Q.getView());
    
    Kokkos::resize(ER_host, bunch.expectedRank.size());
    Kokkos::deep_copy(ER_host, bunch.expectedRank.getView());

    if (Ippl::Comm->rank() == 0) {
        std::cout << "After update:" << std::endl;
    }

    for (int rank = 0; rank < Ippl::Comm->size(); ++rank) {
        if (Ippl::Comm->rank() == rank) {
            std::cout << "------------" << std::endl
                      << "Rank " << rank << std::endl;
            for (size_t i = 0; i < bunch.getLocalNum(); ++i) {
                std::cout << ID_host(i) << " " << R_host(i) << " " << Q_host(i) << " " << ER_host(i) << std::endl;
            }
        }
        Ippl::Comm->barrier();
    }

    return 0;
}
