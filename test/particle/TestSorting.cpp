#include "Ippl.h"
#include <Kokkos_Sort.hpp>

#include <random>

template <class PLayout>
struct Bunch : public ippl::ParticleBase<PLayout> {
    Bunch(PLayout& playout)
        : ippl::ParticleBase<PLayout>(playout) {
        this->addAttribute(Q);
    }

    ~Bunch() {}

    typedef ippl::ParticleAttrib<double> charge_container_type;
    charge_container_type Q;
};

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        typedef ippl::ParticleSpatialLayout<double, 3> playout_type;
        typedef Bunch<playout_type> bunch_type;
        using Mesh_t      = ippl::UniformCartesian<double, 3>;
        using Centering_t = Mesh_t::DefaultCentering;

        int pt = 64;
        ippl::Index I(pt);
        ippl::NDIndex<3> owned(I, I, I);

        std::array<bool, 3> isParallel;
        isParallel.fill(true);

        // all parallel layout, standard domain, normal axis order
        ippl::FieldLayout<3> layout(MPI_COMM_WORLD, owned, isParallel);

        double dx                      = 1.0 / double(pt);
        ippl::Vector<double, 3> hx     = {dx, dx, dx};
        ippl::Vector<double, 3> origin = {0, 0, 0};
        Mesh_t mesh(owned, hx, origin);

        playout_type pl(layout, mesh);

        bunch_type bunch(pl);
        typedef ippl::Field<double, 3, Mesh_t, Centering_t> field_type;

        field_type field;

        field.initialize(mesh, layout);

        bunch.setParticleBC(ippl::BC::PERIODIC);

        int nRanks              = ippl::Comm->size();
        unsigned int nParticles = std::pow(64, 3);

        if (nParticles % nRanks > 0) {
            if (ippl::Comm->rank() == 0) {
                std::cerr << nParticles << " not a multiple of " << nRanks << std::endl;
            }
            return 0;
        }

        unsigned int nLoc = nParticles / nRanks;

        bunch.create(nLoc);

        std::mt19937_64 eng;
        eng.seed(42);
        eng.discard(nLoc * ippl::Comm->rank());
        std::uniform_real_distribution<double> unif(hx[0] / 2, 1 - (hx[0] / 2));

        typename bunch_type::particle_position_type::HostMirror R_host = bunch.R.getHostMirror();
        double sum_coord                                               = 0.0;
        for (unsigned int i = 0; i < nLoc; ++i) {
            ippl::Vector<double, 3> r = {unif(eng), unif(eng), unif(eng)};
            R_host(i)                 = r;
            sum_coord += r[0] + r[1] + r[2];
        }
        Kokkos::deep_copy(bunch.R.getView(), R_host);

        double global_sum_coord = 0.0;
        ippl::Comm->reduce(sum_coord, global_sum_coord, 1, std::plus<double>());

        if (ippl::Comm->rank() == 0) {
            std::cout << "Sum coord: " << global_sum_coord << std::endl;
        }

        bunch.update();
        nLoc = bunch.getLocalNum(); // update might have changed the number of particles


        using hash_type = typename bunch_type::charge_container_type::hash_type;
        Inform msg("ParticleSorting");

        // Sample random charges uniformly (use the same eng from before)
        std::uniform_real_distribution<double> unif_charge(0.5, 10.5);
        typename bunch_type::charge_container_type::HostMirror Q_host = bunch.Q.getHostMirror();
        for (unsigned int i = 0; i < nLoc; ++i) { Q_host(i) = unif_charge(eng); }
        Kokkos::deep_copy(bunch.Q.getView(), Q_host);
        msg << "Charges assigned." << endl;

        // Generate index array
        msg << "Sorting particles by charge." << endl;
        hash_type sortedIndexArr("indices", nLoc);
        Kokkos::View<double*> Q_view = bunch.Q.getView();
        
        // Mirror views for keys and indices --> sorting on host is easier...
        auto host_keys = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Q_view);
        auto host_indices = Kokkos::create_mirror_view(sortedIndexArr);
        // Perform argsort on the host
        std::sort(host_indices.data(), host_indices.data() + nLoc,
                  [&](const size_t i, const size_t j) { return host_keys(i) < host_keys(j); });
        // Copy the sorted indices back to the device
        Kokkos::deep_copy(sortedIndexArr, host_indices);


        // Now we can apply the index array to all attributes
        msg << "Applying sorting permutation to all attributes." << endl;
        bunch.template forAllAttributes([&]<typename Attribute>(Attribute*& attribute) {
            // Ensure indices are in the correct memory space --> copies data ONLY when different memory spaces, so should be efficient
            using memory_space    = typename Attribute::memory_space;
            auto indices_device = Kokkos::create_mirror_view_and_copy(memory_space{}, sortedIndexArr);

            attribute->pack(indices_device);
            attribute->unpack(nLoc, true);
        });

        // Check if the sorting was successful
        Q_view = bunch.Q.getView();
        bool sorted = true;
        Kokkos::parallel_reduce("CheckSorted", nLoc - 1, KOKKOS_LAMBDA(const size_t& i, bool& update) {
            if (Q_view(i) > Q_view(i + 1)) update = false;
        }, Kokkos::LAnd<bool>(sorted));
        
        // Debug output
        ippl::Comm->barrier();
        msg << "Checking if charges are sorted: " << sorted << endl;
        Inform msg2All("ParticleSorting", INFORM_ALL_NODES);
        if (!sorted) {
            msg2All << "Sorting failed." << endl;
            ippl::Comm->abort();
        } else {
            msg2All << "Sorting successful." << endl;
        }
    }
    ippl::finalize();

    return 0;
}
