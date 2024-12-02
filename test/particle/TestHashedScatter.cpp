#include "Ippl.h"

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

        int pt = 512;
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
        unsigned int nParticles = std::pow(256, 3);

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

        Inform msg("TestHashedScatter");
        Inform msg2All("TestHashedScatter", INFORM_ALL_NODES);

        /*
        First test for custom range policy: Only scatter pt/2 particles and compare to the 
        total sum of the field.
        For this test, charges need to be all the same. 
        */
        {
            bunch.Q = 1.0;
            bunch.update();
            field = 0.0;
            int NScatterred = nLoc / 2 + ippl::Comm->rank();
            
            double Q_total = 1.0 * NScatterred;
            ippl::Comm->allreduce(Q_total, 1, std::plus<double>());
            
            Kokkos::RangePolicy<> policy(0, NScatterred);
            scatter(bunch.Q, field, bunch.R, policy);

            msg << "---- Testing scatter with custom range policy ----" << endl;
            ippl::Comm->barrier();

            double Total_charge_field = field.sum();

            msg2All << "Total charge in the field:     " << Total_charge_field << endl;
            msg2All << "Total charge of the particles: " << Q_total << endl;
            msg2All << "Error:                         --> " << std::fabs(Q_total - Total_charge_field) << endl;
        }
        ippl::Comm->barrier();
        
        
        /*
        Second test for custom hash_type: Assign random charges, create an index array, shuffle it 
        and scatter the first half of the particles. Then compute the total charge in a loop and compare
        it to the sum from the field. 
         */
        using hash_type = typename bunch_type::charge_container_type::hash_type;
        {
            // Sample random charges uniformly (use the same eng from before)
            std::uniform_real_distribution<double> unif_charge(0.5, 1.5);
            typename bunch_type::charge_container_type::HostMirror Q_host = bunch.Q.getHostMirror();
            for (unsigned int i = 0; i < nLoc; ++i) { Q_host(i) = unif_charge(eng); }
            Kokkos::deep_copy(bunch.Q.getView(), Q_host);
            
            bunch.update();
            
            // Reset the field
            int NScatterred = nLoc / 2 + ippl::Comm->rank();
            field = 0.0;

            // Create index array using hash_type
            std::vector<int> host_indices(nLoc);
            std::iota(host_indices.begin(), host_indices.end(), 0);
            std::shuffle(host_indices.begin(), host_indices.end(), eng);

            hash_type hash("indexArray", nLoc);
            auto hash_host = Kokkos::create_mirror_view(hash);                          // Create a host mirror of the hash array
            for (unsigned int i = 0; i < nLoc; ++i) { hash_host(i) = host_indices[i]; } // Fill the host mirror with the shuffled index array
            Kokkos::deep_copy(hash, hash_host);                                         // Copy the shuffled host mirror back to the device 

            // The custom range policy
            Kokkos::RangePolicy<> policy(0, NScatterred);

            // Compute total scattered charge manually
            double Q_total = 0.0;
            auto viewQ = bunch.Q.getView();
            Kokkos::parallel_reduce("computeTotalCharge", policy, KOKKOS_LAMBDA(const size_t i, double& val) {
                val += viewQ(hash(i));
            }, Kokkos::Sum<double>(Q_total));
            ippl::Comm->allreduce(Q_total, 1, std::plus<double>());
            
            std::cout << "Q_total: " << Q_total << std::endl;
            std::cout << hash.extent(0) << std::endl;

            // Perform the scatter over the custom hash_type
            scatter(bunch.Q, field, bunch.R, policy, hash);

            msg << "---- Testing scatter with custom range policy and hash_type ----" << endl;

            double Total_charge_field = field.sum();

            msg2All << "Total charge in the field:     " << Total_charge_field << endl;
            msg2All << "Total charge of the particles: " << Q_total << endl;
            msg2All << "Error:                         --> " << std::fabs(Q_total - Total_charge_field) << endl;
        }

    }
    ippl::finalize();

    return 0;
}
