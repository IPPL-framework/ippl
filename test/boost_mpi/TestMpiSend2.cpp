#include <iostream>

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/array.hpp>

#include <Kokkos_Core.hpp>

namespace mpi = boost::mpi;

class Bunch
{
public:
    typedef Kokkos::View<double*> view_type;
    typedef Kokkos::View<size_t*> id_type;


    Bunch(int n)
    : mass_m("mass", n)
    , charge_m("charge", n)
    , id_m("id", n)
    {};


public:
    view_type mass_m;
    view_type charge_m;
    id_type id_m;


private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int /*version*/)
    {
        ar & boost::serialization::make_array(mass_m.data(), mass_m.size());
        ar & boost::serialization::make_array(charge_m.data(), charge_m.size());
        ar & boost::serialization::make_array(id_m.data(), id_m.size());
    }
};


BOOST_IS_MPI_DATATYPE(Bunch)



int main(int argc, char *argv[]) {

    mpi::environment env(argc, argv);
    mpi::communicator world;

    Kokkos::initialize(argc,argv);
    {

        if (world.rank() == 0) {
            Bunch bunch(20);

            Kokkos::parallel_for("assign", 20, KOKKOS_LAMBDA(const size_t i) {
                bunch.id_m(i) = i;
                bunch.mass_m(i) = i + 0.5;
                bunch.charge_m(i) = 0.25;
            });

            Kokkos::fence()

            for (int i = 1; i < world.size(); ++i) {
                world.send(i, 42 /*tag*/, bunch);
            }

        } else {
            Bunch bunch(20);

            world.recv(0, 42 /*tag*/, bunch);

            Bunch::id_type::HostMirror h_id = Kokkos::create_mirror_view(bunch.id_m);
            Kokkos::deep_copy(h_id, bunch.id_m);

            Bunch::view_type::HostMirror h_charge =Kokkos::create_mirror_view(bunch.charge_m);
            Kokkos::deep_copy(h_charge, bunch.charge_m);

            Bunch::view_type::HostMirror h_mass =Kokkos::create_mirror_view(bunch.mass_m);
            Kokkos::deep_copy(h_charge, bunch.mass_m);

            for (size_t i = 0; i < 20; ++i) {
                std::cout << h_id(i) << " " << h_charge(i) << " " << h_mass(i) << std::endl;
            }
        }
    }
    Kokkos::finalize();


    return 0;
}
