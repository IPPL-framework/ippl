#include <iostream>

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/base_object.hpp>

#include <Kokkos_Core.hpp>

namespace mpi = boost::mpi;


class BunchBase
{
public:
    typedef Kokkos::View<double*> view_type;
    typedef Kokkos::View<size_t*> id_type;


    BunchBase(int n)
    : mass_m("mass", n)
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
        ar & boost::serialization::make_array(id_m.data(), id_m.size());
    }
};


BOOST_IS_MPI_DATATYPE(BunchBase)


class BunchDerived : public BunchBase
{
public:

    BunchDerived(int n)
    : BunchBase(n)
    , charge_m("charge", n)
    {};


public:
    view_type charge_m;


private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int /*version*/)
    {
        // 10. November 2020
        // https://www.boost.org/doc/libs/1_74_0/libs/serialization/doc/serialization.html#base

        // we need to serialize base class members
        ar & boost::serialization::base_object<BunchBase>(*this);

        ar & boost::serialization::make_array(charge_m.data(), charge_m.size());
    }
};


BOOST_IS_MPI_DATATYPE(BunchDerived)



int main(int argc, char *argv[]) {

    mpi::environment env(argc, argv);


    Kokkos::initialize(argc,argv);
    {
        mpi::communicator world;

        if (world.rank() == 0) {
            BunchDerived bunch(20);

            Kokkos::parallel_for("assign", 20, KOKKOS_LAMBDA(const size_t i) {
                bunch.id_m(i) = i;
                bunch.mass_m(i) = i + 0.5;
                bunch.charge_m(i) = 0.25;
            });

            for (int i = 1; i < world.size(); ++i) {
                world.send(i, 42 /*tag*/, bunch);
            }

        } else {
            BunchDerived bunch(20);

            world.recv(0, 42 /*tag*/, bunch);

            for (size_t i = 0; i < 20; ++i) {
                std::cout << bunch.id_m(i) << " " << bunch.charge_m(i) << " " << bunch.mass_m(i) << std::endl;
            }
        }
    }
    Kokkos::finalize();


    return 0;
}
