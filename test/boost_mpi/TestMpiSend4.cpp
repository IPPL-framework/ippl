#include <iostream>
#include <vector>

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


    BunchBase(int n, mpi::communicator& world)
    : mass_m("mass", n)
    , id_m("id", n)
    , world_m(world)
    {
        addAttribute(mass_m);
        addAttribute(id_m);
    };


    void addAttribute(Kokkos::View<double*>& pa) {
        attrib_m.push_back(&pa);
    }


    view_type& getView(size_t i) {
        return *attrib_m[i];
    }

    template <class BType>
    void update() {

        // number of particles to send
        size_t n = 5;

        if (world_m.rank() == 0) {
            BType buffer(n, world_m);

            for (size_t j = 0; j < attrib_m.size(); ++j) {

                auto& bview = buffer.getView(j);

                auto& this_view = this->getView(j);

                Kokkos::parallel_for("assign",
                                     Kokkos::RangePolicy(5, 10), KOKKOS_CLASS_LAMBDA(const size_t i) {
                    bview(i-5) = this_view(i);
                });
            }

            for (int i = 1; i < world_m.size(); ++i) {
                world_m.send(i, 42 /*tag*/, buffer);
            }

        } else {
            BType buffer(n, world_m);

            world_m.recv(0, 42 /*tag*/, buffer);


            for (size_t j = 0; j < attrib_m.size(); ++j) {

                auto& bview = buffer.getView(j);

                auto& this_view = this->getView(j);

                Kokkos::resize(this_view, n);

                Kokkos::parallel_for("assign",
                                     n, KOKKOS_CLASS_LAMBDA(const size_t i) {
                    this_view(i) = bview(i);
                });
            }
        }
    }


public:
    view_type mass_m;
    view_type id_m;
    mpi::communicator& world_m;

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int /*version*/)
    {
        for (size_t i = 0; i < attrib_m.size(); ++i) {
            ar & boost::serialization::make_array(attrib_m[i]->data(),
                                                  attrib_m[i]->size());
        }
    }

    std::vector<view_type*> attrib_m;
};

BOOST_IS_MPI_DATATYPE(BunchBase)



class BunchDerived : public BunchBase
{
public:

    BunchDerived(int n, mpi::communicator& world)
    : BunchBase(n, world)
    , charge_m("charge", n)
    {
        addAttribute(charge_m);
    };

    void update() {
        BunchBase::update<BunchDerived>();
    }


public:
    view_type charge_m;
};



int main(int argc, char *argv[]) {

    mpi::environment env(argc, argv);
    mpi::communicator world;

    Kokkos::initialize(argc,argv);
    {
        int n = 0;

        if (world.rank() == 0) {
            n = 20;
        }

        BunchDerived bunch(n, world);

        if (world.rank() == 0) {

            Kokkos::parallel_for("assign", 20, KOKKOS_LAMBDA(const size_t i) {
                bunch.id_m(i) = i;
                bunch.mass_m(i) = i + 0.5;
                bunch.charge_m(i) = 0.25;
            });
        }

        bunch.update();

        if ( world.rank() > 0) {
            for (size_t i = 0; i < bunch.id_m.size(); ++i) {
                std::cout << bunch.id_m(i) << " " << bunch.charge_m(i) << " " << bunch.mass_m(i) << std::endl;
            }
        }
    }
    Kokkos::finalize();


    return 0;
}
