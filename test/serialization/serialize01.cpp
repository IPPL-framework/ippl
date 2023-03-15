#include <iostream>
#include <vector>

#include <mpi.h>
#include <Kokkos_Core.hpp>

template <class BType>
void send(int, int, BType&);

template <class BType>
void recv(int, int, BType&);

class Archive {
public:
    Archive(int size = 0)
        : writepos(0)
        , readpos(0)
        , buffer_m("buffer", size) {}

    template <typename T>
    void operator<<(const Kokkos::View<T*>& val) {
        int s = sizeof(T);
        Kokkos::resize(buffer_m, buffer_m.size() + s * val.size());
        Kokkos::parallel_for(
            "serialize", 10, KOKKOS_CLASS_LAMBDA(const int i) {
                std::memcpy(buffer_m.data() + i * s + writepos, val.data() + i, s);
            });
        writepos += s * val.size();
    }

    template <typename T>
    void operator>>(Kokkos::View<T*>& val) {
        int s = sizeof(T);
        Kokkos::parallel_for(
            "deserialize", 10, KOKKOS_CLASS_LAMBDA(const int i) {
                std::memcpy(val.data() + i, buffer_m.data() + i * s + readpos, s);
            });
        readpos += s * val.size();
    }

    void* getBuffer() const { return buffer_m.data(); }

    size_t getSize() const { return buffer_m.size(); }

    ~Archive() = default;

private:
    size_t writepos;
    size_t readpos;
    Kokkos::View<char*> buffer_m;
};

class BunchBase {
public:
    typedef Kokkos::View<double*> view_type;

    BunchBase(int n)
        : mass_m("mass", n)
        , id_m("id", n) {
        addAttribute(mass_m);
        addAttribute(id_m);
    };

    ~BunchBase() {}

    void addAttribute(Kokkos::View<double*>& pa) { attrib_m.push_back(&pa); }

    view_type& getView(size_t i) { return *attrib_m[i]; }

    template <class BType>
    void update() {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        int size = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // number of particles to send
        size_t n = 10;

        if (rank == 0) {
            BType buffer(n);

            for (size_t j = 0; j < attrib_m.size(); ++j) {
                auto& bview = buffer.getView(j);

                auto& this_view = this->getView(j);

                Kokkos::parallel_for(
                    "assign", Kokkos::RangePolicy<size_t>(5, 15),
                    KOKKOS_LAMBDA(const size_t i) { bview(i - 5) = this_view(i); });
            }

            for (int i = 1; i < size; ++i) {
                send(i, 42 /*tag*/, buffer);
            }

        } else {
            BType buffer(n);

            recv(0, 42 /*tag*/, buffer);

            for (size_t j = 0; j < attrib_m.size(); ++j) {
                auto& bview = buffer.getView(j);

                auto& this_view = this->getView(j);

                Kokkos::resize(this_view, n);

                Kokkos::parallel_for(
                    "assign", n, KOKKOS_LAMBDA(const size_t i) { this_view(i) = bview(i); });
            }
        }
    }

    template <class Archive>
    void serialize(Archive& ar) {
        for (size_t i = 0; i < attrib_m.size(); ++i) {
            ar << *attrib_m[i];
        }
    }

    template <class Archive>
    void deserialize(Archive& ar) {
        for (size_t i = 0; i < attrib_m.size(); ++i) {
            ar >> *attrib_m[i];
        }
    }

public:
    view_type mass_m;
    view_type id_m;

private:
    std::vector<view_type*> attrib_m;
};

class BunchDerived : public BunchBase {
public:
    BunchDerived(int n)
        : BunchBase(n)
        , charge_m("charge", n) {
        addAttribute(charge_m);
    };

    ~BunchDerived() {}
    void update() { BunchBase::update<BunchDerived>(); }

public:
    view_type charge_m;
};

template <class BType>
void send(int dest, int tag, BType& buffer) {
    Archive ar;
    buffer.serialize(ar);
    MPI_Send(ar.getBuffer(), ar.getSize(), MPI_BYTE, dest, tag, MPI_COMM_WORLD);
}

template <class BType>
void recv(int src, int tag, BType& buffer) {
    MPI_Status status;

    MPI_Probe(src, tag, MPI_COMM_WORLD, &status);

    int msize = 0;
    MPI_Get_count(&status, MPI_BYTE, &msize);

    Archive ar(msize);

    MPI_Recv(ar.getBuffer(), ar.getSize(), MPI_BYTE, src, tag, MPI_COMM_WORLD, &status);

    buffer.deserialize(ar);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    Kokkos::initialize(argc, argv);
    {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        int size = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        int n = 0;

        if (rank == 0) {
            n = 20;
        }

        BunchDerived bunch(n);

        if (rank == 0) {
            auto idView     = bunch.id_m;
            auto massView   = bunch.mass_m;
            auto chargeView = bunch.charge_m;
            Kokkos::parallel_for(
                "assign", 20, KOKKOS_LAMBDA(const size_t i) {
                    idView(i)     = i;
                    massView(i)   = i + 0.5;
                    chargeView(i) = 0.25;
                });
        }

        bunch.update();

        if (rank > 0) {
            BunchBase::view_type::HostMirror h_id = Kokkos::create_mirror_view(bunch.id_m);
            Kokkos::deep_copy(h_id, bunch.id_m);

            BunchBase::view_type::HostMirror h_charge = Kokkos::create_mirror_view(bunch.charge_m);
            Kokkos::deep_copy(h_charge, bunch.charge_m);

            BunchBase::view_type::HostMirror h_mass = Kokkos::create_mirror_view(bunch.mass_m);
            Kokkos::deep_copy(h_mass, bunch.mass_m);

            for (size_t i = 0; i < h_id.size(); ++i) {
                std::cout << h_id(i) << " " << h_charge(i) << " " << h_mass(i) << std::endl;
            }
        }
    }
    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}
