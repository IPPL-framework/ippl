#include <algorithm>
#include <iostream>
#include <list>
#include <mpi.h>
#include <numeric>
#include <type_traits>
#include <vector>

// template <typename T>
// struct is_serializable : std::false_type {};
//
// template <typename T>
// struct is_serializable<std::vector<T> > : std::true_type {};
//
// template <typename T>
// struct is_serializable<std::vector<std::list<T> > > : std::false_type {};

//     std::cout << is_serializable<double>::value << std::endl;
//
//     std::cout << is_serializable<std::vector<int>>::value << std::endl;
//     std::cout << is_serializable<std::vector<double>>::value << std::endl;
//
//     std::cout << is_serializable<std::vector<std::list<int> >>::value << std::endl;

// template <class Buffer>
// class Archive {
//
// public:
//
//     virtual void serialize() = 0;
//     virtual void deserialize() = 0;
//
//     virtual void resize_buffer(int size) = 0;
//
//     virtual Buffer::value_type* get_data() = 0;
//
//     virtual int get_size() = 0;
//
// protected:
//     Buffer buffer_m;
//     int size_m;
// };

class Object {  //: public Archive<std::vector<double> > {

public:
    Object() {}

    void fill(int n) {
        a_m.resize(n);
        b_m.resize(n);
        std::iota(a_m.begin(), a_m.end(), 0);
        std::iota(b_m.begin(), b_m.end(), n);
    }

    void print() {
        std::cout << "a_m: ";
        for (double v : a_m) {
            std::cout << v << " ";
        }
        std::cout << std::endl << "b_m: ";
        for (double v : b_m) {
            std::cout << v << " ";
        }
        std::cout << std::endl;
    }

    void serialize() {
        buffer_m.resize(a_m.size() + b_m.size());
        std::copy(a_m.begin(), a_m.end(), buffer_m.begin());
        std::copy(b_m.begin(), b_m.end(), buffer_m.begin() + a_m.size());
    }

    void deserialize() {
        a_m.resize(size_m);
        b_m.resize(size_m);
        std::copy(buffer_m.begin(), buffer_m.begin() + size_m, a_m.begin());
        std::copy(buffer_m.begin() + size_m, buffer_m.end(), b_m.begin());
        buffer_m.clear();
    }

    void resize_buffer(int size) {
        size_m = size;
        buffer_m.resize(2 * size);
    }

    double* get_data() { return buffer_m.data(); }

    int get_size() { return buffer_m.size(); }

private:
    std::vector<double> a_m, b_m;

    std::vector<double> buffer_m;
    int size_m;
};

template <typename T>
void send(T* buf, int /*count*/, int dest, int tag) {
    buf->serialize();
    MPI_Send(buf->get_data(), buf->get_size(), MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
}

template <typename T>
void recv(T* buf, int count, int source, int tag) {
    buf->resize_buffer(count);
    MPI_Recv(buf->get_data(), buf->get_size(), MPI_DOUBLE, source, tag, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    buf->deserialize();
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Object obj;

    if (rank == 0) {
        obj.fill(10);
        std::cout << "Rank 0:" << std::endl;
        obj.print();
        send(&obj, 10, 1, 42);
    } else {
        recv(&obj, 10, 0, 42);
        std::cout << "Rank 1:" << std::endl;
        obj.print();
    }

    MPI_Finalize();

    return 0;
}
