#include "Ippl.h"

#include "gtest/gtest.h"




TEST(FEMVector, ValueAssign) {
    using T = double;

    size_t n = 10; // size of the FEMVector

    // setup fake neighbors and halo related indices
    std::vector<size_t> neighbors(1);
    std::vector< Kokkos::View<size_t*> > sendIdxs(1,Kokkos::View<size_t*>("sendIdxs", 2));
    std::vector< Kokkos::View<size_t*> > recvIdxs(1,Kokkos::View<size_t*>("recvIdxs", 2));
    
    // create the FEMVector
    ippl::FEMVector<T> v(10,neighbors, sendIdxs, recvIdxs);
    // set all the values to 42
    v = 42.;

    // check that every entry now is 42
    auto view = v.getView();
    auto hView = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy (hView, view);

    for (size_t i = 0; i < n; ++i) {
        ASSERT_EQ(hView(i), 42.);
    }
}


TEST(FEMVector, ClearValue) {
    using T = double;

    // setup fake neighbors and halo related indices
    std::vector<size_t> neighbors = {0, 0};
    std::vector< Kokkos::View<size_t*> > sendIdxs(2);
    std::vector< Kokkos::View<size_t*> > recvIdxs(2);
    Kokkos::resize(sendIdxs[0],2);
    Kokkos::resize(sendIdxs[1],2);
    Kokkos::resize(recvIdxs[0],2);
    Kokkos::resize(recvIdxs[1],2);

    // set up all the neighbor indices stuff
    for (size_t i = 0; i < 2; ++i) {
        auto hSendView = Kokkos::create_mirror_view(sendIdxs[i]);
        auto hRecvView = Kokkos::create_mirror_view(recvIdxs[i]);
        for (size_t j = 0; j < 2; ++j) {
            hSendView(j) = i*2 + 2 + j;
            hRecvView(j) = i*2 + 6 + j;
        }
        Kokkos::deep_copy(sendIdxs[i], hSendView);
        Kokkos::deep_copy(recvIdxs[i], hRecvView);
    }
    
    // create the FEMVector
    ippl::FEMVector<T> v(10,neighbors, sendIdxs, recvIdxs);
    // clear the values of the halo to 42
    v.clearHalo(42.);

    // check that every entry now is 42
    auto view = v.getView();
    auto hView = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(hView, view);

    for (size_t n = 0; n < neighbors.size(); ++n) {
        auto hRecvIdxs = Kokkos::create_mirror_view(recvIdxs[n]);
        Kokkos::deep_copy(hRecvIdxs, recvIdxs[n]);
        for (size_t i = 0; i < recvIdxs[n].extent(0); ++i) {
            ASSERT_EQ(hView(hRecvIdxs(i)),42.);
        }
    }
}



TEST(FEMVector, fillHalo) {
    using T = double;
    size_t n = 10;  // size of FEMVector

    // setup fake neighbors and halo related indices
    size_t rank = ippl::Comm->rank();
    size_t size = ippl::Comm->size();
    size_t nr = (rank + 1) % size;
    size_t nl = rank == 0 ? size - 1 : rank -1;
    
    std::vector<size_t> neighbors = {nr, nl};

    std::vector< Kokkos::View<size_t*> > sendIdxs(2);
    std::vector< Kokkos::View<size_t*> > recvIdxs(2);
    Kokkos::resize(sendIdxs[0],2);
    Kokkos::resize(sendIdxs[1],2);
    Kokkos::resize(recvIdxs[0],2);
    Kokkos::resize(recvIdxs[1],2);

    // set up all the neighbor indices stuff
    for (size_t i = 0; i < 2; ++i) {
        auto hSendView = Kokkos::create_mirror_view(sendIdxs[i]);
        auto hRecvView = Kokkos::create_mirror_view(recvIdxs[i]);
        for (size_t j = 0; j < 2; ++j) {
            hSendView(j) = i*2 + 2 + j;
            hRecvView(j) = i*2 + 6 + j;
        }
        Kokkos::deep_copy(sendIdxs[i], hSendView);
        Kokkos::deep_copy(recvIdxs[i], hRecvView);
    }

    // create the FEMVector
    ippl::FEMVector<T> v(n,neighbors, sendIdxs, recvIdxs);
    // set all the values to 42
    v = (double)rank;

    v.fillHalo();

    auto view = v.getView();
    auto hView = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy (hView, view);

    ASSERT_EQ(hView(0), (double)rank);
    ASSERT_EQ(hView(1), (double)rank);
    ASSERT_EQ(hView(2), (double)rank);
    ASSERT_EQ(hView(3), (double)rank);
    ASSERT_EQ(hView(4), (double)rank);
    ASSERT_EQ(hView(5), (double)rank);
    ASSERT_EQ(hView(6), (double)nr);
    ASSERT_EQ(hView(7), (double)nr);
    ASSERT_EQ(hView(8), (double)nl);
    ASSERT_EQ(hView(9), (double)nl);
}


TEST(FEMVector, accumulateHalo) {
    using T = double;
    size_t n = 10;  // size of FEMVector

    // setup fake neighbors and halo related indices
    size_t rank = ippl::Comm->rank();
    size_t size = ippl::Comm->size();
    size_t nr = (rank + 1) % size;
    size_t nl = rank == 0 ? size - 1 : rank -1;
    std::vector<size_t> neighbors = {nr, nl};

    std::vector< Kokkos::View<size_t*> > sendIdxs(2);
    std::vector< Kokkos::View<size_t*> > recvIdxs(2);
    Kokkos::resize(sendIdxs[0],2);
    Kokkos::resize(sendIdxs[1],2);
    Kokkos::resize(recvIdxs[0],2);
    Kokkos::resize(recvIdxs[1],2);

    // set up all the neighbor indices stuff
    for (size_t i = 0; i < 2; ++i) {
        auto hSendView = Kokkos::create_mirror_view(sendIdxs[i]);
        auto hRecvView = Kokkos::create_mirror_view(recvIdxs[i]);
        for (size_t j = 0; j < 2; ++j) {
            hSendView(j) = i*2 + 2 + j;
            hRecvView(j) = i*2 + 6 + j;
        }
        Kokkos::deep_copy(sendIdxs[i], hSendView);
        Kokkos::deep_copy(recvIdxs[i], hRecvView);
    }

    // create the FEMVector
    ippl::FEMVector<T> v(n,neighbors, sendIdxs, recvIdxs);
    // set all the values to 42
    v = (double)rank;

    v.accumulateHalo();

    auto view = v.getView();
    auto hView = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy (hView, view);

    ASSERT_EQ(hView(0), (double)rank);
    ASSERT_EQ(hView(1), (double)rank);
    ASSERT_EQ(hView(2), (double)rank+nr);
    ASSERT_EQ(hView(3), (double)rank+nr);
    ASSERT_EQ(hView(4), (double)rank+nl);
    ASSERT_EQ(hView(5), (double)rank+nl);
    ASSERT_EQ(hView(6), (double)rank);
    ASSERT_EQ(hView(7), (double)rank);
    ASSERT_EQ(hView(8), (double)rank);
    ASSERT_EQ(hView(9), (double)rank);
}


int main(int argc, char* argv[]) {
    int success = 1;
    ippl::initialize(argc, argv);
    {
        ::testing::InitGoogleTest(&argc, argv);
        success = RUN_ALL_TESTS();
    }
    ippl::finalize();
    return success;
}