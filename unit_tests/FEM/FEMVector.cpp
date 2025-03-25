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
        EXPECT_EQ(hView(i), 42.);
    }
}


TEST(FEMVector, FEMVectorAssign) {
    using T = double;

    size_t n = 10; // size of the FEMVector

    // setup fake neighbors and halo related indices
    std::vector<size_t> neighbors(1);
    std::vector< Kokkos::View<size_t*> > sendIdxs(1,Kokkos::View<size_t*>("sendIdxs", 2));
    std::vector< Kokkos::View<size_t*> > recvIdxs(1,Kokkos::View<size_t*>("recvIdxs", 2));
    
    // create the FEMVectors
    ippl::FEMVector<T> a(10,neighbors, sendIdxs, recvIdxs);
    ippl::FEMVector<T> b(10,neighbors, sendIdxs, recvIdxs);
    a = 42.;
    b = a;

    // check that every entry now is 42
    auto view = b.getView();
    auto hView = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy (hView, view);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(hView(i), 42.);
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
            EXPECT_EQ(hView(hRecvIdxs(i)),42.);
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

    EXPECT_EQ(hView(0), (double)rank);
    EXPECT_EQ(hView(1), (double)rank);
    EXPECT_EQ(hView(2), (double)rank);
    EXPECT_EQ(hView(3), (double)rank);
    EXPECT_EQ(hView(4), (double)rank);
    EXPECT_EQ(hView(5), (double)rank);
    EXPECT_EQ(hView(6), (double)nr);
    EXPECT_EQ(hView(7), (double)nr);
    EXPECT_EQ(hView(8), (double)nl);
    EXPECT_EQ(hView(9), (double)nl);
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

    EXPECT_EQ(hView(0), (double)rank);
    EXPECT_EQ(hView(1), (double)rank);
    EXPECT_EQ(hView(2), (double)rank+nr);
    EXPECT_EQ(hView(3), (double)rank+nr);
    EXPECT_EQ(hView(4), (double)rank+nl);
    EXPECT_EQ(hView(5), (double)rank+nl);
    EXPECT_EQ(hView(6), (double)rank);
    EXPECT_EQ(hView(7), (double)rank);
    EXPECT_EQ(hView(8), (double)rank);
    EXPECT_EQ(hView(9), (double)rank);
}


TEST(FEMVector, Arithmetic) {
    using T = double;

    size_t n = 10; // size of the FEMVector

    // setup fake neighbors and halo related indices
    std::vector<size_t> neighbors(1);
    std::vector< Kokkos::View<size_t*> > sendIdxs(1,Kokkos::View<size_t*>("sendIdxs", 2));
    std::vector< Kokkos::View<size_t*> > recvIdxs(1,Kokkos::View<size_t*>("recvIdxs", 2));
    
    // create the FEMVectors
    ippl::FEMVector<T> a(n,neighbors, sendIdxs, recvIdxs);
    ippl::FEMVector<T> b(n,neighbors, sendIdxs, recvIdxs);
    ippl::FEMVector<T> c(n,neighbors, sendIdxs, recvIdxs);
    
    auto aView = a.getView();
    auto haView = Kokkos::create_mirror_view(aView);
    auto bView = b.getView();
    auto hbView = Kokkos::create_mirror_view(bView);
    auto cView = c.getView();
    auto hcView = Kokkos::create_mirror_view(cView);

    for (size_t i = 0; i < n; ++i) {
        haView(i) = i;
        hbView(i) = i+1;
    }
    Kokkos::deep_copy(bView, hbView);
    Kokkos::deep_copy(aView, haView);
    
    // check addition
    c = a + b;
    Kokkos::deep_copy(hcView, cView);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(hcView(i), i+i+1.);
    }
    
    // check subtraction
    c = a - b;
    Kokkos::deep_copy(hcView, cView);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(hcView(i), i - (i+1.));
    }

    // check multiplication
    c = a * b;
    Kokkos::deep_copy(hcView, cView);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(hcView(i), i * (i+1.));
    }

    // check division
    c = a / b;
    Kokkos::deep_copy(hcView, cView);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(hcView(i), i / (i+1.));
    }


    // check scalar addition
    c = a + 10.;
    Kokkos::deep_copy(hcView, cView);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(hcView(i), i + 10.);
    }

    // check scalar subtraction
    c = a - 10.;
    Kokkos::deep_copy(hcView, cView);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(hcView(i), i - 10.);
    }

    // check scalar multiplication
    c = a * 10.;
    Kokkos::deep_copy(hcView, cView);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(hcView(i), i * 10.);
    }

    // check scalar division
    c = a / 10.;
    Kokkos::deep_copy(hcView, cView);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(hcView(i), i / 10.);
    }
    
}

TEST(FEMVector, innerProduct) {
    using T = double;

    size_t n = 10; // size of the FEMVector

    // setup fake neighbors and halo related indices
    std::vector<size_t> neighbors(1);
    std::vector< Kokkos::View<size_t*> > sendIdxs(1,Kokkos::View<size_t*>("sendIdxs", 2));
    std::vector< Kokkos::View<size_t*> > recvIdxs(1,Kokkos::View<size_t*>("recvIdxs", 2));
    
    // create the FEMVectors
    ippl::FEMVector<T> a(n,neighbors, sendIdxs, recvIdxs);
    ippl::FEMVector<T> b(n,neighbors, sendIdxs, recvIdxs);
    
    auto aView = a.getView();
    auto haView = Kokkos::create_mirror_view(aView);
    auto bView = b.getView();
    auto hbView = Kokkos::create_mirror_view(bView);

    for (size_t i = 0; i < n; ++i) {
        haView(i) = i;
        hbView(i) = i+1;
    }
    Kokkos::deep_copy(bView, hbView);
    Kokkos::deep_copy(aView, haView);

    T val = innerProduct(a,b);

    EXPECT_EQ(val, 1./3*(n*n*n - n)*ippl::Comm->size());
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