#include <vector>

#include "Ippl.h"
#include "LinearSolvers/PCG.h"

#include "gtest/gtest.h"

#include "Utility/ParameterList.h"



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
    v.setHalo(42.);

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


TEST(FEMVector, deepCopyShallowCopy) {
    using T = double;
    size_t n = 10;
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

    ippl::FEMVector<T> v(n, neighbors, sendIdxs, recvIdxs);
    v = 10.;
    // create a shallow copy
    ippl::FEMVector<T> shallow = v;

    // create a deep copy
    ippl::FEMVector<T> deep = v.deepCopy();

    v = 2.;
    v.setHalo(5.);

    auto vView = v.getView();
    auto shallowView = shallow.getView();
    auto deepView = deep.getView();
    
    auto hvView = Kokkos::create_mirror_view(vView);
    auto hShallowView = Kokkos::create_mirror_view(shallowView);
    auto hDeepView = Kokkos::create_mirror_view(deepView);
    
    Kokkos::deep_copy(hvView, vView);
    Kokkos::deep_copy(hShallowView, shallowView);
    Kokkos::deep_copy(hDeepView, deepView);

    for (size_t i = 0; i < n; ++i) {
        // we should have that both shallow and v are the same
        EXPECT_EQ(hvView(i), hShallowView(i));

        // we should have that deep is only 10.
        EXPECT_EQ(hDeepView(i), 10.);
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

    // check combination
    c = 1.;
    c = c - 10*a - b;
    Kokkos::deep_copy(hcView, cView);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(hcView(i), 1. - 10.*i - (i+1));
    }
    
}

TEST(FEMVector, innerProduct) {
    using T = double;

    size_t n = 10; // size of the FEMVector

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
            hSendView(j) = i*2 + 6 + j;
            hRecvView(j) = i*2 + j;
        }
        Kokkos::deep_copy(sendIdxs[i], hSendView);
        Kokkos::deep_copy(recvIdxs[i], hRecvView);
    }
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

    val = innerProduct(a,a);
    EXPECT_EQ(val, 1./6*(n-1)*n*(2*n-1)*ippl::Comm->size());
}

TEST(FEMVector, norm) {
    using T = double;

    size_t n = 10; // size of the FEMVector

    // setup fake neighbors and halo related indices
    std::vector<size_t> neighbors(1);
    std::vector< Kokkos::View<size_t*> > sendIdxs(1,Kokkos::View<size_t*>("sendIdxs", 2));
    std::vector< Kokkos::View<size_t*> > recvIdxs(1,Kokkos::View<size_t*>("recvIdxs", 2));
    
    // create the FEMVectors
    ippl::FEMVector<T> a(n,neighbors, sendIdxs, recvIdxs);
    
    auto aView = a.getView();
    auto haView = Kokkos::create_mirror_view(aView);

    for (size_t i = 0; i < n; ++i) {
        haView(i) = i;
    }
    Kokkos::deep_copy(aView, haView);

    T val = norm(a);

    EXPECT_EQ(val, Kokkos::sqrt(ippl::Comm->size()*1/6.*(n-1)*n*(2*n-1)));
}


TEST(FEMVector, CG) {
    using T = double;
    using CG_t = ippl::CG<ippl::FEMVector<T>, ippl::FEMVector<T>, ippl::FEMVector<T>,
                          ippl::FEMVector<T>, ippl::FEMVector<T>, ippl::FEMVector<T>,
                          ippl::FEMVector<T> >;

    ippl::ParameterList params;
    params.add("output_type", 0b01);
    params.add("max_iterations", 1000);
    params.add("tolerance", (T)1e-13);

    CG_t cg;
    
    // create the FEMVectors
    size_t n = 10;
    ippl::FEMVector<T> a(n);
    ippl::FEMVector<T> b(n);
    auto aView = a.getView();
    auto haView = Kokkos::create_mirror_view(aView);
    auto bView = b.getView();
    auto hbView = Kokkos::create_mirror_view(bView);

    // Identity operator
    a = 0.;
    b = 3.;
    const auto operatorId = [](ippl::FEMVector<T> v) -> ippl::FEMVector<T> {
        return v;
    };
    cg.setOperator(operatorId);
    cg(a,b,params);

    // check that every entry now is 3
    Kokkos::deep_copy (haView, aView);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(haView(i), 3.);
    }

    // Some tridiagonal matrix
    {
        std::vector<double> bRef = {61, 33, 75, 15, 69, 12, 47, 19, 53, 11};
        std::vector<double> aRef = {5, 4, 5, 1, 5, 1, 2, 2, 5, 1};
        
        std::vector<double> matRef = {
            10, -1, 3, 0, 0, 0, 0, 0, 0, 0,
            -1, 10, -1, 3, 0, 0, 0, 0, 0, 0,
            3, -1, 10, -1, 3, 0, 0, 0, 0, 0,
            0, 3, -1, 10, -1, 3, 0, 0, 0, 0,
            0, 0, 3, -1, 10, -1, 3, 0, 0, 0,
            0, 0, 0, 3, -1, 10, -1, 3, 0, 0,
            0, 0, 0, 0, 3, -1, 10, -1, 3, 0,
            0, 0, 0, 0, 0, 3, -1, 10, -1, 3,
            0, 0, 0, 0, 0, 0, 3, -1, 10, -1,
            0, 0, 0, 0, 0, 0, 0, 3, -1, 10
        };
        
    
        for (size_t i = 0; i < n; ++i) {
            hbView(i) = bRef[i];
        }
        Kokkos::deep_copy(bView, hbView);


        const auto operatorTriDiagMat = [=](ippl::FEMVector<T> v) -> ippl::FEMVector<T> {
            ippl::FEMVector<T> out(v.size(), std::vector<size_t>(),
                                    std::vector< Kokkos::View<size_t*> >(),
                                    std::vector< Kokkos::View<size_t*> >());
            auto outView = out.getView();
            auto hOutView = Kokkos::create_mirror_view(outView);

            auto vView = v.getView();
            auto hvView = Kokkos::create_mirror_view(vView);
            Kokkos::deep_copy(hvView, vView);

            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    hOutView(i) += hvView(j) * matRef[i*n + j];
                }
            }

            Kokkos::deep_copy(outView, hOutView);

            return out;
        };
        a = 1.;
        cg.setOperator(operatorTriDiagMat);
        cg(a,b,params);


        Kokkos::deep_copy (haView, aView);
        for (size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(haView(i), aRef[i], 1e-6);
        }
    }


    // some random matrix
    {
        std::vector<double> bRef = {23.91243720168186, 28.76946440071826, 11.39405321895077, 
            14.58146738148548, 54.46293935692246, 50.55020649187946, 
            -6.966599074336089, 25.08955228913931, 97.98346398457775, 
            61.88726870993443};
        std::vector<double> aRef = {2, 2, 2, 5, 3, 1, 3, 3, 4, 3};
        
        std::vector<double> matRef = {
            10.15274438556837, -1.575357090699566, 
            0.1033530371385715, -0.8261284642500747, -0.8834242907511218, 
            5.652017827163076, -0.1701548630577168, 0.4131148230083384, 
            2.671939012710765, -1.245594008895713, -1.575357090699566, 
            6.382713346688194, 3.241831081261847, -2.937913800842192, 
            2.297035155593666, 1.98989341982756, -1.977299789149673, 
            2.764215315196782, 2.095562987718326, 
            2.578887104935028, 0.1033530371385715, 3.241831081261847, 
            7.995771133671092, -6.356959260093841, 4.662112232835592, 
            2.01755777428059, -1.221016215206041, 0.8489913292249818, 
            0.8146275942084533, 
            0.7835362745329846, -0.8261284642500747, -2.937913800842192, 
            -6.356959260093841, 
            8.700380945624307, -3.062426877405789, -2.120980515110928, 
            1.607962852195594, -2.671019775396403, 
            2.175940086736792, -0.9615875754267607, -0.8834242907511218, 
            2.297035155593666, 4.662112232835592, -3.062426877405789, 
            8.533392427711236, 3.401747061032478, 0.1010267492617375, 
            2.490997357340004, 0.1648513849035274, 
            6.728741781669867, 5.652017827163076, 1.98989341982756, 
            2.01755777428059, -2.120980515110928, 3.401747061032478, 
            10.80609077728808, -3.017414496852657, 2.325793498580139, 
            6.569789197749419, -1.126484910557991, -0.1701548630577168, 
            -1.977299789149673, -1.221016215206041, 1.607962852195594, 
            0.1010267492617375, -3.017414496852657, 
            6.654116621919533, -4.497004122281946, -3.137962038753028, 
            0.1751244348932011, 0.4131148230083384, 2.764215315196782, 
            0.8489913292249818, -2.671019775396403, 2.490997357340004, 
            2.325793498580139, -4.497004122281946, 6.11956623484971, 
            0.1886720675300253, 4.990282684279192, 2.671939012710765, 
            2.095562987718326, 0.8146275942084525, 2.175940086736793, 
            0.1648513849035272, 6.569789197749419, -3.137962038753028, 
            0.1886720675300255, 
            22.24697534370522, -3.754956817331048, -1.245594008895714, 
            2.578887104935028, 0.7835362745329846, -0.9615875754267607, 
            6.728741781669866, -1.126484910557992, 0.1751244348932012, 
            4.990282684279193, -3.754956817331048, 14.30847110775968
        };
        
    
        for (size_t i = 0; i < n; ++i) {
            hbView(i) = bRef[i];
        }
        Kokkos::deep_copy(bView, hbView);


        const auto operatorRandMat = [=](ippl::FEMVector<T> v) -> ippl::FEMVector<T> {
            ippl::FEMVector<T> out(v.size(), std::vector<size_t>(),
                                    std::vector< Kokkos::View<size_t*> >(),
                                    std::vector< Kokkos::View<size_t*> >());
            auto outView = out.getView();
            auto hOutView = Kokkos::create_mirror_view(outView);

            auto vView = v.getView();
            auto hvView = Kokkos::create_mirror_view(vView);
            Kokkos::deep_copy(hvView, vView);

            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    hOutView(i) += hvView(j) * matRef[i*n + j];
                }
            }

            Kokkos::deep_copy(outView, hOutView);

            return out;
        };
        a = 1.;
        cg.setOperator(operatorRandMat);
        cg(a,b,params);


        Kokkos::deep_copy (haView, aView);
        for (size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(haView(i), aRef[i], 1e-6);
        }
    }

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