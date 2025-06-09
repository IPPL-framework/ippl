// Tests the FEM Poisson solver by solving the 2d problem:
//
// curl(curl(E)) + E = {(1+k^2)sin(k*y), (1+k^2)sin(k*x)} in [-1,1]^2
// n x E = 0 on boundary 
//
// Exact solution is E = {sin(k*y),sin(k*x)}
//
// BCs: Zero dirichlet bc.
//

#include "Ippl.h"

#include "Meshes/Centering.h"
#include "MaxwellSolvers/FEMMaxwellDiffusionSolver.h"

#include <fstream>


template <typename T, typename Layout>
void saveToFile(ippl::Vector<T,2> cellSpacing, ippl::Vector<T,2> origin, const Layout& layout,
    ippl::FEMVector<ippl::Vector<T,2>> data, const std::string& filename) {
    
    // In order to not save the halo cells, what we do is we create a skeleton
    // copy of data and simply set for it the halo to 1, like this we can
    // create a mask indicating which values are halo and which are not.
    ippl::FEMVector<size_t> mask = data.template skeletonCopy<size_t>();
    mask = 0;
    mask.setHalo(1);

    auto view = data.getView();
    auto hView = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(hView, view);
    
    auto maskView = mask.getView();
    auto hMaskView = Kokkos::create_mirror_view(maskView);
    Kokkos::deep_copy(hMaskView, maskView);

    auto ldom = layout.getLocalNDIndex();
    
    auto d = ldom.last() - ldom.first();
    size_t nx = d[0] + 1 + 2;
    size_t ny = d[1] + 1 + 2;
    
    for (size_t r = 0; r < ippl::Comm->size(); ++r) {
        if (r == ippl::Comm->rank()) {
            std::ofstream file;
            if (r == 0) {
                file.open(filename);
                file << "x,y,z,rank,vx,vy,vz\n";
            } else {
                file.open(filename, std::ios::app);
            }
            for (size_t i = 0; i < hView.extent(0); ++i) {

                size_t yOffset = i / (2*nx - 1);
                size_t xOffset = i - (2*nx -1)*yOffset;
                T x = xOffset*cellSpacing[0];
                T y = yOffset*cellSpacing[1];
                if (xOffset < (nx-1)) {
                    // we are parallel to the x axis
                    x += cellSpacing[0]/2.;
                } else {
                    // we are parallel to the y axis
                    y += cellSpacing[1]/2.;
                    x -= (nx-1)*cellSpacing[0];
                }
                // we also remove 1 cell spacing for the fact that we have ghost
                // cells
                x += origin[0] + ldom.first()[0]*cellSpacing[0] - cellSpacing[0];
                y += origin[1] + ldom.first()[1]*cellSpacing[1] - cellSpacing[1];

                if (hMaskView(i) == 0) {
                    file << x << "," << y << ",0," << r << ","
                         << hView(i)[0] << "," << hView(i)[1] << ",0\n";
                    
                }
            }

            file.close();
        }

        ippl::Comm->barrier();
    }
}



template <typename T, typename Layout>
void saveToFile(ippl::Vector<T,3> cellSpacing, ippl::Vector<T,3> origin, const Layout& layout,
    ippl::FEMVector<ippl::Vector<T,3>> data, const std::string& filename) {
    
    // In order to not save the halo cells, what we do is we create a skeleton
    // copy of data and simply set for it the halo to 1, like this we can
    // create a mask indicating which values are halo and which are not.
    ippl::FEMVector<size_t> mask = data.template skeletonCopy<size_t>();
    mask = 0;
    mask.setHalo(1);

    auto view = data.getView();
    auto hView = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(hView, view);
    
    auto maskView = mask.getView();
    auto hMaskView = Kokkos::create_mirror_view(maskView);
    Kokkos::deep_copy(hMaskView, maskView);

    auto ldom = layout.getLocalNDIndex();
    
    auto d = ldom.last() - ldom.first();
    size_t nx = d[0] + 1 + 2;
    size_t ny = d[1] + 1 + 2;
    
    for (size_t r = 0; r < ippl::Comm->size(); ++r) {
        if (r == ippl::Comm->rank()) {
            std::ofstream file;
            if (r == 0) {
                file.open(filename);
                file << "x,y,z,rank,vx,vy,vz\n";
            } else {
                file.open(filename, std::ios::app);
            }
            for (size_t i = 0; i < hView.extent(0); ++i) {
                size_t zOffset = i / (nx*(ny-1) + ny*(nx-1) + nx*ny);
                T z = zOffset*cellSpacing[2];
                T x = 0; 
                T y = 0;

                if (i - (nx*(ny-1) + ny*(nx-1) + nx*ny)*zOffset >= (nx*(ny-1) + ny*(nx-1))) {
                    // we are parallel to z axis
                    size_t f = i - (nx*(ny-1) + ny*(nx-1) + nx*ny)*zOffset
                        - (nx*(ny-1) + ny*(nx-1));
                    z += cellSpacing[2]/2;
                    
                    size_t yOffset = f / nx;
                    y = yOffset*cellSpacing[1];

                    size_t xOffset = f % nx;
                    x = xOffset*cellSpacing[0];
                } else {
                    // are parallel to one of the other axes
                    size_t f = i - (nx*(ny-1) + ny*(nx-1) + nx*ny)*zOffset;
                    size_t yOffset = f / (2*nx - 1);
                    size_t xOffset = f - (2*nx - 1)*yOffset;

                    x = xOffset*cellSpacing[0];
                    y = yOffset*cellSpacing[1];

                    if (xOffset < (nx-1)) {
                        // we are parallel to the x axis
                        x += cellSpacing[0]/2.;
                    } else {
                        // we are parallel to the y axis
                        y += cellSpacing[1]/2.;
                        x -= (nx-1)*cellSpacing[0];
                    }
                }

                // have to subtract one cellspacing for the ghost cells
                x += origin[0] + ldom.first()[0]*cellSpacing[0] - cellSpacing[0];
                y += origin[1] + ldom.first()[1]*cellSpacing[1] - cellSpacing[1];
                z += origin[2] + ldom.first()[2]*cellSpacing[2] - cellSpacing[2];

                if (hMaskView(i) == 0) {
                    file << x << "," << y << "," << z << "," << r << ","
                         << hView(i)[0] << "," << hView(i)[1] << "," << hView(i)[2] << "\n";
                    
                }
            }

            file.close();
        }

        ippl::Comm->barrier();
    }
}

template <typename T, typename Layout>
void saveToFile(ippl::Vector<T,3> cellSpacing, ippl::Vector<T,3> origin, const Layout& layout,
    ippl::FEMVector<T> data, const std::string& filename) {
    
    // In order to not save the halo cells, what we do is we create a skeleton
    // copy of data and simply set for it the halo to 1, like this we can
    // create a mask indicating which values are halo and which are not.
    ippl::FEMVector<size_t> mask = data.template skeletonCopy<size_t>();
    mask = 0;
    mask.setHalo(1);

    auto view = data.getView();
    auto hView = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(hView, view);
    
    auto maskView = mask.getView();
    auto hMaskView = Kokkos::create_mirror_view(maskView);
    Kokkos::deep_copy(hMaskView, maskView);

    auto ldom = layout.getLocalNDIndex();
    
    auto d = ldom.last() - ldom.first();
    size_t nx = d[0] + 1 + 2;
    size_t ny = d[1] + 1 + 2;
    
    for (size_t r = 0; r < ippl::Comm->size(); ++r) {
        if (r == ippl::Comm->rank()) {
            std::ofstream file;
            if (r == 0) {
                file.open(filename);
                file << "x,y,z,rank,val\n";
            } else {
                file.open(filename, std::ios::app);
            }
            for (size_t i = 0; i < hView.extent(0); ++i) {

                size_t zOffset = i / (nx*(ny-1) + ny*(nx-1) + nx*ny);
                T z = zOffset*cellSpacing[2];
                T x = 0; 
                T y = 0;

                if (i - (nx*(ny-1) + ny*(nx-1) + nx*ny)*zOffset >= (nx*(ny-1) + ny*(nx-1))) {
                    // we are parallel to z axis
                    size_t f = i - (nx*(ny-1) + ny*(nx-1) + nx*ny)*zOffset
                        - (nx*(ny-1) + ny*(nx-1));
                    z += cellSpacing[2]/2;
                    
                    size_t yOffset = f / nx;
                    y = yOffset*cellSpacing[1];

                    size_t xOffset = f % nx;
                    x = xOffset*cellSpacing[0];
                } else {
                    // are parallel to one of the other axes
                    size_t f = i - (nx*(ny-1) + ny*(nx-1) + nx*ny)*zOffset;
                    size_t yOffset = f / (2*nx - 1);
                    size_t xOffset = f - (2*nx - 1)*yOffset;

                    x = xOffset*cellSpacing[0];
                    y = yOffset*cellSpacing[1];

                    if (xOffset < (nx-1)) {
                        // we are parallel to the x axis
                        x += cellSpacing[0]/2.;
                    } else {
                        // we are parallel to the y axis
                        y += cellSpacing[1]/2.;
                        x -= (nx-1)*cellSpacing[0];
                    }
                }

                // have to subtract one cellspacing for the ghost cells
                x += origin[0] + ldom.first()[0]*cellSpacing[0] - cellSpacing[0];
                y += origin[1] + ldom.first()[1]*cellSpacing[1] - cellSpacing[1];
                z += origin[2] + ldom.first()[2]*cellSpacing[2] - cellSpacing[2];

                if (hMaskView(i) == 0) {
                    file << x << "," << y << "," << z << "," << r << ","
                         << hView(i) << "\n";
                    
                }
            }

            file.close();
        }
        ippl::Comm->barrier();
    }
}



template <typename T, typename Layout>
void saveToFile(ippl::Vector<T,2> cellSpacing, ippl::Vector<T,2> origin, const Layout& layout,
    ippl::FEMVector<T> data, const std::string& filename) {
    
    // In order to not save the halo cells, what we do is we create a skeleton
    // copy of data and simply set for it the halo to 1, like this we can
    // create a mask indicating which values are halo and which are not.
    ippl::FEMVector<size_t> mask = data.template skeletonCopy<size_t>();
    mask = 0;
    mask.setHalo(1);

    auto view = data.getView();
    auto hView = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(hView, view);
    
    auto maskView = mask.getView();
    auto hMaskView = Kokkos::create_mirror_view(maskView);
    Kokkos::deep_copy(hMaskView, maskView);

    auto ldom = layout.getLocalNDIndex();
    
    auto d = ldom.last() - ldom.first();
    size_t nx = d[0] + 1 + 2;
    size_t ny = d[1] + 1 + 2;
    
    for (size_t r = 0; r < ippl::Comm->size(); ++r) {
        if (r == ippl::Comm->rank()) {
            std::ofstream file;
            if (r == 0) {
                file.open(filename);
                file << "x,y,z,rank,val\n";
            } else {
                file.open(filename, std::ios::app);
            }
            for (size_t i = 0; i < hView.extent(0); ++i) {

                size_t yOffset = i / (2*nx - 1);
                size_t xOffset = i - (2*nx -1)*yOffset;
                T x = xOffset*cellSpacing[0];
                T y = yOffset*cellSpacing[1];
                if (xOffset < (nx-1)) {
                    // we are parallel to the x axis
                    x += cellSpacing[0]/2.;
                } else {
                    // we are parallel to the y axis
                    y += cellSpacing[1]/2.;
                    x -= (nx-1)*cellSpacing[0];
                }
                // we also remove 1 cell spacing for the fact that we have ghost
                // cells
                x += origin[0] + ldom.first()[0]*cellSpacing[0] - cellSpacing[0];
                y += origin[1] + ldom.first()[1]*cellSpacing[1] - cellSpacing[1];

                if (hMaskView(i) == 0) {
                    file << x << "," << y << ",0," << r << ","
                         << hView(i) << "\n";
                    
                }
            }

            file.close();
        }

        ippl::Comm->barrier();
    }
}

template <typename T, typename Layout>
void saveToFile(ippl::Vector<T,2> cellSpacing, ippl::Vector<T,2> origin, const Layout& layout,
    ippl::Field<ippl::Vector<T,2>, 2, ippl::UniformCartesian<T, 2>, Cell> data, const std::string& filename) {
    
    auto ldom = layout.getLocalNDIndex();
    auto d = ldom.last() - ldom.first();
    size_t nx = d[0] + 1 + 2;
    size_t ny = d[1] + 1 + 2;

    auto view = data.getView();
    auto hView = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(hView, view);
    
    for (size_t r = 0; r < ippl::Comm->size(); ++r) {
        if (r == ippl::Comm->rank()) {
            std::ofstream file;
            if (r == 0) {
                file.open(filename);
                file << "x,y,z,rank,vx,vy,vz\n";
            } else {
                file.open(filename, std::ios::app);
            }
            for (size_t i = 1; i < hView.extent(0)-1; ++i) {
                T x = i*cellSpacing[0] + origin[0] + ldom.first()[0]*cellSpacing[0] - cellSpacing[0];
                for (size_t j = 1; j < hView.extent(1)-1; ++j) {
                    T y = j*cellSpacing[1] + origin[1]+ ldom.first()[1]*cellSpacing[1] - cellSpacing[1];
                    file << x << "," << y << ",0," << r << ","
                         << hView(i,j)[0] << "," << hView(i,j)[1] << ",0\n";
                }
                
            }
            file.close();
        }

        ippl::Comm->barrier();
    }

    
}


template <typename T, typename Layout>
void saveToFile(ippl::Vector<T,3> cellSpacing, ippl::Vector<T,3> origin, const Layout& layout,
    ippl::Field<ippl::Vector<T,3>, 3, ippl::UniformCartesian<T, 3>, Cell> data, const std::string& filename) {
    
    auto ldom = layout.getLocalNDIndex();
    auto d = ldom.last() - ldom.first();
    size_t nx = d[0] + 1 + 2;
    size_t ny = d[1] + 1 + 2;

    auto view = data.getView();
    auto hView = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(hView, view);
    
    for (size_t r = 0; r < ippl::Comm->size(); ++r) {
        if (r == ippl::Comm->rank()) {
            std::ofstream file;
            if (r == 0) {
                file.open(filename);
                file << "x,y,z,rank,vx,vy,vz\n";
            } else {
                file.open(filename, std::ios::app);
            }
            for (size_t i = 1; i < hView.extent(0)-1; ++i) {
                T x = i*cellSpacing[0] + origin[0] + ldom.first()[0]*cellSpacing[0] - cellSpacing[0];
                for (size_t j = 1; j < hView.extent(1)-1; ++j) {
                    T y = j*cellSpacing[1] + origin[1]+ ldom.first()[1]*cellSpacing[1] - cellSpacing[1];
                    for (size_t k = 1; k < hView.extent(2)-1; ++k) {
                        T z = k*cellSpacing[2] + origin[2]+ ldom.first()[2]*cellSpacing[2] - cellSpacing[2];
                        
                        file << x << "," << y << "," << z << "," << r << ","
                            << hView(i,j,k)[0] << "," << hView(i,j,k)[1] << "," << hView(i,j,k)[2] <<"\n";
                    }         
                }
                
            }
            file.close();
        }

        ippl::Comm->barrier();
    }

    
}


template <typename T, typename Layout, unsigned Dim>
void saveMPILayout(ippl::Vector<T,2> cellSpacing, ippl::Vector<T,2> origin,
    ippl::FEMVector<T>& data, const Layout& layout) {
    
    data = ippl::Comm->rank();
    data.setHalo(-199+ippl::Comm->rank());
    data.accumulateHalo();
    data.setHalo(-1);
    auto view = data.getView();
    auto hView = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(hView, view);
    auto ldom = layout.getLocalNDIndex();
    
    auto d = ldom.last() - ldom.first();
    size_t nx = d[0] + 1 + 2;
    size_t ny = d[1] + 1 + 2;
    
    for (size_t r = 0; r < ippl::Comm->size(); ++r) {
        if (r == ippl::Comm->rank()) {
            std::ofstream file;
            std::ofstream haloFile("halo_"+std::to_string(r)+".csv");
            std::ofstream recvFile("recv_"+std::to_string(r)+".csv");
            haloFile << "x,y,z\n";
            recvFile << "x,y,z,rank\n";
            if (r == 0) {
                file.open("ranks.csv");
                file << "x,y,z,rank\n";
            } else {
                file.open("ranks.csv", std::ios::app);
            }
            for (size_t i = 0; i < hView.extent(0); ++i) {

                size_t yOffset = i / (2*nx - 1);
                size_t xOffset = i - (2*nx -1)*yOffset;
                T x = xOffset*cellSpacing[0];
                T y = yOffset*cellSpacing[1];
                if (xOffset < (nx-1)) {
                    // we are parallel to the x axis
                    x += cellSpacing[0]/2.;
                } else {
                    // we are parallel to the y axis
                    y += cellSpacing[1]/2.;
                    x -= (nx-1)*cellSpacing[0];
                }
                // we also remove 1 cell spacing for the fact that we have ghost
                // cells
                x += origin[0] + ldom.first()[0]*cellSpacing[0] - cellSpacing[0];
                y += origin[1] + ldom.first()[1]*cellSpacing[1] - cellSpacing[1];

                if (hView(i) == -1) {
                    haloFile << x << "," << y << ",0," << "\n";
                    
                } else {
                    if (hView(i) < 0) {
                        T val = hView(i) - ippl::Comm->rank() + 199;
                        recvFile << x << "," << y << ",0," << val << "\n";
                        hView(i) = ippl::Comm->rank();
                    }
                    file << x << "," << y << ",0," << hView(i) << "\n";
                }
            }

            file.close();
            haloFile.close();
        }

        ippl::Comm->barrier();
    }
}


template <typename T, unsigned Dim, typename Layout>
ippl::FEMVector<T> createFEMVector(const Layout& layout) {
    using indices_t = ippl::Vector<int, Dim>;

    auto ldom = layout.getLocalNDIndex();
    auto doms = layout.getHostLocalDomains();
    //std::cout << "rank " << ippl::Comm->rank() << ": " << ldom.first() << "-" << ldom.last() << "\n";


    auto getDOFIndices = [&](indices_t elementPos) {
        ippl::Vector<size_t, 4> FEMVectorDOFs(0);
        
        // Then we can subtract from it the starting position and add the ghost
        // things
        const auto& ldom = layout.getLocalNDIndex();
        elementPos -= ldom.first();
        elementPos += 1;
        
        indices_t dif(0);
        dif = ldom.last() - ldom.first();
        dif += 1 + 2; // plus 1 for last still being in +2 for ghosts.

        ippl::Vector<size_t, Dim> v(1);
        if constexpr (Dim == 2) {
            size_t nx = dif[0];
            v(1) = 2*nx-1;
        } else if constexpr (Dim == 3) {
            size_t nx = dif[0];
            size_t ny = dif[1];
            v(1) = 2*nx -1;
            v(2) = 3*nx*ny - nx - ny;
        }

        size_t nx = dif[0];
        FEMVectorDOFs(0) = v.dot(elementPos);
        FEMVectorDOFs(1) = FEMVectorDOFs(0) + nx - 1;
        FEMVectorDOFs(2) = FEMVectorDOFs(1) + nx;
        FEMVectorDOFs(3) = FEMVectorDOFs(1) + 1;

        if constexpr (Dim == 3) {
            size_t ny = dif[1];

            FEMVectorDOFs(4) = v(2)*elementPos(2) + 2*nx*ny - nx - ny;
            FEMVectorDOFs(5) = FEMVectorDOFs(4) + 1;
            FEMVectorDOFs(6) = FEMVectorDOFs(4) + nx + 1;
            FEMVectorDOFs(7) = FEMVectorDOFs(4) + nx;
            FEMVectorDOFs(8) = FEMVectorDOFs(0) + 3*nx*ny - nx - ny;
            FEMVectorDOFs(9) = FEMVectorDOFs(8) + nx - 1;
            FEMVectorDOFs(10) = FEMVectorDOFs(9) + nx;
            FEMVectorDOFs(11) = FEMVectorDOFs(9) + 1;
        }
        

        return FEMVectorDOFs;
    };

    // Create the temporaries and so on which will store the MPI information.
    std::vector<size_t> neighbors;
    std::vector< Kokkos::View<size_t*> > sendIdxs;
    std::vector< Kokkos::View<size_t*> > recvIdxs;
    std::vector< std::vector<size_t> > sendIdxsTemp;
    std::vector< std::vector<size_t> > recvIdxsTemp;

    // Here we loop thought all the domains to figure out how we are related to
    // them and if we have to do any kind of exchange.
    for (size_t i = 0; i < doms.extent(0); ++i) {
        if (i == ippl::Comm->rank()) {
            // We are looking at ourself
            continue;
        }
        auto odom = doms(i);

        // East boundary
        if (ldom.last()[0] == odom.first()[0]-1 &&
                !(odom.last()[1] < ldom.first()[1] || odom.first()[1] > ldom.last()[1])) {
            // Extract the range of the boundary.
            int begin = std::max(odom.first()[1], ldom.first()[1]);
            int end = std::min(odom.last()[1], ldom.last()[1]);
            int pos = ldom.last()[0];
            
            // Add this to the neighbour list.
            neighbors.push_back(i);
            sendIdxsTemp.push_back(std::vector<size_t>());
            recvIdxsTemp.push_back(std::vector<size_t>());
            size_t idx = neighbors.size() - 1;
            
            // Add all the halo
            indices_t elementPosHalo(0);
            elementPosHalo(0) = pos;
            indices_t elementPosSend(0);
            elementPosSend(0) = pos;
            for (int k = begin; k <= end; ++k) {
                elementPosHalo(1) = k;
                elementPosSend(1) = k;
                
                auto dofIndicesHalo = getDOFIndices(elementPosHalo);
                recvIdxsTemp[idx].push_back(dofIndicesHalo[3]);

                auto dofIndicesSend = getDOFIndices(elementPosSend);
                sendIdxsTemp[idx].push_back(dofIndicesSend[0]);
                sendIdxsTemp[idx].push_back(dofIndicesSend[1]);
            }
            // Check if on very north
            if (end == layout.getDomain().last()[1] || ldom.last()[1] > odom.last()[1]) {
                elementPosSend(1) = end;
                auto dofIndicesSend = getDOFIndices(elementPosSend);
                // also have to add dof 2
                sendIdxsTemp[idx].push_back(dofIndicesSend[2]);
            }
        }

        // West boundary
        if (ldom.first()[0] == odom.last()[0]+1 &&
                !(odom.last()[1] < ldom.first()[1] || odom.first()[1] > ldom.last()[1])) {
            // Extract the range of the boundary.
            int begin = std::max(odom.first()[1], ldom.first()[1]);
            int end = std::min(odom.last()[1], ldom.last()[1]);
            int pos = ldom.first()[0];
            
            // Add this to the neighbour list.
            neighbors.push_back(i);
            sendIdxsTemp.push_back(std::vector<size_t>());
            recvIdxsTemp.push_back(std::vector<size_t>());
            size_t idx = neighbors.size() - 1;
            
            // Add all the halo
            indices_t elementPosHalo(0);
            elementPosHalo(0) = pos-1;
            indices_t elementPosSend(0);
            elementPosSend(0) = pos;
            for (int k = begin; k <= end; ++k) {
                elementPosHalo(1) = k;
                elementPosSend(1) = k;
                
                auto dofIndicesHalo = getDOFIndices(elementPosHalo);
                recvIdxsTemp[idx].push_back(dofIndicesHalo[0]);
                recvIdxsTemp[idx].push_back(dofIndicesHalo[1]);

                auto dofIndicesSend = getDOFIndices(elementPosSend);
                sendIdxsTemp[idx].push_back(dofIndicesSend[1]);
            }
            // Check if on very north
            if (end == layout.getDomain().last()[1] || odom.last()[1] > ldom.last()[1]) {
                elementPosHalo(1) = end;
                auto dofIndicesHalo = getDOFIndices(elementPosHalo);
                // also have to add dof 2
                recvIdxsTemp[idx].push_back(dofIndicesHalo[2]);
            }
        }

        // North boundary
        if (ldom.last()[1] == odom.first()[1]-1 &&
                !(odom.last()[0] < ldom.first()[0] || odom.first()[0] > ldom.last()[0])) {
            // Extract the range of the boundary.
            int begin = std::max(odom.first()[0], ldom.first()[0]);
            int end = std::min(odom.last()[0], ldom.last()[0]);
            int pos = ldom.last()[1];
            
            // Add this to the neighbour list.
            neighbors.push_back(i);
            sendIdxsTemp.push_back(std::vector<size_t>());
            recvIdxsTemp.push_back(std::vector<size_t>());
            size_t idx = neighbors.size() - 1;
            
            // Add all the halo
            indices_t elementPosHalo(0);
            elementPosHalo(1) = pos;
            indices_t elementPosSend(0);
            elementPosSend(1) = pos;
            for (int k = begin; k <= end; ++k) {
                elementPosHalo(0) = k;
                elementPosSend(0) = k;
                
                auto dofIndicesHalo = getDOFIndices(elementPosHalo);
                recvIdxsTemp[idx].push_back(dofIndicesHalo[2]);

                auto dofIndicesSend = getDOFIndices(elementPosSend);
                sendIdxsTemp[idx].push_back(dofIndicesSend[0]);
                sendIdxsTemp[idx].push_back(dofIndicesSend[1]);
            }
            // Check if on very east
            if (end == layout.getDomain().last()[0] || ldom.last()[0] > odom.last()[0]) {
                elementPosSend(0) = end;
                auto dofIndicesSend = getDOFIndices(elementPosSend);
                // also have to add dof 3
                sendIdxsTemp[idx].push_back(dofIndicesSend[3]);
            }
        }

        // South boundary
        if (ldom.first()[1] == odom.last()[1]+1 &&
                !(odom.last()[0] < ldom.first()[0] || odom.first()[0] > ldom.last()[0])) {
            // Extract the range of the boundary.
            int begin = std::max(odom.first()[0], ldom.first()[0]);
            int end = std::min(odom.last()[0], ldom.last()[0]);
            int pos = ldom.first()[1];
            
            // Add this to the neighbour list.
            neighbors.push_back(i);
            sendIdxsTemp.push_back(std::vector<size_t>());
            recvIdxsTemp.push_back(std::vector<size_t>());
            size_t idx = neighbors.size() - 1;
            
            // Add all the halo
            indices_t elementPosHalo(0);
            elementPosHalo(1) = pos-1;
            indices_t elementPosSend(0);
            elementPosSend(1) = pos;
            for (int k = begin; k <= end; ++k) {
                elementPosHalo(0) = k;
                elementPosSend(0) = k;
                
                auto dofIndicesHalo = getDOFIndices(elementPosHalo);
                recvIdxsTemp[idx].push_back(dofIndicesHalo[0]);
                recvIdxsTemp[idx].push_back(dofIndicesHalo[1]);

                auto dofIndicesSend = getDOFIndices(elementPosSend);
                sendIdxsTemp[idx].push_back(dofIndicesSend[0]);
            }
            // Check if on very east
            if (end == layout.getDomain().last()[0] || odom.last()[0] > ldom.last()[0]) {
                elementPosHalo(0) = end;
                auto dofIndicesHalo = getDOFIndices(elementPosHalo);
                // also have to add dof 3
                recvIdxsTemp[idx].push_back(dofIndicesHalo[3]);
            }
        }
    }



    for (size_t i = 0; i < neighbors.size(); ++i) {
        sendIdxs.push_back(Kokkos::View<size_t*>("FEMvector::sendIdxs[" + std::to_string(i) +
                                                    "]", sendIdxsTemp[i].size()));
        recvIdxs.push_back(Kokkos::View<size_t*>("FEMvector::recvIdxs[" + std::to_string(i) +
                                                    "]", recvIdxsTemp[i].size()));
        auto sendView = sendIdxs[i];
        auto recvView = recvIdxs[i];
        auto hSendView = Kokkos::create_mirror_view(sendView);
        auto hRecvView = Kokkos::create_mirror_view(recvView);

        for (size_t j = 0; j < sendIdxsTemp[i].size(); ++j) {
            hSendView(j) = sendIdxsTemp[i][j];
        }

        for (size_t j = 0; j < recvIdxsTemp[i].size(); ++j) {
            hRecvView(j) = recvIdxsTemp[i][j];
        }

        Kokkos::deep_copy(sendView, hSendView);
        Kokkos::deep_copy(recvView, hRecvView);
    }
    

    
    // Now finaly create the FEMVector
    indices_t extents(0);
    extents = (ldom.last() - ldom.first()) + 3;
    size_t nx = extents(0);
    size_t ny = extents(1);
    size_t n = nx*(ny-1) + ny*(nx-1);
    ippl::FEMVector<T> vec(n, neighbors, sendIdxs, recvIdxs);
    
    return vec;
}



template <typename T, unsigned Dim, typename Layout>
ippl::FEMVector<T> createFEMVector3d(const Layout& layout) {
    using indices_t = ippl::Vector<int, Dim>;

    auto ldom = layout.getLocalNDIndex();
    auto doms = layout.getHostLocalDomains();
    //std::cout << doms.extent(0) << "\n";
    //std::cout << "rank " << ippl::Comm->rank() << ": " << ldom.first() << "-" << ldom.last() << "\n";


    auto getDOFIndices = [&](indices_t elementPos) {
        ippl::Vector<size_t, 12> FEMVectorDOFs(0);
        
        // Then we can subtract from it the starting position and add the ghost
        // things
        const auto& ldom = layout.getLocalNDIndex();
        elementPos -= ldom.first();
        elementPos += 1;
        
        indices_t dif(0);
        dif = ldom.last() - ldom.first();
        dif += 1 + 2; // plus 1 for last still being in +2 for ghosts.

        ippl::Vector<size_t, Dim> v(1);
        if constexpr (Dim == 2) {
            size_t nx = dif[0];
            v(1) = 2*nx-1;
        } else if constexpr (Dim == 3) {
            size_t nx = dif[0];
            size_t ny = dif[1];
            v(1) = 2*nx -1;
            v(2) = 3*nx*ny - nx - ny;
        }

        size_t nx = dif[0];
        FEMVectorDOFs(0) = v.dot(elementPos);
        FEMVectorDOFs(1) = FEMVectorDOFs(0) + nx - 1;
        FEMVectorDOFs(2) = FEMVectorDOFs(1) + nx;
        FEMVectorDOFs(3) = FEMVectorDOFs(1) + 1;

        if constexpr (Dim == 3) {
            size_t ny = dif[1];
            FEMVectorDOFs(4) = v(2)*elementPos(2) + 2*nx*ny - nx - ny
                + elementPos(1)*nx + elementPos(0);
            FEMVectorDOFs(5) = FEMVectorDOFs(4) + 1;
            FEMVectorDOFs(6) = FEMVectorDOFs(4) + nx + 1;
            FEMVectorDOFs(7) = FEMVectorDOFs(4) + nx;
            FEMVectorDOFs(8) = FEMVectorDOFs(0) + 3*nx*ny - nx - ny;
            FEMVectorDOFs(9) = FEMVectorDOFs(8) + nx - 1;
            FEMVectorDOFs(10) = FEMVectorDOFs(9) + nx;
            FEMVectorDOFs(11) = FEMVectorDOFs(9) + 1;
        }
        

        return FEMVectorDOFs;
    };

    // Create the temporaries and so on which will store the MPI information.
    std::vector<size_t> neighbors;
    std::vector< Kokkos::View<size_t*> > sendIdxs;
    std::vector< Kokkos::View<size_t*> > recvIdxs;
    std::vector< std::vector<size_t> > sendIdxsTemp;
    std::vector< std::vector<size_t> > recvIdxsTemp;


    auto flatBoundaryExchange = [&neighbors, &getDOFIndices, &layout](
        size_t i, size_t a, size_t f, size_t s,
        std::vector<std::vector<size_t> >& va, std::vector<std::vector<size_t> >& vb,
        int posA, int posB,
        const std::vector<size_t>& idxsA, const std::vector<size_t>& idxsB,
        ippl::NDIndex<3>& adom, ippl::NDIndex<3>& bdom) {
        
        int beginF = std::max(bdom.first()[f], adom.first()[f]);
        int endF = std::min(bdom.last()[f], adom.last()[f]);
        int beginS = std::max(bdom.first()[s], adom.first()[s]);
        int endS = std::min(bdom.last()[s], adom.last()[s]);
        
        neighbors.push_back(i);
        va.push_back(std::vector<size_t>());
        vb.push_back(std::vector<size_t>());
        size_t idx = neighbors.size() - 1;

        // Add all the halo
        indices_t elementPosA(0);
        elementPosA(a) = posA;
        indices_t elementPosB(0);
        elementPosB(a) = posB;
        for (int k = beginF; k <= endF; ++k) {
            elementPosA(f) = k;
            elementPosB(f) = k;
            for (int l = beginS; l <= endS; ++l) {
                elementPosA(s) = l;
                elementPosB(s) = l;

                auto dofIndicesA = getDOFIndices(elementPosA);
                va[idx].push_back(dofIndicesA[idxsA[0]]);
                va[idx].push_back(dofIndicesA[idxsA[1]]);

                auto dofIndicesB = getDOFIndices(elementPosB);
                vb[idx].push_back(dofIndicesB[idxsB[0]]);
                vb[idx].push_back(dofIndicesB[idxsB[1]]);
                vb[idx].push_back(dofIndicesB[idxsB[2]]);

                
                if (k == endF) {
                    if (endF == layout.getDomain().last()[f] ||
                            bdom.last()[f] > adom.last()[f]) {
                        va[idx].push_back(dofIndicesA[idxsA[2]]);
                    }
                    // Check if we have to add some special stuff
                    if (endF == layout.getDomain().last()[f] ||
                            adom.last()[f] > bdom.last()[f]) {
                        vb[idx].push_back(dofIndicesB[idxsB[3]]);
                        vb[idx].push_back(dofIndicesB[idxsB[4]]);
                    }

                    // call this last, as modifies elementPosA(s) 
                    if (bdom.first()[f] < adom.first()[f]) {
                        indices_t tmpPos = elementPosA;
                        tmpPos(f) = beginF-1;
                        auto dofIndicestmp = getDOFIndices(tmpPos);
                        va[idx].push_back(dofIndicestmp[idxsA[0]]);
                        va[idx].push_back(dofIndicestmp[idxsA[1]]);
                    }
                }
            }
            // Have to add space row to Halo
            if (endS == layout.getDomain().last()[s] || bdom.last()[s] > adom.last()[s]) {
                elementPosA(s) = endS;
                auto dofIndicesA = getDOFIndices(elementPosA);
                va[idx].push_back(dofIndicesA[idxsA[3]]);
            }
            // Check if we have to add some special stuff
            if (endS == layout.getDomain().last()[s] || adom.last()[s] > bdom.last()[s]) {
                elementPosB(s) = endS;
                auto dofIndicesB = getDOFIndices(elementPosB);
                vb[idx].push_back(dofIndicesB[idxsB[5]]);
                vb[idx].push_back(dofIndicesB[idxsB[6]]);
            }

            // call this last, as modifies elementPosA(f);
            if (bdom.first()[f] < adom.first()[f]) {
                indices_t tmpPos = elementPosA;
                tmpPos(s) = beginS-1;
                auto dofIndicestmp = getDOFIndices(tmpPos);
                va[idx].push_back(dofIndicestmp[idxsA[0]]);
                va[idx].push_back(dofIndicestmp[idxsA[1]]);
            }
        }
        // Check if we have to add some special stuff
        if ((endF == layout.getDomain().last()[f] || adom.last()[f] > bdom.last()[f]) && 
            (endS == layout.getDomain().last()[s] || adom.last()[s] > bdom.last()[s])) {
            elementPosB(f) = endF;
            elementPosB(s) = endS;
            auto dofIndicesB = getDOFIndices(elementPosB);
            vb[idx].push_back(dofIndicesB[idxsB[7]]);
        }
    };


    auto negativeDiagonalExchange = [&neighbors, &ldom, &getDOFIndices](
        size_t i, size_t a, size_t f, size_t s, int ao, int bo,
        std::vector<std::vector<size_t> >& va, std::vector<std::vector<size_t> >& vb,
        const std::vector<size_t>& idxsA, const std::vector<size_t>& idxsB,
        ippl::NDIndex<3>& odom) {
        
        neighbors.push_back(i);
        va.push_back(std::vector<size_t>());
        vb.push_back(std::vector<size_t>());
        size_t idx = neighbors.size() - 1;

        indices_t elementPosA(0);
        elementPosA(f) = ldom.last()[f];
        elementPosA(s) = ldom.first()[s] + ao;

        indices_t elementPosB(0);
        elementPosB(f) = ldom.last()[f];
        elementPosB(s) = ldom.first()[s] + bo;

        int begin = std::max(odom.first()[a], ldom.first()[a]);
        int end = std::min(odom.last()[a], ldom.last()[a]);

        for (int k = begin; k <= end; ++k) {
            elementPosA(a) = k;
            elementPosB(a) = k;

            auto dofIndicesA = getDOFIndices(elementPosA);
            va[idx].push_back(dofIndicesA[idxsA[0]]);
            va[idx].push_back(dofIndicesA[idxsA[1]]);

            auto dofIndicesB = getDOFIndices(elementPosB);
            vb[idx].push_back(dofIndicesB[idxsB[0]]);
            vb[idx].push_back(dofIndicesB[idxsB[1]]);
        }
    };

    auto positiveDiagonalExchange = [&neighbors, &ldom, &getDOFIndices](
        size_t i, size_t a, size_t f, size_t s,
        indices_t posA, indices_t posB,
        std::vector<std::vector<size_t> >& va, std::vector<std::vector<size_t> >& vb,
        const std::vector<size_t>& idxsA, const std::vector<size_t>& idxsB,
        ippl::NDIndex<3>& odom) {
        
        neighbors.push_back(i);
        va.push_back(std::vector<size_t>());
        vb.push_back(std::vector<size_t>());
        size_t idx = neighbors.size() - 1;

        indices_t elementPosA(0);
        elementPosA(f) = posA(f);
        elementPosA(s) = posA(s);

        indices_t elementPosB(0);
        elementPosB(f) = posB(f);
        elementPosB(s) = posB(s);

        int begin = std::max(odom.first()[a], ldom.first()[a]);
        int end = std::min(odom.last()[a], ldom.last()[a]);

        for (int k = begin; k <= end; ++k) {
            elementPosA(a) = k;
            elementPosB(a) = k;

            auto dofIndicesA = getDOFIndices(elementPosA);
            va[idx].push_back(dofIndicesA[idxsA[0]]);
            va[idx].push_back(dofIndicesA[idxsA[1]]);
            va[idx].push_back(dofIndicesA[idxsA[2]]);

            auto dofIndicesB = getDOFIndices(elementPosB);
            vb[idx].push_back(dofIndicesB[idxsB[0]]);
        }
    };

    // Here we loop thought all the domains to figure out how we are related to
    // them and if we have to do any kind of exchange.
    for (size_t i = 0; i < doms.extent(0); ++i) {
        if (i == ippl::Comm->rank()) {
            // We are looking at ourself
            continue;
        }
        auto odom = doms(i);

        // East boundary
        if (ldom.last()[0] == odom.first()[0]-1 &&
                !(odom.last()[1] < ldom.first()[1] || odom.first()[1] > ldom.last()[1]) && 
                !(odom.last()[2] < ldom.first()[2] || odom.first()[2] > ldom.last()[2])) {
            
            int pos = ldom.last()[0];
            flatBoundaryExchange(
                i, 0, 1, 2,
                recvIdxsTemp, sendIdxsTemp,
                pos, pos,
                {3,5,6,11}, {0,1,4,2,7,8,9,10},
                ldom, odom
            );
        }

        // West boundary
        if (ldom.first()[0] == odom.last()[0]+1 &&
                !(odom.last()[1] < ldom.first()[1] || odom.first()[1] > ldom.last()[1]) && 
                !(odom.last()[2] < ldom.first()[2] || odom.first()[2] > ldom.last()[2])) {
            
            int pos = ldom.first()[0];
            flatBoundaryExchange(
                i, 0, 1, 2,
                sendIdxsTemp, recvIdxsTemp,
                pos, pos-1,
                {1,4,7,9}, {0,1,4,2,7,8,9,10},
                odom, ldom
            );
        }

        // North boundary
        if (ldom.last()[1] == odom.first()[1]-1 &&
                !(odom.last()[0] < ldom.first()[0] || odom.first()[0] > ldom.last()[0]) &&
                !(odom.last()[2] < ldom.first()[2] || odom.first()[2] > ldom.last()[2])) {

            int pos = ldom.last()[1];
            flatBoundaryExchange(
                i, 1, 0, 2,
                recvIdxsTemp, sendIdxsTemp,
                pos, pos,
                {2,7,6,10}, {0,1,4,3,5,8,9,11},
                ldom, odom
            );
        }

        // South boundary
        if (ldom.first()[1] == odom.last()[1]+1 &&
                !(odom.last()[0] < ldom.first()[0] || odom.first()[0] > ldom.last()[0]) &&
                !(odom.last()[2] < ldom.first()[2] || odom.first()[2] > ldom.last()[2])) {
            
            int pos = ldom.first()[1];
            flatBoundaryExchange(
                i, 1, 0, 2,
                sendIdxsTemp, recvIdxsTemp,
                pos, pos-1,
                {0,4,5,8}, {0,1,4,3,5,8,9,11},
                odom, ldom
            );
            
        }

        // Space boundary
        if (ldom.last()[2] == odom.first()[2]-1 &&
                !(odom.last()[0] < ldom.first()[0] || odom.first()[0] > ldom.last()[0]) && 
                !(odom.last()[1] < ldom.first()[1] || odom.first()[1] > ldom.last()[1])) {
            
            int pos = ldom.last()[2];
            flatBoundaryExchange(
                i, 2, 0, 1,
                recvIdxsTemp, sendIdxsTemp,
                pos, pos,
                {8,9,11,10}, {0,1,4,3,5,2,7,6},
                ldom, odom
            );
        }

        // Ground boundary
        if (ldom.first()[2] == odom.last()[2]+1 &&
                !(odom.last()[0] < ldom.first()[0] || odom.first()[0] > ldom.last()[0]) && 
                !(odom.last()[1] < ldom.first()[1] || odom.first()[1] > ldom.last()[1])) {

            int pos = ldom.first()[2];
            flatBoundaryExchange(
                i, 2, 0, 1,
                sendIdxsTemp, recvIdxsTemp,
                pos, pos-1,
                {0,1,3,2}, {0,1,4,3,5,2,7,6},
                odom, ldom
            );
        }


        
        // Next up we handle all the anoying diagonals
        // The negative ones
        // Parallel to y from space to ground, west to east
        if (ldom.last()[0] == odom.first()[0]-1 && ldom.first()[2] == odom.last()[2]+1 && 
                !(odom.last()[1] < ldom.first()[1] || odom.first()[1] > ldom.last()[1])) {
            
            negativeDiagonalExchange(
                i, 1, 0, 2, 0, -1,
                sendIdxsTemp, recvIdxsTemp,
                {0,1}, {3,5},
                odom
            );
        }

        // Parallel to y from ground to space, east to west
        if (ldom.first()[0] == odom.last()[0]+1 && ldom.last()[2] == odom.first()[2]-1 && 
                !(odom.last()[1] < ldom.first()[1] || odom.first()[1] > ldom.last()[1])) {
            
            negativeDiagonalExchange(
                i, 1, 2, 0, -1, 0,
                recvIdxsTemp, sendIdxsTemp,
                {8,9}, {1,4},
                odom
            );
        }


        // Parallel to x from space to ground, south to north
        if (ldom.last()[1] == odom.first()[1]-1 && ldom.first()[2] == odom.last()[2]+1 && 
                !(odom.last()[0] < ldom.first()[0] || odom.first()[0] > ldom.last()[0])) {
            negativeDiagonalExchange(
                i, 0, 1, 2, 0, -1,
                sendIdxsTemp, recvIdxsTemp,
                {0,1}, {2,7},
                odom
            );
        }

        // Parallel to x from ground to space, north to south
        if (ldom.first()[1] == odom.last()[1]+1 && ldom.last()[2] == odom.first()[2]-1 && 
                !(odom.last()[0] < ldom.first()[0] || odom.first()[0] > ldom.last()[0])) {
            negativeDiagonalExchange(
                i, 0, 2, 1, -1, 0,
                recvIdxsTemp, sendIdxsTemp,
                {8,9}, {0,4},
                odom
            );
        }


        // Parallel to z from west to east, north to south
        if (ldom.last()[0] == odom.first()[0]-1 && ldom.first()[1] == odom.last()[1]+1 && 
                !(odom.last()[2] < ldom.first()[2] || odom.first()[2] > ldom.last()[2])) {
            negativeDiagonalExchange(
                i, 2, 0, 1, 0, -1,
                sendIdxsTemp, recvIdxsTemp,
                {0,4}, {3,5},
                odom
            );
        }

        // Parallel to z from east to west, south to north
        if (ldom.first()[0] == odom.last()[0]+1 && ldom.last()[1] == odom.first()[1]-1 && 
                !(odom.last()[2] < ldom.first()[2] || odom.first()[2] > ldom.last()[2])) {
            negativeDiagonalExchange(
                i, 2, 1, 0, -1, 0,
                recvIdxsTemp, sendIdxsTemp,
                {2,7}, {1,4},
                odom
            );
        }



        // The positive ones
        // Parallel to y from ground to space, west to east
        if (ldom.last()[0] == odom.first()[0]-1 && ldom.last()[2] == odom.first()[2]-1 && 
                !(odom.last()[1] < ldom.first()[1] || odom.first()[1] > ldom.last()[1])) {
            positiveDiagonalExchange(
                i, 1, 0, 2,
                ldom.last(), ldom.last(),
                sendIdxsTemp, recvIdxsTemp,
                {0,1,4}, {11},
                odom
            );
        }

        // Parallel to y from space to ground, east to west
        if (ldom.first()[0] == odom.last()[0]+1 && ldom.first()[2] == odom.last()[2]+1 && 
                !(odom.last()[1] < ldom.first()[1] || odom.first()[1] > ldom.last()[1])) {
            positiveDiagonalExchange(
                i, 1, 0, 2,
                ldom.first()-1, ldom.first(),
                recvIdxsTemp, sendIdxsTemp,
                {0,1,4}, {1},
                odom
            );
        }


        // Parallel to x from ground to space, south to north
        if (ldom.last()[1] == odom.first()[1]-1 && ldom.last()[2] == odom.first()[2]-1 && 
                !(odom.last()[0] < ldom.first()[0] || odom.first()[0] > ldom.last()[0])) {
            positiveDiagonalExchange(
                i, 0, 1, 2,
                ldom.last(), ldom.last(),
                sendIdxsTemp, recvIdxsTemp,
                {0,1,4}, {10},
                odom
            );
        }

        // Parallel to x from space to ground, north to south
        if (ldom.first()[1] == odom.last()[1]+1 && ldom.first()[2] == odom.last()[2]+1 && 
                !(odom.last()[0] < ldom.first()[0] || odom.first()[0] > ldom.last()[0])) {
            positiveDiagonalExchange(
                i, 0, 1, 2,
                ldom.first()-1, ldom.first(),
                recvIdxsTemp, sendIdxsTemp,
                {0,1,4}, {0},
                odom
            );
        }


        // Parallel to z from west to east, south to north
        if (ldom.last()[0] == odom.first()[0]-1 && ldom.last()[1] == odom.first()[1]-1 && 
                !(odom.last()[2] < ldom.first()[2] || odom.first()[2] > ldom.last()[2])) {
            positiveDiagonalExchange(
                i, 2, 0, 1,
                ldom.last(), ldom.last(),
                sendIdxsTemp, recvIdxsTemp,
                {0,1,4}, {6},
                odom
            );
        }

        // Parallel to z from east to west, north to south
        if (ldom.first()[0] == odom.last()[0]+1 && ldom.first()[1] == odom.last()[1]+1 && 
                !(odom.last()[2] < ldom.first()[2] || odom.first()[2] > ldom.last()[2])) {
            positiveDiagonalExchange(
                i, 2, 0, 1,
                ldom.first()-1, ldom.first(),
                recvIdxsTemp, sendIdxsTemp,
                {0,1,4}, {4},
                odom
            );
        }
        
    }
    


    indices_t extents(0);
    extents = (ldom.last() - ldom.first()) + 3;
    size_t nx = extents(0);
    size_t ny = extents(1);
    size_t nz = extents(2);
    size_t n = (nz-1)*(nx*(ny-1) + ny*(nx-1) + nx*ny) + nx*(ny-1) + ny*(nx-1);

    //std::cout << ippl::Comm->rank() << " size: " << n << "\n";

    for (size_t i = 0; i < neighbors.size(); ++i) {
        sendIdxs.push_back(Kokkos::View<size_t*>("FEMvector::sendIdxs[" + std::to_string(i) +
                                                    "]", sendIdxsTemp[i].size()));
        recvIdxs.push_back(Kokkos::View<size_t*>("FEMvector::recvIdxs[" + std::to_string(i) +
                                                    "]", recvIdxsTemp[i].size()));
        auto sendView = sendIdxs[i];
        auto recvView = recvIdxs[i];
        auto hSendView = Kokkos::create_mirror_view(sendView);
        auto hRecvView = Kokkos::create_mirror_view(recvView);
        /*
        std::cout << ippl::Comm->rank() << "-->|" << sendIdxsTemp[i].size() << "|"
                  << neighbors[i] << "\n";
        */
        for (size_t j = 0; j < sendIdxsTemp[i].size(); ++j) {
            hSendView(j) = sendIdxsTemp[i][j];
            
            if (hSendView(j) >= n) {
                std::cout << ippl::Comm->rank() << " sends " << hSendView(j) << " to "
                          << neighbors[i] << " but has size " << n << "\n";
            }
            
        }
        /*
        std::cout << neighbors[i] << "-->|" << recvIdxsTemp[i].size() << "|"
                  << ippl::Comm->rank() << "\n"; 
        */
        for (size_t j = 0; j < recvIdxsTemp[i].size(); ++j) {
            hRecvView(j) = recvIdxsTemp[i][j];
            
            if (hRecvView(j) >= n) {
                std::cout << ippl::Comm->rank() << " receives " << hRecvView(j) << " from "
                          << neighbors[i] << " but has size " << n << "\n";
            }
            
        }

        Kokkos::deep_copy(sendView, hSendView);
        Kokkos::deep_copy(recvView, hRecvView);
    }
    

    
    // Now finaly create the FEMVector
    ippl::FEMVector<T> vec(n, neighbors, sendIdxs, recvIdxs);
    
    return vec;
}



template <typename T, unsigned Dim>
struct Analytical{
    using point_t =  ippl::Vector<T, Dim>;

    T k;
    Analytical(T k) : k(k) {}

    KOKKOS_FUNCTION const point_t operator() (const point_t& pos) const{
        point_t sol(0);
        if constexpr (Dim == 2) {
            T x = pos[0];
            T y = pos[1];
            sol[0] = -(y*y - 1);
            sol[1] = -(x*x - 1);
        } else {
            T x = pos[0];
            T y = pos[1];
            T z = pos[2];
            sol[0] = (y*y - 1)*(z*z - 1);
            sol[1] = (x*x - 1)*(z*z - 1);
            sol[2] = (x*x - 1)*(y*y - 1);
        }
        return sol;
    }
};


template <typename T, unsigned Dim>
void testFEMSolver(const unsigned& numNodesPerDim, const T& domain_start = 0.0,
                   const T& domain_end = 1.0) {
    // start the timer
    static IpplTimings::TimerRef initTimer = IpplTimings::getTimer("initTest");
    IpplTimings::startTimer(initTimer);

    Inform m("");
    Inform msg2all("", INFORM_ALL_NODES);

    using Mesh_t   = ippl::UniformCartesian<T, Dim>;
    using Field_t  = ippl::Field<ippl::Vector<T,Dim>, Dim, Mesh_t, Cell>;
    using BConds_t = ippl::BConds<Field_t, Dim>;
    using point_t =  ippl::Vector<T, Dim>;

    const unsigned numCellsPerDim = numNodesPerDim - 1;
    const unsigned numGhosts      = 1;

    // Domain: [-1, 1]
    const ippl::Vector<unsigned, Dim> nodesPerDimVec(numNodesPerDim);
    ippl::NDIndex<Dim> domain(nodesPerDimVec);
    ippl::Vector<T, Dim> cellSpacing((domain_end - domain_start) / static_cast<T>(numCellsPerDim));
    ippl::Vector<T, Dim> origin(domain_start);
    Mesh_t mesh(domain, cellSpacing, origin);

    // specifies decomposition; here all dimensions are parallel
    std::array<bool, Dim> isParallel;
    isParallel.fill(true);

    ippl::FieldLayout<Dim> layout(MPI_COMM_WORLD, domain, isParallel);
    Field_t lhs(mesh, layout, numGhosts);  // left hand side (updated in the algorithm)
    Field_t rhs(mesh, layout, numGhosts);  // right hand side (set once)

    // Define boundary conditions
    BConds_t bcField;
    for (unsigned int i = 0; i < 2 * Dim; ++i) {
        bcField[i] = std::make_shared<ippl::ZeroFace<Field_t>>(i);
    }
    lhs.setFieldBC(bcField);
    rhs.setFieldBC(bcField);

    //auto vec = createFEMVector<T,Dim, ippl::FieldLayout<Dim>>(layout);

    
    T k = 3.14159265359;


    // Generate the rhs FEMVector
    auto ldom = layout.getLocalNDIndex();
    auto d = ldom.last() - ldom.first() + 1 + 2;

    //saveMPILayout<T,ippl::FieldLayout<Dim>,Dim>(cellSpacing, origin, vec, layout);
    
    std::vector<size_t> neighbors;
    std::vector< Kokkos::View<size_t*> > sendIdxs;
    std::vector< Kokkos::View<size_t*> > recvIdxs;
    size_t n = 0;
    if constexpr(Dim == 2) {
        n = (d[0]*(d[1]-1) + d[1]*(d[0]-1));
    } else {
        n = (d[2]-1)*(d[0]*(d[1]-1) + d[1]*(d[0]-1) + d[0]*d[1]) + d[0]*(d[1]-1) + d[1]*(d[0]-1);
    }
    ippl::FEMVector<ippl::Vector<T,Dim> > rhsVector = //(n , neighbors, sendIdxs, recvIdxs);
        createFEMVector3d<ippl::Vector<T,Dim>,Dim, ippl::FieldLayout<Dim>>(layout);

    auto viewRhs = rhsVector.getView();

    ippl::FEMVector<ippl::Vector<T,Dim> > solutionVector =
        rhsVector.template skeletonCopy<ippl::Vector<T,Dim>>();
    auto viewSolution = solutionVector.getView();

    Analytical<T, Dim> analytical(k);

    Field_t fieldSolution(mesh, layout, 1);
    auto fieldView = fieldSolution.getView();
    using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
    ippl::parallel_for("Assign solution field", fieldSolution.getFieldRangePolicy(),
        KOKKOS_LAMBDA(const index_array_type& args) {
            T x = args[0]*cellSpacing[0] + origin[0] + ldom.first()[0]*cellSpacing[0] - cellSpacing[0];
            T y = args[1]*cellSpacing[1] + origin[1] + ldom.first()[1]*cellSpacing[1] - cellSpacing[1];

            ippl::Vector<T, Dim> t(0);
            t[0] = x;
            t[1] = y;

            if constexpr (Dim == 3) {
                T z = args[2]*cellSpacing[2] + origin[2] + ldom.first()[2]*cellSpacing[2] - cellSpacing[2];
                t[2] = z;
            }

            apply(fieldView, args) = analytical(t);
            
        }
    );

    

    auto rhsFunc = KOKKOS_LAMBDA(const point_t& pos) -> point_t {
        point_t sol(0);
        T dummy = k;
        if constexpr (Dim == 2) {
            T x = pos[0];
            T y = pos[1];
            sol[0] = 2 - (y*y - 1);
            sol[1] = 2 - (x*x - 1);
        }
        if constexpr (Dim == 3) {
            T x = pos[0];
            T y = pos[1];
            T z = pos[2];
            sol[0] = -2*(z*z - 1) - 2*(y*y - 1) + (y*y - 1)*(z*z - 1);
            sol[1] = -2*(x*x - 1) - 2*(z*z - 1) + (x*x - 1)*(z*z - 1);
            sol[2] = -2*(y*y - 1) - 2*(x*x - 1) + (x*x - 1)*(y*y - 1);
        }
        return sol;
    };



    if constexpr (Dim == 2) {
        Kokkos::parallel_for("Assign RHS", rhsVector.size(),
            KOKKOS_LAMBDA(size_t i) {
                size_t nx = d[0];
                size_t ny = d[1];
                size_t yOffset = i / (2*nx - 1);
                size_t xOffset = i - (2*nx -1)*yOffset;
                T x = xOffset*cellSpacing[0];
                T y = yOffset*cellSpacing[1];
                if (xOffset < (nx-1)) {
                    // we are parallel to the x axis
                    x += cellSpacing[0]/2.;
                } else {
                    // we are parallel to the y axis
                    y += cellSpacing[1]/2.;
                    x -= (nx-1)*cellSpacing[0];
                }
                // have to subtract one cellspacing for the ghost cells
                x += origin[0] + ldom.first()[0]*cellSpacing[0] - cellSpacing[0];
                y += origin[1] + ldom.first()[1]*cellSpacing[1] - cellSpacing[1];
                
                ippl::Vector<T, Dim> t(0);
                t(0) = x;
                t(1) = y;
                
                viewRhs(i) = rhsFunc(t);
                viewSolution(i) = analytical(t);
            }
        );

    } else {
        Kokkos::parallel_for("Assign RHS", rhsVector.size(),
            KOKKOS_LAMBDA(size_t i) {
                size_t nx = d[0];
                size_t ny = d[1];
                size_t nz = d[2];

                size_t zOffset = i / (nx*(ny-1) + ny*(nx-1) + nx*ny);
                T z = zOffset*cellSpacing[2];
                T x = 0;
                T y = 0;

                if (i - (nx*(ny-1) + ny*(nx-1) + nx*ny)*zOffset >= (nx*(ny-1) + ny*(nx-1))) {
                    // we are parallel to z axis
                    size_t f = i - (nx*(ny-1) + ny*(nx-1) + nx*ny)*zOffset
                        - (nx*(ny-1) + ny*(nx-1));
                    z += cellSpacing[2]/2;
                    
                    size_t yOffset = f / nx;
                    y = yOffset*cellSpacing[1];

                    size_t xOffset = f % nx;
                    x = xOffset*cellSpacing[0];
                } else {
                    // are parallel to one of the other axes
                    size_t f = i - (nx*(ny-1) + ny*(nx-1) + nx*ny)*zOffset;
                    size_t yOffset = f / (2*nx - 1);
                    size_t xOffset = f - (2*nx - 1)*yOffset;

                    x = xOffset*cellSpacing[0];
                    y = yOffset*cellSpacing[1];

                    if (xOffset < (nx-1)) {
                        // we are parallel to the x axis
                        x += cellSpacing[0]/2.;
                    } else {
                        // we are parallel to the y axis
                        y += cellSpacing[1]/2.;
                        x -= (nx-1)*cellSpacing[0];
                    }
                }

                // have to subtract one cellspacing for the ghost cells
                x += origin[0] + ldom.first()[0]*cellSpacing[0] - cellSpacing[0];
                y += origin[1] + ldom.first()[1]*cellSpacing[1] - cellSpacing[1];
                z += origin[2] + ldom.first()[2]*cellSpacing[2] - cellSpacing[2];
                
                ippl::Vector<T, Dim> t(0);
                t(0) = x;
                t(1) = y;
                t(2) = z;

                viewRhs(i) = rhsFunc(t);
                viewSolution(i) = analytical(t);
            }
        );
    }

    //saveToFile(cellSpacing, origin, layout, rhsVector, "rhs.csv");
    //saveToFile(cellSpacing, origin, layout, solutionVector, "solution.csv");

    IpplTimings::stopTimer(initTimer);

    // initialize the solver
    ippl::FEMMaxwellDiffusionSolver<Field_t> solver(lhs, rhs, rhsVector, rhsFunc);
    //saveToFile(cellSpacing, origin,layout, *(solver.rhsVector_m), "solver_rhs.csv");

    // set the parameters
    ippl::ParameterList params;
    params.add("tolerance", 1e-13);
    params.add("max_iterations", 10000);
    solver.mergeParameters(params);
    
    // solve the problem
    ippl::FEMVector<ippl::Vector<T,Dim> > result = solver.solve();
    
    
    ippl::FEMVector<ippl::Vector<T,Dim> > diff = result.template skeletonCopy<ippl::Vector<T,Dim>>();
    diff  = result - solutionVector;

    //saveToFile(cellSpacing, origin, layout, result, "result.csv");
    //saveToFile(nx, ny, cellSpacing, origin, diff, "diff.csv");
    //saveToFile(cellSpacing, origin, layout, *(solver.lhsVector_m), "solver_lhs.csv");
    //saveToFile(cellSpacing, origin, layout, lhs, "field_result.csv");
    //saveToFile(cellSpacing, origin, layout, fieldSolution, "field_solution.csv");
    //fieldSolution = lhs - fieldSolution;
    //saveToFile(cellSpacing, origin, layout, fieldSolution, "field_diff.csv");

    T coefError = solver.getL2ErrorCoeff(*(solver.lhsVector_m), analytical);
    if (ippl::Comm->rank() == 0) {
        std::cout << std::setw(10) << numNodesPerDim;
        std::cout << std::setw(25) << std::setprecision(16) << cellSpacing[0];
        std::cout << std::setw(25) << std::setprecision(16) << coefError;
        std::cout << std::setw(25) << std::setprecision(16) << solver.getResidue();
        std::cout << std::setw(15) << std::setprecision(16) << solver.getIterationCount();
        std::cout << "\n";
    }
    

    /*
    // start the timer
    static IpplTimings::TimerRef errorTimer = IpplTimings::getTimer("computeError");
    IpplTimings::startTimer(errorTimer);

    // Compute the error
    AnalyticSol<T, Dim> analytic;
    const T relError = solver.getL2Error(analytic);

    if (ippl::Comm->rank() == 0) {
        std::cout << std::setw(10) << numNodesPerDim;
        std::cout << std::setw(25) << std::setprecision(16) << cellSpacing[0];
        std::cout << std::setw(25) << std::setprecision(16) << relError;
        std::cout << std::setw(25) << std::setprecision(16) << solver.getResidue();
        std::cout << std::setw(15) << std::setprecision(16) << solver.getIterationCount();
        std::cout << "\n";
    }
    IpplTimings::stopTimer(errorTimer);
    */
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg("");

        using T = double;

        unsigned dim = 3;

        if (argc > 1 && std::atoi(argv[1]) == 1) {
            dim = 1;
        } else if (argc > 1 && std::atoi(argv[1]) == 2) {
            dim = 2;
        }

        // start the timer
        static IpplTimings::TimerRef allTimer = IpplTimings::getTimer("allTimer");
        IpplTimings::startTimer(allTimer);

        msg << std::setw(10) << "Size";
        msg << std::setw(25) << "Spacing";
        msg << std::setw(25) << "Relative Error";
        msg << std::setw(25) << "Residue";
        msg << std::setw(15) << "Iterations";
        msg << endl;

        if (ippl::Comm->rank() == 0) {
            std::cout << std::setw(10) << "num_nodes"
                      << std::setw(25) << "cell_spacing"
                      //<< std::setw(25) << "value_error"
                      //<< std::setw(25) << "interp_error"
                      << std::setw(25) << "interp_error_coef"
                      << std::setw(25) << "solver_residue"
                      << std::setw(15) << "num_it\n";
        }
        /*
        if (dim == 1) {
            // 1D Sinusoidal
            for (unsigned n = 1 << 5; n <= 1 << 12; n = n << 1) {
                testFEMSolver<T, 1>(n, 1.0, 3.0);
            }
        } else if (dim == 2) {
            // 2D Sinusoidal
            for (unsigned n = 1 << 5; n <= 1 << 12; n = n << 1) {
                testFEMSolver<T, 2>(n, 1.0, 3.0);
            }
        } else {
            // 3D Sinusoidal
            for (unsigned n = 1 << 5; n <= 1 << 12; n = n << 1) {
                testFEMSolver<T, 3>(n, 1.0, 3.0);
            }
        }
        */
        
        
        
        
        for (unsigned n = 8; n <= 100; n += 1) {
            testFEMSolver<T, 3>(n, -1.0, 1.0);
        }
          
        
        //testFEMSolver<T, 3>(7, 1.0, 3.0);

        // stop the timer
        IpplTimings::stopTimer(allTimer);

        // print the timers
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
    }
    ippl::finalize();

    return 0;
}
