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

template <typename T, unsigned Dim, typename Layout>
void saveToFile(ippl::Vector<T,2> cellSpacing, ippl::Vector<T,2> origin, const Layout& layout,
    ippl::Field<ippl::Vector<T,Dim>, Dim, ippl::UniformCartesian<T, Dim>, Cell> data, const std::string& filename) {
    
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



template <typename T, unsigned Dim, typename Field>
ippl::FEMVector<T> createFEMVector(const Field& field) {
    using Layout_t = ippl::FieldLayout<Dim>;
    auto view = field.getView();

    // figure out the total number of elements in the field.
    auto& layout = field.getLayout();
    auto ldom = layout.getLocalNDIndex();
    ippl::Vector<size_t, Dim> extents(0);
    for (size_t d = 0; d < Dim; ++d) {
        extents(d) = view.extent(d);
    }
    size_t nx = extents(0);
    size_t ny = extents(1);
    size_t n = nx*(ny-1) + ny*(nx-1);


    // Next up we need to create the neighbor thing and get all the indices
    auto neighbors = layout.getNeighbors();
    auto neighborSendRange = layout.getNeighborsSendRange();
    auto neighborRecvRange = layout.getNeighborsRecvRange();
    std::vector<size_t> neighborsFV;
    std::vector< Kokkos::View<size_t*> > sendIdxs;
    std::vector< Kokkos::View<size_t*> > recvIdxs;
    std::vector< std::vector<size_t> > sendIdxsTemp;
    std::vector< std::vector<size_t> > recvIdxsTemp;

    auto inbetween = [&](size_t a, size_t b) -> size_t {
        if (b < a) {
            size_t t = a;
            a = b;
            b = t;
        }

        ippl::Vector<size_t, Dim> ap(0);
        ippl::Vector<size_t, Dim> bp(0);

        ap[0] = a % nx;
        ap[1] = a / nx;

        bp[0] = b % nx;
        bp[1] = b / nx;

        ippl::Vector<size_t, Dim> dif = bp - ap;

        if (dif[0] == 0) {
            return a + (ap[1] + 1) * (nx - 1);
        } else {
            return a + ap[1] * (nx - 1);
        }
    };
    
    for (size_t i = 0; i < neighbors.size(); ++i) {
        const auto& componentNeighbors = neighbors[i];
        for (size_t j = 0; j < componentNeighbors.size(); ++j) {
            int rank = componentNeighbors[j];
            // check if we already have a rank in there
            const auto it = std::find(neighborsFV.begin(), neighborsFV.end(), rank);
            size_t idx = it - neighborsFV.begin();
            if (it == neighborsFV.end()) {
                // it is not yet in
                neighborsFV.push_back(rank);
                sendIdxsTemp.push_back(std::vector<size_t>());
                recvIdxsTemp.push_back(std::vector<size_t>());
            }



            typename Layout_t::bound_type recvRange = neighborRecvRange[i][j];
            typename Layout_t::bound_type sendRange = neighborSendRange[i][j];
            
            
            if constexpr (Dim == 2) {
                
                // First we do this for the halo cells
                if (recvRange.hi[0] - recvRange.lo[0] == 1) {
                    // are on y axis
                    if (recvRange.lo[0] == extents(0)-1) {
                        // Are on east boundary.
                        // Therefore we have to the the inbetweens parallel to the y axis
                        
                        std::cout << ippl::Comm->rank() << " recv: "
                                  << ldom.first() << "-" << ldom.last() << " " << extents
                                  << " with range "
                                  << "(" <<recvRange.lo[0]<<","<< recvRange.lo[1]<< ")"
                                  << "-" << "("<<recvRange.hi[0]<<","<<recvRange.hi[1]<<")"
                                  << " -> east\n";
                        
                        size_t x = recvRange.lo[0];
                        // Note that we are going one y further, this is just
                        // because we have one DOF on the edge going up which
                        // we need to add. This does not lead to problems with
                        // the domain boundary as this never part of the mpi
                        // boundaries anyhow.
                        // Similar thing goes for the lower end.
                        for (size_t y = recvRange.lo[1];
                                y < recvRange.hi[1]; ++y) {
                            size_t a = y * nx + x;
                            size_t b = (y+1) * nx + x;

                            size_t pos = inbetween(a,b);
                            if (pos < n) {
                                recvIdxsTemp[idx].push_back(pos);
                            }
                            
                            if (pos > n-1)
                                std::cout << pos << " / " << n << ", (" << x << "," << y << ")"
                                          << " - (" << x << "," << y+1 << ")\n";
                        } 
                    } else {
                        // Are on west boundary
                        // Therefore have to take the inbetweens on the y axis
                        // and the inbetweens on the x-axis.
                        
                        std::cout << ippl::Comm->rank() << " recv: "
                                  << ldom.first() << "-" << ldom.last() << " " << extents
                                  << " with range "
                                  << "(" <<recvRange.lo[0]<<","<< recvRange.lo[1]<< ")"
                                  << "-" << "("<<recvRange.hi[0]<<","<<recvRange.hi[1]<<")"
                                  << " -> west\n";
                        
                        size_t x = recvRange.lo[0];
                        // the ones on the y axis
                        // Note that we are going one y further, this is just
                        // because we have one DOF on the edge going up which
                        // we need to add. This does not lead to problems with
                        // the domain boundary as this never part of the mpi
                        // boundaries anyhow.
                        for (size_t y = recvRange.lo[1];
                                //y < recvRange.hi[1]; ++y) {
                                y < recvRange.hi[1]-1; ++y) {
                            size_t a = y * nx + x;
                            size_t b = (y+1) * nx + x;
                            size_t pos = inbetween(a,b);
                            if (pos < n) {
                                recvIdxsTemp[idx].push_back(pos);
                            }
                            
                            if (pos > n-1)
                                std::cout << pos << " / " << n << ", (" << x << "," << y << ")"
                                          << " - (" << x << "," << y+1 << ")\n";
                        }
                        
                        // the ones on the x-axis
                        // Note that we are going one y further, this is just
                        // because we have one DOF on the edge going up which
                        // we need to add. This does not lead to problems with
                        // the domain boundary as this never part of the mpi
                        // boundaries anyhow.
                        for (size_t y = recvRange.lo[1];
                                y < recvRange.hi[1]; ++y) {
                            size_t a = y * nx + x;
                            size_t b = y * nx + (x+1);
                            size_t pos = inbetween(a,b);
                            if (pos < n) {
                                recvIdxsTemp[idx].push_back(pos);
                            }
                            
                            if (pos > n-1)
                                std::cout << pos << " / " << n << ", (" << x << "," << y << ")"
                                          << " - (" << x+1 << "," << y << ")\n";
                        }
                    }
                } else {
                    // are on x axis
                    if (recvRange.lo[1] == extents(1)-1) {
                        // Are on north boundary.
                        // Therefore we have to take the in betweens parallel to
                        // the x axis.
                        
                        std::cout << ippl::Comm->rank() << " recv: "
                                  << ldom.first() << "-" << ldom.last() << " " << extents
                                  << " with range "
                                  << "(" <<recvRange.lo[0]<<","<< recvRange.lo[1]<< ")"
                                  << "-" << "("<<recvRange.hi[0]<<","<<recvRange.hi[1]<<")"
                                  << " -> north\n";
                        
                        size_t y = recvRange.lo[1];
                        // Note that we are going one x further, this is just
                        // because we have one DOF on the edge going east which
                        // we need to add. This does not lead to problems with
                        // the domain boundary as this never part of the mpi
                        // boundaries anyhow.
                        for (size_t x = recvRange.lo[0];
                                x < recvRange.hi[0]; ++x) {
                            size_t a = y * nx + x;
                            size_t b = y * nx + x + 1;

                            size_t pos = inbetween(a,b);
                            if (pos < n) {
                                recvIdxsTemp[idx].push_back(pos);
                            }
                            
                            if (pos > n-1)
                                std::cout << pos << " / " << n << ", (" << x << "," << y << ")"
                                          << " - (" << x+1 << "," << y << ")\n";
                        } 
                    } else {
                        // Are on south boundary.
                        // Therefore have to take the inbetweens on the x axis
                        // and the ones on the y axis.
                        
                        std::cout << ippl::Comm->rank() << " recv: "
                                  << ldom.first() << "-" << ldom.last() << " " << extents
                                  << " with range "
                                  << "(" <<recvRange.lo[0]<<","<< recvRange.lo[1]<< ")"
                                  << "-" << "("<<recvRange.hi[0]<<","<<recvRange.hi[1]<<")"
                                  << " -> south\n";
                        
                        size_t y = recvRange.lo[1];

                        // x-axis
                        // Note that we are going one x further, this is just
                        // because we have one DOF on the edge going east which
                        // we need to add. This does not lead to problems with
                        // the domain boundary as this never part of the mpi
                        // boundaries anyhow.
                        for (size_t x = recvRange.lo[0];
                                //x < recvRange.hi[0]; ++x) {
                                x < recvRange.hi[0]-1; ++x) {
                            size_t a = y * nx + x;
                            size_t b = y * nx + x + 1;
                            size_t pos = inbetween(a,b);
                            if (pos < n) {
                                recvIdxsTemp[idx].push_back(pos);
                            }
                            
                            if (pos > n-1)
                                std::cout << pos << " / " << n << ", (" << x << "," << y << ")"
                                          << " - (" << x+1 << "," << y << ")\n";
                        } 

                        // y axis
                        // Note that we are going one x further, this is just
                        // because we have one DOF on the edge going east which
                        // we need to add. This does not lead to problems with
                        // the domain boundary as this never part of the mpi
                        // boundaries anyhow.
                        for (size_t x = recvRange.lo[0];
                                x < recvRange.hi[0]; ++x) {
                            size_t a = y * nx + x;
                            size_t b = (y+1) * nx + x;
                            size_t pos = inbetween(a,b);
                            if (pos < n) {
                                recvIdxsTemp[idx].push_back(pos);
                            }
                            
                            if (pos > n-1)
                                std::cout << pos << " / " << n << ", (" << x << "," << y << ")"
                                          << " - (" << x << "," << y+1 << ")\n";
                        } 
                    }
                }



                // First we do this for the halo cells
                if (sendRange.hi[0] - sendRange.lo[0] == 1) {
                    // are on y axis
                    if (sendRange.lo[0] == extents(0)-2) {
                        // Are on east boundary.
                        // Therefore we have to the the inbetweens parallel to the y axis
                        
                        std::cout << ippl::Comm->rank() << " send: "
                                  << ldom.first() << "-" << ldom.last() << " " << extents
                                  << " with range "
                                  << "(" <<sendRange.lo[0]<<","<< sendRange.lo[1]<< ")"
                                  << "-" << "("<<sendRange.hi[0]<<","<<sendRange.hi[1]<<")"
                                  << " -> east\n";
                        
                        size_t x = sendRange.lo[0];
                        // the ones on the y axis
                        // Note that we are going one y further, this is just
                        // because we have one DOF on the edge going up which
                        // we need to add. This does not lead to problems with
                        // the domain boundary as this never part of the mpi
                        // boundaries anyhow.
                        for (size_t y = sendRange.lo[1];
                                //y < sendRange.hi[1]; ++y) {
                                y < sendRange.hi[1]-1; ++y) {
                            size_t a = y * nx + x;
                            size_t b = (y+1) * nx + x;
                            size_t pos = inbetween(a,b);
                            if (pos < n) {
                                sendIdxsTemp[idx].push_back(pos);
                            }
                        }
                        
                        // the ones on the x-axis
                        // Note that we are going one y further, this is just
                        // because we have one DOF on the edge going up which
                        // we need to add. This does not lead to problems with
                        // the domain boundary as this never part of the mpi
                        // boundaries anyhow.
                        for (size_t y = sendRange.lo[1];
                                y < sendRange.hi[1]; ++y) {
                            size_t a = y * nx + x;
                            size_t b = y * nx + (x+1);
                            size_t pos = inbetween(a,b);
                            if (pos < n) {
                                sendIdxsTemp[idx].push_back(pos);
                            }
                        } 
                    } else {
                        // Are on west boundary
                        // Therefore have to take the inbetweens on the y axis
                        // and the inbetweens on the x-axis.
                        
                        std::cout << ippl::Comm->rank() << " send: "
                                  << ldom.first() << "-" << ldom.last() << " " << extents
                                  << " with range "
                                  << "(" <<sendRange.lo[0]<<","<< sendRange.lo[1]<< ")"
                                  << "-" << "("<<sendRange.hi[0]<<","<<sendRange.hi[1]<<")"
                                  << " -> west\n";
                        
                        size_t x = sendRange.lo[0];
                        // Note that we are going one y further, this is just
                        // because we have one DOF on the edge going up which
                        // we need to add. This does not lead to problems with
                        // the domain boundary as this never part of the mpi
                        // boundaries anyhow.
                        // Similar thing goes for the lower end.
                        for (size_t y = sendRange.lo[1];
                                y < sendRange.hi[1]; ++y) {
                            size_t a = y * nx + x;
                            size_t b = (y+1) * nx + x;

                            size_t pos = inbetween(a,b);
                            if (pos < n) {
                                sendIdxsTemp[idx].push_back(pos);
                            }
                            
                            if (pos > n-1)
                                std::cout << pos << " / " << n << ", (" << x << "," << y << ")"
                                          << " - (" << x << "," << y+1 << ")\n";
                        } 
                    }
                } else {
                    // are on x axis
                    if (sendRange.lo[1] == extents(1)-2) {
                        // Are on north boundary.
                        // Therefore we have to take the in betweens parallel to
                        // the x axis.
                        
                        std::cout << ippl::Comm->rank() << " send: "
                                  << ldom.first() << "-" << ldom.last() << " " << extents
                                  << " with range "
                                  << "(" <<sendRange.lo[0]<<","<< sendRange.lo[1]<< ")"
                                  << "-" << "("<<sendRange.hi[0]<<","<<sendRange.hi[1]<<")"
                                  << " -> north\n";
                        
                        size_t y = sendRange.lo[1];
                        // x-axis
                        // Note that we are going one x further, this is just
                        // because we have one DOF on the edge going east which
                        // we need to add. This does not lead to problems with
                        // the domain boundary as this never part of the mpi
                        // boundaries anyhow.
                        for (size_t x = sendRange.lo[0];
                                //x < sendRange.hi[0]; ++x) {
                                x < sendRange.hi[0]-1; ++x) {
                            size_t a = y * nx + x;
                            size_t b = y * nx + x + 1;
                            size_t pos = inbetween(a,b);
                            if (pos < n) {
                                sendIdxsTemp[idx].push_back(pos);
                            }
                        } 

                        // y axis
                        // Note that we are going one x further, this is just
                        // because we have one DOF on the edge going east which
                        // we need to add. This does not lead to problems with
                        // the domain boundary as this never part of the mpi
                        // boundaries anyhow.
                        for (size_t x = sendRange.lo[0];
                                x < sendRange.hi[0]; ++x) {
                            size_t a = y * nx + x;
                            size_t b = (y+1) * nx + x;
                            size_t pos = inbetween(a,b);
                            if (pos < n) {
                                sendIdxsTemp[idx].push_back(pos);
                            }
                        }

                    } else {
                        // Are on south boundary.
                        // Therefore have to take the inbetweens on the x axis
                        // and the ones on the y axis.
                        
                        std::cout << ippl::Comm->rank() << " send: "
                                  << ldom.first() << "-" << ldom.last() << " " << extents
                                  << " with range "
                                  << "(" <<sendRange.lo[0]<<","<< sendRange.lo[1]<< ")"
                                  << "-" << "("<<sendRange.hi[0]<<","<<sendRange.hi[1]<<")"
                                  << " -> south\n";
                        
                        size_t y = sendRange.lo[1]; 

                        // Note that we are going one x further, this is just
                        // because we have one DOF on the edge going east which
                        // we need to add. This does not lead to problems with
                        // the domain boundary as this never part of the mpi
                        // boundaries anyhow.
                        for (size_t x = sendRange.lo[0];
                                x < sendRange.hi[0]; ++x) {
                            size_t a = y * nx + x;
                            size_t b = y * nx + x + 1;

                            size_t pos = inbetween(a,b);
                            if (pos < n) {
                                sendIdxsTemp[idx].push_back(pos);
                            }
                        } 
                    }
                }
            } else if constexpr (Dim == 3) {

            }            
            
        }
    }

    

    // now copy all the data from the Temps to the real deal
    
    for (size_t i = 0; i < neighborsFV.size(); ++i) {
        sendIdxs.push_back(Kokkos::View<size_t*>("FEMvector::sendIdxs[" + std::to_string(i) +
                                                    "]", sendIdxsTemp[i].size()));
        recvIdxs.push_back(Kokkos::View<size_t*>("FEMvector::recvIdxs[" + std::to_string(i) +
                                                    "]", recvIdxsTemp[i].size()));
        auto sendView = sendIdxs[i];
        auto recvView = recvIdxs[i];
        auto hSendView = Kokkos::create_mirror_view(sendView);
        auto hRecvView = Kokkos::create_mirror_view(recvView);
        /*
        std::cout << "rank " << ippl::Comm->rank() << ": "
                  << sendIdxsTemp[i].size() << " vs " << recvIdxsTemp[i].size() << "\n";
        */
        for (size_t j = 0; j < sendIdxsTemp[i].size(); ++j) {
            hSendView(j) = sendIdxsTemp[i][j];
            assert(hSendView(j) < n);
        }

        for (size_t j = 0; j < recvIdxsTemp[i].size(); ++j) {
            hRecvView(j) = recvIdxsTemp[i][j];
            assert(hRecvView(j) < n);
        }

        Kokkos::deep_copy(sendView, hSendView);
        Kokkos::deep_copy(recvView, hRecvView);
    }
    

    
    // Now finaly create the FEMVector
    ippl::FEMVector<T> vec(n, neighborsFV, sendIdxs, recvIdxs);
    /*
    // set the values of the FEMVector
    auto vecView = vec.getView();
    auto fieldView = field.getView();
    using index_array_type =
            typename RangePolicy<Dim, typename FieldRHS::execution_space>::index_array_type;
    ippl::parallel_for(
        "LagrangeSpaceFEMVector::interpolateToFEMVector", getRangePolicy(fieldView),
        KOKKOS_LAMBDA(const index_array_type& args) {
            size_t idx = 0;
            for (unsigned i = 0; i < Dim; ++i) {
                idx += args[i]*v[i];
            }
            vecView[idx] = apply(fieldView, args);
        }
    );
    */
    return vec;
}


template <typename T, unsigned Dim>
struct Analytical{
    using point_t =  ippl::Vector<T, Dim>;

    T k;
    Analytical(T k) : k(k) {}

    KOKKOS_FUNCTION const point_t operator() (const point_t& pos) const{
        point_t sol(0);
        sol[0] = Kokkos::sin(k*pos[1]);
        sol[1] = Kokkos::sin(k*pos[0]);
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

    auto vec = createFEMVector<T,Dim, Field_t>(lhs);


    
    T k = 3.14159265359;


    // Generate the rhs FEMVector
    auto ldom = layout.getLocalNDIndex();
    auto d = ldom.last() - ldom.first();
    size_t nx = d[0] + 1 + 2;
    size_t ny = d[1] + 1 + 2;

    saveMPILayout<T,ippl::FieldLayout<Dim>,Dim>(cellSpacing, origin, vec, layout);
 
    ippl::FEMVector<ippl::Vector<T,Dim> > rhsVector = createFEMVector<ippl::Vector<T,Dim>,Dim, Field_t>(lhs);
    auto viewRhs = rhsVector.getView();

    ippl::FEMVector<ippl::Vector<T,Dim> > solutionVector = createFEMVector<ippl::Vector<T,Dim>,Dim, Field_t>(lhs);
    auto viewSolution = solutionVector.getView();

    Field_t fieldSolution(mesh, layout, 0);
    auto fieldView = fieldSolution.getView();
    using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
    ippl::parallel_for("Assign solution field", fieldSolution.getFieldRangePolicy(),
        KOKKOS_LAMBDA(const index_array_type& args) {
            T x = args[0]*cellSpacing[0] + origin[0] + ldom.first()[0]*cellSpacing[0];
            T y = args[1]*cellSpacing[1] + origin[1] + ldom.first()[1]*cellSpacing[1];

            apply(fieldView, args)[0] = Kokkos::sin(k*y);
            apply(fieldView, args)[1] = Kokkos::sin(k*x);
            
        }
    );

    

    auto rhsFunc = KOKKOS_LAMBDA(const point_t& pos) -> point_t {
        point_t sol(0);
        sol[0] = (1. + k*k)*Kokkos::sin(k*pos[1]);
        sol[1] = (1. + k*k)*Kokkos::sin(k*pos[0]);
        return sol;
    };



    Kokkos::parallel_for("Assign RHS", rhsVector.size(),
        KOKKOS_LAMBDA(size_t i) {

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
            
            viewRhs(i)[0] = (1. + k*k)*Kokkos::sin(k*y);
            viewRhs(i)[1] = (1. + k*k)*Kokkos::sin(k*x);

            if (x < 1.0 || x > 3.0) viewRhs(i) *= 0;
            if (y < 1.0 || y > 3.0) viewRhs(i) *= 0;

            viewSolution(i)[0] = Kokkos::sin(k*y);
            viewSolution(i)[1] = Kokkos::sin(k*x);

            if (x < 1.0 || x > 3.0) viewSolution(i) *= 0;
            if (y < 1.0 || y > 3.0) viewSolution(i) *= 0;
        }
    );

    saveToFile(cellSpacing, origin, layout, rhsVector, "rhs.csv");
    saveToFile(cellSpacing, origin, layout, solutionVector, "solution.csv");

    IpplTimings::stopTimer(initTimer);

    // initialize the solver
    ippl::FEMMaxwellDiffusionSolver<Field_t> solver(lhs, rhs, rhsVector, rhsFunc);
    saveToFile(cellSpacing, origin,layout, *(solver.rhsVector_m), "solver_rhs.csv");

    // set the parameters
    ippl::ParameterList params;
    params.add("tolerance", 1e-13);
    params.add("max_iterations", 10000);
    solver.mergeParameters(params);

    // solve the problem
    ippl::FEMVector<ippl::Vector<T,Dim> > result = solver.solve();
    ippl::FEMVector<ippl::Vector<T,Dim> > diff = result.template skeletonCopy<ippl::Vector<T,Dim>>();
    diff  = result - solutionVector;
    saveToFile(cellSpacing, origin, layout, result, "result.csv");
    //saveToFile(nx, ny, cellSpacing, origin, diff, "diff.csv");
 
    saveToFile(cellSpacing, origin, layout, *(solver.lhsVector_m), "solver_lhs.csv");

    saveToFile(cellSpacing, origin, layout, lhs, "field_result.csv");
    //saveToFile(nx, ny, cellSpacing, origin, fieldSolution, "field_solution.csv");
    //fieldSolution = lhs - fieldSolution;
    //saveToFile(nx, ny, cellSpacing, origin, fieldSolution, "field_diff.csv");

    auto fieldDifView = fieldSolution.getView();
    auto hFieldDifView = Kokkos::create_mirror_view(fieldDifView);
    Kokkos::deep_copy(hFieldDifView, fieldDifView);
    T difError = 0;
    for (size_t i = 0; i < hFieldDifView.extent(0); ++i) {
        for (size_t j = 0; j < hFieldDifView.extent(1); ++j) {
            difError += Kokkos::sqrt(hFieldDifView(i,j).dot(hFieldDifView(i,j)));
        }
    }
    difError /= hFieldDifView.extent(0);


    ippl::FEMVector<ippl::Vector<T,Dim> > dummy = result.template skeletonCopy<ippl::Vector<T,Dim>>(); 

    auto resultView = result.getView();
    auto solutionView = solutionVector.getView();
    auto hResultView = Kokkos::create_mirror_view(resultView);
    auto hSolutionView = Kokkos::create_mirror_view(solutionView);
    Kokkos::deep_copy(hResultView, resultView);
    Kokkos::deep_copy(hSolutionView, solutionView);

    T s = 0;
    for (size_t i = 0; i < hSolutionView.extent(0); ++i) {
        auto a = hResultView(i) - hSolutionView(i);
        s += dot(a,a).apply();
    }

    s = Kokkos::sqrt(s)/(Dim*nx);
    
    //T error = solver.getL2Error(result, Analytical<T, Dim>(k));
    T coefError = solver.getL2ErrorCoeff(*(solver.lhsVector_m), Analytical<T, Dim>(k));
    
    if (ippl::Comm->rank() == 0) {
        std::cout << std::setw(10) << numNodesPerDim;
        std::cout << std::setw(25) << std::setprecision(16) << cellSpacing[0];
        std::cout << std::setw(25) << std::setprecision(16) << difError;
        //std::cout << std::setw(25) << std::setprecision(16) << error;
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
                      << std::setw(25) << "value_error"
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
        
        
        /*
        for (unsigned n = 8; n <= 256; n *= 2) {
            testFEMSolver<T, 2>(n, 0, 3.0);
        }
            */
        
        
          
        
        testFEMSolver<T, 2>(103, 1.0, 3.0);

        // stop the timer
        IpplTimings::stopTimer(allTimer);

        // print the timers
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
    }
    ippl::finalize();

    return 0;
}
