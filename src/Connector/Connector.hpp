#ifndef CONNECTOR_H
#define CONNECTOR_H

// Copyright (c) 2021, Andreas Adelmann
// Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//

#include "PICManager/PICManager.hpp"

/**
 *
 * @FixMe Inform write to file does not work
 * @note CSV output if serial only for the moment. Use this for debug purposes
 **/

namespace Connector {

    enum CONNECTION_TYPE {
        VTK,
        HDF5,
        CSV,
        INSITU
    };

    template <typename T, unsigned Dim = 3>
    class Connector {
        const char* testName_m;
        const CONNECTION_TYPE connType_m;
        // std::map<int, std::unique_ptr<Inform> > streams_m;
        std::map<int, std::unique_ptr<std::ofstream> > streams_m;

        int streamIdCounter_m;

    public:
        Inform* msg;

        Connector(const char* TestName, const CONNECTION_TYPE connType)
            : testName_m(TestName)
            , connType_m(connType)
            , streamIdCounter_m(0)
            , msg(nullptr) {
            //
            msg = new Inform("Connector ");
        }

        ~Connector() {
            if (ippl::Comm->rank() == 0) {
                for (int i = 0; i < streamIdCounter_m; i++)
                    deleteStream(i);
            }
            if (msg)
                delete msg;
        }

        const char* getName() { return testName_m; }

        const char* getConnTypeName() {
            switch (connType_m) {
                case CONNECTION_TYPE::VTK:
                    return "VTK";
                case CONNECTION_TYPE::HDF5:
                    return "HDF5 not yet avaidable";
                case CONNECTION_TYPE::CSV:
                    return "CSV";
                case CONNECTION_TYPE::INSITU:
                    return "INSITY not yet avaidable";
            };

            /* soon we can do this :)
               #include <iostream>
               #include <reflexpr>   (not yet in C++20
               return std::meta::get_base_name_v<
                  std::meta::get_element_m<
                  std::meta::get_enumerators_m<reflexpr(connType_m)>,
                  0>
               >;
            */
        }

        /**
         * @brief Request a stream from the pool.
         *
         * @return Reference to the requested stream.
         */
        int requestStreamId(std::string fileName) {
            // std::unique_ptr<Inform> outputStream(
            // new Inform(NULL, fileName.c_str(), Inform::OVERWRITE, ippl::Comm->rank()));

            std::unique_ptr<std::ofstream> outputStream(new std::ofstream(fileName.c_str()));
            outputStream->precision(10);
            outputStream->setf(std::ios::scientific, std::ios::floatfield);
            int streamId = streamIdCounter_m;
            streamIdCounter_m++;
            streams_m[streamId] = std::move(outputStream);
            return streamId;
        }

        /**
         * @brief Write a message to the stream with the given ID.
         *
         * @param streamId The ID of the stream to write to.
         * @param message The message to write.
         */

        void write(int streamId, const std::string& message) {
            if (ippl::Comm->rank() == 0) {
                auto it = streams_m.find(streamId);
                if (it != streams_m.end()) {
                    (*it->second) << message;
                }
            }
        }

        /**
         * @brief Delete a stream from the pool.
         *
         * @param streamId The ID of the stream to delete.
         */
        void deleteStream(int streamId) {
            if (ippl::Comm->rank() == 0) {
                auto it = streams_m.find(streamId);
                if (it != streams_m.end()) {
                    it->second->flush();
                    streams_m.erase(it);
                }
            }
        }
    };

    template <typename T, unsigned Dim = 3>
    class VTKConnector : public Connector<T, Dim> {
        int myVtkVFieldStreamId_m;
        int myVtkSFieldStreamId_m;

    public:
        VTKConnector(const char* TestName, int iteration)
            : Connector<T, Dim>(TestName, VTK)
            , myVtkVFieldStreamId_m(-1)
            , myVtkSFieldStreamId_m(-1) {
            std::stringstream pname;

            if (ippl::Comm->size() > 1) {
                *Connector<double, Dim>::msg << "Parallel VTK output not yet supported" << endl;
                return;
            }

            /// vector field
            if (ippl::Comm->rank() == 0) {
                pname << "data/"
                      << "ef_" << std::setw(4) << std::setfill('0') << iteration;
                pname << ".vtk";

                myVtkVFieldStreamId_m = Connector<double, Dim>::requestStreamId(pname.str());
            }
            pname = std::stringstream();

            /// scalar field
            if (ippl::Comm->rank() == 0) {
                pname << "data/"
                      << "scalar_" << std::setw(4) << std::setfill('0') << iteration;
                pname << ".vtk";

                myVtkSFieldStreamId_m = Connector<double, Dim>::requestStreamId(pname.str());
            }
        }

        ~VTKConnector() {}

        void dumpVTK(VField_t<T, 3>& E, int nx, int ny, int nz, double dx, double dy, double dz) {
            if (ippl::Comm->size() > 1) {
                return;
            }
            //
            typename VField_t<T, 3>::view_type::host_mirror_type host_view = E.getHostMirror();
            std::stringstream ss;
            Kokkos::deep_copy(host_view, E.getView());

            // start with header
            ss << "# vtk DataFile Version 2.0" << std::endl;
            ss << Connector<T, Dim>::testName_m << std::endl;
            ss << "ASCII" << std::endl;
            ss << "DATASET STRUCTURED_POINTS" << std::endl;
            ss << "DIMENSIONS " << nx + 3 << " " << ny + 3 << " " << nz + 3 << std::endl;
            ss << "ORIGIN " << -dx << " " << -dy << " " << -dz << std::endl;
            ss << "SPACING " << dx << " " << dy << " " << dz << std::endl;
            ss << "CELL_DATA " << (nx + 2) * (ny + 2) * (nz + 2) << std::endl;
            ss << "VECTORS E-Field float" << std::endl;
            Connector<double, Dim>::write(myVtkVFieldStreamId_m, ss.str());
            //
            for (int z = 0; z < nz + 2; z++) {
                for (int y = 0; y < ny + 2; y++) {
                    for (int x = 0; x < nx + 2; x++) {
                        ss << host_view(x, y, z)[0] << "\t" << host_view(x, y, z)[1] << "\t"
                           << host_view(x, y, z)[2] << std::endl;
                    }
                    Connector<double, Dim>::write(myVtkVFieldStreamId_m, ss.str());
                    ss = std::stringstream();
                }
            }
            Connector<double, Dim>::deleteStream(myVtkVFieldStreamId_m);
        }

        void dumpVTK(Field_t<3>& rho, int nx, int ny, int nz, double dx, double dy, double dz) {
            //
            if (ippl::Comm->size() > 1) {
                return;
            }
            typename Field_t<3>::view_type::host_mirror_type host_view = rho.getHostMirror();
            std::stringstream ss;
            Kokkos::deep_copy(host_view, rho.getView());

            // start with header

            ss << "# vtk DataFile Version 2.0" << std::endl;
            ss << Connector<T, Dim>::testName_m << std::endl;
            ss << "ASCII" << std::endl;
            ss << "DATASET STRUCTURED_POINTS" << std::endl;
            ss << "DIMENSIONS " << nx + 3 << " " << ny + 3 << " " << nz + 3 << std::endl;
            ss << "ORIGIN " << -dx << " " << -dy << " " << -dz << std::endl;
            ss << "SPACING " << dx << " " << dy << " " << dz << std::endl;
            ss << "CELL_DATA " << (nx + 2) * (ny + 2) * (nz + 2) << std::endl;
            ss << "SCALARS Rho float" << std::endl;
            ss << "LOOKUP_TABLE default" << std::endl;
            Connector<double, Dim>::write(myVtkSFieldStreamId_m, ss.str());
            //
            for (int z = 0; z < nz + 2; z++) {
                for (int y = 0; y < ny + 2; y++) {
                    for (int x = 0; x < nx + 2; x++) {
                        ss << host_view(x, y, z) << std::endl;
                    }
                    Connector<double, Dim>::write(myVtkSFieldStreamId_m, ss.str());
                    ss = std::stringstream();
                }
            }
            Connector<double, Dim>::deleteStream(myVtkSFieldStreamId_m);
        }
    };

    template <typename T, unsigned Dim = 3>
    class StatisticsConnector : public Connector<T, Dim> {
        size_type totalNumParts_m;
        int myStatStreamId_m;
        int myLbStreamId_m;
        int myLdStreamId_m;
        int myBtStreamId_m;

    public:
        StatisticsConnector(const char* TestName, size_type totalNumParts)
            : Connector<T, Dim>(TestName, CSV)
            , totalNumParts_m(totalNumParts)
            , myStatStreamId_m(-1)
            , myLbStreamId_m(-1)
            , myLdStreamId_m(-1)
            , myBtStreamId_m(-1) {
            std::stringstream pname;

            /// Statistics
            if (ippl::Comm->rank() == 0) {
                pname << "data/"
                      << "ParticleField_" << ippl::Comm->rank();
                pname << ".csv";

                myStatStreamId_m = Connector<double, Dim>::requestStreamId(pname.str());
            }
            pname = std::stringstream();

            /// Load Balancing
            if (ippl::Comm->rank() == 0) {
                pname << "data/"
                      << "LoadBalance_" << ippl::Comm->rank();
                pname << ".csv";

                myLbStreamId_m = Connector<double, Dim>::requestStreamId(pname.str());
            }
            pname = std::stringstream();

            /// Landau Damping
            if (ippl::Comm->rank() == 0) {
                pname << "data/"
                      << "FieldLandau_"
                      << ".csv";

                myLdStreamId_m = Connector<double, Dim>::requestStreamId(pname.str());
            }
            pname = std::stringstream();

            /// Bump on Tail
            if (ippl::Comm->rank() == 0) {
                pname << "data/"
                      << "FieldBumponTail_"
                      << ".csv ";
                myBtStreamId_m = Connector<double, Dim>::requestStreamId(pname.str());
            }
        }

        ~StatisticsConnector() {}

        void gatherFieldStatistics(auto vel, auto rhs, auto vField, Vector_t<T, Dim> hr,
                                   const size_type localNum, double time) {
            auto Vview = vel.getView();

            double kinEnergy = 0.0;
            double potEnergy = 0.0;

            potEnergy = 0.5 * hr[0] * hr[1] * hr[2] * rhs.sum();

            Kokkos::parallel_reduce(
                "Particle Kinetic Energy", localNum,
                KOKKOS_LAMBDA(const int i, double& valL) {
                    double myVal = dot(Vview(i), Vview(i)).apply();
                    valL += myVal;
                },
                Kokkos::Sum<double>(kinEnergy));

            kinEnergy *= 0.5;
            double gkinEnergy = 0.0;

            MPI_Reduce(&kinEnergy, &gkinEnergy, 1, MPI_DOUBLE, MPI_SUM, 0,
                       ippl::Comm->getCommunicator());

            const int nghostE = vField.getNghost();
            auto Eview        = vField.getView();
            Vector_t<T, Dim> normE;

            using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
            for (unsigned d = 0; d < Dim; ++d) {
                T temp = 0.0;
                ippl::parallel_reduce(
                    "Vector E reduce", ippl::getRangePolicy(Eview, nghostE),
                    KOKKOS_LAMBDA(const index_array_type& args, T& valL) {
                        // ippl::apply accesses the view at the given indices and obtains a
                        // reference; see src/Expression/IpplOperations.h
                        T myVal = std::pow(ippl::apply(Eview, args)[d], 2);
                        valL += myVal;
                    },
                    Kokkos::Sum<T>(temp));
                T globaltemp          = 0.0;
                MPI_Datatype mpi_type = get_mpi_datatype<T>(temp);
                MPI_Reduce(&temp, &globaltemp, 1, mpi_type, MPI_SUM, 0,
                           ippl::Comm->getCommunicator());
                normE[d] = std::sqrt(globaltemp);
            }

            std::stringstream ss;
            if (ippl::Comm->rank() == 0) {
                if (time == 0.0) {
                    ss << "time, Potential energy, Kinetic energy, Total energy, Rho_norm2, "
                          "Ex_norm2, Ey_norm2, Ez_norm2"
                       << std::endl;
                } else {
                    double rhoNorm_m = 0.0;  /// FixMe
                    ss << std::setprecision(10) << time << " " << potEnergy << " " << gkinEnergy
                       << " " << potEnergy + gkinEnergy << " " << rhoNorm_m << " ";
                    for (unsigned d = 0; d < Dim; d++) {
                        ss << normE[d] << " ";
                    }
                    ss << std::endl;
                }
                Connector<double, Dim>::write(myStatStreamId_m, ss.str());
            }
            ippl::Comm->barrier();
        }

        void gatherLocalDomainStatistics(const FieldLayout_t<Dim>& fl, const unsigned int step) {
            if (ippl::Comm->rank() == 0) {
                const typename FieldLayout_t<Dim>::host_mirror_type domains =
                    fl.getHostLocalDomains();

                std::ofstream myfile;
                myfile.open("data/domains" + std::to_string(step) + ".txt");
                for (unsigned int i = 0; i < domains.size(); ++i) {
                    for (unsigned d = 0; d < Dim; d++) {
                        myfile << domains[i][d].first() << " " << domains[i][d].last() << " ";
                    }
                    myfile << std::endl;
                }
                myfile.close();
            }
            ippl::Comm->barrier();
        }

        void gatherLoadBalancingStatistics(size_type localNum, double time) {
            std::vector<double> imb(ippl::Comm->size());
            double equalPart = (double)totalNumParts_m / ippl::Comm->size();
            double dev       = (std::abs((double)localNum - equalPart) / totalNumParts_m) * 100.0;
            MPI_Gather(&dev, 1, MPI_DOUBLE, imb.data(), 1, MPI_DOUBLE, 0,
                       ippl::Comm->getCommunicator());

            std::stringstream ss;
            if (ippl::Comm->rank() == 0) {
                if (time == 0.0) {
                    ss << "time, rank, imbalance percentage";
                } else {
                    ss << std::setprecision(10);
                    for (int r = 0; r < ippl::Comm->size(); ++r) {
                        ss << time << " " << r << " " << imb[r];
                    }
                }
                ss << std::endl;
                Connector<double, Dim>::write(myLbStreamId_m, ss.str());
            }
            ippl::Comm->barrier();
        }

        /*

            Application specific routines, could go into seperate classes
            but for now I keep them here

           */

        void dumpLandau(auto Vfield, Vector_t<double, Dim> hr, double time) {
            const int nghostE = Vfield.getNghost();
            auto vFieldView   = Vfield.getView();

            using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
            double localEx2 = 0, localExNorm = 0;
            ippl::parallel_reduce(
                "Ex stats", ippl::getRangePolicy(vFieldView, nghostE),
                KOKKOS_LAMBDA(const index_array_type& args, double& E2, double& ENorm) {
                    // ippl::apply<unsigned> accesses the view at the given indices and obtains
                    // a reference; see src/Expression/IpplOperations.h
                    double val = ippl::apply(vFieldView, args)[0];
                    double e2  = Kokkos::pow(val, 2);
                    E2 += e2;

                    double norm = Kokkos::fabs(ippl::apply(vFieldView, args)[0]);
                    if (norm > ENorm) {
                        ENorm = norm;
                    }
                },
                Kokkos::Sum<double>(localEx2), Kokkos::Max<double>(localExNorm));

            double globaltemp = 0.0;
            MPI_Reduce(&localEx2, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, 0,
                       ippl::Comm->getCommunicator());
            double fieldEnergy =
                std::reduce(hr.begin(), hr.end(), globaltemp, std::multiplies<double>());

            double ExAmp = 0.0;
            MPI_Reduce(&localExNorm, &ExAmp, 1, MPI_DOUBLE, MPI_MAX, 0,
                       ippl::Comm->getCommunicator());

            std::stringstream ss;
            if (ippl::Comm->rank() == 0) {
                if (time == 0.0) {
                    ss << "time, Ex_field_energy, Ex_max_norm";
                } else {
                    ss << std::setprecision(10) << std::ios::scientific << std::ios::floatfield
                       << time << " " << fieldEnergy << " " << ExAmp;
                }
                ss << std::endl;
                Connector<double, Dim>::write(myLdStreamId_m, ss.str());
            }
            ippl::Comm->barrier();
        }

        void dumpBumponTail(auto Vfield, Vector_t<double, Dim> hr, double time) {
            const int nghostE = Vfield.getNghost();
            auto Eview        = Vfield.getView();
            double fieldEnergy, EzAmp;

            using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
            double temp            = 0.0;
            ippl::parallel_reduce(
                "Ex inner product", ippl::getRangePolicy(Eview, nghostE),
                KOKKOS_LAMBDA(const index_array_type& args, double& valL) {
                    // ippl::apply accesses the view at the given indices and obtains a
                    // reference; see src/Expression/IpplOperations.h
                    double myVal = std::pow(ippl::apply(Eview, args)[Dim - 1], 2);
                    valL += myVal;
                },
                Kokkos::Sum<double>(temp));
            double globaltemp = 0.0;
            MPI_Reduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, 0,
                       ippl::Comm->getCommunicator());
            fieldEnergy = std::reduce(hr.begin(), hr.end(), globaltemp, std::multiplies<double>());

            double tempMax = 0.0;
            ippl::parallel_reduce(
                "Ex max norm", ippl::getRangePolicy(Eview, nghostE),
                KOKKOS_LAMBDA(const index_array_type& args, double& valL) {
                    // ippl::apply accesses the view at the given indices and obtains a
                    // reference; see src/Expression/IpplOperations.h
                    double myVal = std::fabs(ippl::apply(Eview, args)[Dim - 1]);
                    if (myVal > valL) {
                        valL = myVal;
                    }
                },
                Kokkos::Max<double>(tempMax));
            EzAmp = 0.0;
            MPI_Reduce(&tempMax, &EzAmp, 1, MPI_DOUBLE, MPI_MAX, 0, ippl::Comm->getCommunicator());
            std::stringstream ss;
            if (ippl::Comm->rank() == 0) {
                if (time == 0.0) {
                    ss << "time, Ez_field_energy, Ez_max_norm";
                } else
                    ss << std::setprecision(10) << time << " " << fieldEnergy << " " << EzAmp;
                ss << std::endl;
                Connector<double, Dim>::write(myBtStreamId_m, ss.str());
            }
            ippl::Comm->barrier();
        }
    };

    template <typename T, unsigned Dim = 3>
    class PhaseSpaceConnector : public Connector<T, Dim> {
        size_type totalNumParts_m;
        int myStreamId_m;

    public:
        PhaseSpaceConnector(const char* TestName, size_type totalNumParts)
            : Connector<T, Dim>(TestName, CSV)
            , totalNumParts_m(totalNumParts)
            , myStreamId_m(-1) {
            //
            if (ippl::Comm->size() > 1) {
                *Connector<double, Dim>::msg << "Parallel phase space output not yet supported"
                                             << endl;
                return;
            }

            std::stringstream pname;
            if (ippl::Comm->rank() == 0) {
                pname << "data/"
                      << "ParticleIC_" << ippl::Comm->rank();
                pname << ".csv";
                myStreamId_m = Connector<double, Dim>::requestStreamId(pname.str());
            }
        }

        ~PhaseSpaceConnector() {
            if (myStreamId_m != -1) {
                Connector<double, Dim>::deleteStream(myStreamId_m);
                myStreamId_m = -1;
            }
        }

        void dumpParticleData(auto RHost, auto VHost, size_type localNum) {
            // FixMe does not work in parallel
            if (ippl::Comm->size() > 1) {
                return;
            }
            *Connector<double, Dim>::msg << "PhaseSpaceConnector write N= " << localNum << endl;
            std::stringstream ss;
            ss << "R_x, R_y, R_z, V_x, V_y, V_z" << std::endl;
            Connector<double, Dim>::write(myStreamId_m, ss.str());
            for (size_type i = 0; i < localNum; i++) {
                std::stringstream stream;
                for (unsigned d = 0; d < Dim; d++)
                    stream << std::setprecision(10) << RHost(i)[d] << " ";
                for (unsigned d = 0; d < Dim; d++)
                    stream << std::setprecision(10) << VHost(i)[d] << " ";
                stream << std::endl;
                Connector<double, Dim>::write(myStreamId_m, stream.str());
            }
            ippl::Comm->barrier();
        }
    };
}  // namespace Connector
#endif
