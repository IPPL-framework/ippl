#ifndef IPPL_FEL_HDF5_WRITER_H
#define IPPL_FEL_HDF5_WRITER_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <hdf5.h>

#include <Kokkos_Core.hpp>

#include "Config.h"
#include "LorentzTransform.h"
#include "datatypes.h"
#include "units.h"

template <typename T, unsigned Dim>
class FELHDF5Writer {
public:
    void open(const config& cfg) {
        if (cfg.output_rhythm == 0 || ippl::Comm->rank() != 0) {
            return;
        }

        std::filesystem::create_directories(cfg.output_path);
        std::stringstream fname;
        fname << cfg.output_path << "fel_output_" << ippl::Comm->size() << ".h5";
        file_ = H5Fcreate(fname.str().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    }

    void close() {
        if (file_ >= 0) {
            H5Fclose(file_);
            file_ = -1;
        }
    }

    template <typename FieldContainer_t, typename ParticleContainer_t, typename Frame_t>
    void writeFrame(int it, double time, FieldContainer_t& fc, ParticleContainer_t& pc,
                    const Frame_t& frame, const config& cfg, const Vector_t<int, Dim>& nr) {
        const int rank = ippl::Comm->rank();
        const int size = ippl::Comm->size();
        const int nx   = nr[0];
        const int nz   = nr[2];

        std::vector<double> localPoyntingMag(nx * nz, 0.0);
        std::vector<double> localPoyntingZ(nx * nz, 0.0);

        auto& mesh      = fc.getMesh();
        const auto h    = mesh.getMeshSpacing();
        auto eHost      = fc.getE().getHostMirror();
        auto bHost      = fc.getB().getHostMirror();
        const auto ldom = fc.getFL().getLocalNDIndex();
        Kokkos::deep_copy(eHost, fc.getE().getView());
        Kokkos::deep_copy(bHost, fc.getB().getView());

        const int nghost = fc.getE().getNghost();
        const int gy     = nr[1] / 2;
        const int ly     = gy - ldom.first()[1] + nghost;

        if (gy >= ldom.first()[1] && gy <= ldom.last()[1]) {
            for (int gx = 0; gx < nx; ++gx) {
                if (gx < ldom.first()[0] || gx > ldom.last()[0]) {
                    continue;
                }
                const int lx = gx - ldom.first()[0] + nghost;
                for (int gz = ldom.first()[2]; gz <= ldom.last()[2]; ++gz) {
                    if (gz < 0 || gz >= nz) {
                        continue;
                    }
                    const int lz = gz - ldom.first()[2] + nghost;
                    ippl::Vector<T, 3> E = eHost(lx, ly, lz);
                    ippl::Vector<T, 3> B = bHost(lx, ly, lz);
                    auto eblab = frame.inverse_transform_EB(
                        Kokkos::make_pair<ippl::Vector<T, 3>, ippl::Vector<T, 3>>(
                            ippl::Vector<T, 3>(E), ippl::Vector<T, 3>(B)));
                    ippl::Vector<T, 3> poynting = ippl::cross(eblab.first, eblab.second);
                    const size_t idx            = static_cast<size_t>(gx) * nz + gz;
                    localPoyntingMag[idx]       = std::sqrt(poynting.Pnorm());
                    localPoyntingZ[idx]         = poynting[2];
                }
            }
        }

        std::vector<double> globalPoyntingMag;
        std::vector<double> globalPoyntingZ;
        if (rank == 0) {
            globalPoyntingMag.resize(localPoyntingMag.size(), 0.0);
            globalPoyntingZ.resize(localPoyntingZ.size(), 0.0);
        }
        MPI_Reduce(localPoyntingMag.data(), rank == 0 ? globalPoyntingMag.data() : nullptr,
                   static_cast<int>(localPoyntingMag.size()), MPI_DOUBLE, MPI_SUM, 0,
                   ippl::Comm->getCommunicator());
        MPI_Reduce(localPoyntingZ.data(), rank == 0 ? globalPoyntingZ.data() : nullptr,
                   static_cast<int>(localPoyntingZ.size()), MPI_DOUBLE, MPI_SUM, 0,
                   ippl::Comm->getCommunicator());

        const size_t nLocal = pc.getLocalNum();
        auto rView          = Kokkos::subview(pc.R.getView(), Kokkos::make_pair(size_t(0), nLocal));
        auto rHost          = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rView);
        const int localParticleValues = static_cast<int>(pc.getLocalNum() * 2);
        std::vector<double> localParticles(std::max(localParticleValues, 1), 0.0);
        for (size_t i = 0; i < nLocal; ++i) {
            localParticles[2 * i]     = rHost(i)[0] * unit_length_in_meters;
            localParticles[2 * i + 1] = rHost(i)[2] * unit_length_in_meters;
        }

        std::vector<int> counts(size, 0);
        MPI_Gather(&localParticleValues, 1, MPI_INT, counts.data(), 1, MPI_INT, 0,
                   ippl::Comm->getCommunicator());

        std::vector<int> displs(size, 0);
        int totalParticleValues = 0;
        if (rank == 0) {
            for (int i = 0; i < size; ++i) {
                displs[i] = totalParticleValues;
                totalParticleValues += counts[i];
            }
        }

        std::vector<double> particles(std::max(totalParticleValues, 1), 0.0);
        MPI_Gatherv(localParticles.data(), localParticleValues, MPI_DOUBLE, particles.data(),
                    counts.data(), displs.data(), MPI_DOUBLE, 0, ippl::Comm->getCommunicator());

        if (rank == 0 && file_ >= 0) {
            writeStepGroup(it, time, cfg, nr, h, globalPoyntingMag, globalPoyntingZ, particles,
                           totalParticleValues / 2);
        }
    }

private:
    template <typename MeshSpacing_t>
    void writeStepGroup(int it, double time, const config& cfg, const Vector_t<int, Dim>& nr,
                        const MeshSpacing_t& h, const std::vector<double>& poyntingMag,
                        const std::vector<double>& poyntingZ,
                        const std::vector<double>& particles, int numParticles) {
        std::stringstream groupName;
        groupName << "/step_" << it;
        hid_t group = H5Gcreate2(file_, groupName.str().c_str(), H5P_DEFAULT, H5P_DEFAULT,
                                 H5P_DEFAULT);
        if (group < 0) {
            return;
        }

        writeScalarAttribute(group, "iteration", it);
        writeScalarAttribute(group, "time", time);
        writeScalarAttribute(group, "unit_length_m", unit_length_in_meters);

        const double extents[3] = {cfg.extents[0] * unit_length_in_meters,
                                   cfg.extents[1] * unit_length_in_meters,
                                   cfg.extents[2] * unit_length_in_meters};
        const double spacing[3] = {h[0] * unit_length_in_meters, h[1] * unit_length_in_meters,
                                   h[2] * unit_length_in_meters};
        const int resolution[3] = {nr[0], nr[1], nr[2]};
        writeVectorAttribute(group, "extents_m", extents, 3);
        writeVectorAttribute(group, "spacing_m", spacing, 3);
        writeVectorAttribute(group, "resolution", resolution, 3);

        const hsize_t planeDims[2] = {static_cast<hsize_t>(nr[0]), static_cast<hsize_t>(nr[2])};
        writeDataset(group, "poynting_magnitude", planeDims, 2, poyntingMag.data());
        writeDataset(group, "poynting_z", planeDims, 2, poyntingZ.data());

        const hsize_t particleDims[2] = {static_cast<hsize_t>(std::max(numParticles, 0)),
                                         static_cast<hsize_t>(2)};
        writeDataset(group, "particle_xz_m", particleDims, 2, particles.data());

        H5Gclose(group);
    }

    template <typename Value>
    void writeScalarAttribute(hid_t object, const char* name, Value value) {
        hid_t space = H5Screate(H5S_SCALAR);
        hid_t attr  = H5Acreate2(object, name, hdf5Type<Value>(), space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr, hdf5Type<Value>(), &value);
        H5Aclose(attr);
        H5Sclose(space);
    }

    template <typename Value>
    void writeVectorAttribute(hid_t object, const char* name, const Value* values, hsize_t size) {
        hid_t space = H5Screate_simple(1, &size, nullptr);
        hid_t attr  = H5Acreate2(object, name, hdf5Type<Value>(), space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr, hdf5Type<Value>(), values);
        H5Aclose(attr);
        H5Sclose(space);
    }

    void writeDataset(hid_t group, const char* name, const hsize_t* dims, int rank,
                      const double* values) {
        hid_t space   = H5Screate_simple(rank, dims, nullptr);
        hid_t dataset = H5Dcreate2(group, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT,
                                   H5P_DEFAULT);
        H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values);
        H5Dclose(dataset);
        H5Sclose(space);
    }

    template <typename Value>
    hid_t hdf5Type() {
        if constexpr (std::is_same_v<Value, double>) {
            return H5T_NATIVE_DOUBLE;
        } else if constexpr (std::is_same_v<Value, int>) {
            return H5T_NATIVE_INT;
        } else {
            static_assert(std::is_same_v<Value, double> || std::is_same_v<Value, int>,
                          "Unsupported HDF5 attribute type");
        }
    }

    hid_t file_ = -1;
};

#endif
