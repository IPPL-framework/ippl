#ifndef IPPL_ALVINE_VTK_DUMP_HPP
#define IPPL_ALVINE_VTK_DUMP_HPP

#include <Kokkos_Core.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

#include "Ippl.h"

namespace alvine::vtk {

inline std::string stepString(int step) {
    std::ostringstream os;
    os << std::setw(6) << std::setfill('0') << step;
    return os.str();
}

inline std::filesystem::path legacyFileName(const std::string& outputDir, const std::string& name,
                                            int step) {
    std::ostringstream os;
    os << name << "_" << stepString(step);
    if (ippl::Comm->size() > 1) {
        os << "_rank" << ippl::Comm->rank();
    }
    os << ".vtk";
    return std::filesystem::path(outputDir) / os.str();
}

inline void ensureOutputDirectory(const std::string& outputDir) {
    if (ippl::Comm->rank() == 0) {
        std::filesystem::create_directories(outputDir);
    }
    ippl::Comm->barrier();
}

template <typename FieldType, typename VectorType>
void writeScalarField2D(const std::string& outputDir, const std::string& name, FieldType& field,
                        const VectorType& origin, const VectorType& spacing, int step) {
    static_assert(FieldType::dim == 2, "Legacy Alvine VTK output expects a 2D field.");

    ensureOutputDirectory(outputDir);

    auto host = field.getHostMirror();
    Kokkos::deep_copy(host, field.getView());

    const auto& local = field.getLayout().getLocalNDIndex();
    const int nghost  = field.getNghost();
    const int nx      = local[0].last() - local[0].first() + 1;
    const int ny      = local[1].last() - local[1].first() + 1;

    const auto file = legacyFileName(outputDir, name, step);
    std::ofstream vtkout(file, std::ios::out);
    if (!vtkout) {
        throw std::runtime_error("Could not open VTK file: " + file.string());
    }

    vtkout.precision(10);
    vtkout.setf(std::ios::scientific, std::ios::floatfield);

    vtkout << "# vtk DataFile Version 2.0\n";
    vtkout << name << "\n";
    vtkout << "ASCII\n";
    vtkout << "DATASET STRUCTURED_POINTS\n";
    vtkout << "DIMENSIONS " << nx + 1 << " " << ny + 1 << " 2\n";
    vtkout << "ORIGIN " << origin[0] + local[0].first() * spacing[0] << " "
           << origin[1] + local[1].first() * spacing[1] << " 0\n";
    vtkout << "SPACING " << spacing[0] << " " << spacing[1] << " 1\n";
    vtkout << "CELL_DATA " << nx * ny << "\n";
    vtkout << "SCALARS " << name << " float\n";
    vtkout << "LOOKUP_TABLE default\n";

    for (int j = local[1].first(); j <= local[1].last(); ++j) {
        for (int i = local[0].first(); i <= local[0].last(); ++i) {
            const int li = i - local[0].first() + nghost;
            const int lj = j - local[1].first() + nghost;
            vtkout << host(li, lj) << "\n";
        }
    }
}

template <typename FieldType, typename VectorType>
void writeVectorField2D(const std::string& outputDir, const std::string& name, FieldType& field,
                        const VectorType& origin, const VectorType& spacing, int step) {
    static_assert(FieldType::dim == 2, "Legacy Alvine VTK output expects a 2D field.");

    ensureOutputDirectory(outputDir);

    auto host = field.getHostMirror();
    Kokkos::deep_copy(host, field.getView());

    const auto& local = field.getLayout().getLocalNDIndex();
    const int nghost  = field.getNghost();
    const int nx      = local[0].last() - local[0].first() + 1;
    const int ny      = local[1].last() - local[1].first() + 1;

    const auto file = legacyFileName(outputDir, name, step);
    std::ofstream vtkout(file, std::ios::out);
    if (!vtkout) {
        throw std::runtime_error("Could not open VTK file: " + file.string());
    }

    vtkout.precision(10);
    vtkout.setf(std::ios::scientific, std::ios::floatfield);

    vtkout << "# vtk DataFile Version 2.0\n";
    vtkout << name << "\n";
    vtkout << "ASCII\n";
    vtkout << "DATASET STRUCTURED_POINTS\n";
    vtkout << "DIMENSIONS " << nx + 1 << " " << ny + 1 << " 2\n";
    vtkout << "ORIGIN " << origin[0] + local[0].first() * spacing[0] << " "
           << origin[1] + local[1].first() * spacing[1] << " 0\n";
    vtkout << "SPACING " << spacing[0] << " " << spacing[1] << " 1\n";
    vtkout << "CELL_DATA " << nx * ny << "\n";
    vtkout << "VECTORS " << name << " float\n";

    for (int j = local[1].first(); j <= local[1].last(); ++j) {
        for (int i = local[0].first(); i <= local[0].last(); ++i) {
            const int li = i - local[0].first() + nghost;
            const int lj = j - local[1].first() + nghost;
            vtkout << host(li, lj)[0] << "\t" << host(li, lj)[1] << "\t0\n";
        }
    }
}

template <typename ParticleContainerType>
void writeParticles2D(const std::string& outputDir, const std::string& name,
                      ParticleContainerType& particles, int step) {
    ensureOutputDirectory(outputDir);

    auto rHost     = particles.R.getHostMirror();
    auto omegaHost = particles.omega.getHostMirror();

    Kokkos::deep_copy(rHost, particles.R.getView());
    Kokkos::deep_copy(omegaHost, particles.omega.getView());

    const auto nlocal = particles.getLocalNum();
    const auto file   = legacyFileName(outputDir, name, step);

    std::ofstream vtkout(file, std::ios::out);
    if (!vtkout) {
        throw std::runtime_error("Could not open VTK file: " + file.string());
    }

    vtkout.precision(10);
    vtkout.setf(std::ios::scientific, std::ios::floatfield);

    vtkout << "# vtk DataFile Version 2.0\n";
    vtkout << name << "\n";
    vtkout << "ASCII\n";
    vtkout << "DATASET POLYDATA\n";
    vtkout << "POINTS " << nlocal << " float\n";

    for (ippl::detail::size_type i = 0; i < nlocal; ++i) {
        vtkout << rHost(i)[0] << "\t" << rHost(i)[1] << "\t0\n";
    }

    vtkout << "VERTICES " << nlocal << " " << 2 * nlocal << "\n";
    for (ippl::detail::size_type i = 0; i < nlocal; ++i) {
        vtkout << "1 " << i << "\n";
    }

    vtkout << "POINT_DATA " << nlocal << "\n";
    vtkout << "SCALARS omega float\n";
    vtkout << "LOOKUP_TABLE default\n";
    for (ippl::detail::size_type i = 0; i < nlocal; ++i) {
        vtkout << omegaHost(i) << "\n";
    }
}

}  // namespace alvine::vtk

#endif
