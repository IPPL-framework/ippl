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

inline std::string pieceFileName(const std::string& name, int step, int rank) {
    std::ostringstream os;
    os << name << "_" << stepString(step) << "_rank" << rank << ".vti";
    return os.str();
}

inline std::string pvtiFileName(const std::string& name, int step) {
    std::ostringstream os;
    os << name << "_" << stepString(step) << ".pvti";
    return os.str();
}

template <unsigned Dim>
void writeExtent(std::ostream& out, const ippl::NDIndex<Dim>& nd, bool pointExtent) {
    static_assert(Dim == 2, "Alvine VTI output currently supports 2D fields only.");

    out << nd[0].first() << " " << nd[0].last() + (pointExtent ? 1 : 0) << " "
        << nd[1].first() << " " << nd[1].last() + (pointExtent ? 1 : 0) << " "
        << 0 << " " << 0;
}

template <typename FieldType, typename VectorType>
void writeScalarPiece(const std::filesystem::path& file, const std::string& dataName,
                      FieldType& field, const VectorType& origin, const VectorType& spacing) {
    static_assert(FieldType::dim == 2, "Scalar VTI output expects a 2D field.");

    auto host = field.getHostMirror();
    Kokkos::deep_copy(host, field.getView());

    const auto& local = field.getLayout().getLocalNDIndex();
    const int nghost  = field.getNghost();

    std::ofstream out(file);
    if (!out) {
        throw std::runtime_error("Could not open VTI piece file: " + file.string());
    }

    out.precision(17);
    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "  <ImageData WholeExtent=\"";
    writeExtent(out, local, true);
    out << "\" Origin=\"" << origin[0] << " " << origin[1] << " 0\" Spacing=\"" << spacing[0]
        << " " << spacing[1] << " 1\">\n";
    out << "    <Piece Extent=\"";
    writeExtent(out, local, true);
    out << "\">\n";
    out << "      <PointData/>\n";
    out << "      <CellData Scalars=\"" << dataName << "\">\n";
    out << "        <DataArray type=\"Float64\" Name=\"" << dataName << "\" format=\"ascii\">\n";

    for (int j = local[1].first(); j <= local[1].last(); ++j) {
        for (int i = local[0].first(); i <= local[0].last(); ++i) {
            const int li = i - local[0].first() + nghost;
            const int lj = j - local[1].first() + nghost;
            out << "          " << host(li, lj) << "\n";
        }
    }

    out << "        </DataArray>\n";
    out << "      </CellData>\n";
    out << "    </Piece>\n";
    out << "  </ImageData>\n";
    out << "</VTKFile>\n";
}

template <typename FieldType, typename VectorType>
void writeVectorPiece(const std::filesystem::path& file, const std::string& dataName,
                      FieldType& field, const VectorType& origin, const VectorType& spacing) {
    static_assert(FieldType::dim == 2, "Vector VTI output expects a 2D field.");

    auto host = field.getHostMirror();
    Kokkos::deep_copy(host, field.getView());

    const auto& local = field.getLayout().getLocalNDIndex();
    const int nghost  = field.getNghost();

    std::ofstream out(file);
    if (!out) {
        throw std::runtime_error("Could not open VTI piece file: " + file.string());
    }

    out.precision(17);
    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "  <ImageData WholeExtent=\"";
    writeExtent(out, local, true);
    out << "\" Origin=\"" << origin[0] << " " << origin[1] << " 0\" Spacing=\"" << spacing[0]
        << " " << spacing[1] << " 1\">\n";
    out << "    <Piece Extent=\"";
    writeExtent(out, local, true);
    out << "\">\n";
    out << "      <PointData/>\n";
    out << "      <CellData Vectors=\"" << dataName << "\">\n";
    out << "        <DataArray type=\"Float64\" Name=\"" << dataName
        << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";

    for (int j = local[1].first(); j <= local[1].last(); ++j) {
        for (int i = local[0].first(); i <= local[0].last(); ++i) {
            const int li = i - local[0].first() + nghost;
            const int lj = j - local[1].first() + nghost;
            out << "          " << host(li, lj)[0] << " " << host(li, lj)[1] << " 0\n";
        }
    }

    out << "        </DataArray>\n";
    out << "      </CellData>\n";
    out << "    </Piece>\n";
    out << "  </ImageData>\n";
    out << "</VTKFile>\n";
}

template <typename FieldType, typename VectorType>
void writePvti(const std::filesystem::path& file, const std::string& dataName,
               const std::string& collectionName, FieldType& field, const VectorType& origin,
               const VectorType& spacing, int step, bool isVector) {
    const auto& layout = field.getLayout();
    const auto& domain = layout.getDomain();

    std::ofstream out(file);
    if (!out) {
        throw std::runtime_error("Could not open PVTI file: " + file.string());
    }

    out.precision(17);
    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"PImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "  <PImageData WholeExtent=\"";
    writeExtent(out, domain, true);
    out << "\" GhostLevel=\"0\" Origin=\"" << origin[0] << " " << origin[1]
        << " 0\" Spacing=\"" << spacing[0] << " " << spacing[1] << " 1\">\n";
    out << "    <PPointData/>\n";
    out << "    <PCellData " << (isVector ? "Vectors" : "Scalars") << "=\"" << dataName
        << "\">\n";
    out << "      <PDataArray type=\"Float64\" Name=\"" << dataName << "\"";
    if (isVector) {
        out << " NumberOfComponents=\"3\"";
    }
    out << "/>\n";
    out << "    </PCellData>\n";

    for (int rank = 0; rank < ippl::Comm->size(); ++rank) {
        const auto& local = layout.getLocalNDIndex(rank);
        out << "    <Piece Extent=\"";
        writeExtent(out, local, true);
        out << "\" Source=\"" << pieceFileName(collectionName, step, rank) << "\"/>\n";
    }

    out << "  </PImageData>\n";
    out << "</VTKFile>\n";
}

template <typename FieldType, typename VectorType>
void writeScalarField2D(const std::string& outputDir, const std::string& name, FieldType& field,
                        const VectorType& origin, const VectorType& spacing, int step) {
    std::filesystem::create_directories(outputDir);

    const int rank = ippl::Comm->rank();
    const auto piecePath = std::filesystem::path(outputDir) / pieceFileName(name, step, rank);
    writeScalarPiece(piecePath, name, field, origin, spacing);

    ippl::Comm->barrier();

    if (rank == 0) {
        const auto pvtiPath = std::filesystem::path(outputDir) / pvtiFileName(name, step);
        writePvti(pvtiPath, name, name, field, origin, spacing, step, false);
    }

    ippl::Comm->barrier();
}

template <typename FieldType, typename VectorType>
void writeVectorField2D(const std::string& outputDir, const std::string& name, FieldType& field,
                        const VectorType& origin, const VectorType& spacing, int step) {
    std::filesystem::create_directories(outputDir);

    const int rank = ippl::Comm->rank();
    const auto piecePath = std::filesystem::path(outputDir) / pieceFileName(name, step, rank);
    writeVectorPiece(piecePath, name, field, origin, spacing);

    ippl::Comm->barrier();

    if (rank == 0) {
        const auto pvtiPath = std::filesystem::path(outputDir) / pvtiFileName(name, step);
        writePvti(pvtiPath, name, name, field, origin, spacing, step, true);
    }

    ippl::Comm->barrier();
}

}  // namespace alvine::vtk

#endif
