#ifndef LANGEVINIO_HPP
#define LANGEVINIO_HPP

#include "LangevinHelpers.hpp"

// Dumping functions currently only work for serial MPI executions

/////////////////
// VTK DUMPING //
/////////////////

void dumpVTKScalar(Field_t<Dim>& F, VectorD_t cellSpacing, VectorD<size_t> nCells, VectorD_t origin,
                   int iteration, double scalingFactor, std::string out_dir, std::string label) {
    int nx = nCells[0];
    int ny = nCells[1];
    int nz = nCells[2];

    const int nghost = F.getNghost();

    typename Field_t<Dim>::view_type::host_mirror_type host_view = F.getHostMirror();

    std::stringstream fname;
    fname << out_dir;
    fname << "/";
    fname << label;
    fname << "_it";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".vtk";

    Kokkos::deep_copy(host_view, F.getView());

    Inform vtkout(NULL, fname.str().c_str(), Inform::OVERWRITE);
    vtkout.precision(10);
    vtkout.setf(std::ios::scientific, std::ios::floatfield);

    // start with header
    vtkout << "# vtk DataFile Version 2.0" << endl;
    vtkout << TestName << endl;
    vtkout << "ASCII" << endl;
    vtkout << "DATASET STRUCTURED_POINTS" << endl;
    vtkout << "DIMENSIONS " << nx + nghost << " " << ny + nghost << " " << nz + nghost << endl;
    vtkout << "ORIGIN " << origin[0] << " " << origin[1] << " " << origin[2] << endl;
    vtkout << "SPACING " << cellSpacing[0] << " " << cellSpacing[1] << " " << cellSpacing[2]
           << endl;
    vtkout << "CELL_DATA " << nx * ny * nz << endl;
    vtkout << "SCALARS " << label << " float" << endl;
    vtkout << "LOOKUP_TABLE default" << endl;
    for (int z = nghost; z < nz + nghost; z++) {
        for (int y = nghost; y < ny + nghost; y++) {
            for (int x = nghost; x < nx + nghost; x++) {
                vtkout << scalingFactor * host_view(x, y, z) << endl;
            }
        }
    }
}

template <typename T>
void dumpVTKVector(VField_t<T, Dim>& F, VectorD_t cellSpacing, VectorD<size_t> nCells,
                   VectorD_t origin, int iteration, double scalingFactor, std::string out_dir,
                   std::string label) {
    int nx = nCells[0];
    int ny = nCells[1];
    int nz = nCells[2];

    const int nghost = F.getNghost();

    typename VField_t<T, Dim>::view_type::host_mirror_type host_view = F.getHostMirror();

    std::stringstream fname;
    fname << out_dir;
    fname << "/";
    fname << label;
    fname << "_it";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".vtk";

    Kokkos::deep_copy(host_view, F.getView());

    Inform vtkout(NULL, fname.str().c_str(), Inform::OVERWRITE);
    vtkout.precision(10);
    vtkout.setf(std::ios::scientific, std::ios::floatfield);

    // start with header
    vtkout << "# vtk DataFile Version 2.0" << endl;
    vtkout << TestName << endl;
    vtkout << "ASCII" << endl;
    vtkout << "DATASET STRUCTURED_POINTS" << endl;
    vtkout << "DIMENSIONS " << nx + nghost << " " << ny + nghost << " " << nz + nghost << endl;
    vtkout << "ORIGIN " << origin[0] << " " << origin[1] << " " << origin[2] << endl;
    vtkout << "SPACING " << cellSpacing[0] << " " << cellSpacing[1] << " " << cellSpacing[2]
           << endl;
    vtkout << "CELL_DATA " << nx * ny * nz << endl;
    vtkout << "VECTORS " << label << " float" << endl;
    for (int z = nghost; z < nz + nghost; z++) {
        for (int y = nghost; y < ny + nghost; y++) {
            for (int x = nghost; x < nx + nghost; x++) {
                vtkout << scalingFactor * host_view(x, y, z)[0] << "\t"
                       << scalingFactor * host_view(x, y, z)[1] << "\t"
                       << scalingFactor * host_view(x, y, z)[2] << endl;
            }
        }
    }
}

/////////////////
// CSV DUMPING //
/////////////////

void dumpCSVMatrix(MatrixD_t m, std::string matrixPrefix, size_type iteration, bool create_header,
                   std::string folder) {
    std::stringstream fname;
    fname << folder << "/";
    fname << matrixPrefix;
    fname << ".csv";

    Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
    csvout.precision(10);
    csvout.setf(std::ios::scientific, std::ios::floatfield);

    if (create_header) {
        // Write header
        csvout << "nv,";

        csvout << matrixPrefix << "_xx," << matrixPrefix << "_xy," << matrixPrefix << "_xz,"
               << matrixPrefix << "_yx," << matrixPrefix << "_yy," << matrixPrefix << "_yz,"
               << matrixPrefix << "_zx," << matrixPrefix << "_zy," << matrixPrefix << "_zz" << endl;
    }

    // And dump into file
    csvout << iteration << ",";
    for (size_type i = 0; i < Dim; ++i) {
        for (size_type j = 0; j < Dim; ++j) {
            csvout << m[i][j];
            if (i == Dim - 1 && j == Dim - 1) {
                csvout << endl;
            } else {
                csvout << ",";
            }
        }
    }
}

void dumpCSVVector(VectorD_t v, std::string vectorPrefix, size_type iteration, bool create_header,
                   std::string folder) {
    std::stringstream fname;
    fname << folder << "/";
    fname << vectorPrefix;
    fname << ".csv";

    Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
    csvout.precision(10);
    csvout.setf(std::ios::scientific, std::ios::floatfield);

    if (create_header) {
        csvout << "nv,";

        // Write header
        csvout << vectorPrefix << "_x," << vectorPrefix << "_y," << vectorPrefix << "_z" << endl;
    }

    // And dump into file
    csvout << iteration << ",";
    for (size_type i = 0; i < Dim; ++i) {
        csvout << v[i];
        if (i == Dim - 1) {
            csvout << endl;
        } else {
            csvout << ",";
        }
    }
}

void dumpCSVScalar(double val, std::string scalarPrefix, size_type iteration, bool create_header,
                   std::string folder) {
    std::stringstream fname;
    fname << folder << "/";
    fname << scalarPrefix;
    fname << ".csv";

    Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
    csvout.precision(10);
    csvout.setf(std::ios::scientific, std::ios::floatfield);

    if (create_header) {
        csvout << "nv,";

        // Write header
        csvout << scalarPrefix << endl;
    }

    // And dump into file
    csvout << iteration << ",";
    csvout << val << endl;
}

template <typename T>
void dumpCSVMatrixField(VField_t<T, Dim>& M0, VField_t<T, Dim>& M1, VField_t<T, Dim>& M2,
                        ippl::Vector<size_type, Dim> nx, std::string matrixPrefix,
                        size_type iteration, std::string folder) {
    VField_view_t M0View = M0.getView();
    VField_view_t M1View = M1.getView();
    VField_view_t M2View = M2.getView();

    typename VField_view_t::host_mirror_type hostM0View = M0.getHostMirror();
    typename VField_view_t::host_mirror_type hostM1View = M1.getHostMirror();
    typename VField_view_t::host_mirror_type hostM2View = M2.getHostMirror();

    Kokkos::deep_copy(hostM0View, M0View);
    Kokkos::deep_copy(hostM1View, M1View);
    Kokkos::deep_copy(hostM2View, M2View);

    const int nghost = M0.getNghost();

    std::stringstream fname;
    fname << folder << "/";
    fname << matrixPrefix << "field_it";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".csv";

    Inform csvout(NULL, fname.str().c_str(), Inform::OVERWRITE);
    csvout.precision(10);
    csvout.setf(std::ios::scientific, std::ios::floatfield);

    // Write header
    csvout << matrixPrefix << "0_x," << matrixPrefix << "0_y," << matrixPrefix << "0_z,"
           << matrixPrefix << "1_x," << matrixPrefix << "1_y," << matrixPrefix << "1_z,"
           << matrixPrefix << "2_x," << matrixPrefix << "2_y," << matrixPrefix << "2_z" << endl;

    // And dump into file
    for (unsigned x = nghost; x < nx[0] + nghost; x++) {
        for (unsigned y = nghost; y < nx[1] + nghost; y++) {
            for (unsigned z = nghost; z < nx[2] + nghost; z++) {
                csvout << hostM0View(x, y, z)[0] << "," << hostM0View(x, y, z)[1] << ","
                       << hostM0View(x, y, z)[2] << "," << hostM1View(x, y, z)[0] << ","
                       << hostM1View(x, y, z)[1] << "," << hostM1View(x, y, z)[2] << ","
                       << hostM2View(x, y, z)[0] << "," << hostM2View(x, y, z)[1] << ","
                       << hostM2View(x, y, z)[2] << endl;
            }
        }
    }
}

void dumpCSVMatrixAttr(ParticleAttrib<VectorD_t>& M0, ParticleAttrib<VectorD_t>& M1,
                       ParticleAttrib<VectorD_t>& M2, size_type particleNum,
                       std::string matrixPrefix, size_type iteration, std::string folder) {
    attr_Dview_t M0View = M0.getView();
    attr_Dview_t M1View = M1.getView();
    attr_Dview_t M2View = M2.getView();

    typename attr_Dview_t::host_mirror_type hostM0View = M0.getHostMirror();
    typename attr_Dview_t::host_mirror_type hostM1View = M1.getHostMirror();
    typename attr_Dview_t::host_mirror_type hostM2View = M2.getHostMirror();

    Kokkos::deep_copy(hostM0View, M0View);
    Kokkos::deep_copy(hostM1View, M1View);
    Kokkos::deep_copy(hostM2View, M2View);

    std::stringstream fname;
    fname << folder << "/";
    fname << matrixPrefix << "attr_it";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".csv";

    Inform csvout(NULL, fname.str().c_str(), Inform::OVERWRITE);
    csvout.precision(10);
    csvout.setf(std::ios::scientific, std::ios::floatfield);

    // Write header
    csvout << matrixPrefix << "0_x," << matrixPrefix << "0_y," << matrixPrefix << "0_z,"
           << matrixPrefix << "1_x," << matrixPrefix << "1_y," << matrixPrefix << "1_z,"
           << matrixPrefix << "2_x," << matrixPrefix << "2_y," << matrixPrefix << "2_z" << endl;

    // And dump into file
    for (size_type i = 0; i < particleNum; ++i) {
        csvout << hostM0View(i)[0] << "," << hostM0View(i)[1] << "," << hostM0View(i)[2] << ","
               << hostM1View(i)[0] << "," << hostM1View(i)[1] << "," << hostM1View(i)[2] << ","
               << hostM2View(i)[0] << "," << hostM2View(i)[1] << "," << hostM2View(i)[2] << endl;
    }
}

#endif /* LANGEVINIO_HPP */
