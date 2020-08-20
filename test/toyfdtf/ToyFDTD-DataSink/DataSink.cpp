#include "DataSink.h"
#include "Ippl.h"

#include <fstream>
#include <cstdio>
#include <string>

#ifdef __ICC
#include <stdint.h>
#else
#include <cstdint>
#endif

DataSink::DataSink(UniformCartesian<3>& mesh, CenteredFieldLayout<3,UniformCartesian<3>,Cell>& field_layout, const int& guard_cell_size, 
                   int &nx, int &ny, int &nz,
                   double &dx, double& dy, double &dz,
                   bool &binary, NDIndex<3> &lDom):
    Nx_m(nx),
    Ny_m(ny),
    Nz_m(nz),
    dx_m(dx),
    dy_m(dy),
    dz_m(dz),
    binary_m(binary),
    iteration_m(0),
    NBV(mesh, field_layout, GuardCellSizes<3>(guard_cell_size)),
    lDoms(6*Ippl::getNodes(),0)
{
    int lxf, lxl, lyf, lyl, lzf, lzl;

    if (lDom[0].first() == 0) {
        lxf = 0;
    } else {
        lxf = lDom[0].first() - 1;
    }

    if (lDom[0].last() == Nx_m) {
        lxl = Nx_m;// + 1;
    } else {
        lxl = lDom[0].last() + 2;
    }

    if (lDom[1].first() == 0) {
        lyf = 0;
    } else {
        lyf = lDom[1].first() - 1;
    }

    if (lDom[1].last() == Ny_m) {
        lyl = Ny_m;// + 1;
    } else {
        lyl = lDom[1].last() + 2;
    }

    if (lDom[2].first() == 0) {
        lzf = 0;
    } else {
        lzf = lDom[2].first() - 1;
    }

    if (lDom[2].last() == Nz_m) {
        lzl = Nz_m;// + 1;
    } else {
        lzl = lDom[2].last() + 2;
    }

//     lDoms = new int[6*Ippl::getNodes()];
//     for (int l = 0; l < 6*Ippl::getNodes(); ++ l) lDoms[l] = 0;

    lDoms[6 * Ippl::myNode()] = lxf;
    lDoms[6 * Ippl::myNode() + 1] = lxl;
    lDoms[6 * Ippl::myNode() + 2] = lyf;
    lDoms[6 * Ippl::myNode() + 3] = lyl;
    lDoms[6 * Ippl::myNode() + 4] = lzf;
    lDoms[6 * Ippl::myNode() + 5] = lzl;

    reduce(&(lDoms[0]), &(lDoms[0]) + 6*Ippl::getNodes(), &(lDoms[0]), OpAddAssign());

}

DataSink::~DataSink()
{ 
    //    delete[] lDoms;
}

void DataSink::dump(Field<Vektor<double,3>,3> &EFD, Field<Vektor<double,3>,3> &HFD)
{
    if (binary_m) {
        dumpBinary(EFD, HFD);
    } else {
        dumpASCII(EFD, HFD);
    }

}

void DataSink::dumpASCII(Field<Vektor<double,3>,3> &EFD, Field<Vektor<double,3>,3> &HFD)
{
    int i, j, k, l;
    int lxf, lxl, lyf, lyl, lzf, lzl;

    Vektor<double,3> maxE = 0.0, maxH = 0.0;
    NDIndex<3> elem;
    Index II, JJ, KK;
    
    short EndianTest_s;
    unsigned char *EndianTest = (unsigned char*)&EndianTest_s;
    EndianTest[0] = 1;
    EndianTest[1] = 0;

    std::ofstream vtkout;
    std::stringstream fname;

    lxf = lDoms[6*Ippl::myNode()];
    lxl = lDoms[6*Ippl::myNode() + 1];
    lyf = lDoms[6*Ippl::myNode() + 2];
    lyl = lDoms[6*Ippl::myNode() + 3];
    lzf = lDoms[6*Ippl::myNode() + 4];
    lzl = lDoms[6*Ippl::myNode() + 5];
    
    vtkout.precision(5);
    vtkout.setf(std::ios_base::scientific, std::ios_base::floatfield);

    fname << "Data/c_";
    fname << std::setw(4) << std::setfill('0') << iteration_m;
    fname << "_node" << Ippl::myNode() << ".vtr";

    //SERIAL at the moment
    //if (Ippl::myNode() == 0) {

    // open a new data file for this iteration
    // and start with header
    vtkout.open(fname.str().c_str(), std::ios::out);
    vtkout << "<?xml version=\"1.0\"?>" << std::endl;
    vtkout << indent_l0 << "<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"" << (EndianTest_s == 1? "LittleEndian": "BigEndian") << "\">" << std::endl;
    vtkout << indent_l1 << "<RectilinearGrid WholeExtent=\"" 
           << lxf << " " << lxl - 1<< " "
           << lyf << " " << lyl - 1<< " "
           << lzf << " " << lzl - 1<< "\">" << std::endl;
    vtkout << indent_l2 << "<Piece Extent=\"" 
           << lxf << " " << lxl - 1<< " "
           << lyf << " " << lyl - 1<< " "
           << lzf << " " << lzl - 1<< "\">" << std::endl;
    vtkout << indent_l3 << "<PointData Vectors=\"E-Field\">" << std::endl;
    vtkout << indent_l4 << "<DataArray type=\"Float32\" Name=\"E-Field\" NumberOfComponents=\"3\" format=\"ascii\">\n";

    //interpolateFieldValues2(EFD);
    NBV = EFD;
    for (k = lzf; k < lzl; ++ k) {
        elem[2]=Index(k,k);
        for (j = lyf; j < lyl; ++ j) {
            elem[1]=Index(j,j);
            for (i = lxf; i < lxl; ++ i) {
                elem[0]=Index(i,i);
                Vektor<double, 3> tmp = EFD.localElement(elem);
                for (l = 0; l < 3; ++ l) {
                    if (fabs(tmp(l)) > maxE(l)) maxE(l) = fabs(tmp(l));
                }
                tmp = HFD.localElement(elem);
                for (l = 0; l < 3; ++ l) {
                    if (fabs(tmp(l)) > maxH(l)) maxH(l) = fabs(tmp(l));
                }
            }
        }
    }

    //interpolateFieldValues1(EFD);
    NBV = HFD;
    for (k = lzf; k < lzl; ++ k) {
        elem[2]=Index(k,k);
        for (j = lyf; j < lyl; ++ j) {
            elem[1]=Index(j,j);
            for (i = lxf; i < lxl; ++ i) {
                elem[0]=Index(i,i);
                Vektor<double, 3> tmp = EFD.localElement(elem);
                for (l = 0; l < 3; ++ l) {
                    if (fabs(tmp(l)) / maxE(l) < 1.0e-6) {
                        vtkout << 0.0 << " ";
                    } else {
                        vtkout << tmp(l) << " ";
                    }
                }
            }
        }
    }


    vtkout << "\n"
           << indent_l4 << "</DataArray>\n"
           << indent_l4 << "<DataArray type=\"Float32\" Name=\"H-Field\" NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;

    for (k = lzf; k < lzl; ++ k) {
        elem[2]=Index(k,k);
        for (j = lyf; j < lyl; ++ j) {
            elem[1]=Index(j,j);
            for (i = lxf; i < lxl; ++ i) {
                elem[0]=Index(i,i);
                Vektor<double, 3> tmp = HFD.localElement(elem);
                for (l = 0; l < 3; ++ l) {
                    if (fabs(tmp(l)) / maxH(l) < 1.0e-6) {
                        vtkout << 0.0 << " ";
                    } else {
                        vtkout << tmp(l) << " ";
                    }
                }
            }
        }
    }

    vtkout << "\n"
           << indent_l4 << "</DataArray>\n"
           << indent_l3 << "</PointData>\n" 
           << indent_l3 << "<Coordinates>\n"
           << indent_l4 << "<DataArray type=\"Float32\" name=\"X_COORDINATES\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
    
    for (i = lxf; i < lxl; ++ i) {
        vtkout << i * dx_m << " ";
    }

    vtkout << indent_l4 << "</DataArray>\n"
           << indent_l4 << "<DataArray type=\"Float32\" name=\"Y_COORDINATES\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
    
    for (j = lyf; j < lyl; ++ j) {
        vtkout <<  j * dy_m << " ";
    }

    vtkout << indent_l4 << "</DataArray>\n"
           << indent_l4 << "<DataArray type=\"Float32\" name=\"Z_COORDINATES\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;

    for (k = lzf; k < lzl; ++ k) {
        vtkout << k * dz_m << " ";
    }

    vtkout << indent_l4 << "</DataArray>\n"
           << indent_l3 << "</Coordinates>\n" 
           << indent_l2 << "</Piece>\n"
           << indent_l1 << "</RectilinearGrid>" << std::endl;


    vtkout << indent_l0 << "</VTKFile>" << std::endl;

    // close the output file for this iteration:
    vtkout.close();

    if (Ippl::myNode() == 0) {
        fname.str("");
        fname << "Data/c_";
        fname << std::setw(4) << std::setfill('0') << iteration_m;
        fname << ".pvtr";

        vtkout.open(fname.str().c_str(), std::ios::out);
        vtkout << indent_l0 << "<?xml version=\"1.0\"?>\n"
               << indent_l0 << "<VTKFile type=\"PRectilinearGrid\" version=\"0.1\" byte_order=\"" << (EndianTest_s == 1? "LittleEndian":"BigEndian") << "\">\n"
               << indent_l1 << "<PRectilinearGrid WholeExtent=\""
               << 0 << " " << Nx_m-1 << " "
               << 0 << " " << Ny_m-1 << " "
               << 0 << " " << Nz_m-1 
//                << 0 << " " << Nx_m << " "
//                << 0 << " " << Ny_m << " "
//                << 0 << " " << Nz_m 
               << "\" GhostLevel=\"1\">\n"
               << indent_l2 << "<PPointData Vectors=\"E-Field\">\n"
               << indent_l3 << "<PDataArray type=\"Float32\" Name=\"E-Field\" NumberOfComponents=\"3\"/>\n"
               << indent_l3 << "<PDataArray type=\"Float32\" Name=\"H-Field\" NumberOfComponents=\"3\"/>\n"
               << indent_l2 << "</PPointData>\n"
               << indent_l2 << "<PCoordinates>\n"
               << indent_l3 << "<PDataArray type=\"Float32\" name=\"X_COORDINATES\" NumberOfComponents=\"1\"/>\n"
               << indent_l3 << "<PDataArray type=\"Float32\" name=\"Y_COORDINATES\" NumberOfComponents=\"1\"/>\n"
               << indent_l3 << "<PDataArray type=\"Float32\" name=\"Z_COORDINATES\" NumberOfComponents=\"1\"/>\n"
               << indent_l2 << "</PCoordinates>" << std::endl;
        
        for (l = 0; l < Ippl::getNodes(); ++ l) {
            fname.str("");
            fname << "c_" << std::setw(4) << std::setfill('0') << iteration_m << "_node" << l << ".vtr";
            vtkout << indent_l2 << "<Piece Extent=\"" 
                   << lDoms[6 * l] << " " << lDoms[6 * l + 1] << " " 
                   << lDoms[6 * l + 2] << " " << lDoms[6 * l + 3] << " " 
                   << lDoms[6 * l + 4] << " " << lDoms[6 * l + 5] << "\" Source=\"" << fname.str() << "\"/>\n";
        }
        vtkout << indent_l1 << "</PRectilinearGrid>\n"
               << indent_l0 << "</VTKFile>" << std::endl;
    }
    
}

void DataSink::dumpBinary(Field<Vektor<double,3>,3> &EFD, Field<Vektor<double,3>,3> &HFD)
{
    
    int i, j, k, l;
    int lxf, lxl, lyf, lyl, lzf, lzl;
    int lnx, lny, lnz; 

    uint32_t size_dataset;

    Vektor<double,3> maxE = 0.0, maxH = 0.0;
    float dummy_flt;
    NDIndex<3> elem;
    Index II, JJ, KK;
    
    short EndianTest_s;
    unsigned char *EndianTest = (unsigned char*)&EndianTest_s;
    EndianTest[0] = 1;
    EndianTest[1] = 0;

    std::ofstream vtkout;
    std::stringstream fname;

    lxf = lDoms[6*Ippl::myNode()];
    lxl = lDoms[6*Ippl::myNode() + 1];
    lyf = lDoms[6*Ippl::myNode() + 2];
    lyl = lDoms[6*Ippl::myNode() + 3];
    lzf = lDoms[6*Ippl::myNode() + 4];
    lzl = lDoms[6*Ippl::myNode() + 5];
    
    lnx = lxl - lxf;
    lny = lyl - lyf;
    lnz = lzl - lzf;
    size_dataset = 3 * lnx * lny * lnz * sizeof(float);

    vtkout.precision(5);
    vtkout.setf(std::ios_base::scientific, std::ios_base::floatfield);

    fname << "Data/c_";
    fname << std::setw(4) << std::setfill('0') << iteration_m;
    fname << "_node" << Ippl::myNode() << ".vtr";

    //SERIAL at the moment
    //if (Ippl::myNode() == 0) {

    // open a new data file for this iteration
    // and start with header
    vtkout.open(fname.str().c_str(), std::ios::out);
    vtkout << "<?xml version=\"1.0\"?>" << std::endl;
    vtkout << indent_l0 << "<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"" << (EndianTest_s == 1? "LittleEndian": "BigEndian") << "\">" << std::endl;
    vtkout << indent_l1 << "<RectilinearGrid WholeExtent=\"" 
           << lxf << " " << lxl - 1<< " "
           << lyf << " " << lyl - 1<< " "
           << lzf << " " << lzl - 1<< "\">" << std::endl;
    vtkout << indent_l2 << "<Piece Extent=\"" 
           << lxf << " " << lxl - 1<< " "
           << lyf << " " << lyl - 1<< " "
           << lzf << " " << lzl - 1<< "\">" << std::endl;
    vtkout << indent_l3 << "<PointData Vectors=\"E-Field\">" << std::endl;
    vtkout << indent_l4 << "<DataArray type=\"Float32\" Name=\"E-Field\" NumberOfComponents=\"3\" format=\"appended\" offset=\"0\"/>\n" 
           << indent_l4 << "<DataArray type=\"Float32\" Name=\"H-Field\" NumberOfComponents=\"3\" format=\"appended\" offset=\"" << size_dataset + sizeof(uint32_t)<< "\"/>\n" 
           << indent_l3 << "</PointData>\n" 
           << indent_l3 << "<Coordinates>\n"
           << indent_l4 << "<DataArray type=\"Float32\" name=\"X_COORDINATES\" NumberOfComponents=\"1\" format=\"appended\" offset=\"" << 2 * size_dataset + 2 * sizeof(uint32_t)<< "\"/>" << std::endl
           << indent_l4 << "<DataArray type=\"Float32\" name=\"Y_COORDINATES\" NumberOfComponents=\"1\" format=\"appended\" offset=\"" << 2 * size_dataset + lnx * sizeof(float) + 3 * sizeof(uint32_t)<< "\"/>" << std::endl
           << indent_l4 << "<DataArray type=\"Float32\" name=\"Z_COORDINATES\" NumberOfComponents=\"1\" format=\"appended\" offset=\"" << 2 * size_dataset + (lnx + lny) * sizeof(float) + 4 * sizeof(uint32_t)<< "\"/>" << std::endl
           << indent_l3 << "</Coordinates>\n" 
           << indent_l2 << "</Piece>\n"
           << indent_l1 << "</RectilinearGrid>" << std::endl;

    //interpolateFieldValues1(EFD);
    NBV = EFD;
    vtkout << indent_l1 << "<AppendedData encoding=\"raw\">\n"
           << indent_l2 << "_";

    vtkout.write(reinterpret_cast<char* >(&size_dataset), sizeof(uint32_t));

    for (k = lzf; k < lzl; ++ k) {
        elem[2]=Index(k,k);
        for (j = lyf; j < lyl; ++ j) {
            elem[1]=Index(j,j);
            for (i = lxf; i < lxl; ++ i) {
                elem[0]=Index(i,i);
                Vektor<double, 3> tmp_d = NBV.localElement(elem);
                Vektor<float, 3> tmp_f(static_cast<float>(tmp_d(0)),
                                       static_cast<float>(tmp_d(1)),
                                       static_cast<float>(tmp_d(2)));
                vtkout.write(reinterpret_cast<char* >(&tmp_f(0)), sizeof(float));
                vtkout.write(reinterpret_cast<char* >(&tmp_f(1)), sizeof(float));
                vtkout.write(reinterpret_cast<char* >(&tmp_f(2)), sizeof(float));
            }
        }
    }

    //interpolateFieldValues2(HFD);
    NBV = HFD;
    vtkout.write(reinterpret_cast<char* >(&size_dataset), sizeof(uint32_t));
    for (k = lzf; k < lzl; ++ k) {
        elem[2]=Index(k,k);
        for (j = lyf; j < lyl; ++ j) {
            elem[1]=Index(j,j);
            for (i = lxf; i < lxl; ++ i) {
                elem[0]=Index(i,i);
                Vektor<double, 3> tmp_d = NBV.localElement(elem);
                Vektor<float, 3> tmp_f(static_cast<float>(tmp_d(0)),
                                       static_cast<float>(tmp_d(1)),
                                       static_cast<float>(tmp_d(2)));
                vtkout.write(reinterpret_cast<char* >(&tmp_f(0)), sizeof(float));
                vtkout.write(reinterpret_cast<char* >(&tmp_f(1)), sizeof(float));
                vtkout.write(reinterpret_cast<char* >(&tmp_f(2)), sizeof(float));
            }
        }
    }

    size_dataset = lnx * sizeof(float);
    vtkout.write(reinterpret_cast<char* >(&size_dataset), sizeof(uint32_t));
    for (i = lxf; i < lxl; ++ i) {
        dummy_flt = static_cast<float>(i * dx_m);
        vtkout.write(reinterpret_cast<char* >(&dummy_flt), sizeof(float));
    }
    size_dataset = lny * sizeof(float);
    vtkout.write(reinterpret_cast<char* >(&size_dataset), sizeof(uint32_t));
    for (j = lyf; j < lyl; ++ j) {
        dummy_flt = static_cast<float>(j * dy_m);
        vtkout.write(reinterpret_cast<char* >(&dummy_flt), sizeof(float));
    }
    size_dataset = lnz * sizeof(float);
    vtkout.write(reinterpret_cast<char* >(&size_dataset), sizeof(uint32_t));
    for (k = lzf; k < lzl; ++ k) {
        dummy_flt = static_cast<float>(k * dz_m);
        vtkout.write(reinterpret_cast<char* >(&dummy_flt), sizeof(float));
    }

    vtkout << "\n"
           << indent_l1 << "</AppendedData>\n"
           << indent_l0 << "</VTKFile>" << std::endl;

    // close the output file for this iteration:
    vtkout.close();

    if (Ippl::myNode() == 0) {
        fname.str("");
        fname << "Data/c_";
        fname << std::setw(4) << std::setfill('0') << iteration_m;
        fname << ".pvtr";

        vtkout.open(fname.str().c_str(), std::ios::out);
        vtkout << indent_l0 << "<?xml version=\"1.0\"?>\n"
               << indent_l0 << "<VTKFile type=\"PRectilinearGrid\" version=\"0.1\" byte_order=\"" << (EndianTest_s == 1? "LittleEndian":"BigEndian") << "\">\n"
               << indent_l1 << "<PRectilinearGrid WholeExtent=\""
               << 0 << " " << Nx_m-1 << " "
               << 0 << " " << Ny_m-1 << " "
               << 0 << " " << Nz_m-1 
//                << 0 << " " << Nx_m << " "
//                << 0 << " " << Ny_m << " "
//                << 0 << " " << Nz_m 
               << "\" GhostLevel=\"1\">\n"
               << indent_l2 << "<PPointData Vectors=\"E-Field\">\n"
               << indent_l3 << "<PDataArray type=\"Float32\" Name=\"E-Field\" NumberOfComponents=\"3\"/>\n"
               << indent_l3 << "<PDataArray type=\"Float32\" Name=\"H-Field\" NumberOfComponents=\"3\"/>\n"
               << indent_l2 << "</PPointData>\n"
               << indent_l2 << "<PCoordinates>\n"
               << indent_l3 << "<PDataArray type=\"Float32\" name=\"X_COORDINATES\" NumberOfComponents=\"1\"/>\n"
               << indent_l3 << "<PDataArray type=\"Float32\" name=\"Y_COORDINATES\" NumberOfComponents=\"1\"/>\n"
               << indent_l3 << "<PDataArray type=\"Float32\" name=\"Z_COORDINATES\" NumberOfComponents=\"1\"/>\n"
               << indent_l2 << "</PCoordinates>" << std::endl;
        
        for (l = 0; l < Ippl::getNodes(); ++ l) {
            fname.str("");
            fname << "c_" << std::setw(4) << std::setfill('0') << iteration_m << "_node" << l << ".vtr";
            vtkout << indent_l2 << "<Piece Extent=\"" 
                   << lDoms[6 * l] << " " << lDoms[6 * l + 1] << " " 
                   << lDoms[6 * l + 2] << " " << lDoms[6 * l + 3] << " " 
                   << lDoms[6 * l + 4] << " " << lDoms[6 * l + 5] << "\" Source=\"" << fname.str() << "\"/>\n";
        }
        vtkout << indent_l1 << "</PRectilinearGrid>\n"
               << indent_l0 << "</VTKFile>" << std::endl;
    }
    
}

void DataSink::interpolateFieldValues1(const Field<Vektor<double, 3>,3> &EFD) 
{
    int li, lj, lk, hi, hj, hk;
    int lDom[6];
    for (int i = 0; i < 6; ++ i) {
        lDom[i] = lDoms[6*Ippl::myNode() + i];
    }

    if (lDom[0] == 0) {
        li = 1;
    } else {
        li = lDom[0];
    }

    if (lDom[1] == Nx_m) {
        hi = Nx_m;
    } else {
        hi = lDom[1];
    }

    if (lDom[2] == 0) {
        lj = 1;
    } else {
        lj = lDom[2];
    }

    if (lDom[3] == Ny_m) {
        hj = Ny_m;
    } else {
        hj = lDom[3];
    }

    if (lDom[4] == 0) {
        lk = 1;
    } else {
        lk = lDom[4];
    }

    if (lDom[5] == Nz_m) {
        hk = Nz_m;
    } else {
        hk = lDom[5];
    }

    NDIndex<3> elem;
    int px[3] = {1,0,0}, py[3] = {0,1,0}, pz[3] = {0,0,1};
    NBV = EFD;
    /// corners
    if (lDom[0] == 0 && lDom[2] == 0 && lDom[4] == 0) {
        elem = setNDIndex(0,0,0);
        NBV.localElement(elem) += 0.5 * EFD.localElement(elem);
        NBV.localElement(elem)(0) -= 0.5 * EFD.localElement(elem + px)(0);
        NBV.localElement(elem)(1) -= 0.5 * EFD.localElement(elem + py)(1);
        NBV.localElement(elem)(2) -= 0.5 * EFD.localElement(elem + pz)(2);
    }

    if (lDom[1] == Nx_m && lDom[2] == 0 && lDom[4] == 0) {
        elem = setNDIndex(Nx_m, 0, 0);
        NBV.localElement(elem) += 0.5 * EFD.localElement(elem);
        NBV.localElement(elem)(0) -= 0.5 * EFD.localElement(elem - px)(0);
        NBV.localElement(elem)(1) -= 0.5 * EFD.localElement(elem + py)(1);
        NBV.localElement(elem)(2) -= 0.5 * EFD.localElement(elem + pz)(2);
    }        

    if (lDom[0] == 0 && lDom[3] == Ny_m && lDom[4] == 0) {
        elem = setNDIndex(0, Ny_m, 0);
        NBV.localElement(elem) += 0.5 * EFD.localElement(elem);
        NBV.localElement(elem)(0) -= 0.5 * EFD.localElement(elem + px)(0);
        NBV.localElement(elem)(1) -= 0.5 * EFD.localElement(elem - py)(1);
        NBV.localElement(elem)(2) -= 0.5 * EFD.localElement(elem + pz)(2);
    }        
 
    if (lDom[1] == Nx_m && lDom[3] == Ny_m && lDom[4] == 0) {
        elem = setNDIndex(Nx_m, Ny_m, 0);
        NBV.localElement(elem) += 0.5 * EFD.localElement(elem);
        NBV.localElement(elem)(0) -= 0.5 * EFD.localElement(elem - px)(0);
        NBV.localElement(elem)(1) -= 0.5 * EFD.localElement(elem - py)(1);
        NBV.localElement(elem)(2) -= 0.5 * EFD.localElement(elem + pz)(2);
    }        

    if (lDom[0] == 0 && lDom[2] == 0 && lDom[5] == Nz_m) {
        elem = setNDIndex(0, 0, Nz_m);
        NBV.localElement(elem) += 0.5 * EFD.localElement(elem);
        NBV.localElement(elem)(0) -= 0.5 * EFD.localElement(elem + px)(0);
        NBV.localElement(elem)(1) -= 0.5 * EFD.localElement(elem + py)(1);
        NBV.localElement(elem)(2) -= 0.5 * EFD.localElement(elem - pz)(2);
    }        

    if (lDom[1] == Nx_m && lDom[2] == 0 && lDom[5] == Nz_m) {
        elem = setNDIndex(Nx_m, 0, Nz_m);
        NBV.localElement(elem) += 0.5 * EFD.localElement(elem);
        NBV.localElement(elem)(0) -= 0.5 * EFD.localElement(elem - px)(0);
        NBV.localElement(elem)(1) -= 0.5 * EFD.localElement(elem + py)(1);
        NBV.localElement(elem)(2) -= 0.5 * EFD.localElement(elem - pz)(2);
    }        

     if (lDom[0] == 0 && lDom[3] == Ny_m && lDom[5] == Nz_m) {
         elem = setNDIndex(0, Ny_m, Nz_m);
         NBV.localElement(elem) += 0.5 * EFD.localElement(elem);
         NBV.localElement(elem)(0) -= 0.5 * EFD.localElement(elem + px)(0);
         NBV.localElement(elem)(1) -= 0.5 * EFD.localElement(elem - py)(1);
         NBV.localElement(elem)(2) -= 0.5 * EFD.localElement(elem - pz)(2);
    }        

     if (lDom[1] == Nx_m && lDom[3] == Ny_m && lDom[5] == Nz_m) {
         elem = setNDIndex(Nx_m, Ny_m, Nz_m);
         NBV.localElement(elem) += 0.5 * EFD.localElement(elem);
         NBV.localElement(elem)(0) -= 0.5 * EFD.localElement(elem - px)(0);
         NBV.localElement(elem)(1) -= 0.5 * EFD.localElement(elem - py)(1);
         NBV.localElement(elem)(2) -= 0.5 * EFD.localElement(elem - pz)(2);
    }        

     /// edges
     /// z direction
     if (lDom[0] == 0) {
         if (lDom[2] == 0) {
             elem = NDIndex<3>(Index(0,0), Index(0,0), Index(lk,hk-1));
             NBV.localElement(elem)(0) += 0.5 * (EFD.localElement(elem)(0) - EFD.localElement(elem + px)(0));
             NBV.localElement(elem)(1) += 0.5 * (EFD.localElement(elem)(1) - EFD.localElement(elem + py)(1));
             NBV.localElement(elem)(2) -= 0.5 * (EFD.localElement(elem)(2) - EFD.localElement(elem - pz)(2));
         }

         if (lDom[3] == Ny_m) {
             elem = NDIndex<3>(Index(0,0), Index(Ny_m,Ny_m), Index(lk,hk-1));
             NBV.localElement(elem)(0) += 0.5 * (EFD.localElement(elem)(0) - EFD.localElement(elem + px)(0));
             NBV.localElement(elem)(1) += 0.5 * (EFD.localElement(elem)(1) - EFD.localElement(elem - py)(1));
             NBV.localElement(elem)(2) -= 0.5 * (EFD.localElement(elem)(2) - EFD.localElement(elem - pz)(2));
         }
     }

     if (lDom[1] == Nx_m) {
         if (lDom[2] == 0) {
             elem = NDIndex<3>(Index(Nx_m,Nx_m), Index(0,0), Index(lk,hk-1));
             NBV.localElement(elem)(0) += 0.5 * (EFD.localElement(elem)(0) - EFD.localElement(elem - px)(0));
             NBV.localElement(elem)(1) += 0.5 * (EFD.localElement(elem)(1) - EFD.localElement(elem + py)(1));
             NBV.localElement(elem)(2) -= 0.5 * (EFD.localElement(elem)(2) - EFD.localElement(elem - pz)(2));
         }

         if (lDom[3] == Ny_m) {
             elem = NDIndex<3>(Index(Nx_m,Nx_m), Index(Ny_m,Ny_m), Index(lk,hk-1));
             NBV.localElement(elem)(0) += 0.5 * (EFD.localElement(elem)(0) - EFD.localElement(elem - px)(0));
             NBV.localElement(elem)(1) += 0.5 * (EFD.localElement(elem)(1) - EFD.localElement(elem - py)(1));
             NBV.localElement(elem)(2) -= 0.5 * (EFD.localElement(elem)(2) - EFD.localElement(elem - pz)(2));
         }
     }

     /// y direction
     if (lDom[4] == 0) {
         if (lDom[0] == 0) {
             elem = NDIndex<3>(Index(0,0), Index(lj, hj-1), Index(0,0));
             NBV.localElement(elem)(0) += 0.5 * (EFD.localElement(elem)(0) - EFD.localElement(elem + px)(0));
             NBV.localElement(elem)(1) -= 0.5 * (EFD.localElement(elem)(1) - EFD.localElement(elem - py)(1));
             NBV.localElement(elem)(2) += 0.5 * (EFD.localElement(elem)(2) - EFD.localElement(elem + pz)(2));
         }
         if (lDom[1] == Nx_m) {
             elem = NDIndex<3>(Index(Nx_m,Nx_m), Index(lj, hj-1), Index(0,0));
             NBV.localElement(elem)(0) += 0.5 * (EFD.localElement(elem)(0) - EFD.localElement(elem - px)(0));
             NBV.localElement(elem)(1) -= 0.5 * (EFD.localElement(elem)(1) - EFD.localElement(elem - py)(1));
             NBV.localElement(elem)(2) += 0.5 * (EFD.localElement(elem)(2) - EFD.localElement(elem + pz)(2));
         }
     }

     if (lDom[5] == Nz_m) {
         if (lDom[0] == 0) {
             elem = NDIndex<3>(Index(0,0), Index(lj, hj-1), Index(Nz_m,Nz_m));
             NBV.localElement(elem)(0) += 0.5 * (EFD.localElement(elem)(0) - EFD.localElement(elem + px)(0));
             NBV.localElement(elem)(1) -= 0.5 * (EFD.localElement(elem)(1) - EFD.localElement(elem - py)(1));
             NBV.localElement(elem)(2) += 0.5 * (EFD.localElement(elem)(2) - EFD.localElement(elem - pz)(2));
         }
         if (lDom[1] == Nx_m) {
             elem = NDIndex<3>(Index(Nx_m,Nx_m), Index(lj, hj-1), Index(Nz_m,Nz_m));
             NBV.localElement(elem)(0) += 0.5 * (EFD.localElement(elem)(0) - EFD.localElement(elem - px)(0));
             NBV.localElement(elem)(1) -= 0.5 * (EFD.localElement(elem)(1) - EFD.localElement(elem - py)(1));
             NBV.localElement(elem)(2) += 0.5 * (EFD.localElement(elem)(2) - EFD.localElement(elem - pz)(2));
         }
     }

     /// x direction
     if (lDom[2] == 0) {
         if (lDom[4] == 0) {
             elem = NDIndex<3>(Index(li,hi-1), Index(0,0), Index(0,0));
             NBV.localElement(elem)(0) -= 0.5 * (EFD.localElement(elem)(0) - EFD.localElement(elem - px)(0));
             NBV.localElement(elem)(1) += 0.5 * (EFD.localElement(elem)(1) - EFD.localElement(elem + py)(1));
             NBV.localElement(elem)(2) += 0.5 * (EFD.localElement(elem)(2) - EFD.localElement(elem + pz)(2));
         }
         if (lDom[5] == Nz_m) {
             elem = NDIndex<3>(Index(li,hi-1), Index(0,0), Index(Nz_m,Nz_m));
             NBV.localElement(elem)(0) -= 0.5 * (EFD.localElement(elem)(0) - EFD.localElement(elem - px)(0));
             NBV.localElement(elem)(1) += 0.5 * (EFD.localElement(elem)(1) - EFD.localElement(elem + py)(1));
             NBV.localElement(elem)(2) += 0.5 * (EFD.localElement(elem)(2) - EFD.localElement(elem - pz)(2));
         }
     }             
     if (lDom[3] == Ny_m) {
         if (lDom[4] == 0) {
             elem = NDIndex<3>(Index(li,hi-1), Index(Ny_m,Ny_m), Index(0,0));
             NBV.localElement(elem)(0) -= 0.5 * (EFD.localElement(elem)(0) - EFD.localElement(elem - px)(0));
             NBV.localElement(elem)(1) += 0.5 * (EFD.localElement(elem)(1) - EFD.localElement(elem - py)(1));
             NBV.localElement(elem)(2) += 0.5 * (EFD.localElement(elem)(2) - EFD.localElement(elem + pz)(2));
         }
         if (lDom[5] == Nz_m) {
             elem = NDIndex<3>(Index(li,hi-1), Index(Ny_m,Ny_m), Index(Nz_m,Nz_m));
             NBV.localElement(elem)(0) -= 0.5 * (EFD.localElement(elem)(0) - EFD.localElement(elem - px)(0));
             NBV.localElement(elem)(1) += 0.5 * (EFD.localElement(elem)(1) - EFD.localElement(elem - py)(1));
             NBV.localElement(elem)(2) += 0.5 * (EFD.localElement(elem)(2) - EFD.localElement(elem - pz)(2));
         }
     }             

     /// faces
     /// yz faces
     if (lDom[0] == 0) {
         elem = NDIndex<3>(Index(0,0), Index(lj,hj-1), Index(lk,hk-1));
         NBV.localElement(elem)(0) += 0.5 * (EFD.localElement(elem)(0) - EFD.localElement(elem + px)(0));
         NBV.localElement(elem)(1) -= 0.5 * (EFD.localElement(elem)(1) - EFD.localElement(elem - py)(1));
         NBV.localElement(elem)(2) -= 0.5 * (EFD.localElement(elem)(2) - EFD.localElement(elem - pz)(2));
     }

     if (lDom[1] == Nx_m) {
         elem = NDIndex<3>(Index(Nx_m,Nx_m), Index(lj,hj-1), Index(lk,hk-1));
         NBV.localElement(elem)(0) += 0.5 * (EFD.localElement(elem)(0) - EFD.localElement(elem - px)(0));
         NBV.localElement(elem)(1) -= 0.5 * (EFD.localElement(elem)(1) - EFD.localElement(elem - py)(1));
         NBV.localElement(elem)(2) -= 0.5 * (EFD.localElement(elem)(2) - EFD.localElement(elem - pz)(2));
     }
          
     /// xz faces
     if (lDom[2] == 0) {
         elem = NDIndex<3>(Index(0,0), Index(lj,hj-1), Index(lk,hk-1));
         NBV.localElement(elem)(0) -= 0.5 * (EFD.localElement(elem)(0) - EFD.localElement(elem - px)(0));
         NBV.localElement(elem)(1) += 0.5 * (EFD.localElement(elem)(1) - EFD.localElement(elem + py)(1));
         NBV.localElement(elem)(2) -= 0.5 * (EFD.localElement(elem)(2) - EFD.localElement(elem - pz)(2));
     }

     if (lDom[3] == Ny_m) {
         elem = NDIndex<3>(Index(li,hi-1), Index(Ny_m,Ny_m), Index(lk,hk-1));
         NBV.localElement(elem)(0) -= 0.5 * (EFD.localElement(elem)(0) - EFD.localElement(elem - px)(0));
         NBV.localElement(elem)(1) += 0.5 * (EFD.localElement(elem)(1) - EFD.localElement(elem - py)(1));
         NBV.localElement(elem)(2) -= 0.5 * (EFD.localElement(elem)(2) - EFD.localElement(elem - pz)(2));
     }

     /// xy faces
     if (lDom[4] == 0) {
         elem = NDIndex<3>(Index(li,hi-1), Index(lj,hj-1), Index(0,0));
         NBV.localElement(elem)(0) -= 0.5 * (EFD.localElement(elem)(0) - EFD.localElement(elem - px)(0));
         NBV.localElement(elem)(1) -= 0.5 * (EFD.localElement(elem)(1) - EFD.localElement(elem - py)(1));
         NBV.localElement(elem)(2) += 0.5 * (EFD.localElement(elem)(2) - EFD.localElement(elem + pz)(2));
     }

     if (lDom[5] == Nz_m) {
         elem = NDIndex<3>(Index(Nx_m,Nx_m), Index(lj,hj-1), Index(lk,hk-1));
         NBV.localElement(elem)(0) -= 0.5 * (EFD.localElement(elem)(0) - EFD.localElement(elem - px)(0));
         NBV.localElement(elem)(1) -= 0.5 * (EFD.localElement(elem)(1) - EFD.localElement(elem - py)(1));
         NBV.localElement(elem)(2) += 0.5 * (EFD.localElement(elem)(2) - EFD.localElement(elem - pz)(2));
     }

     /// inner
     elem = NDIndex<3>(Index(li,hi-1), Index(lj,hj-1), Index(lk,hk-1));
     NBV.localElement(elem)(0) -= 0.5 * (EFD.localElement(elem)(0) - EFD.localElement(elem-px)(0));
     NBV.localElement(elem)(1) -= 0.5 * (EFD.localElement(elem)(1) - EFD.localElement(elem-py)(1));
     NBV.localElement(elem)(2) -= 0.5 * (EFD.localElement(elem)(2) - EFD.localElement(elem-pz)(2));
}

void DataSink::interpolateFieldValues2(const Field<Vektor<double, 3>,3> &HFD) 
{
    int li, lj, lk, hi, hj, hk;
    int lDom[6];
    for (int i = 0; i < 6; ++ i) {
        lDom[i] = lDoms[i + 6*Ippl::myNode()];
    }

    if (lDom[0] == 0) {
        li = 1;
    } else {
        li = lDom[0];
    }

    if (lDom[1] == Nx_m) {
        hi = Nx_m;
    } else {
        hi = lDom[1];
    }

    if (lDom[2] == 0) {
        lj = 1;
    } else {
        lj = lDom[2];
    }

    if (lDom[3] == Ny_m) {
        hj = Ny_m;
    } else {
        hj = lDom[3];
    }

    if (lDom[4] == 0) {
        lk = 1;
    } else {
        lk = lDom[4];
    }

    if (lDom[5] == Nz_m) {
        hk = Nz_m;
    } else {
        hk = lDom[5];
    }

    NDIndex<3> elem;
    int px[3] = {1,0,0}, py[3] = {0,1,0}, pz[3] = {0,0,1};
    NBV = 0.5 * HFD;
    /// corners
    if (lDom[0] == 0 && lDom[2] == 0 && lDom[4] == 0) {
        elem = setNDIndex(0,0,0);
        NBV.localElement(elem) +=  1.75 * HFD.localElement(elem);
        NBV.localElement(elem)(0) -= 0.75 * HFD.localElement(elem + py)(0) + 0.75 * HFD.localElement(elem + pz)(0) - 0.25 * HFD.localElement(elem + py + pz)(0);
        NBV.localElement(elem)(1) -= 0.75 * HFD.localElement(elem + pz)(1) + 0.75 * HFD.localElement(elem + px)(1) - 0.25 * HFD.localElement(elem + pz + px)(1);
        NBV.localElement(elem)(2) -= 0.75 * HFD.localElement(elem + px)(2) + 0.75 * HFD.localElement(elem + py)(2) - 0.25 * HFD.localElement(elem + px + py)(2);
    }

    if (lDom[1] == Nx_m && lDom[2] == 0 && lDom[4] == 0) {
        elem = setNDIndex(Nx_m, 0, 0);
        NBV.localElement(elem) += 1.75 * HFD.localElement(elem);
        NBV.localElement(elem)(0) -= 0.75 * HFD.localElement(elem + py)(0) + 0.75 * HFD.localElement(elem + pz)(0) - 0.25 * HFD.localElement(elem + py + pz)(0);
        NBV.localElement(elem)(1) -= 0.75 * HFD.localElement(elem + pz)(1) + 0.75 * HFD.localElement(elem - px)(1) - 0.25 * HFD.localElement(elem + pz - px)(1);
        NBV.localElement(elem)(2) -= 0.75 * HFD.localElement(elem - px)(2) + 0.75 * HFD.localElement(elem + py)(2) - 0.25 * HFD.localElement(elem - px + py)(2);
    }        
 
    if (lDom[0] == 0 && lDom[3] == Ny_m && lDom[4] == 0) {
        elem = setNDIndex(0, Ny_m, 0);
        NBV.localElement(elem) += 1.75 * HFD.localElement(elem);
        NBV.localElement(elem)(0) -= 0.75 * HFD.localElement(elem - py)(0) + 0.75 * HFD.localElement(elem + pz)(0) - 0.25 * HFD.localElement(elem - py + pz)(0);
        NBV.localElement(elem)(1) -= 0.75 * HFD.localElement(elem + pz)(1) + 0.75 * HFD.localElement(elem + px)(1) - 0.25 * HFD.localElement(elem + pz + px)(1);
        NBV.localElement(elem)(2) -= 0.75 * HFD.localElement(elem + px)(2) + 0.75 * HFD.localElement(elem - py)(2) - 0.25 * HFD.localElement(elem + px - py)(2);
    }        
 
    if (lDom[1] == Nx_m && lDom[3] == Ny_m && lDom[4] == 0) {
        elem = setNDIndex(Nx_m, Ny_m, 0);
        NBV.localElement(elem) += 1.75 * HFD.localElement(elem);
        NBV.localElement(elem)(0) -= 0.75 * HFD.localElement(elem - py)(0) + 0.75 * HFD.localElement(elem + pz)(0) - 0.25 * HFD.localElement(elem - py + pz)(0);
        NBV.localElement(elem)(1) -= 0.75 * HFD.localElement(elem + pz)(1) + 0.75 * HFD.localElement(elem - px)(1) - 0.25 * HFD.localElement(elem + pz - px)(1);
        NBV.localElement(elem)(2) -= 0.75 * HFD.localElement(elem - px)(2) + 0.75 * HFD.localElement(elem - py)(2) - 0.25 * HFD.localElement(elem - px - py)(2);
    }        
 
    if (lDom[0] == 0 && lDom[2] == 0 && lDom[5] == Nz_m) {
        elem = setNDIndex(0, 0, Nz_m);
        NBV.localElement(elem) += 1.75 * HFD.localElement(elem);
        NBV.localElement(elem)(0) -= 0.75 * HFD.localElement(elem + py)(0) + 0.75 * HFD.localElement(elem - pz)(0) - 0.25 * HFD.localElement(elem + py - pz)(0);
        NBV.localElement(elem)(1) -= 0.75 * HFD.localElement(elem - pz)(1) + 0.75 * HFD.localElement(elem + px)(1) - 0.25 * HFD.localElement(elem - pz + px)(1);
        NBV.localElement(elem)(2) -= 0.75 * HFD.localElement(elem + px)(2) + 0.75 * HFD.localElement(elem + py)(2) - 0.25 * HFD.localElement(elem + px + py)(2);
    }        
    if (lDom[1] == Nx_m && lDom[2] == 0 && lDom[5] == Nz_m) {
        elem = setNDIndex(Nx_m, 0, Nz_m);
        NBV.localElement(elem) += 1.75 * HFD.localElement(elem);
        NBV.localElement(elem)(0) -= 0.75 * HFD.localElement(elem + py)(0) + 0.75 * HFD.localElement(elem - pz)(0) - 0.25 * HFD.localElement(elem + py - pz)(0);
        NBV.localElement(elem)(1) -= 0.75 * HFD.localElement(elem - pz)(1) + 0.75 * HFD.localElement(elem - px)(1) - 0.25 * HFD.localElement(elem - pz - px)(1);
        NBV.localElement(elem)(2) -= 0.75 * HFD.localElement(elem - px)(2) + 0.75 * HFD.localElement(elem + py)(2) - 0.25 * HFD.localElement(elem - px + py)(2);
    }        
    if (lDom[0] == 0 && lDom[3] == Ny_m && lDom[5] == Nz_m) {
        elem = setNDIndex(0, Ny_m, Nz_m);
        NBV.localElement(elem) += 1.75 * HFD.localElement(elem);
        NBV.localElement(elem)(0) -= 0.75 * HFD.localElement(elem - py)(0) + 0.75 * HFD.localElement(elem - pz)(0) - 0.25 * HFD.localElement(elem - py - pz)(0);
        NBV.localElement(elem)(1) -= 0.75 * HFD.localElement(elem - pz)(1) + 0.75 * HFD.localElement(elem + px)(1) - 0.25 * HFD.localElement(elem - pz + px)(1);
        NBV.localElement(elem)(2) -= 0.75 * HFD.localElement(elem + px)(2) + 0.75 * HFD.localElement(elem - py)(2) - 0.25 * HFD.localElement(elem + px - py)(2);
    }        
    if (lDom[1] == Nx_m && lDom[3] == Ny_m && lDom[5] == Nz_m) {
        elem = setNDIndex(Nx_m, Ny_m, Nz_m);
        NBV.localElement(elem) += 1.75 * HFD.localElement(elem);
        NBV.localElement(elem)(0) -= 0.75 * HFD.localElement(elem - py)(0) + 0.75 * HFD.localElement(elem - pz)(0) - 0.25 * HFD.localElement(elem - py - pz)(0);
        NBV.localElement(elem)(1) -= 0.75 * HFD.localElement(elem - pz)(1) + 0.75 * HFD.localElement(elem - px)(1) - 0.25 * HFD.localElement(elem - pz - px)(1);
        NBV.localElement(elem)(2) -= 0.75 * HFD.localElement(elem - px)(2) + 0.75 * HFD.localElement(elem - py)(2) - 0.25 * HFD.localElement(elem - px - py)(2);
    }        
 
    /// edges
    /// z direction
    if (lDom[0] == 0) {
        if (lDom[2] == 0) {
            elem = NDIndex<3>(Index(0,0), Index(0,0), Index(lk,hk-1));
            NBV.localElement(elem) += HFD.localElement(elem);
            NBV.localElement(elem)(0) -= 0.5 * HFD.localElement(elem + py + pz)(0);
            NBV.localElement(elem)(1) -= 0.5 * HFD.localElement(elem + pz + px)(1);
            NBV.localElement(elem)(2) -= 0.5 * HFD.localElement(elem + px + py)(2);
        }
     
        if (lDom[3] == Ny_m) {
            elem = NDIndex<3>(Index(0,0), Index(Ny_m,Ny_m), Index(lk,hk-1));
            NBV.localElement(elem) += HFD.localElement(elem);
            NBV.localElement(elem)(0) -= 0.5 * HFD.localElement(elem - py + pz)(0);
            NBV.localElement(elem)(1) -= 0.5 * HFD.localElement(elem + pz + px)(1);
            NBV.localElement(elem)(2) -= 0.5 * HFD.localElement(elem + px - py)(2);
        }
    }
 
    if (lDom[1] == Nx_m) {
        if (lDom[2] == 0) {
            elem = NDIndex<3>(Index(Nx_m,Nx_m), Index(0,0), Index(lk,hk-1));
            NBV.localElement(elem) += HFD.localElement(elem);
            NBV.localElement(elem)(0) -= 0.5 * HFD.localElement(elem + py + pz)(0);
            NBV.localElement(elem)(1) -= 0.5 * HFD.localElement(elem + pz - px)(1);
            NBV.localElement(elem)(2) -= 0.5 * HFD.localElement(elem - px + py)(2);
        }
     
        if (lDom[3] == Ny_m) {
            elem = NDIndex<3>(Index(Nx_m,Nx_m), Index(Ny_m,Ny_m), Index(lk,hk-1));
            NBV.localElement(elem) += HFD.localElement(elem);
            NBV.localElement(elem)(0) -= 0.5 * HFD.localElement(elem - py + pz)(0);
            NBV.localElement(elem)(1) -= 0.5 * HFD.localElement(elem + pz - px)(1);
            NBV.localElement(elem)(2) -= 0.5 * HFD.localElement(elem - px - py)(2);
        }
    }
 
 
    /// y direction
    if (lDom[4] == 0) {
        if (lDom[0] == 0) {
            elem = NDIndex<3>(Index(0,0), Index(lj, hj-1), Index(0,0));
            NBV.localElement(elem) += HFD.localElement(elem);
            NBV.localElement(elem)(0) -= 0.5 * HFD.localElement(elem + py + pz)(0);
            NBV.localElement(elem)(1) -= 0.5 * HFD.localElement(elem + pz + px)(1);
            NBV.localElement(elem)(2) -= 0.5 * HFD.localElement(elem + px + py)(2);
        }
        if (lDom[1] == Nx_m) {
            elem = NDIndex<3>(Index(Nx_m,Nx_m), Index(lj, hj-1), Index(0,0));
            NBV.localElement(elem) += HFD.localElement(elem);
            NBV.localElement(elem)(0) -= 0.5 * HFD.localElement(elem + py + pz)(0);
            NBV.localElement(elem)(1) -= 0.5 * HFD.localElement(elem + pz - px)(1);
            NBV.localElement(elem)(2) -= 0.5 * HFD.localElement(elem - px + py)(2);
        }
    }
 
    if (lDom[5] == Nz_m) {
        if (lDom[0] == 0) {
            elem = NDIndex<3>(Index(0,0), Index(lj, hj-1), Index(Nz_m,Nz_m));
            NBV.localElement(elem) += HFD.localElement(elem);
            NBV.localElement(elem)(0) -= 0.5 * HFD.localElement(elem + py - pz)(0);
            NBV.localElement(elem)(1) -= 0.5 * HFD.localElement(elem - pz + px)(1);
            NBV.localElement(elem)(2) -= 0.5 * HFD.localElement(elem + px + py)(2);
        }
        if (lDom[1] == Nx_m) {
            elem = NDIndex<3>(Index(Nx_m,Nx_m), Index(lj, hj-1), Index(Nz_m,Nz_m));
            NBV.localElement(elem) += HFD.localElement(elem);
            NBV.localElement(elem)(0) -= 0.5 * HFD.localElement(elem + py - pz)(0);
            NBV.localElement(elem)(1) -= 0.5 * HFD.localElement(elem - pz - px)(1);
            NBV.localElement(elem)(2) -= 0.5 * HFD.localElement(elem - px + py)(2);
        }
    }
 
 
    /// x direction
    if (lDom[2] == 0) {
        if (lDom[4] == 0) {
            elem = NDIndex<3>(Index(li,hi-1), Index(0,0), Index(0,0));
            NBV.localElement(elem) += HFD.localElement(elem);
            NBV.localElement(elem)(0) -= 0.5 * HFD.localElement(elem + py + pz)(0);
            NBV.localElement(elem)(1) -= 0.5 * HFD.localElement(elem + pz + px)(1);
            NBV.localElement(elem)(2) -= 0.5 * HFD.localElement(elem + px + py)(2);
        }
        if (lDom[5] == Nz_m) {
            elem = NDIndex<3>(Index(li,hi-1), Index(0,0), Index(Nz_m,Nz_m));
            NBV.localElement(elem) += HFD.localElement(elem);
            NBV.localElement(elem)(0) -= 0.5 * HFD.localElement(elem + py - pz)(0);
            NBV.localElement(elem)(1) -= 0.5 * HFD.localElement(elem - pz + px)(1);
            NBV.localElement(elem)(2) -= 0.5 * HFD.localElement(elem + px + py)(2);
        }
    }             
    if (lDom[3] == Ny_m) {
        if (lDom[4] == 0) {
            elem = NDIndex<3>(Index(li,hi-1), Index(Ny_m,Ny_m), Index(0,0));
            NBV.localElement(elem) += HFD.localElement(elem);
            NBV.localElement(elem)(0) -= 0.5 * HFD.localElement(elem - py + pz)(0);
            NBV.localElement(elem)(1) -= 0.5 * HFD.localElement(elem + pz + px)(1);
            NBV.localElement(elem)(2) -= 0.5 * HFD.localElement(elem + px - py)(2);
        }
        if (lDom[5] == Nz_m) {
            elem = NDIndex<3>(Index(li,hi-1), Index(Ny_m,Ny_m), Index(Nz_m,Nz_m));
            NBV.localElement(elem) += HFD.localElement(elem);
            NBV.localElement(elem)(0) -= 0.5 * HFD.localElement(elem - py - pz)(0);
            NBV.localElement(elem)(1) -= 0.5 * HFD.localElement(elem - pz + px)(1);
            NBV.localElement(elem)(2) -= 0.5 * HFD.localElement(elem + px - py)(2);
        }
    }             

     /// faces
     /// yz faces
     if (lDom[0] == 0) {
         elem = NDIndex<3>(Index(0,0), Index(lj,hj-1), Index(lk,hk-1));
         NBV.localElement(elem)(0) += 0.25 * (- HFD.localElement(elem)(0) + HFD.localElement(elem - py)(0)
                                              + HFD.localElement(elem - pz)(0) + HFD.localElement(elem - py - pz)(0));
         NBV.localElement(elem)(1) += 0.25 * (HFD.localElement(elem)(1) - HFD.localElement(elem + px)(1)
                                             + 3. * HFD.localElement(elem - pz)(1) - HFD.localElement(elem + px - pz)(1));
         NBV.localElement(elem)(2) += 0.25 * (HFD.localElement(elem)(2) - HFD.localElement(elem + px)(2)
                                              + 3. * HFD.localElement(elem - py)(2) - HFD.localElement(elem + px - py)(2));
     }

     if (lDom[1] == Nx_m) {
         elem = NDIndex<3>(Index(Nx_m,Nx_m), Index(lj,hj-1), Index(lk,hk-1));
         NBV.localElement(elem)(0) += 0.25 * (- HFD.localElement(elem)(0) + HFD.localElement(elem - py)(0)
                                              + HFD.localElement(elem - pz)(0) + HFD.localElement(elem - py - pz)(0));
         NBV.localElement(elem)(1) += 0.25 * (HFD.localElement(elem)(1) - HFD.localElement(elem - px)(1)
                                             + 3. * HFD.localElement(elem - pz)(1) - HFD.localElement(elem - px - pz)(1));
         NBV.localElement(elem)(2) += 0.25 * (HFD.localElement(elem)(2) - HFD.localElement(elem - px)(2)
                                              + 3. * HFD.localElement(elem - py)(2) - HFD.localElement(elem - px - py)(2));
     }
      
     /// xz faces
     if (lDom[2] == 0) {
         elem = NDIndex<3>(Index(0,0), Index(lj,hj-1), Index(lk,hk-1));
         NBV.localElement(elem)(0) += 0.25 * (HFD.localElement(elem)(0) - HFD.localElement(elem + py)(0)
                                              + 3. * HFD.localElement(elem - pz)(0) - HFD.localElement(elem + py - pz)(0));
         NBV.localElement(elem)(1) += 0.25 * (- HFD.localElement(elem)(1) + HFD.localElement(elem - px)(1)
                                              + HFD.localElement(elem - pz)(1) + HFD.localElement(elem - px - pz)(1));
         NBV.localElement(elem)(2) += 0.25 * (HFD.localElement(elem)(2) - HFD.localElement(elem + py)(2)
                                              + 3. * HFD.localElement(elem - px)(2) - HFD.localElement(elem + py - px)(2));
     }

     if (lDom[3] == Ny_m) {
         elem = NDIndex<3>(Index(li,hi-1), Index(Ny_m,Ny_m), Index(lk,hk-1));
         NBV.localElement(elem)(0) += 0.25 * (HFD.localElement(elem)(0) - HFD.localElement(elem - py)(0)
                                              + 3. * HFD.localElement(elem - pz)(0) - HFD.localElement(elem - py - pz)(0));
         NBV.localElement(elem)(1) += 0.25 * (- HFD.localElement(elem)(1) + HFD.localElement(elem - px)(1)
                                              + HFD.localElement(elem - pz)(1) + HFD.localElement(elem - px - pz)(1));
         NBV.localElement(elem)(2) += 0.25 * (HFD.localElement(elem)(2) - HFD.localElement(elem - py)(2)
                                              + 3. * HFD.localElement(elem - px)(2) - HFD.localElement(elem - py - px)(2));
     }

     /// xy faces
     if (lDom[4] == 0) {
         elem = NDIndex<3>(Index(li,hi-1), Index(lj,hj-1), Index(0,0));
         NBV.localElement(elem)(0) += 0.25 * (HFD.localElement(elem)(0) - HFD.localElement(elem + pz)(0)
                                              + 3. * HFD.localElement(elem - py)(0) - HFD.localElement(elem - py + pz)(0));
         NBV.localElement(elem)(1) += 0.25 * (HFD.localElement(elem)(1) - HFD.localElement(elem + pz)(1)
                                              + 3. * HFD.localElement(elem - px)(1) - HFD.localElement(elem + pz - px)(1));
         NBV.localElement(elem)(2) += 0.25 * (- HFD.localElement(elem)(2) + HFD.localElement(elem - px)(2)
                                              + HFD.localElement(elem - py)(2) + HFD.localElement(elem - px - py)(2));
     }

     if (lDom[5] == Nz_m) {
         elem = NDIndex<3>(Index(Nx_m,Nx_m), Index(lj,hj-1), Index(lk,hk-1));
         NBV.localElement(elem)(0) += 0.25 * (HFD.localElement(elem)(0) - HFD.localElement(elem - pz)(0)
                                              + 3. * HFD.localElement(elem - py)(0) - HFD.localElement(elem - py - pz)(0));
         NBV.localElement(elem)(1) += 0.25 * (HFD.localElement(elem)(1) - HFD.localElement(elem - pz)(1)
                                              + 3. * HFD.localElement(elem - px)(1) - HFD.localElement(elem - pz - px)(1));
         NBV.localElement(elem)(2) += 0.25 * (- HFD.localElement(elem)(2) + HFD.localElement(elem - px)(2)
                                              + HFD.localElement(elem - py)(2) + HFD.localElement(elem - px - py)(2));
     }

     /// inner
     elem = NDIndex<3>(Index(li,hi-1), Index(lj,hj-1), Index(lk,hk-1));
     NBV.localElement(elem)(0) += 0.25 * (-HFD.localElement(elem)(0) + HFD.localElement(elem - py)(0)
                                          + HFD.localElement(elem - pz)(0) + HFD.localElement(elem - py - pz)(0));
     NBV.localElement(elem)(1) += 0.25 * (-HFD.localElement(elem)(1) + HFD.localElement(elem - px)(1)
                                          + HFD.localElement(elem - pz)(1) + HFD.localElement(elem - px - pz)(1));
     NBV.localElement(elem)(2) += 0.25 * (-HFD.localElement(elem)(2) + HFD.localElement(elem - px)(2)
                                          + HFD.localElement(elem - py)(2) + HFD.localElement(elem - px - py)(2));

}
