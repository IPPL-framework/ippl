//
// File VTKFieldWriterParallel
//   The functions in this file are used by the P3M applications.
//
// Benjamin Ulmer, ETH Zürich (2016)
// Implemented as part of the Master thesis
// "The P3M Model on Emerging Computer Architectures With Application to Microbunching"
// (http://amas.web.psi.ch/people/aadelmann/ETH-Accel-Lecture-1/projectscompleted/cse/thesisBUlmer.pdf)
//
#ifndef VTK_FIELD_WRITER_
#define VTK_FIELD_WRITER_

#include <H5hut.h>

template <typename FieldType, typename ParticleType>
void dumpVTKVector(FieldType& f, const ParticleType& p, int iteration = 0,
                   std::string label = "EField") {
    if (Ippl::myNode() == 0) {
        NDIndex<3> lDom = f.getLayout().getLocalNDIndex();
        int nx          = lDom[0].length();
        int ny          = lDom[1].length();
        int nz          = lDom[2].length();
        double dx       = p->hr_m[0];
        double dy       = p->hr_m[1];
        double dz       = p->hr_m[2];

        std::string filename;
        filename = "data/";
        filename += label;
        filename += "_nod_";
        filename += std::to_string(Ippl::myNode());
        filename += "_it_";
        filename += std::to_string(iteration);
        filename += ".vtk";

        Inform vtkout(NULL, filename.c_str(), Inform::OVERWRITE);
        vtkout.precision(15);
        vtkout.setf(std::ios::scientific, std::ios::floatfield);

        vtkout << "# vtk DataFile Version 2.0" << endl;
        vtkout << "toyfdtd" << endl;
        vtkout << "ASCII" << endl;
        vtkout << "DATASET STRUCTURED_POINTS" << endl;
        vtkout << "DIMENSIONS " << nx << " " << ny << " " << nz << endl;
        // vtkout << "ORIGIN 0 0 0" << endl;
        // vtkout << "ORIGIN "<< p->extend_l[0]+.5*dx <<" " << p->extend_l[1]+.5*dy << " " <<
        // p->extend_l[2]+.5*dy << endl;
        vtkout << "ORIGIN " << p->extend_l[0] + .5 * dx + lDom[0].first() * dx << " "
               << p->extend_l[1] + .5 * dy + lDom[1].first() * dy << " "
               << p->extend_l[2] + .5 * dy + lDom[2].first() * dz << endl;
        vtkout << "SPACING " << dx << " " << dy << " " << dz << endl;
        vtkout << "POINT_DATA " << nx * ny * nz << endl;
        vtkout << "VECTORS Vector_Value float" << endl;
        for (int z = lDom[2].first(); z <= lDom[2].last(); z++) {
            for (int y = lDom[1].first(); y <= lDom[1].last(); y++) {
                for (int x = lDom[0].first(); x <= lDom[0].last(); x++) {
                    Vektor<double, 3> tmp = f[x][y][z].get();
                    vtkout << tmp(0) << "\t" << tmp(1) << "\t" << tmp(2) << endl;
                }
            }
        }
    }
}

template <typename FieldType, typename ParticleType>
void dumpVTKScalar(FieldType& f, const ParticleType& p, int iteration = 0,
                   std::string label = "RhoField") {
    NDIndex<3> lDom = f.getLayout().getLocalNDIndex();
    int nx          = lDom[0].length();
    int ny          = lDom[1].length();
    int nz          = lDom[2].length();
    double dx       = p->hr_m[0];
    double dy       = p->hr_m[1];
    double dz       = p->hr_m[2];

    std::string filename;
    filename = "data/";
    filename += label;
    filename += "_nod_";
    filename += std::to_string(Ippl::myNode());
    filename += "_it_";
    filename += std::to_string(iteration);
    filename += ".vtk";

    Inform vtkout(NULL, filename.c_str(), Inform::OVERWRITE);
    vtkout.precision(10);
    vtkout.setf(std::ios::scientific, std::ios::floatfield);

    vtkout << "# vtk DataFile Version 2.0" << endl;
    vtkout << "toyfdtd" << endl;
    vtkout << "ASCII" << endl;
    vtkout << "DATASET STRUCTURED_POINTS" << endl;
    vtkout << "DIMENSIONS " << nx << " " << ny << " " << nz << endl;
    // vtkout << "DIMENSIONS " << lDom[0].length()  << " " <<  lDom[1].length()  << " " <<
    // lDom[2].length()  << endl; vtkout << "ORIGIN 0 0 0" << endl;
    vtkout << "ORIGIN " << p->extend_l[0] + .5 * dx + lDom[0].first() * dx << " "
           << p->extend_l[1] + .5 * dy + lDom[1].first() * dy << " "
           << p->extend_l[2] + .5 * dy + lDom[2].first() * dz << endl;
    // vtkout << "ORIGIN "<< p->rmin_m[0]<<" " << p->rmin_m[1] << " " << p->rmin_m[2] << endl;
    vtkout << "SPACING " << dx << " " << dy << " " << dz << endl;
    vtkout << "POINT_DATA " << nx * ny * nz << endl;
    vtkout << "SCALARS Scalar_Value float" << endl;
    vtkout << "LOOKUP_TABLE default" << endl;
    for (int z = lDom[2].first(); z <= lDom[2].last(); z++) {
        for (int y = lDom[1].first(); y <= lDom[1].last(); y++) {
            for (int x = lDom[0].first(); x <= lDom[0].last(); x++) {
                std::complex<double> tmp = f[x][y][z].get();
                vtkout << real(tmp) << endl;
            }
        }
    }

    // Write the sum of the rho field to separate file
    std::stringstream fname;
    fname << "data/" << label << "Sum";
    fname << ".csv";

    Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
    csvout.precision(10);
    csvout.setf(std::ios::scientific, std::ios::floatfield);

    // start with header
    if (iteration == 0) {
        csvout << "it,FieldSum" << endl;
    }
    csvout << iteration << ", " << sum(f) << endl;
}

template <typename ParticleType>
void dumpParticlesOPAL(const ParticleType& p, int iteration = 0) {
    //    std::cout <<" Node " << std::to_string(Ippl::myNode()) << " has cached particles : " <<
    //    p->getGhostNum() << std::endl;
    std::stringstream fname;
    fname << "data/dist";
    fname << std::setw(1) << Ippl::myNode();
    fname << std::setw(5) << "_it_";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".dat";

    Inform csvout(NULL, fname.str().c_str(), Inform::OVERWRITE);
    csvout.precision(10);
    csvout.setf(std::ios::scientific, std::ios::floatfield);

    csvout << p->getLocalNum() << endl;

    for (unsigned i = 0; i < p->getLocalNum() + p->getGhostNum(); ++i) {
        csvout << p->R[i][0] << "\t" << p->v[i][0] << "\t" << p->R[i][1] << "\t" << p->v[i][1]
               << "\t" << p->R[i][2] << "\t" << p->v[i][2] << endl;
    }
}

template <typename ParticleType>
void dumpParticlesCSV(const ParticleType& p, int iteration = 0) {
    //	std::cout <<" Node " << std::to_string(Ippl::myNode()) << " has cached particles : " <<
    // p->getGhostNum() << std::endl;
    std::stringstream fname;
    fname << "data/charges_nod_";
    fname << std::setw(1) << Ippl::myNode();
    fname << std::setw(5) << "_it_";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".csv";

    // open a new data file for this iteration
    // and start with header
    Inform csvout(NULL, fname.str().c_str(), Inform::OVERWRITE);
    csvout.precision(10);
    csvout.setf(std::ios::scientific, std::ios::floatfield);

    csvout << "x coord, y coord, z coord, charge, EfieldMagnitude, ID, vx, vy, vz, f" << endl;

    for (unsigned i = 0; i < p->getLocalNum() + p->getGhostNum(); ++i) {
        double distributionf =
            p->v[i][2] * p->v[i][2]
            * exp(-(p->v[i][0] * p->v[i][0] + p->v[i][1] * p->v[i][1] + p->v[i][2] * p->v[i][2])
                  / 2.);

        csvout << p->R[i][0] << "," << p->R[i][1] << "," << p->R[i][2] << "," << p->Q[i] << ","
               << sqrt(p->EF[i][0] * p->EF[i][0] + p->EF[i][1] * p->EF[i][1]
                       + p->EF[i][2] * p->EF[i][2])
               << "," << p->ID[i] << "," << p->v[i][0] << "," << p->v[i][1] << "," << p->v[i][2]
               << "," << distributionf << endl;
    }
    csvout << endl;
}

template <typename ParticleType>
void dumpParticlesCSVp(const ParticleType& p, int iteration = 0) {
    if (Ippl::myNode() == 0) {
        //		std::cout <<" Node " << std::to_string(Ippl::myNode()) << " has cached
        // particles : " << p->getGhostNum() << std::endl;
        std::stringstream fname;
        fname << "data/charges_nod_";
        fname << std::setw(1) << Ippl::myNode();
        fname << std::setw(5) << "_it_";
        fname << std::setw(4) << std::setfill('0') << iteration;
        fname << ".csv";

        // open a new data file for this iteration
        // and start with header
        Inform csvout(NULL, fname.str().c_str(), Inform::OVERWRITE);
        csvout.precision(15);
        csvout.setf(std::ios::scientific, std::ios::floatfield);

        csvout << "x coord, y coord, z coord, z_lab, z_dispersions,charge, EfieldMagnitude, ID, "
                  "px, py, pz, pz_lab"
               << endl;

        for (unsigned i = 0; i < p->getTotalNum(); ++i) {
            // for (unsigned i=0; i<p->getLocalNum(); ++i) {
            // for (unsigned i=0; i<100000; ++i) {
            // if(p->R[i][0]>60*1e-6 && p->R[i][0]<120*1e-6 && p->R[i][1]>60*1e-6 &&
            // p->R[i][1]<120*1e-6){
            double zlab   = 1. / p->gamma * p->R[i][2];
            double pz_lab = p->gamma * p->p[i][2];
            double pz0    = p->beta0 * p->gamma * p->m0;
            csvout << p->R[i][0] << "," << p->R[i][1] << "," << p->R[i][2] << "," << zlab << ","
                   << zlab + p->R56 * pz_lab / pz0 << "," << p->Q[i] << ","
                   << sqrt(p->EF[i][0] * p->EF[i][0] + p->EF[i][1] * p->EF[i][1]
                           + p->EF[i][2] * p->EF[i][2])
                   << "," << p->ID[i] << "," << p->p[i][0] << "," << p->p[i][1] << "," << p->p[i][2]
                   << "," << pz_lab << endl;
            //}
        }
        csvout << endl;
    }
}

template <typename ParticleType>
void writeBeamStatistics(const ParticleType& p, int iteration) {
    if (Ippl::myNode() == 0) {
        std::stringstream fname;
        fname << "data/BeamSTatistics_seedID_";
        fname << std::setw(2) << std::setfill('0') << p->seedID;
        fname << ".csv";

        // open a new data file for this iteration
        // and start with header
        Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
        csvout.precision(10);
        csvout.setf(std::ios::scientific, std::ios::floatfield);

        if (iteration == 0) {
            csvout << "it,rrmsX[microns], rrmsY[microns], rrmsZ[microns], "
                      "prmsX[MeV/c],prmsY[MeV/c],prmsZ[MeV/"
                      "c],rmeanX[microns],rmeanY[microns],rmeanZ[microns],pmeanX[MeV/c],pmeanY[MeV/"
                      "c],pmeanZ[MeV/c],epsX[mm mrad],epsY[mm mrad],epsZ[mm mrad],epsnormX[mm "
                      "mrad],epsnormY[mm mrad],epsnormZ[mm "
                      "mrad],rprmsX[MeV*m/c],rprmsY[MeV*m/c],rprmsZ[MeV*m/c],eps6x6[mm^3 "
                      "mrad^3],eps6x6Normalized[mm^3 mrad^3],epsnorm_no_corell[mm^3 mrad^3]"
                   << endl;
        }
        csvout << iteration << ", " << p->rrms_m * 1e6 << "," << p->prms_m << ","
               << p->rmean_m * 1e6 << "," << p->pmean_m << "," << p->eps_m * 1e6 << ","
               << p->eps_norm_m * 1e6 << "," << p->rprms_m << "," << p->eps6x6_m * 1e18 << ","
               << p->eps6x6_normalized_m * 1e18 << ","
               << p->eps_norm_m[0] * 1e6 * p->eps_norm_m[1] * 1e6 * p->eps_norm_m[2] * 1e6 << endl;
    }
}

template <typename ParticleType>
void writeBeamStatisticsVelocity(const ParticleType& p, int iteration) {
    if (Ippl::myNode() == 0) {
        std::stringstream fname;
        fname << "data/BeamStatistics";
        fname << ".csv";

        // open a new data file for this iteration
        // and start with header
        Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
        csvout.precision(10);
        csvout.setf(std::ios::scientific, std::ios::floatfield);

        if (iteration == 0) {
            csvout << "it,rrmsX, rrmsY, rrmsZ, "
                      "vrmsX,vrmsY,vrmsZ,rmeanX,rmeanY,rmeanZ,vmeanX,vmeanY,vmeanZ,epsX,epsY,epsZ,"
                      "rvrmsX,rvrmsY,rvrmsZ"
                   << endl;
        }
        csvout << iteration << " " << p->rrms_m << " " << p->vrms_m << " " << p->rmean_m << " "
               << p->vmean_m << " " << p->eps_m << " " << p->rvrms_m << endl;
    }
}

template <typename ParticleType>
void writezcoordCSV(const ParticleType& p) {
    std::stringstream fname;
    fname << "data/zcoords";
    fname << ".csv";

    // open a new data file for this iteration
    // and start with header
    Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
    csvout.precision(10);
    csvout.setf(std::ios::scientific, std::ios::floatfield);

    for (unsigned i = 0; i < p->getLocalNum(); ++i) {
        csvout << p->R[i][2];
        if (i != p->getLocalNum() - 1)
            csvout << ",";
    }
    csvout << endl;
}

template <typename ParticleType>
void writeEzCSV(const ParticleType& p) {
    std::stringstream fname;
    fname << "data/Ez";
    fname << ".csv";

    // open a new data file for this iteration
    // and start with header
    Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
    csvout.precision(10);
    csvout.setf(std::ios::scientific, std::ios::floatfield);

    for (unsigned i = 0; i < p->getLocalNum(); ++i) {
        csvout << p->EF[i][2];
        if (i != p->getLocalNum() - 1)
            csvout << ",";
    }
    csvout << endl;
}

template <typename ParticleType>
void writeEnergy(const ParticleType& p, int iteration) {
    if (Ippl::myNode() == 0) {
        std::stringstream fname;
        fname << "data/energy";
        fname << ".csv";

        // open a new data file for this iteration
        // and start with header
        Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
        csvout.precision(10);
        csvout.setf(std::ios::scientific, std::ios::floatfield);

        if (iteration == 0) {
            // csvout << "it,Efield,Ekin,Epot,Etot,rhomax" << endl;
            csvout << "it,Efield,Ekin,Etot,Epot" << endl;
        }
        csvout << iteration << ", " << p->field_energy << "," << p->kinetic_energy << ","
               << p->field_energy + p->kinetic_energy << "," << p->integral_phi_m << endl;
    }
}
template <typename ParticleType>
void writeTemperature(const ParticleType& p, int iteration) {
    if (Ippl::myNode() == 0) {
        std::stringstream fname;
        fname << "data/Temperature";
        fname << ".csv";

        // open a new data file for this iteration
        // and start with header
        Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
        csvout.precision(10);
        csvout.setf(std::ios::scientific, std::ios::floatfield);
        if (iteration == 0) {
            csvout << "it,temp_x, temp_y, temp_z" << endl;
        }
        csvout << iteration << ", " << p->temperature[0] << "," << p->temperature[1] << ","
               << p->temperature[2] << endl;
    }
}

template <typename ParticleType>
void dumpConservedQuantities(const ParticleType& p, int iteration) {
    std::stringstream fname;
    fname << "data/conservedQuantities_nod_";
    fname << std::setw(1) << Ippl::myNode();
    fname << ".csv";

    // open a new data file for this iteration
    // and start with header
    Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
    csvout.precision(10);
    csvout.setf(std::ios::scientific, std::ios::floatfield);

    if (iteration == 1) {
        csvout << "it,EfieldPart,EpotPart,Rhosum,EfieldSum,PhiSum" << endl;
    }
    csvout << iteration << ", " << sum(dot(p->EF, p->EF)) << "," << sum(p->Phi) << "," << p->RhoSum
           << "," << sum(dot(p->eg_m, p->eg_m)) << "," << sum(p->phi_m) << endl;
}

template <typename ParticleType>
void writeEamplitude(const ParticleType& p, int iteration) {
    std::stringstream fname;
    fname << "data/Eamplitude";
    fname << ".csv";

    // open a new data file for this iteration
    // and start with header
    Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
    csvout.precision(10);
    csvout.setf(std::ios::scientific, std::ios::floatfield);

    if (iteration == 0) {
        csvout << "it,max(|E|),max(|Ez|)" << endl;
    }
    csvout << iteration << ", " << p->AmplitudeEfield << "," << p->AmplitudeEFz << endl;
}

template <typename FieldType2d, typename ParticleType>
void write_f_field(FieldType2d& f, const ParticleType& p, int iteration = 0,
                   std::string /*label*/ = "fInterpol") {
    Vektor<double, 3> dx = (p->extend_r - p->extend_l) / (p->Nx);
    Vektor<double, 3> dv = 2. * p->Vmax / (p->Nv);
    std::stringstream fname;
    fname << "data/f_mesh_";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".csv";

    // open a new data file for this iteration
    // and start with header
    Inform csvout(NULL, fname.str().c_str(), Inform::OVERWRITE);
    csvout.precision(10);
    csvout.setf(std::ios::scientific, std::ios::floatfield);

    csvout << "z, vz, f" << endl;
    NDIndex<2> lDom = p->domain2d_m;
    for (int i = lDom[0].first(); i <= lDom[0].last(); i++) {
        for (int j = lDom[1].first(); j <= lDom[1].last(); j++) {
            csvout << (i + 0.5) * dx[2] << "," << (j + 0.5) * dv[2] - p->Vmax[2] << ","
                   << f[i][j].get() << endl;
        }
    }

    // Write the sum of the rho field to separate file
    std::stringstream name;
    name << "data/fSum";
    name << ".csv";

    // open a new data file for this iteration
    // and start with header
    Inform out(NULL, fname.str().c_str(), Inform::APPEND);
    out.precision(10);
    out.setf(std::ios::scientific, std::ios::floatfield);

    if (iteration == 0) {
        out << "it,FieldSum" << endl;
    }
    out << iteration << ", " << sum(f) << endl;
}

template <typename ParticleType>
void dumpH5part(const ParticleType& p, unsigned int iteration = 0) {
    const size_t nl = p->getLocalNum();

    void* varray   = malloc(nl * sizeof(double));
    double* farray = (double*)varray;

    h5_int64_t rc = 0;

    H5PartSetNumParticles(p->H5f_m, nl);

    H5SetStep(p->H5f_m, iteration);

    for (size_t i = 0; i < nl; i++)
        farray[i] = p->R[i](0);
    rc = H5PartWriteDataFloat64(p->H5f_m, "x", farray);
    if (rc != H5_SUCCESS)
        ERRORMSG("H5 rc= " << rc << " in " << __FILE__ << " @ line " << __LINE__ << endl);

    for (size_t i = 0; i < nl; i++)
        farray[i] = p->R[i](1);
    rc = H5PartWriteDataFloat64(p->H5f_m, "y", farray);
    if (rc != H5_SUCCESS)
        ERRORMSG("H5 rc= " << rc << " in " << __FILE__ << " @ line " << __LINE__ << endl);

    for (size_t i = 0; i < nl; i++)
        farray[i] = p->R[i](2);
    rc = H5PartWriteDataFloat64(p->H5f_m, "z", farray);
    if (rc != H5_SUCCESS)
        ERRORMSG("H5 rc= " << rc << " in " << __FILE__ << " @ line " << __LINE__ << endl);
    for (size_t i = 0; i < nl; i++)
        farray[i] = 1. / p->gamma * p->R[i](2);
    rc = H5PartWriteDataFloat64(p->H5f_m, "zLab", farray);
    if (rc != H5_SUCCESS)
        ERRORMSG("H5 rc= " << rc << " in " << __FILE__ << " @ line " << __LINE__ << endl);

    for (size_t i = 0; i < nl; i++)
        farray[i] = p->p[i](0);
    rc = H5PartWriteDataFloat64(p->H5f_m, "px", farray);
    if (rc != H5_SUCCESS)
        ERRORMSG("H5 rc= " << rc << " in " << __FILE__ << " @ line " << __LINE__ << endl);

    for (size_t i = 0; i < nl; i++)
        farray[i] = p->p[i](1);
    rc = H5PartWriteDataFloat64(p->H5f_m, "py", farray);
    if (rc != H5_SUCCESS)
        ERRORMSG("H5 rc= " << rc << " in " << __FILE__ << " @ line " << __LINE__ << endl);

    for (size_t i = 0; i < nl; i++)
        farray[i] = p->p[i](2);
    rc = H5PartWriteDataFloat64(p->H5f_m, "pz", farray);
    if (rc != H5_SUCCESS)
        ERRORMSG("H5 rc= " << rc << " in " << __FILE__ << " @ line " << __LINE__ << endl);

    for (size_t i = 0; i < nl; i++)
        farray[i] = p->gamma * p->p[i](2);
    rc = H5PartWriteDataFloat64(p->H5f_m, "pzLab", farray);
    if (rc != H5_SUCCESS)
        ERRORMSG("H5 rc= " << rc << " in " << __FILE__ << " @ line " << __LINE__ << endl);
    double pz0 = p->beta0 * p->gamma * p->m0;
    for (size_t i = 0; i < nl; i++)
        farray[i] = 1. / p->gamma * p->R[i](2) + farray[i] / pz0 * p->R56;
    rc = H5PartWriteDataFloat64(p->H5f_m, "zR56", farray);
    if (rc != H5_SUCCESS)
        ERRORMSG("H5 rc= " << rc << " in " << __FILE__ << " @ line " << __LINE__ << endl);

    /*
                                    NDIndex<3> idx = p->getFieldLayout().getLocalNDIndex();
                                    h5_err_t herr = H5Block3dSetView(
                                                    p->H5f_m,
                                                    idx[0].min(), idx[0].max(),
                                                    idx[1].min(), idx[1].max(),
                                                    idx[2].min(), idx[2].max());
                                    if(herr != H5_SUCCESS)
                                            ERRORMSG("H5 herr= " << herr << " in " << __FILE__ << "
    @ line " << __LINE__ << endl); std::unique_ptr<h5_float64_t[]> data(new
    h5_float64_t[(idx[0].max() + 1)  * (idx[1].max() + 1) * (idx[2].max() + 1)]); size_t ii = 0;
                                    for(int i = idx[2].min(); i <= idx[2].max(); ++ i) {
                                            for(int j = idx[1].min(); j <= idx[1].max(); ++ j) {
                                                    for(int k = idx[0].min(); k <= idx[0].max(); ++
    k) { data[ii] = p->getEFDMag(k, j, i);
                                                            ++ ii;
                                                    }
                                            }
                                    }
                                    herr = H5Block3dWriteScalarFieldFloat64(p->H5f_m, "EFDMag",
    data.get()); if(herr != H5_SUCCESS) ERRORMSG("H5 herr= " << herr << " in " << __FILE__ << " @
    line " << __LINE__ << endl);
    i*/
}

template <typename ParticleType>
void dumpH5partVelocity(const ParticleType& p, unsigned int iteration = 0) {
    const size_t nl = p->getLocalNum();

    void* varray   = malloc(nl * sizeof(double));
    double* farray = (double*)varray;

    h5_int64_t rc = 0;

    H5PartSetNumParticles(p->H5f_m, nl);

    H5SetStep(p->H5f_m, iteration);

    for (size_t i = 0; i < nl; i++)
        farray[i] = p->R[i](0);
    rc = H5PartWriteDataFloat64(p->H5f_m, "x", farray);
    if (rc != H5_SUCCESS)
        ERRORMSG("H5 rc= " << rc << " in " << __FILE__ << " @ line " << __LINE__ << endl);

    for (size_t i = 0; i < nl; i++)
        farray[i] = p->R[i](1);
    rc = H5PartWriteDataFloat64(p->H5f_m, "y", farray);
    if (rc != H5_SUCCESS)
        ERRORMSG("H5 rc= " << rc << " in " << __FILE__ << " @ line " << __LINE__ << endl);

    for (size_t i = 0; i < nl; i++)
        farray[i] = p->R[i](2);
    rc = H5PartWriteDataFloat64(p->H5f_m, "z", farray);
    if (rc != H5_SUCCESS)
        ERRORMSG("H5 rc= " << rc << " in " << __FILE__ << " @ line " << __LINE__ << endl);

    for (size_t i = 0; i < nl; i++)
        farray[i] = p->v[i](0);
    rc = H5PartWriteDataFloat64(p->H5f_m, "vx", farray);
    if (rc != H5_SUCCESS)
        ERRORMSG("H5 rc= " << rc << " in " << __FILE__ << " @ line " << __LINE__ << endl);

    for (size_t i = 0; i < nl; i++)
        farray[i] = p->v[i](1);
    rc = H5PartWriteDataFloat64(p->H5f_m, "vy", farray);
    if (rc != H5_SUCCESS)
        ERRORMSG("H5 rc= " << rc << " in " << __FILE__ << " @ line " << __LINE__ << endl);

    for (size_t i = 0; i < nl; i++)
        farray[i] = p->v[i](2);
    rc = H5PartWriteDataFloat64(p->H5f_m, "vz", farray);
    if (rc != H5_SUCCESS)
        ERRORMSG("H5 rc= " << rc << " in " << __FILE__ << " @ line " << __LINE__ << endl);

    for (size_t i = 0; i < nl; i++)
        farray[i] = p->EF[i](0);
    rc = H5PartWriteDataFloat64(p->H5f_m, "EFx", farray);
    if (rc != H5_SUCCESS)
        ERRORMSG("H5 rc= " << rc << " in " << __FILE__ << " @ line " << __LINE__ << endl);

    for (size_t i = 0; i < nl; i++)
        farray[i] = p->EF[i](1);
    rc = H5PartWriteDataFloat64(p->H5f_m, "EFy", farray);
    if (rc != H5_SUCCESS)
        ERRORMSG("H5 rc= " << rc << " in " << __FILE__ << " @ line " << __LINE__ << endl);

    for (size_t i = 0; i < nl; i++)
        farray[i] = p->EF[i](2);
    rc = H5PartWriteDataFloat64(p->H5f_m, "EFz", farray);
    if (rc != H5_SUCCESS)
        ERRORMSG("H5 rc= " << rc << " in " << __FILE__ << " @ line " << __LINE__ << endl);
}

#endif
