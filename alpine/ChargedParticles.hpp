// ChargedParticles header file
//   Defines a particle attribute for charged particles to be used in
//   test programs
//
// Copyright (c) 2021 Paul Scherrer Institut, Villigen PSI, Switzerland
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

#include "Ippl.h"

#include "Solver/FFTPeriodicPoissonSolver.h"

// some typedefs
template <unsigned Dim = 3>
using PLayout_t = ippl::ParticleSpatialLayout<double, Dim>;

template <unsigned Dim = 3>
using Mesh_t = ippl::UniformCartesian<double, Dim>;

template <unsigned Dim = 3>
using Centering_t = typename Mesh_t<Dim>::DefaultCentering;

template <unsigned Dim = 3>
using FieldLayout_t = ippl::FieldLayout<Dim>;

template <unsigned Dim = 3>
using ORB = ippl::OrthogonalRecursiveBisection<double, Dim, Mesh_t<Dim>, Centering_t<Dim>>;

using size_type = ippl::detail::size_type;

template <typename T, unsigned Dim = 3>
using Vector = ippl::Vector<T, Dim>;

template <typename T, unsigned Dim = 3>
using Field = ippl::Field<T, Dim, Mesh_t<Dim>, Centering_t<Dim>>;

template <typename T>
using ParticleAttrib = ippl::ParticleAttrib<T>;

template <unsigned Dim = 3>
using Vector_t = Vector<double, Dim>;

template <unsigned Dim = 3>
using Field_t = Field<double, Dim>;

template <unsigned Dim = 3>
using VField_t = Field<Vector_t<Dim>, Dim>;

template <unsigned Dim = 3>
using Solver_t =
    ippl::FFTPeriodicPoissonSolver<Vector_t<Dim>, double, Dim, Mesh_t<Dim>, Centering_t<Dim>>;

const double pi = std::acos(-1.0);

// Test programs have to define this variable for VTK dump purposes
extern const char* TestName;

void dumpVTK(VField_t<3>& E, int nx, int ny, int nz, int iteration, double dx, double dy,
             double dz) {
    typename VField_t<3>::view_type::host_mirror_type host_view = E.getHostMirror();

    std::stringstream fname;
    fname << "data/ef_";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".vtk";

    Kokkos::deep_copy(host_view, E.getView());

    Inform vtkout(NULL, fname.str().c_str(), Inform::OVERWRITE);
    vtkout.precision(10);
    vtkout.setf(std::ios::scientific, std::ios::floatfield);

    // start with header
    vtkout << "# vtk DataFile Version 2.0" << endl;
    vtkout << TestName << endl;
    vtkout << "ASCII" << endl;
    vtkout << "DATASET STRUCTURED_POINTS" << endl;
    vtkout << "DIMENSIONS " << nx + 3 << " " << ny + 3 << " " << nz + 3 << endl;
    vtkout << "ORIGIN " << -dx << " " << -dy << " " << -dz << endl;
    vtkout << "SPACING " << dx << " " << dy << " " << dz << endl;
    vtkout << "CELL_DATA " << (nx + 2) * (ny + 2) * (nz + 2) << endl;

    vtkout << "VECTORS E-Field float" << endl;
    for (int z = 0; z < nz + 2; z++) {
        for (int y = 0; y < ny + 2; y++) {
            for (int x = 0; x < nx + 2; x++) {
                vtkout << host_view(x, y, z)[0] << "\t" << host_view(x, y, z)[1] << "\t"
                       << host_view(x, y, z)[2] << endl;
            }
        }
    }
}

void dumpVTK(Field_t<3>& rho, int nx, int ny, int nz, int iteration, double dx, double dy,
             double dz) {
    typename Field_t<3>::view_type::host_mirror_type host_view = rho.getHostMirror();

    std::stringstream fname;
    fname << "data/scalar_";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".vtk";

    Kokkos::deep_copy(host_view, rho.getView());

    Inform vtkout(NULL, fname.str().c_str(), Inform::OVERWRITE);
    vtkout.precision(10);
    vtkout.setf(std::ios::scientific, std::ios::floatfield);

    // start with header
    vtkout << "# vtk DataFile Version 2.0" << endl;
    vtkout << TestName << endl;
    vtkout << "ASCII" << endl;
    vtkout << "DATASET STRUCTURED_POINTS" << endl;
    vtkout << "DIMENSIONS " << nx + 3 << " " << ny + 3 << " " << nz + 3 << endl;
    vtkout << "ORIGIN " << -dx << " " << -dy << " " << -dz << endl;
    vtkout << "SPACING " << dx << " " << dy << " " << dz << endl;
    vtkout << "CELL_DATA " << (nx + 2) * (ny + 2) * (nz + 2) << endl;

    vtkout << "SCALARS Rho float" << endl;
    vtkout << "LOOKUP_TABLE default" << endl;
    for (int z = 0; z < nz + 2; z++) {
        for (int y = 0; y < ny + 2; y++) {
            for (int x = 0; x < nx + 2; x++) {
                vtkout << host_view(x, y, z) << endl;
            }
        }
    }
}

template <class PLayout, unsigned Dim = 3>
class ChargedParticles : public ippl::ParticleBase<PLayout> {
public:
    VField_t<Dim> E_m;
    Field_t<Dim> rho_m;

    // ORB
    ORB<Dim> orb;

    Vector<int, Dim> nr_m;

    ippl::e_dim_tag decomp_m[Dim];

    Vector_t<Dim> hr_m;
    Vector_t<Dim> rmin_m;
    Vector_t<Dim> rmax_m;

    double Q_m;

    std::string stype_m;

    std::shared_ptr<Solver_t<Dim>> solver_mp;

    double time_m;

    double rhoNorm_m;

    unsigned int loadbalancefreq_m;

    double loadbalancethreshold_m;

public:
    ParticleAttrib<double> q;                                        // charge
    typename ippl::ParticleBase<PLayout>::particle_position_type P;  // particle velocity
    typename ippl::ParticleBase<PLayout>::particle_position_type
        E;  // electric field at particle position

    ParticleAttrib<Vector<double, 2>> phaseCoords;  // phase space coordinates

    /*
      This constructor is mandatory for all derived classes from
      ParticleBase as the bunch buffer uses this
    */
    ChargedParticles(PLayout& pl, bool trackPhase = false)
        : ippl::ParticleBase<PLayout>(pl) {
        // register the particle attributes
        this->addAttribute(q);
        this->addAttribute(P);
        this->addAttribute(E);
        if (trackPhase)
            this->addAttribute(phaseCoords);
    }

    ChargedParticles(PLayout& pl, Vector_t<Dim> hr, Vector_t<Dim> rmin, Vector_t<Dim> rmax,
                     ippl::e_dim_tag decomp[Dim], double Q, bool trackPhase = false)
        : ippl::ParticleBase<PLayout>(pl)
        , hr_m(hr)
        , rmin_m(rmin)
        , rmax_m(rmax)
        , Q_m(Q) {
        // register the particle attributes
        this->addAttribute(q);
        this->addAttribute(P);
        this->addAttribute(E);
        setupBCs();
        for (unsigned int i = 0; i < Dim; i++)
            decomp_m[i] = decomp[i];
        if (trackPhase)
            this->addAttribute(phaseCoords);
    }

    ~ChargedParticles() {}

    void setupBCs() { setBCAllPeriodic(); }

    void updateLayout(FieldLayout_t<Dim>& fl, Mesh_t<Dim>& mesh,
                      ChargedParticles<PLayout, Dim>& buffer, bool& isFirstRepartition) {
        // Update local fields
        static IpplTimings::TimerRef tupdateLayout = IpplTimings::getTimer("updateLayout");
        IpplTimings::startTimer(tupdateLayout);
        this->E_m.updateLayout(fl);
        this->rho_m.updateLayout(fl);

        // Update layout with new FieldLayout
        PLayout& layout = this->getLayout();
        layout.updateLayout(fl, mesh);
        IpplTimings::stopTimer(tupdateLayout);
        static IpplTimings::TimerRef tupdatePLayout = IpplTimings::getTimer("updatePB");
        IpplTimings::startTimer(tupdatePLayout);
        if (!isFirstRepartition) {
            layout.update(*this, buffer);
        }
        IpplTimings::stopTimer(tupdatePLayout);
    }

    void initializeORB(FieldLayout_t<Dim>& fl, Mesh_t<Dim>& mesh) {
        orb.initialize(fl, mesh, rho_m);
    }

    void repartition(FieldLayout_t<Dim>& fl, Mesh_t<Dim>& mesh,
                     ChargedParticles<PLayout, Dim>& buffer, bool& isFirstRepartition) {
        // Repartition the domains
        bool res = orb.binaryRepartition(this->R, fl, isFirstRepartition);

        if (res != true) {
            std::cout << "Could not repartition!" << std::endl;
            return;
        }
        // Update
        this->updateLayout(fl, mesh, buffer, isFirstRepartition);
        this->solver_mp->setRhs(rho_m);
    }

    bool balance(size_type totalP, const unsigned int nstep) {
        if (std::strcmp(TestName, "UniformPlasmaTest") == 0) {
            return (nstep % loadbalancefreq_m == 0);
        } else {
            int local = 0;
            std::vector<int> res(Ippl::Comm->size());
            double equalPart = (double)totalP / Ippl::Comm->size();
            double dev       = std::abs((double)this->getLocalNum() - equalPart) / totalP;
            if (dev > loadbalancethreshold_m)
                local = 1;
            MPI_Allgather(&local, 1, MPI_INT, res.data(), 1, MPI_INT, Ippl::getComm());

            for (unsigned int i = 0; i < res.size(); i++) {
                if (res[i] == 1)
                    return true;
            }
            return false;
        }
    }

    void gatherStatistics(size_type totalP) {
        std::vector<double> imb(Ippl::Comm->size());
        double equalPart = (double)totalP / Ippl::Comm->size();
        double dev       = (std::abs((double)this->getLocalNum() - equalPart) / totalP) * 100.0;
        MPI_Gather(&dev, 1, MPI_DOUBLE, imb.data(), 1, MPI_DOUBLE, 0, Ippl::getComm());

        if (Ippl::Comm->rank() == 0) {
            std::stringstream fname;
            fname << "data/LoadBalance_";
            fname << Ippl::Comm->size();
            fname << ".csv";

            Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
            csvout.precision(5);
            csvout.setf(std::ios::scientific, std::ios::floatfield);

            if (time_m == 0.0) {
                csvout << "time, rank, imbalance percentage" << endl;
            }

            for (int r = 0; r < Ippl::Comm->size(); ++r) {
                csvout << time_m << " " << r << " " << imb[r] << endl;
            }
        }

        Ippl::Comm->barrier();
    }

    void gatherCIC() { gather(this->E, E_m, this->R); }

    void scatterCIC(size_type totalP, unsigned int iteration, Vector_t<Dim>& hrField) {
        Inform m("scatter ");

        rho_m = 0.0;
        scatter(q, rho_m, this->R);

        static IpplTimings::TimerRef sumTimer = IpplTimings::getTimer("Check");
        IpplTimings::startTimer(sumTimer);
        double Q_grid = rho_m.sum();

        size_type Total_particles = 0;
        size_type local_particles = this->getLocalNum();

        MPI_Reduce(&local_particles, &Total_particles, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0,
                   Ippl::getComm());

        double rel_error = std::fabs((Q_m - Q_grid) / Q_m);
        m << "Rel. error in charge conservation = " << rel_error << endl;

        if (Ippl::Comm->rank() == 0) {
            if (Total_particles != totalP || rel_error > 1e-10) {
                m << "Time step: " << iteration << endl;
                m << "Total particles in the sim. " << totalP << " "
                  << "after update: " << Total_particles << endl;
                m << "Rel. error in charge conservation: " << rel_error << endl;
                std::abort();
            }
        }

        double h = 1;
        for (const auto& hr : hrField)
            h *= hr;
        rho_m = rho_m / h;

        rhoNorm_m = norm(rho_m);
        IpplTimings::stopTimer(sumTimer);

        // dumpVTK(rho_m,nr_m[0],nr_m[1],nr_m[2],iteration,hrField[0],hrField[1],hrField[2]);

        // rho = rho_e - rho_i
        double size = 1;
        for (unsigned d = 0; d < Dim; d++)
            size *= rmax_m[d] - rmin_m[d];
        rho_m = rho_m - (Q_m / size);
    }

    void initSolver() {
        Inform m("solver ");
        if (stype_m == "FFT")
            initFFTSolver();
        else
            m << "No solver matches the argument" << endl;
    }

    void initFFTSolver() {
        ippl::ParameterList sp;
        sp.add("output_type", Solver_t<Dim>::GRAD);
        sp.add("use_heffte_defaults", false);
        sp.add("use_pencils", true);
        sp.add("use_reorder", false);
        sp.add("use_gpu_aware", true);
        sp.add("comm", ippl::p2p_pl);
        sp.add("r2c_direction", 0);

        solver_mp = std::make_shared<Solver_t<Dim>>();

        solver_mp->mergeParameters(sp);

        solver_mp->setRhs(rho_m);

        solver_mp->setLhs(E_m);
    }

    void dumpData() {
        auto Pview = P.getView();

        double Energy = 0.0;

        Kokkos::parallel_reduce(
            "Particle Energy", this->getLocalNum(),
            KOKKOS_LAMBDA(const int i, double& valL) {
                double myVal = dot(Pview(i), Pview(i)).apply();
                valL += myVal;
            },
            Kokkos::Sum<double>(Energy));

        Energy *= 0.5;
        double gEnergy = 0.0;

        MPI_Reduce(&Energy, &gEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());

        const int nghostE = E_m.getNghost();
        auto Eview        = E_m.getView();
        Vector_t<Dim> normE;

        using index_array_type = typename ippl::detail::RangePolicy<Dim>::index_array_type;
        for (unsigned d = 0; d < Dim; ++d) {
            double temp = 0.0;
            Kokkos::parallel_reduce(
                "Vector E reduce", ippl::detail::getRangePolicy<Dim>(Eview, nghostE),
                ippl::detail::functorize<ippl::detail::REDUCE, Dim, double>(
                    KOKKOS_LAMBDA(const index_array_type& args, double& valL) {
                        double myVal = std::pow(ippl::apply<Dim>(Eview, args)[d], 2);
                        valL += myVal;
                    }),
                Kokkos::Sum<double>(temp));
            double globaltemp = 0.0;
            MPI_Reduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());
            normE[d] = std::sqrt(globaltemp);
        }

        if (Ippl::Comm->rank() == 0) {
            std::stringstream fname;
            fname << "data/ParticleField_";
            fname << Ippl::Comm->size();
            fname << ".csv";

            Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
            csvout.precision(10);
            csvout.setf(std::ios::scientific, std::ios::floatfield);

            if (time_m == 0.0) {
                csvout << "time, Kinetic energy, Rho_norm2, Ex_norm2, Ey_norm2, Ez_norm2";
                for (unsigned d = 0; d < Dim; d++)
                    csvout << "E" << d << "norm2, ";
                csvout << endl;
            }

            csvout << time_m << " " << gEnergy << " " << rhoNorm_m << " ";
            for (unsigned d = 0; d < Dim; d++)
                csvout << normE[d] << " ";
            csvout << endl;
        }

        Ippl::Comm->barrier();
    }

    void dumpLandau() {
        const int nghostE = E_m.getNghost();
        auto Eview        = E_m.getView();
        double fieldEnergy, ExAmp;

        using index_array_type = typename ippl::detail::RangePolicy<Dim>::index_array_type;
        double temp            = 0.0;
        Kokkos::parallel_reduce(
            "Ex inner product", ippl::detail::getRangePolicy<Dim>(Eview, nghostE),
            ippl::detail::functorize<ippl::detail::REDUCE, Dim, double>(
                KOKKOS_LAMBDA(const index_array_type& args, double& valL) {
                    double myVal = std::pow(ippl::apply<Dim>(Eview, args)[0], 2);
                    valL += myVal;
                }),
            Kokkos::Sum<double>(temp));
        double globaltemp = 0.0;
        MPI_Reduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());
        fieldEnergy = globaltemp;
        for (const auto& h : hr_m)
            fieldEnergy *= h;

        double tempMax = 0.0;
        Kokkos::parallel_reduce("Ex max norm", ippl::detail::getRangePolicy<Dim>(Eview, nghostE),
                                ippl::detail::functorize<ippl::detail::REDUCE, Dim, double>(
                                    KOKKOS_LAMBDA(const index_array_type& args, double& valL) {
                                        double myVal = std::fabs(ippl::apply<Dim>(Eview, args)[0]);
                                        if (myVal > valL)
                                            valL = myVal;
                                    }),
                                Kokkos::Max<double>(tempMax));
        ExAmp = 0.0;
        MPI_Reduce(&tempMax, &ExAmp, 1, MPI_DOUBLE, MPI_MAX, 0, Ippl::getComm());

        if (Ippl::Comm->rank() == 0) {
            std::stringstream fname;
            fname << "data/FieldLandau_";
            fname << Ippl::Comm->size();
            fname << ".csv";

            Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
            csvout.precision(10);
            csvout.setf(std::ios::scientific, std::ios::floatfield);

            if (time_m == 0.0) {
                csvout << "time, Ex_field_energy, Ex_max_norm" << endl;
            }

            csvout << time_m << " " << fieldEnergy << " " << ExAmp << endl;
        }

        Ippl::Comm->barrier();
    }

    void dumpBumponTail() {
        const int nghostE = E_m.getNghost();
        auto Eview        = E_m.getView();
        double fieldEnergy, EzAmp;

        using index_array_type = typename ippl::detail::RangePolicy<Dim>::index_array_type;
        double temp            = 0.0;
        Kokkos::parallel_reduce(
            "Ex inner product", ippl::detail::getRangePolicy<Dim>(Eview, nghostE),
            ippl::detail::functorize<ippl::detail::REDUCE, Dim, double>(
                KOKKOS_LAMBDA(const index_array_type& args, double& valL) {
                    double myVal = std::pow(ippl::apply<Dim>(Eview, args)[2], 2);
                    valL += myVal;
                }),
            Kokkos::Sum<double>(temp));
        double globaltemp = 0.0;
        MPI_Reduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());
        fieldEnergy = globaltemp;
        for (const auto& h : hr_m)
            fieldEnergy *= h;

        double tempMax = 0.0;
        Kokkos::parallel_reduce("Ex max norm", ippl::detail::getRangePolicy<Dim>(Eview, nghostE),
                                ippl::detail::functorize<ippl::detail::REDUCE, Dim, double>(
                                    KOKKOS_LAMBDA(const index_array_type& args, double& valL) {
                                        double myVal = std::fabs(ippl::apply<Dim>(Eview, args)[2]);
                                        if (myVal > valL)
                                            valL = myVal;
                                    }),
                                Kokkos::Max<double>(tempMax));
        EzAmp = 0.0;
        MPI_Reduce(&tempMax, &EzAmp, 1, MPI_DOUBLE, MPI_MAX, 0, Ippl::getComm());

        if (Ippl::Comm->rank() == 0) {
            std::stringstream fname;
            fname << "data/FieldBumponTail_";
            fname << Ippl::Comm->size();
            fname << ".csv";

            Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
            csvout.precision(10);
            csvout.setf(std::ios::scientific, std::ios::floatfield);

            if (time_m == 0.0) {
                csvout << "time, Ez_field_energy, Ez_max_norm" << endl;
            }

            csvout << time_m << " " << fieldEnergy << " " << EzAmp << endl;
        }

        Ippl::Comm->barrier();
    }

    void dumpParticleData() {
        typename ParticleAttrib<Vector_t<Dim>>::HostMirror R_host = this->R.getHostMirror();
        typename ParticleAttrib<Vector_t<Dim>>::HostMirror P_host = this->P.getHostMirror();
        Kokkos::deep_copy(R_host, this->R.getView());
        Kokkos::deep_copy(P_host, P.getView());
        std::stringstream pname;
        pname << "data/ParticleIC_";
        pname << Ippl::Comm->rank();
        pname << ".csv";
        Inform pcsvout(NULL, pname.str().c_str(), Inform::OVERWRITE, Ippl::Comm->rank());
        pcsvout.precision(10);
        pcsvout.setf(std::ios::scientific, std::ios::floatfield);
        pcsvout << "R_x, R_y, R_z, V_x, V_y, V_z" << endl;
        for (size_type i = 0; i < this->getLocalNum(); i++) {
            for (unsigned d = 0; d < Dim; d++)
                pcsvout << R_host(i)[d] << " ";
            for (unsigned d = 0; d < Dim; d++)
                pcsvout << P_host(i)[d] << " ";
            pcsvout << endl;
        }
        Ippl::Comm->barrier();
    }

    void dumpLocalDomains(const FieldLayout_t<Dim>& fl, const unsigned int step) {
        if (Ippl::Comm->rank() == 0) {
            const typename FieldLayout_t<Dim>::host_mirror_type domains = fl.getHostLocalDomains();
            std::ofstream myfile;
            myfile.open("data/domains" + std::to_string(step) + ".txt");
            for (unsigned int i = 0; i < domains.size(); ++i) {
                for (unsigned d = 0; d < Dim; d++)
                    myfile << domains[i][d].first() << " " << domains[i][d].last() << " ";
                myfile << "\n";
            }
            myfile.close();
        }
        Ippl::Comm->barrier();
    }

private:
    void setBCAllPeriodic() { this->setParticleBC(ippl::BC::PERIODIC); }
};
