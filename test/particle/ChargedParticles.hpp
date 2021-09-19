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

// dimension of our positions
constexpr unsigned Dim = 3;

// some typedefs
typedef ippl::ParticleSpatialLayout<double,Dim>   PLayout_t;
typedef ippl::UniformCartesian<double, Dim>        Mesh_t;
typedef ippl::FieldLayout<Dim> FieldLayout_t;
typedef ippl::OrthogonalRecursiveBisection<double, Dim, Mesh_t> ORB;

using size_type = ippl::detail::size_type;

template<typename T, unsigned Dim>
using Vector = ippl::Vector<T, Dim>;

template<typename T, unsigned Dim>
using Field = ippl::Field<T, Dim>;

template<typename T>
using ParticleAttrib = ippl::ParticleAttrib<T>;

typedef Vector<double, Dim>  Vector_t;
typedef Field<double, Dim>   Field_t;
typedef Field<Vector_t, Dim> VField_t;
typedef ippl::FFTPeriodicPoissonSolver<Vector_t, double, Dim> Solver_t;

const double pi = acos(-1.0);

// Test programs have to define this variable for VTK dump purposes
extern const char* TestName;

void dumpVTK(VField_t& E, int nx, int ny, int nz, int iteration,
             double dx, double dy, double dz) {


    typename VField_t::view_type::host_mirror_type host_view = E.getHostMirror();

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
    vtkout << "DIMENSIONS " << nx+3 << " " << ny+3 << " " << nz+3 << endl;
    vtkout << "ORIGIN "     << -2*dx  << " " << -2*dy  << " "  << -2*dz << endl;
    vtkout << "SPACING " << dx << " " << dy << " " << dz << endl;
    vtkout << "CELL_DATA " << (nx+2)*(ny+2)*(nz+2) << endl;

    vtkout << "VECTORS E-Field float" << endl;
    for (int z=0; z<nz+2; z++) {
        for (int y=0; y<ny+2; y++) {
            for (int x=0; x<nx+2; x++) {

                vtkout << host_view(x,y,z)[0] << "\t"
                       << host_view(x,y,z)[1] << "\t"
                       << host_view(x,y,z)[2] << endl;
            }
        }
    }
}

void dumpVTK(Field_t& rho, int nx, int ny, int nz, int iteration,
             double dx, double dy, double dz) {

    typename Field_t::view_type::host_mirror_type host_view = rho.getHostMirror();

    std::stringstream fname;
    fname << "data/scalar_";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".vtk";

    Kokkos::deep_copy(host_view, rho.getView());

    Inform vtkout(NULL, fname.str().c_str(), Inform::OVERWRITE);
    vtkout.precision(10);
    vtkout.setf(std::ios::scientific, std::ios::floatfield);

    //double vol = dx*dy*dz;

    // start with header
    vtkout << "# vtk DataFile Version 2.0" << endl;
    vtkout << TestName << endl;
    vtkout << "ASCII" << endl;
    vtkout << "DATASET STRUCTURED_POINTS" << endl;
    vtkout << "DIMENSIONS " << nx+3 << " " << ny+3 << " " << nz+3 << endl;
    vtkout << "ORIGIN " << -2*dx << " " << -2*dy << " " << -2*dz << endl;
    vtkout << "SPACING " << dx << " " << dy << " " << dz << endl;
    vtkout << "CELL_DATA " << (nx+2)*(ny+2)*(nz+2) << endl;

    vtkout << "SCALARS Rho float" << endl;
    vtkout << "LOOKUP_TABLE default" << endl;
    for (int z=0; z<nz+2; z++) {
        for (int y=0; y<ny+2; y++) {
            for (int x=0; x<nx+2; x++) {

                vtkout << host_view(x,y,z) << endl;
            }
        }
    }
}

template<class PLayout>
class ChargedParticles : public ippl::ParticleBase<PLayout> {
public:
    VField_t E_m;
    Field_t rho_m;

    // ORB
    ORB orb;

    Vector<int, Dim> nr_m;

    ippl::e_dim_tag decomp_m[Dim];

    Vector_t hr_m;
    Vector_t rmin_m;
    Vector_t rmax_m;

    double Q_m;

    std::string stype_m;

    std::shared_ptr<Solver_t> solver_mp;

    double time_m;

    double rhoNorm_m;


public:
    ParticleAttrib<double>     q; // charge
    typename ippl::ParticleBase<PLayout>::particle_position_type P;  // particle velocity
    typename ippl::ParticleBase<PLayout>::particle_position_type E;  // electric field at particle position


    /*
      This constructor is mandatory for all derived classes from
      ParticleBase as the update function invokes this
    */
    ChargedParticles(PLayout& pl)
    : ippl::ParticleBase<PLayout>(pl)
    {
        // register the particle attributes
        this->addAttribute(q);
        this->addAttribute(P);
        this->addAttribute(E);
    }

    ChargedParticles(PLayout& pl,
                     Vector_t hr,
                     Vector_t rmin,
                     Vector_t rmax,
                     ippl::e_dim_tag decomp[Dim],
                     double Q)
    : ippl::ParticleBase<PLayout>(pl)
    , hr_m(hr)
    , rmin_m(rmin)
    , rmax_m(rmax)
    , Q_m(Q)
    {
        // register the particle attributes
        this->addAttribute(q);
        this->addAttribute(P);
        this->addAttribute(E);
        setupBCs();
        for (unsigned int i = 0; i < Dim; i++)
            decomp_m[i]=decomp[i];
    }

    ~ChargedParticles(){ }

    void setupBCs() {
        setBCAllPeriodic();
    }

    void updateLayout(FieldLayout_t& fl, Mesh_t& mesh, ChargedParticles<PLayout>& buffer) {
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
        layout.update(*this, buffer);
        IpplTimings::stopTimer(tupdatePLayout);
    }

    void initializeORB(FieldLayout_t& fl, Mesh_t& mesh) {
        orb.initialize(fl, mesh);
    }

    void repartition(FieldLayout_t& fl, Mesh_t& mesh, ChargedParticles<PLayout>& buffer) {
        // Repartition the domains
        bool res = orb.binaryRepartition(this->R, fl);

        if (res != true) {
           std::cout << "Could not repartition!" << std::endl;
           return;
        }
        // Update
        this->updateLayout(fl, mesh, buffer);
        this->solver_mp->setRhs(rho_m);
    }

    bool balance(size_type totalP){
        int local = 0;
        std::vector<int> res(Ippl::Comm->size());
        double threshold = 0.01;
        double equalPart = (double) totalP / Ippl::Comm->size();
        double dev = std::abs((double)this->getLocalNum() - equalPart) / totalP;
        if (dev > threshold)
            local = 1;
        MPI_Allgather(&local, 1, MPI_INT, res.data(), 1, MPI_INT, Ippl::getComm());

        for (unsigned int i = 0; i < res.size(); i++) {
            if (res[i] == 1)
                return true;
        }
        return false;
    }

    void gatherStatistics(size_type totalP) {
        auto prec = std::cout.precision();
        std::cout << "Rank " << Ippl::Comm->rank() << " has "
                  << std::setprecision(4)
                  << (double)this->getLocalNum() / totalP * 100.0
                  << "% of the total particles " << std::endl
                  << std::setprecision(prec);
    }

    void gatherCIC() {

        gather(this->E, E_m, this->R);

    }

    void scatterCIC(size_type totalP, int iteration, Vector_t& hrField) {


         Inform m("scatter ");

         rho_m = 0.0;
         scatter(q, rho_m, this->R);

         static IpplTimings::TimerRef sumTimer = IpplTimings::getTimer("Check");
         IpplTimings::startTimer(sumTimer);
         double Q_grid = rho_m.sum();

         size_type Total_particles = 0;
         size_type local_particles = this->getLocalNum();

         MPI_Reduce(&local_particles, &Total_particles, 1,
                       MPI_UNSIGNED_LONG, MPI_SUM, 0, Ippl::getComm());

         double rel_error = std::fabs((Q_m-Q_grid)/Q_m);
         m << "Rel. error in charge conservation = " << rel_error << endl;

         if(Ippl::Comm->rank() == 0) {
             if(Total_particles != totalP || rel_error > 1e-10) {
                 std::cout << "Total particles in the sim. " << totalP
                           << " " << "after update: "
                           << Total_particles << std::endl;
                 std::cout << "Total particles not matched in iteration: "
                           << iteration << std::endl;
                 std::cout << "Rel. error in charge conservation: "
                           << rel_error << std::endl;
                 exit(1);
             }
         }

         rho_m = rho_m / (hrField[0] * hrField[1] * hrField[2]);

         rhoNorm_m = norm(rho_m);
         IpplTimings::stopTimer(sumTimer);

         //dumpVTK(rho_m,nr_m[0],nr_m[1],nr_m[2],iteration,hrField[0],hrField[1],hrField[2]);

         //rho = rho_e - rho_i
         rho_m = rho_m - (Q_m/((rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2])));
    }

    void initSolver() {

        Inform m("solver ");
        if(stype_m == "FFT")
            initFFTSolver();
        else
            m << "No solver matches the argument" << endl;

    }

    void initFFTSolver() {
        ippl::ParameterList sp;
        sp.add("output_type", Solver_t::GRAD);

        ippl::FFTParams fftParams;

#ifdef Heffte_ENABLE_CUDA
        fftParams.setAllToAll( false );
#else
        fftParams.setAllToAll( true );
#endif
        fftParams.setPencils( true );
        fftParams.setReorder( false );
        fftParams.setRCDirection( 0 );

        solver_mp = std::make_shared<Solver_t>(fftParams);

        solver_mp->mergeParameters(sp);

        solver_mp->setRhs(rho_m);

        solver_mp->setLhs(E_m);
    }



     void dumpData() {

        auto Pview = P.getView();

        double Energy = 0.0;

        Kokkos::parallel_reduce("Particle Energy", this->getLocalNum(),
                                KOKKOS_LAMBDA(const int i, double& valL){
                                    double myVal = dot(Pview(i), Pview(i)).apply();
                                    valL += myVal;
                                }, Kokkos::Sum<double>(Energy));

        Energy *= 0.5;
        double gEnergy = 0.0;

        MPI_Reduce(&Energy, &gEnergy, 1,
                    MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());


        const int nghostE = E_m.getNghost();
        auto Eview = E_m.getView();
        Vector_t normE;
        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

        for (unsigned d=0; d<Dim; ++d) {

        double temp = 0.0;
        Kokkos::parallel_reduce("Vector E reduce",
                                mdrange_type({nghostE, nghostE, nghostE},
                                             {Eview.extent(0) - nghostE,
                                              Eview.extent(1) - nghostE,
                                              Eview.extent(2) - nghostE}),
                                KOKKOS_LAMBDA(const size_t i, const size_t j,
                                              const size_t k, double& valL)
                                {
                                    double myVal = pow(Eview(i, j, k)[d], 2);
                                    valL += myVal;
                                }, Kokkos::Sum<double>(temp));
            double globaltemp = 0.0;
            MPI_Reduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());
            normE[d] = sqrt(globaltemp);
        }

        if (Ippl::Comm->rank() == 0) {
            std::stringstream fname;
            fname << "data/ParticleField_";
            fname << Ippl::Comm->size();
            fname << ".csv";

            Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
            csvout.precision(10);
            csvout.setf(std::ios::scientific, std::ios::floatfield);

            if(time_m == 0.0) {
                csvout << "time, Kinetic energy, Rho_norm2, Ex_norm2, Ey_norm2, Ez_norm2" << endl;
            }

            csvout << time_m << " "
                   << gEnergy << " "
                   << rhoNorm_m << " "
                   << normE[0] << " "
                   << normE[1] << " "
                   << normE[2] << endl;
        }

        Ippl::Comm->barrier();
     }

     void dumpLandau() {

        const int nghostE = E_m.getNghost();
        auto Eview = E_m.getView();
        double fieldEnergy, ExAmp;
        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

        double temp = 0.0;
        Kokkos::parallel_reduce("Ex inner product",
                                mdrange_type({nghostE, nghostE, nghostE},
                                             {Eview.extent(0) - nghostE,
                                              Eview.extent(1) - nghostE,
                                              Eview.extent(2) - nghostE}),
                                KOKKOS_LAMBDA(const size_t i, const size_t j,
                                              const size_t k, double& valL)
                                {
                                    double myVal = pow(Eview(i, j, k)[0], 2);
                                    valL += myVal;
                                }, Kokkos::Sum<double>(temp));
        double globaltemp = 0.0;
        MPI_Reduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());
        fieldEnergy = 0.5 * globaltemp * hr_m[0] * hr_m[1] * hr_m[2];

        double tempMax = 0.0;
        Kokkos::parallel_reduce("Ex max norm",
                                mdrange_type({nghostE, nghostE, nghostE},
                                             {Eview.extent(0) - nghostE,
                                              Eview.extent(1) - nghostE,
                                              Eview.extent(2) - nghostE}),
                                KOKKOS_LAMBDA(const size_t i, const size_t j,
                                              const size_t k, double& valL)
                                {
                                    double myVal = abs(Eview(i, j, k)[0]);
                                    if(myVal > valL) valL = myVal;
                                }, Kokkos::Max<double>(tempMax));
        ExAmp = 0.0;
        MPI_Reduce(&tempMax, &ExAmp, 1, MPI_DOUBLE, MPI_MAX, 0, Ippl::getComm());

        if (Ippl::Comm->rank() == 0) {
            std::stringstream fname;
            fname << "data/ParticleField_";
            fname << Ippl::Comm->size();
            fname << ".csv";

            Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
            csvout.precision(10);
            csvout.setf(std::ios::scientific, std::ios::floatfield);

            if(time_m == 0.0) {
                csvout << "time, Ex_field_energy, Ex_max_norm" << endl;
            }

            csvout << time_m << " "
                   << fieldEnergy << " "
                   << ExAmp << endl;
        }

        Ippl::Comm->barrier();
     }

private:
    void setBCAllPeriodic() {

        this->setParticleBC(ippl::BC::PERIODIC);
    }

};
