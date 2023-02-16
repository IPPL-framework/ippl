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
#include "Solver/FFTPoissonSolver.h"

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
typedef ippl::FFTPoissonSolver<Vector_t, double, Dim> VSolver_t;


typedef Vector<Vector_t, Dim> Matrix_t;
typedef Field<Matrix_t, Dim> MField_t;

const double pi = std::acos(-1.0);

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
    vtkout << "ORIGIN "     << -dx  << " " << -dy  << " "  << -dz << endl;
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

    // start with header
    vtkout << "# vtk DataFile Version 2.0" << endl;
    vtkout << TestName << endl;
    vtkout << "ASCII" << endl;
    vtkout << "DATASET STRUCTURED_POINTS" << endl;
    vtkout << "DIMENSIONS " << nx+3 << " " << ny+3 << " " << nz+3 << endl;
    vtkout << "ORIGIN " << -dx << " " << -dy << " " << -dz << endl;
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
    std::shared_ptr<VSolver_t> solver_mp2;

    double time_m;

    double rhoNorm_m;

    unsigned int loadbalancefreq_m;
    
    double loadbalancethreshold_m;


public:
    ParticleAttrib<double>     q; // charge
    ParticleAttrib<Vector_t> P;  // particle velocity
    ParticleAttrib<Vector_t> E;  // electric field at particle position

    // typename ippl::ParticleBase<PLayout>::particle_position_type P;  // particle velocity
    // typename ippl::ParticleBase<PLayout>::particle_position_type E;  // electric field at particle position

    // EXCL_LANGEVIN ... 
    //ORB orb_v;     //NEW
    unsigned int nP; // NEW
    double pMass; // NEW


    double KAPPA; //NEW

    Field_t   fr_m; //NEW
    Field_t   fv_mv; //NEW
    VField_t  gradRBH_mv; //(=Fd) NEW
    VField_t diffCoeffArr_mv[3];//NEW
    MField_t diffusionCoeff_mv;//NEW

    std::shared_ptr<VSolver_t> solver_mvRB_H; //NEW
    std::shared_ptr<VSolver_t> solver_mvRB_G; //NEW

    ParticleAttrib<double> fr;//NEW == 1
    ParticleAttrib<double> fv;//NEW == 1
    ParticleAttrib<Vector_t> Fd;//NEW
    ParticleAttrib<Vector_t> D0;//NEW
    ParticleAttrib<Vector_t> D1;//NEW
    ParticleAttrib<Vector_t> D2;//NEW

    Vector<int, Dim> nv_mv; //NEW
    Vector_t hv_mv; //NEW
    Vector_t vmin_mv; //NEW
    Vector_t vmax_mv; //NEW



    /*
      This constructor is mandatory for all derived classes from
      ParticleBase as the bunch buffer uses this
    */
    ChargedParticles(PLayout& pl)
    : ippl::ParticleBase<PLayout>(pl)
    {
        // register the particle attributes
        this->addAttribute(q);
        this->addAttribute(P);
        this->addAttribute(E);
        //EXCL_LANGEVIN
        this->addAttribute(fr);
        this->addAttribute(fv);
        this->addAttribute(Fd);
        this->addAttribute(D0);
        this->addAttribute(D1);
        this->addAttribute(D2);
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
        //EXCL_LANGEVIN
        this->addAttribute(fr);
        this->addAttribute(fv);
        this->addAttribute(Fd);
        this->addAttribute(D0);
        this->addAttribute(D1);
        this->addAttribute(D2);


        setupBCs();
        for (unsigned int i = 0; i < Dim; i++)
            decomp_m[i]=decomp[i];
    }

    ~ChargedParticles(){ }

    void setupBCs() {
        setBCAllPeriodic();
    }

    void updateLayout(FieldLayout_t& fl, Mesh_t& mesh, ChargedParticles<PLayout>& buffer,
                      bool& isFirstRepartition) {
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
        if(!isFirstRepartition) {
            layout.update(*this, buffer);
        }
        IpplTimings::stopTimer(tupdatePLayout);
    }

    void initializeORB(FieldLayout_t& fl, Mesh_t& mesh) {
        orb.initialize(fl, mesh, rho_m);
    }

    void repartition(FieldLayout_t& fl, Mesh_t& mesh, ChargedParticles<PLayout>& buffer, 
                     bool& isFirstRepartition) {
        // Repartition the domains
        bool res = orb.binaryRepartition(this->R, fl, isFirstRepartition);

        if (res != true) {
           std::cout << "Could not repartition!" << std::endl;
           return;
        }
        // Update
        this->updateLayout(fl, mesh, buffer, isFirstRepartition);
        this->solver_mp->setRhs(rho_m); //does not exist for current solver used??
        // #######################################################################
    }

    bool balance(size_type totalP, const unsigned int nstep){
        if(std::strcmp(TestName,"UniformPlasmaTest") == 0) {
            return (nstep % loadbalancefreq_m == 0);
        }
        else {
            int local = 0;
            std::vector<int> res(Ippl::Comm->size());
            double equalPart = (double) totalP / Ippl::Comm->size();
            double dev = std::abs((double)this->getLocalNum() - equalPart) / totalP;
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
        double equalPart = (double) totalP / Ippl::Comm->size();
        double dev = (std::abs((double)this->getLocalNum() - equalPart) 
                     / totalP) * 100.0;
        MPI_Gather(&dev, 1, MPI_DOUBLE, imb.data(), 1, MPI_DOUBLE, 0, 
                   Ippl::getComm());
    
        if (Ippl::Comm->rank() == 0) {
            std::stringstream fname;
            fname << "data/LoadBalance_";
            fname << Ippl::Comm->size();
            fname << ".csv";

            Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
            csvout.precision(5);
            csvout.setf(std::ios::scientific, std::ios::floatfield);

            if(time_m == 0.0) {
                csvout << "time, rank, imbalance percentage" << endl;
            }

            for(int r=0; r < Ippl::Comm->size(); ++r) { 
                csvout << time_m << " "
                       << r << " "
                       << imb[r] << endl;
            }
        }

        Ippl::Comm->barrier();
    
    }

    void gatherCIC() {

        gather(this->E, E_m, this->R);

    }

    void scatterCIC(size_type totalP, unsigned int iteration, Vector_t& hrField) {


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
                 m << "Time step: " << iteration << endl;
                 m << "Total particles in the sim. " << totalP
                   << " " << "after update: "
                   << Total_particles << endl;
                 m << "Rel. error in charge conservation: "
                   << rel_error << endl;
                 std::abort();
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
        sp.add("use_heffte_defaults", false);  
        sp.add("use_pencils", true);  
        sp.add("use_reorder", false);  
        sp.add("use_gpu_aware", true);  
        sp.add("comm", ippl::p2p_pl);  
        sp.add("r2c_direction", 0);  


        // sp.add("output_type", Solver_t::GRAD);
        //open boundary condion:

        solver_mp2 = std::make_shared<VSolver_t>(this->E_m, this->rho_m, sp, "VICO");
        
        // solver_mp2 = std::make_shared<VSolver_t>(this->E_m, this->rho_m, sp, "HOCKNEY");
        solver_mp2 = std::make_shared<VSolver_t>(this->E_m, this->rho_m, sp, "VICO");


        sp.add("output_type", Solver_t::GRAD);
        solver_mp = std::make_shared<Solver_t>(); //periodic
        solver_mp->mergeParameters(sp);
        solver_mp->setLhs(E_m);
        solver_mp->setRhs(rho_m);




    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////
    //===================================================================================================
//===================================================================================================
//===================================================================================================
    ////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
// Possible algorithms: "HOCKNEY" or "VICO".

    void initRosenbluthSolver(){

        ippl::ParameterList sp;

        sp.add("use_pencils", true);
        sp.add("comm", ippl::p2p_pl);  
        sp.add("use_reorder", false); 
        sp.add("use_heffte_defaults", false);  
        sp.add("use_gpu_aware", true);  
        sp.add("r2c_direction", 0); 
        // sp.add("output_type", VSolver_t::SOL); //GRAD

        solver_mvRB_H = std::make_shared<VSolver_t>(this->gradRBH_mv,  this->fv_mv, sp, "VICO");
        solver_mvRB_G = std::make_shared<VSolver_t>(this->fv_mv, sp, "BIHARMONIC");
    }

        // Inform m("scatterVEL");
        // how to check kinetic energy conservations...
        //  m << "Rel. error in Ekin conservation = " << rel_error << endl;
        //there exist a reduction for views and particle attributes in ippl?
        // iuess we can check for integration over velocity space conservation...         
        // ???????????????????????????????????????????????????????????


   
    void scatterPhaseSpaceDist(Vector_t& hrField) {


         Inform m("scatter 2");

        fv = 1.0;
        fv_mv = 0.0;
        scatter(this->fv, this->fv_mv, this->P);   
        fv_mv = fv_mv / (this->hv_mv[0] * this->hv_mv[1] * this->hv_mv[2]*this->nP);
        // fv_mv = fv_mv - (double(nP)/((vmax_mv[0] - vmin_mv[0]) * (vmax_mv[1] - vmin_mv[1]) * (vmax_mv[2] - vmin_mv[2])));

        fr = 1.0;
        fr_m = 0.0;
        scatter(this->fr, this->fr_m, this->R);     
        fr_m = fr_m / (hrField[0] * hrField[1] * hrField[2]);// / this->nP);
        // fr_m = fr_m - (double(nP)/((rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2]))); 

        
        double parts_grid_m = fr_m.sum()*(hrField[0] * hrField[1] * hrField[2]);
        double parts_grid_v = fv_mv.sum()*(this->hv_mv[0] * this->hv_mv[1] * this->hv_mv[2]);
        m << parts_grid_m << "  " << parts_grid_v << endl;
    }
    void rescatterVelocityDist() {

        Inform m("scatter 3");
        fv = 1.0;
        fv_mv = 0.0;
        scatter(this->fv, this->fv_mv, this->P);   
        fv_mv = fv_mv / (this->hv_mv[0] * this->hv_mv[1] * this->hv_mv[2]*this->nP);
    }
    void gather_Fd_D_Density() {

        gather(this->fr, this->fr_m, this->R); 

        gather(this->Fd, this->gradRBH_mv, this->P);
        gather(this->D0, diffCoeffArr_mv[0], this->P);
        gather(this->D1, diffCoeffArr_mv[1], this->P);
        gather(this->D2, diffCoeffArr_mv[2], this->P);



    }




    // void gather22(Vector_t& hrField) {
        
    //     Inform m("gatter 22");

    //     double parts_grid_1 = fr_m.sum();
    //     fr_m = fr_m / 156055;
    //     double parts_grid_2 = fr_m.sum();
    //     fr_m = fr_m * 156055;


    //     fr_m = fr_m / (hrField[0] * hrField[1] * hrField[2]);
    //     double parts_grid_mt = fr_m.sum();

    //     m << parts_grid_1 << "  "
    //     << parts_grid_2 << "  " 
    //     << parts_grid_mt << endl;

    //     gather(this->fr, this->fr_m, this->R); 
    // }


    ////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////
    //===================================================================================================
//===================================================================================================
//===================================================================================================
    ////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////
//===================================================================================================
//===================================================================================================
//===================================================================================================
    ////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////

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
                                    double myVal = std::pow(Eview(i, j, k)[d], 2);
                                    valL += myVal;
                                }, Kokkos::Sum<double>(temp));
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
                                    double myVal = std::pow(Eview(i, j, k)[0], 2);
                                    valL += myVal;
                                }, Kokkos::Sum<double>(temp));
        double globaltemp = 0.0;
        MPI_Reduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());
        fieldEnergy = globaltemp * hr_m[0] * hr_m[1] * hr_m[2];

        double tempMax = 0.0;
        Kokkos::parallel_reduce("Ex max norm",
                                mdrange_type({nghostE, nghostE, nghostE},
                                             {Eview.extent(0) - nghostE,
                                              Eview.extent(1) - nghostE,
                                              Eview.extent(2) - nghostE}),
                                KOKKOS_LAMBDA(const size_t i, const size_t j,
                                              const size_t k, double& valL)
                                {
                                    double myVal = std::fabs(Eview(i, j, k)[0]);
                                    if(myVal > valL) valL = myVal;
                                }, Kokkos::Max<double>(tempMax));
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

            if(time_m == 0.0) {
                csvout << "time, Ex_field_energy, Ex_max_norm" << endl;
            }

            csvout << time_m << " "
                   << fieldEnergy << " "
                   << ExAmp << endl;

        }
        
        Ippl::Comm->barrier();
     }

//===================================================================================================
//===================================================================================================
//===================================================================================================
    ////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////
//===================================================================================================
//===================================================================================================
//===================================================================================================
    ////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////
//additionally print vmax or vmin??? ... 
//print particle data every other step to for analysis ...


     void dumpLangevin(unsigned int iteration,  size_type N, std::string folder) {
         Inform m("DUMPLangevin");


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
                                    double myVal = std::pow(Eview(i, j, k)[0], 2);
                                    valL += myVal;
                                }, Kokkos::Sum<double>(temp));
                                Kokkos::fence();
        double globaltemp = 0.0;
        MPI_Reduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());
        fieldEnergy = globaltemp * hr_m[0] * hr_m[1] * hr_m[2];

        double tempMax = 0.0;
        Kokkos::parallel_reduce("Ex max norm",
                                mdrange_type({nghostE, nghostE, nghostE},
                                             {Eview.extent(0) - nghostE,
                                              Eview.extent(1) - nghostE,
                                              Eview.extent(2) - nghostE}),
                                KOKKOS_LAMBDA(const size_t i, const size_t j,
                                              const size_t k, double& valL)
                                {
                                    double myVal = std::fabs(Eview(i, j, k)[0]);
                                    if(myVal > valL) valL = myVal;
                                }, Kokkos::Max<double>(tempMax));
                                Kokkos::fence();
        ExAmp = 0.0;
        MPI_Reduce(&tempMax, &ExAmp, 1, MPI_DOUBLE, MPI_MAX, 0, Ippl::getComm());
//////////////////////////////////////////////////////////////////////////////////////////////////////
m << "temperature" << endl;
//=======start TEMPERATURE CALCULATION======  
        double locVELsum[Dim]={0.0,0.0,0.0};
        double globVELsum[Dim];
        double avgVEL[Dim];
        double locT[Dim]={0.0,0.0,0.0};
        double globT[Dim];       
	    Vector_t temperature;

	    const size_t locNp = static_cast<size_t>(this->getLocalNum());
	  
        //TODO get Rid of auto
	    auto pRView = this->R.getView();
        auto pPView = this->P.getView();
        auto pVMirror = this->P.getHostMirror();
        Kokkos::deep_copy(pVMirror, pPView);

        for(unsigned d = 0; d<Dim; ++d){
		    Kokkos::parallel_reduce("get local velocity sum", 
		    			 locNp, 
		    			 KOKKOS_LAMBDA(const int k, double& valL){
                                       	double myVal = pVMirror(k)[d];
                                        valL += myVal;
                                        //valL += pVMirror(k)[d];
                                    	},                    			
		    			 Kokkos::Sum<double>(locVELsum[d])
		    			);
    	Kokkos::fence();
	    }
	   	MPI_Allreduce(locVELsum, globVELsum, 3, MPI_DOUBLE, MPI_SUM, Ippl::getComm());	
        for(unsigned d=0; d<Dim; ++d) avgVEL[d]=globVELsum[d]/N;

        for(unsigned d = 0; d<Dim; ++d){
		    Kokkos::parallel_reduce("get local velocity sum", 
					 locNp,
		    			 KOKKOS_LAMBDA(const int k, double& valL){
                                       	double myVal = (pVMirror(k)[d]-avgVEL[d])*(pVMirror(k)[d]-avgVEL[d]);
                                        valL += myVal;
                                         //valL += (pVMirror(k)[d]/mass-avgVEL[d])*(pVMirror(k)[d]/mass-avgVEL[d]);
                                    	},                    			
		    			 Kokkos::Sum<double>(locT[d])
		     			);
    		Kokkos::fence();
	    }

    	MPI_Reduce(locT, globT, 3, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());	
        if (Ippl::Comm->rank() == 0) for(unsigned d=0; d<Dim; ++d)    temperature[d]=globT[d]/N;

//======= end  TEMPERATURE CALCULATION    ======  
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//====== start  CALCULATING BEAM STATISTICS    &&    EMITTANCE ======
    m << "BeamStatistics" << endl;
    const double zero = 0.0;

	double     centroid[2 * Dim]={};
	double       moment[2 * Dim][2 * Dim]={};
	double     Ncentroid[2 * Dim]={};
	double       Nmoment[2 * Dim][2 * Dim]={};

	double loc_centroid[2 * Dim]={};
	double   loc_moment[2 * Dim][2 * Dim]={};
        
	for(unsigned i = 0; i < 2 * Dim; i++) {
            loc_centroid[i] = 0.0;
            for(unsigned j = 0; j <= i; j++) {
                loc_moment[i][j] = 0.0;
                loc_moment[j][i] = 0.0;
            }
   	 }
    

	for(unsigned i = 0; i< 2*Dim; ++i){

	Kokkos::parallel_reduce("write Emittance 1 redcution",
				locNp,
				KOKKOS_LAMBDA(const int k,
						double& cent,
						double& mom0,
						double& mom1,
						double& mom2,
						double& mom3,
						double& mom4,
						double& mom5
                        ){ 
					double  part[2 * Dim];
	            			part[0] = pRView(k)[0];
	            			part[1] = pVMirror(k)[0];
	            			part[2] = pRView(k)[1];
	            			part[3] = pVMirror(k)[1];
	            			part[4] = pRView(k)[2];
	            			part[5] = pVMirror(k)[2];

	            			
					cent += part[i];
					mom0 += part[i]*part[0];
					mom1 += part[i]*part[1];
					mom2 += part[i]*part[2];
					mom3 += part[i]*part[3];
					mom4 += part[i]*part[4];
					mom5 += part[i]*part[5];

				},
				Kokkos::Sum<double>(loc_centroid[i]),
				Kokkos::Sum<double>(loc_moment[i][0]),
				Kokkos::Sum<double>(loc_moment[i][1]),
				Kokkos::Sum<double>(loc_moment[i][2]),
				Kokkos::Sum<double>(loc_moment[i][3]),
				Kokkos::Sum<double>(loc_moment[i][4]),
				Kokkos::Sum<double>(loc_moment[i][5])
		);	
	Kokkos::fence();
	}
    Ippl::Comm->barrier();
    MPI_Allreduce(loc_moment, moment, 2 * Dim * 2 * Dim, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
    MPI_Allreduce(loc_centroid, centroid, 2 * Dim, MPI_DOUBLE, MPI_SUM, Ippl::getComm());

    Ippl::Comm->barrier();
    	for(unsigned i = 0; i < 2 * Dim; i++) {
            loc_centroid[i] = 0.0;
            for(unsigned j = 0; j <= i; j++) {
                loc_moment[i][j] = 0.0;
                loc_moment[j][i] = 0.0;
            }
   	 }

        

	for(unsigned i = 0; i< 2*Dim; ++i){
        // NORMALIZED EMITTANCE!!!!!
    	Kokkos::parallel_reduce("write Emittance 1 redcution",
				locNp,
				KOKKOS_LAMBDA(const int k,
						double& cent,
						double& mom0,
						double& mom1,
						double& mom2,
						double& mom3,
						double& mom4,
						double& mom5
						
                        ){ 
                            // double c = 3e9; //(cm/ms) used most of the time...
                            double c = 2.99792458e9;
                            double v2 =       pVMirror(k)[0]*pVMirror(k)[0]
                                            + pVMirror(k)[1]*pVMirror(k)[1]
                                            + pVMirror(k)[2]*pVMirror(k)[2];
                            double lorentz = 1.0/(sqrt(1.0-v2/c));

					double  part[2 * Dim];
	            			part[0] = pRView(k)[0];
	            			part[1] = (pVMirror(k)[0]/c)*lorentz;
	            			part[2] = pRView(k)[1];
	            			part[3] = (pVMirror(k)[1]/c)*lorentz;
	            			part[4] = pRView(k)[2];
	            			part[5] = (pVMirror(k)[2]/c)*lorentz;
                            
	            			
					cent += part[i];
					mom0 += part[i]*part[0];
					mom1 += part[i]*part[1];
					mom2 += part[i]*part[2];
					mom3 += part[i]*part[3];
					mom4 += part[i]*part[4];
					mom5 += part[i]*part[5];

				},
				Kokkos::Sum<double>(loc_centroid[i]),
				Kokkos::Sum<double>(loc_moment[i][0]),
				Kokkos::Sum<double>(loc_moment[i][1]),
				Kokkos::Sum<double>(loc_moment[i][2]),
				Kokkos::Sum<double>(loc_moment[i][3]),
				Kokkos::Sum<double>(loc_moment[i][4]),
				Kokkos::Sum<double>(loc_moment[i][5])
		);	
	Kokkos::fence();
	}

    Ippl::Comm->barrier();
    MPI_Allreduce(loc_moment, Nmoment, 2 * Dim * 2 * Dim, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
    MPI_Allreduce(loc_centroid, Ncentroid, 2 * Dim, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
    Ippl::Comm->barrier();
    /////////////////////////////////////////////////////////


    
        Vector_t eps2, fac;
        Vector_t rsqsum, vsqsum, rvsum;
        Vector_t rmean, vmean, rrms, vrms, eps, rvrms;
        Vector_t norm;

        Vector_t Neps2, Nfac;
        Vector_t Nrsqsum, Nvsqsum, Nrvsum;
        Vector_t Nrmean, Nvmean, Nrrms, Nvrms, Neps, Nrvrms;
        Vector_t Nnorm;

    if (Ippl::Comm->rank() == 0)
    {



        	for(unsigned int i = 0 ; i < Dim; i++) {
        	    rmean(i) = centroid[2 * i] / N;
        	    vmean(i) = centroid[(2 * i) + 1] / N;
        	    rsqsum(i) = moment[2 * i][2 * i] - N * rmean(i) * rmean(i);
        	    vsqsum(i) = moment[(2 * i) + 1][(2 * i) + 1] - N * vmean(i) * vmean(i);
        	    if(vsqsum(i) < 0)   vsqsum(i) = 0;
        	    rvsum(i) = (moment[(2 * i)][(2 * i) + 1] - N * rmean(i) * vmean(i));

                Nrmean(i) = Ncentroid[2 * i] / N;
        	    Nvmean(i) = Ncentroid[(2 * i) + 1] / N;
        	    Nrsqsum(i) = Nmoment[2 * i][2 * i] - N * Nrmean(i) * Nrmean(i);
        	    Nvsqsum(i) = Nmoment[(2 * i) + 1][(2 * i) + 1] - N * Nvmean(i) * Nvmean(i);
        	    if(Nvsqsum(i) < 0)   Nvsqsum(i) = 0;
        	    Nrvsum(i) = (Nmoment[(2 * i)][(2 * i) + 1] - N * Nrmean(i) * Nvmean(i));


        	}


        //coefficient wise
        eps2  = (rsqsum * vsqsum - rvsum * rvsum) / (N * N);
        rvsum = rvsum/ N;

        Neps2  = (Nrsqsum * Nvsqsum - Nrvsum * Nrvsum) / (N * N);
        Nrvsum = Nrvsum/ N;




        	for(unsigned int i = 0 ; i < Dim; i++) {
                //  rvsum(i) /= N;
   	    	     rrms(i) = sqrt(rsqsum(i) / N);
   	    	     vrms(i) = sqrt(vsqsum(i) / N);
                //  eps2(i) = rsqsum(i)*vsqsum(i)/(N*N) - rvsum(i)*rvsum(i);

   	    	     eps(i)  =  std::sqrt(std::max(eps2(i), zero));
   	    	     double tmpry = rrms(i) * vrms(i);
   	    	     fac(i) = (tmpry == 0.0) ? zero : 1.0/tmpry;


   	    	     Nrrms(i) = sqrt(rsqsum(i) / N);
   	    	     Nvrms(i) = sqrt(vsqsum(i) / N);

   	    	     Neps(i)  =  std::sqrt(std::max(Neps2(i), zero));
   	    	     tmpry = Nrrms(i) * Nvrms(i);
   	    	     Nfac(i) = (tmpry == 0.0) ? zero : 1.0/tmpry;
   	    	}
        Nrvrms = Nrvsum * Nfac;


    }

//VMAX VMIN////////////////////////////////////////////////////////////////////////////
    double vmax_loc[Dim];
    double vmin_loc[Dim];
    double vmax[Dim];
    double vmin[Dim];

	for(unsigned d = 0; d<Dim; ++d){

		Kokkos::parallel_reduce("vel max", 
					 this->getLocalNum(),
					 KOKKOS_LAMBDA(const int i, double& mm){   
              		                    mm = std::max(pPView(i)[d], mm);
                                	 },                    			
					  Kokkos::Max<double>(vmax_loc[d])
					);

		Kokkos::parallel_reduce("vel min", 
					 this->getLocalNum(),
					 KOKKOS_LAMBDA(const int i, double& mm){   
              		                    mm = std::min(pPView(i)[d], mm);

                                	 },                    			
					  Kokkos::Min<double>(vmin_loc[d])
					);
	}
	Kokkos::fence();
	MPI_Allreduce(vmax_loc, vmax, Dim , MPI_DOUBLE, MPI_MAX, Ippl::getComm());
	MPI_Allreduce(vmin_loc, vmin, Dim , MPI_DOUBLE, MPI_MIN, Ippl::getComm());
    Ippl::Comm->barrier();

    // RMAX RMIN    // ??? cant use  coefficient wise given reductions here?
    double rmax_loc[Dim];
    double rmin_loc[Dim];
    double rmax[Dim];
    double rmin[Dim];

	for(unsigned d = 0; d<Dim; ++d){

		Kokkos::parallel_reduce("rel max", 
					 this->getLocalNum(),
					 KOKKOS_LAMBDA(const int i, double& mm){   
              		                    mm = std::max(pRView(i)[d], mm);
                                	 },                    			
					  Kokkos::Max<double>(rmax_loc[d])
					);

		Kokkos::parallel_reduce("rel min", 
					 this->getLocalNum(),
					 KOKKOS_LAMBDA(const int i, double& mm){   
              		                    mm = std::min(pRView(i)[d], mm);

                                	 },                    			
					  Kokkos::Min<double>(rmin_loc[d])
					);
	}
	Kokkos::fence();
	MPI_Allreduce(rmax_loc, rmax, Dim , MPI_DOUBLE, MPI_MAX, Ippl::getComm());
	MPI_Allreduce(rmin_loc, rmin, Dim , MPI_DOUBLE, MPI_MIN, Ippl::getComm());
    Ippl::Comm->barrier();
//==    ==== end  CALCULATING BEAM STATISTICS    &&    EMITTANCE ======
/////////////////////////////////////////////////////////////////////////////////
// ===== start PRINTING  =========


    if (Ippl::Comm->rank() == 0)
    {
    

            // std::string folder2 = folder;
            std::stringstream fname;
            fname << "/FieldLangevin_";
            fname << Ippl::Comm->size();
            fname << ".csv";
            Inform csvout(NULL, (folder+fname.str()).c_str(), Inform::APPEND);
            csvout.precision(10);
            csvout.setf(std::ios::scientific, std::ios::floatfield);

            std::stringstream fname2;
            fname2 << "/All_FieldLangevin_";
            fname2 << Ippl::Comm->size();
            fname2 << ".csv";
            Inform csvout2(NULL, (folder+fname2.str()).c_str(), Inform::APPEND);
            csvout2.precision(10);
            csvout2.setf(std::ios::scientific, std::ios::floatfield);


            if(time_m == 0.0) {
        csvout  <<          
                    "iteration, "           <<
                    "time,   "              << 
                    // "E_X_field_energy, " << 
                    // "E_X_max_norm, "     << 
                    "T_X,    "              <<
                    "rvrms_X,"              <<
                    "eps_X,  "              <<
                    "Neps_X,"              <<  
                    endl;

		csvout2 <<  
                    "iteration,"            << 
                    "vmaxX,vmaxY,vmaxZ,"    <<"                                        "<<
                    "vminX,vminY,vminZ,"    <<"                                        "<<
                    "rmaxX,rmaxY,rmaxZ,"    <<"                                        "<<
                    "rminX,rminY,rminZ,"    <<"                                        "<<
                    "vrmsX,vrmsY,vrmsZ,"    <<"                                        "<<
                    "Tx,Ty,Tz,"             <<"                                        "<<
                    "epsX,epsY,epsZ,"       <<"                                        "<<
                    "epsX2,epsY2,epsZ2,"    <<"                                        "<<
                    "rvrmsX,rvrmsY,rvrmsZ," <<"                                        "<<
                    "rrmsX,rrmsY,rrmsZ,"    <<"                                        "<<
                    "rmeanX,rmeanY,rmeanZ," <<"                                        "<<
                    "vmeanX,vmeanY,vmeanZ," <<"                                        "<<
                    "time,"                 <<
                    "Ex_field_energy,"      <<
                    "Ex_max_norm,"          <<
                    endl;
	    }     

            csvout<<    
                    iteration       <<","<<
                    time_m          <<","<< 
                    // fieldEnergy     <<" "<< 
                    // ExAmp           <<" "<< 
                    temperature[0]  <<","<< 
                    rvrms[0]        <<","<<
                    eps[0]          <<","<<
                    Neps[0]          <<","<<
    	 	endl;	

            csvout2<<   
                    iteration   <<","<<
                            vmax[0]<<","<<        vmax[1]<<","<<        vmax[2]<<",     "<<
                            vmin[0]<<","<<        vmin[1]<<","<<        vmin[2]<<",     "<<	
                            rmax[0]<<","<<        rmax[1]<<","<<        rmax[2]<<",     "<<
                            rmin[0]<<","<<        rmin[1]<<","<<        rmin[2]<<",     "<<	
                    vrms        (0)<<","<<vrms        (1)<<","<<vrms        (2)<<",     "<<
                    temperature (0)<<","<<temperature (1)<<","<<temperature (2)<<",     "<<
                    eps         (0)<<","<<eps         (1)<<","<<eps         (2)<<",     "<<
                    eps2        (0)<<","<<eps2        (1)<<","<<eps2        (2)<<",     "<<
                    rvrms       (0)<<","<<rvrms       (1)<<","<<rvrms       (2)<<",     "<<	
                    rrms        (0)<<","<<rrms        (1)<<","<<rrms        (2)<<",     "<<
                    rmean       (0)<<","<<rmean       (1)<<","<<rmean       (2)<<",     "<<
                    vmean       (0)<<","<<vmean       (1)<<","<<vmean       (2)<<",     "<<
                    time_m      <<","<< 
                    fieldEnergy <<","<< 
                    ExAmp       <<","<< 
		endl;
        }
        
        Ippl::Comm->barrier();
 }

//===================================================================================================
//===================================================================================================
//===================================================================================================
    ////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////
     
     void dumpBumponTail() {

        const int nghostE = E_m.getNghost();
        auto Eview = E_m.getView();
        double fieldEnergy, EzAmp;
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
                                    double myVal = std::pow(Eview(i, j, k)[2], 2);
                                    valL += myVal;
                                }, Kokkos::Sum<double>(temp));
        double globaltemp = 0.0;
        MPI_Reduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());
        fieldEnergy = globaltemp * hr_m[0] * hr_m[1] * hr_m[2];

        double tempMax = 0.0;
        Kokkos::parallel_reduce("Ex max norm",
                                mdrange_type({nghostE, nghostE, nghostE},
                                             {Eview.extent(0) - nghostE,
                                              Eview.extent(1) - nghostE,
                                              Eview.extent(2) - nghostE}),
                                KOKKOS_LAMBDA(const size_t i, const size_t j,
                                              const size_t k, double& valL)
                                {
                                    double myVal = std::fabs(Eview(i, j, k)[2]);
                                    if(myVal > valL) valL = myVal;
                                }, Kokkos::Max<double>(tempMax));
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

            if(time_m == 0.0) {
                csvout << "time, Ez_field_energy, Ez_max_norm" << endl;
            }

            csvout << time_m << " "
                   << fieldEnergy << " "
                   << EzAmp << endl;

        }
        
        Ippl::Comm->barrier();
     }

     void dumpParticleData() {

        typename ParticleAttrib<Vector_t>::HostMirror R_host = this->R.getHostMirror();
        typename ParticleAttrib<Vector_t>::HostMirror P_host = this->P.getHostMirror();
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
        for (size_type i = 0; i< this->getLocalNum(); i++) {
            pcsvout << R_host(i)[0] << " "
                    << R_host(i)[1] << " "
                    << R_host(i)[2] << " "
                    << P_host(i)[0] << " "
                    << P_host(i)[1] << " "
                    << P_host(i)[2] << endl;
        }
        Ippl::Comm->barrier();
     }
     
     void dumpLocalDomains(const FieldLayout_t& fl, const unsigned int step) {

        if (Ippl::Comm->rank() == 0) {
            const typename FieldLayout_t::host_mirror_type domains = fl.getHostLocalDomains();
            std::ofstream myfile;
            myfile.open("data/domains" + std::to_string(step) + ".txt");
            for (unsigned int i = 0; i < domains.size(); ++i) {
                myfile << domains[i][0].first() << " " << domains[i][1].first() << " " << domains[i][2].first() << " "
                       << domains[i][0].first() << " " << domains[i][1].last() << " " << domains[i][2].first() << " "
                       << domains[i][0].last() << " " << domains[i][1].first() << " " << domains[i][2].first() << " "
                       << domains[i][0].first() << " " << domains[i][1].first() << " " << domains[i][2].last()
                       << "\n";
            }
            myfile.close();
        }
        Ippl::Comm->barrier();
     }

private:
    void setBCAllPeriodic() {

        this->setParticleBC(ippl::BC::PERIODIC);
    }

};
