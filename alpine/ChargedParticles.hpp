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

    double time_m;

    double rhoNorm_m;

    unsigned int loadbalancefreq_m;
    
    double loadbalancethreshold_m;


public:
    ParticleAttrib<double>     q; // charge
    typename ippl::ParticleBase<PLayout>::particle_position_type P;  // particle velocity
    typename ippl::ParticleBase<PLayout>::particle_position_type E;  // electric field at particle position


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
        this->solver_mp->setRhs(rho_m);
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
        sp.add("output_type", Solver_t::GRAD);
        sp.add("use_heffte_defaults", false);  
        sp.add("use_pencils", true);  
        sp.add("use_reorder", false);  
        sp.add("use_gpu_aware", true);  
        sp.add("comm", ippl::p2p_pl);  
        sp.add("r2c_direction", 0);  

        solver_mp = std::make_shared<Solver_t>();

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

     void dumpLangevin(unsigned int iteration, double mass, size_type N) {

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
//////////////////////////////////////////////////////////////////////////////////////////////////////
//=======start TEMPERATURE CALCULATION======  
        double locVELsum[Dim]={0.0,0.0,0.0};
        double globVELsum[Dim];
        double avgVEL[Dim];
        double locT[Dim]={0.0,0.0,0.0};
        double globT[Dim];       
	    Vector_t temperature;
	  
	    auto pRMirror = this->R.getView();
	    //  auto pRMirror = this->R.getHostMirror();
        auto pPMirror = this->P.getView();
        //	auto pPMirror = this->P.getHostMirror();
        //  deep copy...
   
	    const size_t locNp = static_cast<size_t>(this->getLocalNum());

        for(unsigned d = 0; d<Dim; ++d){
		    Kokkos::parallel_reduce("get local velocity sum", 
		    			 locNp, 
		    			 KOKKOS_LAMBDA(const int k, double& valL){
                                       	double myVal = pPMirror[k](d)/mass;
                                        valL += myVal;
                                        	//valL += pPMirror[i](d)/mass;
                                    	},                    			
		    			 Kokkos::Sum<double>(locVELsum[d])
		    			);
    	Kokkos::fence();
	    }
        // for(unsigned long k = 0; k < this->getLocalNum(); ++k) {
        //   for(unsigned d = 0; d < Dim; d++) {
        //     // loc_avg_vel[d]   += this->v[k](d);
        //     locVELsum[d] += pPMirror[k](d)/mass;
        //   }
        // }
	   	MPI_Allreduce(locVELsum, globVELsum, 3, MPI_DOUBLE, MPI_SUM, Ippl::getComm());	
        for(unsigned d=0; d<Dim; ++d) avgVEL[d]=globVELsum[d]/N;

        for(unsigned d = 0; d<Dim; ++d){
		    Kokkos::parallel_reduce("get local velocity sum", 
					 locNp,
		    			 KOKKOS_LAMBDA(const int k, double& valL){
                                       	double myVal = (pPMirror[k](d)/mass-avgVEL[d])*(pPMirror[k](d)/mass-avgVEL[d]);
                                        valL += myVal;
                                         //valL += (pPMirror[i](d)/mass-avgVEL[d])*(pPMirror[i](d)/mass-avgVEL[d]);
                                    	},                    			
		    			 Kokkos::Sum<double>(locT[d])
		     			);
    		Kokkos::fence();
	    }
        // for(unsigned long k = 0; k < this->getLocalNum(); ++k) {
        //   for(unsigned d = 0; d < Dim; d++) {
        //     // loc_temp[d]   += (this->v[k](d)-avg_vel[d])*(this->v[k](d)-avg_vel[d]);
        //     locT[d] += (pPMirror[k](d)/mass-avgVEL[d])*(pPMirror[k](d)/mass-avgVEL[d]);
        //   }
        // }
    	MPI_Reduce(locT, globT, 3, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());	
        if (Ippl::Comm->rank() == 0) for(unsigned d=0; d<Dim; ++d)    temperature[d]=globT[d]/N;

//======= end  TEMPERATURE CALCULATION    ======  
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//====== start  CALCULATING BEAM STATISTICS    &&    EMITTANCE ======

	double     centroid[2 * Dim];
	double       moment[2 * Dim][2 * Dim];

	double loc_centroid[2 * Dim];
	double   loc_moment[2 * Dim][2 * Dim];
        
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
					double    part[2 * Dim];
	            			part[1] = pPMirror[k](0);
	            			part[3] = pPMirror[k](1);
	            			part[5] = pPMirror[k](2);
	            			part[0] = pRMirror[k](0);
	            			part[2] = pRMirror[k](1);
	            			part[4] = pRMirror[k](2);
                if(k==0){
					cent += loc_centroid[i];
					mom0 += loc_moment[i][0];
					mom1 += loc_moment[i][1];
					mom2 += loc_moment[i][2];
					mom3 += loc_moment[i][3];
					mom4 += loc_moment[i][4];
					mom5 += loc_moment[i][5];
                }
	            			
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
        // for(unsigned long k = 0; k < locNp; ++k) {
        //     double    part[2 * Dim];
	    //         			part[1] = pPMirror[k](0);
	    //         			part[3] = pPMirror[k](1);
	    //         			part[5] = pPMirror[k](2);
	    //         			part[0] = pRMirror[k](0);
	    //         			part[2] = pRMirror[k](1);
	    //         			part[4] = pRMirror[k](2);

        //     for(unsigned i = 0; i < 2 * Dim; i++) {
        //         loc_centroid[i]   += part[i];
        //         for(unsigned j = 0; j <= i; j++) {
        //             loc_moment[i][j]   += part[i] * part[j];
        //         }
        //     }
        // }
    
    	// for(unsigned i = 0; i < 2 * Dim; i++) {
    	//     for(unsigned j = 0; j < i; j++) {
    	//         loc_moment[j][i] = loc_moment[i][j];
    	//     }
    	// }

    MPI_Allreduce(loc_moment, moment, 2 * Dim * 2 * Dim, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
    MPI_Allreduce(loc_centroid, centroid, 2 * Dim, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
    
    const double zero = 0.0;
    Vector_t eps2, fac, rsqsum, vsqsum, rvsum;
    Vector_t rmean, vmean, rrms, vrms, eps, rvrms;

    	for(unsigned int i = 0 ; i < Dim; i++) {
    	    rmean(i) = centroid[2 * i] / N;
    	    vmean(i) = centroid[(2 * i) + 1] / N;
    	    rsqsum(i) = moment[2 * i][2 * i] - N * rmean(i) * rmean(i);
    	    vsqsum(i) = moment[(2 * i) + 1][(2 * i) + 1] - N * vmean(i) * vmean(i);
    	    if(vsqsum(i) < 0)
    	        vsqsum(i) = 0;
    	    rvsum(i) = moment[(2 * i)][(2 * i) + 1] - N * rmean(i) * vmean(i);
    	}

    eps2 = (rsqsum * vsqsum - rvsum * rvsum) / (N * N);
    rvsum = rvsum/double(N);

    	for(unsigned int i = 0 ; i < Dim; i++) {
   		     rrms(i) = sqrt(rsqsum(i) / N);
   		     vrms(i) = sqrt(vsqsum(i) / N);
   		     eps(i)  =  std::sqrt(std::max(eps2(i), zero));
   		     double tmp = rrms(i) * vrms(i);
   		     fac(i) = (tmp == 0) ? zero : 1.0 / tmp;
   		 }
    rvrms = rvsum * fac;
    //!!!since we are using momentum and not velocity v<->p!!!

//====== end  CALCULATING BEAM STATISTICS    &&    EMITTANCE ======
/////////////////////////////////////////////////////////////////////////////////
// ===== start PRINTING  =========

	if (Ippl::Comm->rank() == 0) {
            std::stringstream fname;
            fname << "data/FieldLangevin_";
            fname << Ippl::Comm->size();
            fname << ".csv";
            Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
            csvout.precision(10);
            csvout.setf(std::ios::scientific, std::ios::floatfield);

            std::stringstream fname2;
            fname2 << "data/All_FieldLangevin_";
            fname2 << Ippl::Comm->size();
            fname2 << ".csv";
            Inform csvout2(NULL, fname2.str().c_str(), Inform::APPEND);
            csvout2.precision(10);
            csvout2.setf(std::ios::scientific, std::ios::floatfield);


            if(time_m == 0.0) {
                csvout  <<  "iteration " 	    <<
                            "time " 		    << 
                            "E_X_field_energy " << 
                            "E_X_max_norm " 	<< 
                            "T_X "        	    <<
		            	    "eps_X "            << 
                            "rprms_X"           <<
		        endl;

		csvout2 <<  "iteration" 	    <<" "<< 
                    "time" 		        <<" "<< 
                    "Ex_field_energy"   <<" "<< 
                    "Ex_max_norm" 	    <<" "<< 
                    "Tx Ty Tz "  	    <<
			        "epsX epsY epsZ "   <<
                    "rrmsX rrmsY rrmsZ "<<
			        "vrmsX vrmsY vrmsZ "<<
			        "rmeanX rmeanY rmeanZ "<<
			        "vmeanX vmeanY vmeanZ "<<
			        "rvrmsX rvrmsY rvrmsZ" <<
		endl;
	    }     

            csvout  <<  iteration 	    <<" "<<
                        time_m 		    <<" "<< 
                        fieldEnergy	    <<" "<< 
                        ExAmp		    <<" "<< 
                        temperature[0]  <<" "<< 
		            	eps[0]	        <<" "<<
		            	rvrms[0]	    <<" "<<
    	 	endl;	

	    csvout2 <<      iteration 	<<" "<<
                        time_m 		<<" "<< 
                        fieldEnergy	<<" "<< 
                        ExAmp		<<" "<< 
                        temperature  <<" "<<
			               eps		    <<" "<<
                        rrms		    <<" "<<
			               vrms		    <<" "<<
			               rmean		<<" "<<
			               vmean		<<" "<<
			               rvrms 		<<	
		endl;
        }
        
        Ippl::Comm->barrier();
 }

//===================================================================================================
//===================================================================================================
//===================================================================================================
     
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
