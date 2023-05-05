#ifndef LANGEVINPARTICLES_HPP
#define LANGEVINPARTICLES_HPP

// Langevin header file
//   Defines a particle attribute for particles existing in velocity space
//   This is used when discretising the velocity distribution of a Langevin
//   type equation with the Cloud-In-Cell (CIC) Method
//   Assumption on the Particle's attributes:
//   - single species particles (same mass and charge)
//   It computes Rosenbluth Potentials which are needed to evolve the
//   velocity particles in velocity space
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
#include "ChargedParticles.hpp"
#include "Solver/FFTPeriodicPoissonSolver.h"
#include "Solver/FFTPoissonSolver.h"

#include <Kokkos_Random.hpp>

#include "LangevinHelpers.hpp"

typedef Field<double, Dim>   Field_t;
typedef Field<VectorD_t, Dim> VField_t;
typedef ippl::FFTPeriodicPoissonSolver<VectorD_t, double, Dim> Solver_t;
typedef ippl::FFTPoissonSolver<VectorD_t, double, Dim> VSolver_t;

typedef Field<MatrixD_t, Dim> MField_t;


template<class PLayout>
class LangevinParticles : public ChargedParticles<PLayout> {

    typedef Solver_t PeriodicSolver_t;
    typedef ippl::FFTPoissonSolver<VectorD_t, double, Dim> OpenSolver_t;
    typedef Kokkos::Random_XorShift64_Pool<> KokkosRNG_t;

    // View types (of particle attributes)
    typedef ParticleAttrib<double>::view_type attr_view_t; // Scalar particle attributes 
    typedef ParticleAttrib<VectorD_t>::view_type attr_Dview_t; // D-dimensional particle attributes
    typedef ParticleAttrib<double>::HostMirror attr_mirror_t; // Scalar particle attributes
    typedef ParticleAttrib<VectorD_t>::HostMirror attr_Dmirror_t; // D-dimensional particle attributes

    // View types (of Fields)
    typedef ippl::detail::ViewType<double, Dim>::view_type field_view_t; // Scalar Fields
    typedef ippl::detail::ViewType<VectorD_t, Dim>::view_type field_Dview_t; // D-dimensional Fields

public:
    /*
      This constructor is mandatory for all derived classes from
      ParticleBase as the bunch buffer uses this
    */
    LangevinParticles(PLayout& pl)
      : ChargedParticles<PLayout>(pl){
    }

    LangevinParticles(PLayout& pl,
                      VectorD_t hr,
                      VectorD_t rmin,
                      VectorD_t rmax,
                      ippl::e_dim_tag configSpaceDecomp[Dim],
                      double pCharge,
                      double pMass,
                      double Q,
                      size_t globalNumParticles,
                      double dt)
        : ChargedParticles<PLayout>(pl,hr,rmin,rmax,configSpaceDecomp,Q),
        pCharge_m(pCharge),
        pMass_m(pMass),
        globalNum_m(globalNumParticles),
        dt_m(dt) {
    }

    void initAllSolvers() {
        initSpaceChargeSolver();
    }

    void initSpaceChargeSolver() {
        // Initializing the solvers defined by the ChargedParticles Class
        // [Hockney Periodic Poisson Solver]
        ChargedParticles<PLayout>::initSolver(Solver_t::GRAD);
    }

    size_type getGlobParticleNum() const { return globalNum_m; }

    VectorD_t compAvgSCForce(double beamRadius) {
        Inform m("computeAvgSpaceChargeForces ");

        VectorD_t avgEF;
        double locEFsum[Dim] = {};
        double globEFsum[Dim];
        double locQ, globQ;
        double locCheck, globCheck;

        attr_view_t  pq_view = this->q.getView();
        attr_Dview_t pE_view = this->E.getView();
        attr_Dview_t pR_view = this->R.getView();

        for (unsigned d = 0; d < Dim; ++d) {
            Kokkos::parallel_reduce(
                    "get local EField sum", this->getLocalNum(),
                    KOKKOS_LAMBDA(const int i, double& lefsum) { lefsum += fabs(pE_view(i)[d]); },
                    Kokkos::Sum<double>(locEFsum[d]));
        }
        Kokkos::parallel_reduce(
                "check charge", this->getLocalNum(),
                KOKKOS_LAMBDA(const int i, double& qsum) { qsum += pq_view[i]; }, Kokkos::Sum<double>(locQ));
        Kokkos::parallel_reduce(
                "check  positioning", this->getLocalNum(),
                KOKKOS_LAMBDA(const int i, double& check) {
                check += int(L2Norm(pR_view[i]) <= beamRadius);
                },
                Kokkos::Sum<double>(locCheck));

        Kokkos::fence();
        MPI_Allreduce(locEFsum, globEFsum, Dim, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
        MPI_Allreduce(&locQ, &globQ, 1, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
        MPI_Allreduce(&locCheck, &globCheck, 1, MPI_DOUBLE, MPI_SUM, Ippl::getComm());

        for (unsigned d = 0; d < Dim; ++d) {
            avgEF[d] = globEFsum[d] / this->getGlobParticleNum();
        }

        m << "Position Check = " << globCheck << endl;
        m << "globQ = " << globQ << endl;
        m << "globSumEF = " << globEFsum[0] << " " << globEFsum[1] << " " << globEFsum[2] << endl;
        m << "AVG Electric SC Force = " << avgEF << endl;

        return avgEF;
    }

    void applyConstantFocusing(const double focus_fct,
                               const double beamRadius,
                               const VectorD_t avgEF) {
        attr_Dview_t pE_view = this->E.getView();
        attr_Dview_t pR_view = this->R.getView();
        double focusingEnergy = L2Norm(avgEF) * focus_fct / beamRadius;
        Kokkos::parallel_for(
                "Apply Constant Focusing", this->getLocalNum(),
                KOKKOS_LAMBDA(const int i) { pE_view(i) += pR_view(i) * focusingEnergy; });
        Kokkos::fence();
    }


    void dumpBeamStatistics(unsigned int iteration, std::string folder) {
        Inform m("DUMPLangevin");

        // Usefull constants
        const size_t pN_glob = getGlobParticleNum();
        const double c_inv = 1.0 / c_m;
        const double c2_inv = c_inv * c_inv;


        // Views/Mirrors for Particle Attributes
        //attr_view_t  pq_view = this->q.getView();
        attr_Dview_t pR_view = this->R.getView();
        attr_Dview_t pP_view = this->P.getView();
        attr_Dview_t pE_view = this->E.getView();

        //attr_mirror_t  pq_mirror = this->q.getHostMirror();
        //attr_Dmirror_t pE_mirror = this->E.getHostMirror();
        //attr_Dmirror_t pR_mirror = this->R.getHostMirror();

        //Kokkos::deep_copy(pE_mirror, pE_view);
        //Kokkos::deep_copy(pR_mirror, pR_view);
        
        // Views/Mirrors for Fields
        field_Dview_t E_view = this->E_m.getView();

        ////////////////////////////////////////////////////////////
        // Gather E-field in x-direction (via particle attribute) //
        ////////////////////////////////////////////////////////////

        // Compute the Avg E-Field over the particle Attribute
        VectorD_t avgEF_particle;
        double locEFsum[Dim] = {};
        double globEFsum[Dim];

        for (unsigned d = 0; d < Dim; ++d) {
            Kokkos::parallel_reduce(
                "get local EField sum", static_cast<size_t>(this->getLocalNum()),
                KOKKOS_LAMBDA(const int i, double& lefsum) { lefsum += std::fabs(pE_view(i)[d]); },
                Kokkos::Sum<double>(locEFsum[d]));
        }

        Kokkos::fence();
        MPI_Allreduce(locEFsum, globEFsum, Dim, MPI_DOUBLE, MPI_SUM, Ippl::getComm());

        for (unsigned d = 0; d < Dim; ++d)
            avgEF_particle[d] = globEFsum[d] / pN_glob;



        /////////////////////////////////////////////////////////////
        // Gather E-field in x-direction (via Field datastructure) //
        /////////////////////////////////////////////////////////////

        const int nghostE = this->E_m.getNghost();
        double ExAmp;
        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

        double temp = 0.0;
        Kokkos::parallel_reduce("Ex inner product",
                mdrange_type({nghostE, nghostE, nghostE},
                    {E_view.extent(0) - nghostE,
                    E_view.extent(1) - nghostE,
                    E_view.extent(2) - nghostE}),
                KOKKOS_LAMBDA(const size_t i, const size_t j,
                    const size_t k, double& valL)
                {
                double myVal = std::pow(E_view(i, j, k)[0], 2);
                valL += myVal;
                }, Kokkos::Sum<double>(temp));
        Kokkos::fence();
        double globaltemp = 0.0;
        MPI_Reduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());
        double fieldEnergy = globaltemp * this->hr_m[0] * this->hr_m[1] * this->hr_m[2];

        // TODO Remove
        m << fieldEnergy << endl;
        m << iteration << endl;
        m << folder << endl;
        m << pN_glob << endl;

        /////////////////////////////////////////////
        // Gather E-Field amplitude in x-direction //
        /////////////////////////////////////////////

        double tempMax = 0.0;
        Kokkos::parallel_reduce("Ex max norm",
                mdrange_type({nghostE, nghostE, nghostE},
                    {E_view.extent(0) - nghostE,
                    E_view.extent(1) - nghostE,
                    E_view.extent(2) - nghostE}),
                KOKKOS_LAMBDA(const size_t i, const size_t j,
                    const size_t k, double& valL)
                {
                double myVal = std::fabs(E_view(i, j, k)[0]);
                if(myVal > valL) valL = myVal;
                }, Kokkos::Max<double>(tempMax));
        Kokkos::fence();
        ExAmp = 0.0;
        MPI_Reduce(&tempMax, &ExAmp, 1, MPI_DOUBLE, MPI_MAX, 0, Ippl::getComm());

        //////////////////////////////////
        // Calculate Global Temperature //
        //////////////////////////////////

        double locVELsum[Dim]={0.0,0.0,0.0};
        double globVELsum[Dim];
        double avgVEL[Dim];
        double locT[Dim]={0.0,0.0,0.0};
        double globT[Dim];       
        VectorD_t temperature;

        const size_t locNp = static_cast<size_t>(this->getLocalNum());

        for(unsigned d = 0; d<Dim; ++d){
            Kokkos::parallel_reduce("get local velocity sum",
                    locNp, 
                    KOKKOS_LAMBDA(const int k, double& valL){
                    double myVal = pP_view(k)[d];
                    valL += myVal;
                    },                             
                    Kokkos::Sum<double>(locVELsum[d])
                    );
            Kokkos::fence();
        }
        MPI_Allreduce(locVELsum, globVELsum, 3, MPI_DOUBLE, MPI_SUM, Ippl::getComm()); 
        for(unsigned d=0; d<Dim; ++d) avgVEL[d]=globVELsum[d]/pN_glob;

        for(unsigned d = 0; d<Dim; ++d){
            Kokkos::parallel_reduce("get local velocity sum", 
                    locNp,
                    KOKKOS_LAMBDA(const int k, double& valL){
                    double myVal = (pP_view(k)[d]-avgVEL[d])*(pP_view(k)[d]-avgVEL[d]);
                    valL += myVal;
                    },                             
                    Kokkos::Sum<double>(locT[d])
                    );
            Kokkos::fence();
        }

        MPI_Reduce(locT, globT, 3, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());   
        if (Ippl::Comm->rank() == 0) for(unsigned d=0; d<Dim; ++d)    temperature[d]=globT[d]/pN_glob;

        //////////////////////////////////
        //// Calculate Global Lorentz  ///
        //////////////////////////////////

        double lorentzAvg = 0.0;
        double lorentzMax = 0.0;
        double loc_lorentzAvg = 0.0;
        double loc_lorentzMax = 0.0;

        Kokkos::parallel_reduce("Lorentz reductions (avg, max)",
                locNp,
                KOKKOS_LAMBDA(const int k,
                    double& lAvg,
                    double& lMax
                    ){
                double lorentz = 1.0 / sqrt(1.0 - c2_inv*(pP_view(k)[0]*pP_view(k)[0] +
                            pP_view(k)[1]*pP_view(k)[1] +
                            pP_view(k)[2]*pP_view(k)[2]));
                lAvg += lorentz / pN_glob;
                lMax = lorentz > lMax ? lorentz : lMax;
                },
                Kokkos::Sum<double>(loc_lorentzAvg),
                Kokkos::Max<double>(loc_lorentzMax)
                );

        MPI_Allreduce(&loc_lorentzAvg, &lorentzAvg, 1, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
        MPI_Allreduce(&loc_lorentzMax, &lorentzMax, 1, MPI_DOUBLE, MPI_MAX, Ippl::getComm());

        ////////////////////////////////////
        //// Calculate Moments of R and V //
        ////////////////////////////////////

        m << "Moments" << endl;
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
                    part[0] = pR_view(k)[0];
                    part[1] = pP_view(k)[0];
                    part[2] = pR_view(k)[1];
                    part[3] = pP_view(k)[1];
                    part[4] = pR_view(k)[2];
                    part[5] = pP_view(k)[2];


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

        //////////////////////////////////////
        //// Calculate Normalized Emittance //
        //////////////////////////////////////

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
                    double v2 = pP_view(k)[0]*pP_view(k)[0]
                                + pP_view(k)[1]*pP_view(k)[1]
                                + pP_view(k)[2]*pP_view(k)[2];
                    double lorentz = 1.0/(sqrt(1.0-v2*c2_inv));

                    double  part[2 * Dim];
                    part[0] = pR_view(k)[0];
                    part[1] = (pP_view(k)[0]*c_inv)*lorentz;
                    part[2] = pR_view(k)[1];
                    part[3] = (pP_view(k)[1]*c_inv)*lorentz;
                    part[4] = pR_view(k)[2];
                    part[5] = (pP_view(k)[2]*c_inv)*lorentz;


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

        MPI_Allreduce(loc_moment, Nmoment, 2 * Dim * 2 * Dim, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
        MPI_Allreduce(loc_centroid, Ncentroid, 2 * Dim, MPI_DOUBLE, MPI_SUM, Ippl::getComm());

        VectorD_t eps2, fac;
        VectorD_t rsqsum, vsqsum, rvsum;
        VectorD_t rmean, vmean, rrms, vrms, eps, rvrms;
        VectorD_t norm;

        VectorD_t Neps2, Nfac;
        VectorD_t Nrsqsum, Nvsqsum, Nrvsum;
        VectorD_t Nrmean, Nvmean, Nrrms, Nvrms, Neps, Nrvrms;
        VectorD_t Nnorm;

        if (Ippl::Comm->rank() == 0)
        {
            for(unsigned int i = 0 ; i < Dim; i++) {
                rmean(i) = centroid[2 * i] / pN_glob;
                vmean(i) = centroid[(2 * i) + 1] / pN_glob;
                rsqsum(i) = moment[2 * i][2 * i] - pN_glob * rmean(i) * rmean(i);
                vsqsum(i) = moment[(2 * i) + 1][(2 * i) + 1] - pN_glob * vmean(i) * vmean(i);
                if(vsqsum(i) < 0)   vsqsum(i) = 0;
                rvsum(i) = (moment[(2 * i)][(2 * i) + 1] - pN_glob * rmean(i) * vmean(i));

                Nrmean(i) = Ncentroid[2 * i] / pN_glob;
                Nvmean(i) = Ncentroid[(2 * i) + 1] / pN_glob;
                Nrsqsum(i) = Nmoment[2 * i][2 * i] - pN_glob * Nrmean(i) * Nrmean(i);
                Nvsqsum(i) = Nmoment[(2 * i) + 1][(2 * i) + 1] - pN_glob * Nvmean(i) * Nvmean(i);
                if(Nvsqsum(i) < 0)   Nvsqsum(i) = 0;
                Nrvsum(i) = (Nmoment[(2 * i)][(2 * i) + 1] - pN_glob * Nrmean(i) * Nvmean(i));
            }

            // Coefficient wise calculation
            eps2  = (rsqsum * vsqsum - rvsum * rvsum) / (pN_glob * pN_glob);
            rvsum = rvsum / pN_glob;

            Neps2  = (Nrsqsum * Nvsqsum - Nrvsum * Nrvsum) / (pN_glob * pN_glob);
            Nrvsum = Nrvsum / pN_glob;

            for(unsigned int i = 0 ; i < Dim; i++) {
                rrms(i) = sqrt(rsqsum(i) / pN_glob);
                vrms(i) = sqrt(vsqsum(i) / pN_glob);

                eps(i)  =  std::sqrt(std::max(eps2(i), zero));
                double tmpry = rrms(i) * vrms(i);
                fac(i) = (tmpry == 0.0) ? zero : 1.0/tmpry;


                Nrrms(i) = sqrt(rsqsum(i) / pN_glob);
                Nvrms(i) = sqrt(vsqsum(i) / pN_glob);

                Neps(i)  =  std::sqrt(std::max(Neps2(i), zero));
                tmpry = Nrrms(i) * Nvrms(i);
                Nfac(i) = (tmpry == 0.0) ? zero : 1.0/tmpry;
            }
            rvrms = rvsum * fac;
            Nrvrms = Nrvsum * Nfac;
        }

        /////////////////////////////////
        //// Calculate Velocity Bounds //
        /////////////////////////////////

        double vmax_loc[Dim];
        double vmin_loc[Dim];
        double vmax[Dim];
        double vmin[Dim];

        for(unsigned d = 0; d<Dim; ++d){

            Kokkos::parallel_reduce("vel max", 
                    this->getLocalNum(),
                    KOKKOS_LAMBDA(const int i, double& mm){   
                    double tmp_vel = pP_view(i)[d];
                    mm = tmp_vel > mm ? tmp_vel : mm;
                    },                             
                    Kokkos::Max<double>(vmax_loc[d])
                    );

            Kokkos::parallel_reduce("vel min", 
                    this->getLocalNum(),
                    KOKKOS_LAMBDA(const int i, double& mm){   
                    double tmp_vel = pP_view(i)[d];
                    mm = tmp_vel < mm ? tmp_vel : mm;
                    },                             
                    Kokkos::Min<double>(vmin_loc[d])
                    );
        }
        Kokkos::fence();
        MPI_Allreduce(vmax_loc, vmax, Dim , MPI_DOUBLE, MPI_MAX, Ippl::getComm());
        MPI_Allreduce(vmin_loc, vmin, Dim , MPI_DOUBLE, MPI_MIN, Ippl::getComm());

        /////////////////////////////////
        //// Calculate Position Bounds //
        /////////////////////////////////

        double rmax_loc[Dim];
        double rmin_loc[Dim];
        double rmax[Dim];
        double rmin[Dim];

        for(unsigned d = 0; d<Dim; ++d){

            Kokkos::parallel_reduce("rel max", 
                    this->getLocalNum(),
                    KOKKOS_LAMBDA(const int i, double& mm){   
                    double tmp_vel = pR_view(i)[d];
                    mm = tmp_vel > mm ? tmp_vel : mm;
                    },                             
                    Kokkos::Max<double>(rmax_loc[d])
                    );

            Kokkos::parallel_reduce("rel min", 
                    this->getLocalNum(),
                    KOKKOS_LAMBDA(const int i, double& mm){   
                    double tmp_vel = pR_view(i)[d];
                    mm = tmp_vel < mm ? tmp_vel : mm;
                    },                             
                    Kokkos::Min<double>(rmin_loc[d])
                    );
        }
        Kokkos::fence();
        MPI_Allreduce(rmax_loc, rmax, Dim , MPI_DOUBLE, MPI_MAX, Ippl::getComm());
        MPI_Allreduce(rmin_loc, rmin, Dim , MPI_DOUBLE, MPI_MIN, Ippl::getComm());

        ////////////////////////////
        //// Write to output file //
        ////////////////////////////
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


             if(iteration == 0) {
                 csvout  <<          
                     "iteration,"           <<
                     "time,"              << 
                     "T_X,"              <<
                     "rvrms_X,"              <<
                     "eps_X,"              <<
                     "Neps_X"              <<  
                     endl;

                 csvout2 <<  
                     "iteration,"            << 
                     "vmaxX,vmaxY,vmaxZ,"    <<
                     "vminX,vminY,vminZ,"    <<
                     "rmaxX,rmaxY,rmaxZ,"    <<
                     "rminX,rminY,rminZ,"    <<
                     "vrmsX,vrmsY,vrmsZ,"    <<
                     "Tx,Ty,Tz,"             <<
                     "epsX,epsY,epsZ,"       <<
                     "NepsX,NepsY,NepsZ,"       <<
                     "epsX2,epsY2,epsZ2,"    <<
                     "rvrmsX,rvrmsY,rvrmsZ," <<
                     "rrmsX,rrmsY,rrmsZ,"    <<
                     "rmeanX,rmeanY,rmeanZ," <<
                     "vmeanX,vmeanY,vmeanZ," <<
                     "time,"                 <<
                     "Ex_field_energy,"      <<
                     "Ex_max_norm,"          <<
                     "lorentz_avg,"          <<
                     "lorentz_max,"          <<
                     //"avgPotential,"         <<
                     //"avgEfield_x,"          <<
                     //"avgEfield_y,"          <<
                     //"avgEfield_z,"           <<
                     "avgEfield_particle_x,"          <<
                     "avgEfield_particle_y,"          <<
                     "avgEfield_particle_z"           <<
                     endl;
             }     

             csvout<<    
                 iteration       <<","<<
                 iteration*dt_m          <<","<< 
                 temperature[0]  <<","<< 
                 rvrms[0]        <<","<<
                 eps[0]          <<","<<
                 Neps[0]         <<
                 endl;  

             csvout2<<   
                 iteration   <<","<<
                 vmax[0]<<","<<        vmax[1]<<","<<        vmax[2]<<","<<
                 vmin[0]<<","<<        vmin[1]<<","<<        vmin[2]<<","<< 
                 rmax[0]<<","<<        rmax[1]<<","<<        rmax[2]<<","<<
                 rmin[0]<<","<<        rmin[1]<<","<<        rmin[2]<<","<< 
                 vrms        (0)<<","<<vrms        (1)<<","<<vrms        (2)<<","<<
                 temperature (0)<<","<<temperature (1)<<","<<temperature (2)<<","<<
                 eps         (0)<<","<<eps         (1)<<","<<eps         (2)<<","<<
                 Neps        (0)<<","<<Neps        (1)<<","<<Neps        (2)<<","<<
                 eps2        (0)<<","<<eps2        (1)<<","<<eps2        (2)<<","<<
                 rvrms       (0)<<","<<rvrms       (1)<<","<<rvrms       (2)<<","<< 
                 rrms        (0)<<","<<rrms        (1)<<","<<rrms        (2)<<","<<
                 rmean       (0)<<","<<rmean       (1)<<","<<rmean       (2)<<","<<
                 vmean       (0)<<","<<vmean       (1)<<","<<vmean       (2)<<","<<
                 iteration*dt_m <<","<< 
                 fieldEnergy    <<","<< 
                 ExAmp          <<","<< 
                 lorentzAvg     <<","<<
                 lorentzMax     <<","<<
                 //avgPot      <<","<<
                 //avgEF[0]    <<","<<
                 //avgEF[1]    <<","<<
                 //avgEF[2]    <<","<<
                 avgEF_particle[0]  <<","<<
                 avgEF_particle[1]  <<","<<
                 avgEF_particle[2]  <<
                 endl;
         }

        Ippl::Comm->barrier();
    }


private:
    // Particle Charge
    double pCharge_m;
    // Mass of the individual particles
    double pMass_m;
    // Total number of global particles
    double globalNum_m;
    // Timestep (used during timestepping in Diffusion term)
    double dt_m;
    // Speed of light in [cm/s]
    const double c_m = 2.99792458e10;
};

#endif /* LANGEVINPARTICLES_HPP */
