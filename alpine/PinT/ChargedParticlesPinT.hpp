// ChargedParticlesPinT header file
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

using size_type = ippl::detail::size_type;

template<typename T, unsigned Dim>
using Vector = ippl::Vector<T, Dim>;

template<typename T, unsigned Dim>
using Field = ippl::Field<T, Dim>;

template<typename T>
using ParticleAttrib = ippl::ParticleAttrib<T>;

typedef Vector<double, Dim>  Vector_t;
typedef Field<double, Dim>   Field_t;
typedef Field<Kokkos::complex<double>, Dim>   CxField_t;
typedef Field<Vector_t, Dim> VField_t;
typedef ippl::FFTPeriodicPoissonSolver<Vector_t, double, Dim> Solver_t;

typedef ippl::FFT<ippl::RCTransform, Dim, double> FFT_t;

const double pi = std::acos(-1.0);

// Test programs have to define this variable for VTK dump purposes
extern const char* TestName;

template<class PLayout>
class ChargedParticlesPinT : public ippl::ParticleBase<PLayout> {
public:

    CxField_t rhoPIF_m;
    CxField_t rhoPIFhalf_m;
    Field_t rhoPIFreal_m;
    Field_t Sk_m;
    Field_t rhoPIC_m;
    VField_t EfieldPIC_m;

    Vector<int, Dim> nr_m;
    Vector<int, Dim> nm_m;

    ippl::e_dim_tag decomp_m[Dim];

    Vector_t hr_m;
    Vector_t rmin_m;
    Vector_t rmax_m;

    double Q_m;

    size_type Np_m;
    
    std::shared_ptr<Solver_t> solver_mp;
    std::shared_ptr<FFT_t> fft_mp;
    
    double time_m;

    std::string shapetype_m;

    std::string coarsetype_m;

    int shapedegree_m;

    std::shared_ptr<ippl::FFT<ippl::NUFFTransform, 3, double>> nufftType1Fine_mp,nufftType2Fine_mp,
                                                               nufftType1Coarse_mp,nufftType2Coarse_mp;

public:
    ParticleAttrib<double>     q; // charge
    typename ippl::ParticleBase<PLayout>::particle_position_type P;  // G(P^(k)_n)
    typename ippl::ParticleBase<PLayout>::particle_position_type E;  // electric field at particle position

    typename ippl::ParticleBase<PLayout>::particle_position_type R0;  // Initial particle positions at t=0
    typename ippl::ParticleBase<PLayout>::particle_position_type P0;  // Initial particle velocities at t=0

    typename ippl::ParticleBase<PLayout>::particle_position_type RprevIter;  // G(R^(k-1)_n)
    typename ippl::ParticleBase<PLayout>::particle_position_type PprevIter;  // G(P^(k-1)_n)

    ///*
    //  This constructor is mandatory for all derived classes from
    //  ParticleBase as the bunch buffer uses this
    //*/
    //ChargedParticlesPinT(PLayout& pl)
    //: ippl::ParticleBase<PLayout>(pl)
    //{
    //    // register the particle attributes
    //    this->addAttribute(q);
    //    this->addAttribute(P);
    //    this->addAttribute(E);
    //    this->addAttribute(R0);
    //    this->addAttribute(P0);
    //    this->addAttribute(RprevIter);
    //    this->addAttribute(PprevIter);
    //    //this->addAttribute(Rfine);
    //    //this->addAttribute(Pfine);
    //}
    
    ChargedParticlesPinT(PLayout& pl,
                     Vector_t hr,
                     Vector_t rmin,
                     Vector_t rmax,
                     ippl::e_dim_tag decomp[Dim],
                     double Q,
                     size_type Np)
    : ippl::ParticleBase<PLayout>(pl)
    , hr_m(hr)
    , rmin_m(rmin)
    , rmax_m(rmax)
    , Q_m(Q)
    , Np_m(Np)
    {
        // register the particle attributes
        this->addAttribute(q);
        this->addAttribute(P);
        this->addAttribute(E);
        this->addAttribute(R0);
        this->addAttribute(P0);
        this->addAttribute(RprevIter);
        this->addAttribute(PprevIter);
        setupBCs();
        for (unsigned int i = 0; i < Dim; i++)
            decomp_m[i]=decomp[i];
    }

    ~ChargedParticlesPinT(){ }

    void setupBCs() {
        setBCAllPeriodic();
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

        solver_mp->setRhs(rhoPIC_m);

        solver_mp->setLhs(EfieldPIC_m);
    }


    void initNUFFTs(FieldLayout_t& FLPIF, double& coarseTol,
                    double& fineTol) {
        
        ippl::ParameterList fftCoarseParams,fftFineParams;

        fftFineParams.add("gpu_method", 1);
        fftFineParams.add("gpu_sort", 0);
        fftFineParams.add("gpu_kerevalmeth", 1);
        fftFineParams.add("tolerance", fineTol);

        fftCoarseParams.add("gpu_method", 1);
        fftCoarseParams.add("gpu_sort", 0);
        fftCoarseParams.add("gpu_kerevalmeth", 1);
        fftCoarseParams.add("tolerance", coarseTol);

        fftFineParams.add("use_cufinufft_defaults", false);
        fftCoarseParams.add("use_cufinufft_defaults", false);
        
        nufftType1Fine_mp = std::make_shared<ippl::FFT<ippl::NUFFTransform, 3, double>>(FLPIF, this->getLocalNum(), 1, fftFineParams);
        nufftType2Fine_mp = std::make_shared<ippl::FFT<ippl::NUFFTransform, 3, double>>(FLPIF, this->getLocalNum(), 2, fftFineParams);

        nufftType1Coarse_mp = std::make_shared<ippl::FFT<ippl::NUFFTransform, 3, double>>(FLPIF, this->getLocalNum(), 1, fftCoarseParams);
        nufftType2Coarse_mp = std::make_shared<ippl::FFT<ippl::NUFFTransform, 3, double>>(FLPIF, this->getLocalNum(), 2, fftCoarseParams);
    }
    
    void dumpFieldEnergy(const unsigned int& nc, const unsigned int& iter, int rankTime, int rankSpace) {
       
        double fieldEnergy = 0.0; 
        double EzAmp = 0.0;

        auto rhoview = rhoPIF_m.getView();
        const int nghost = rhoPIF_m.getNghost();
        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<Dim>>;
      
        const FieldLayout_t& layout = rhoPIF_m.getLayout(); 
        const Mesh_t& mesh = rhoPIF_m.get_mesh();
        const Vector<double, Dim>& dx = mesh.getMeshSpacing();
        const auto& domain = layout.getDomain();
        Vector<double, Dim> Len;
        Vector<int, Dim> N;

        for (unsigned d=0; d < Dim; ++d) {
            N[d] = domain[d].length();
            Len[d] = dx[d] * N[d];
        }


        Kokkos::complex<double> imag = {0.0, 1.0};
        double pi = std::acos(-1.0);
        Kokkos::parallel_reduce("Ez energy and Max",
                              mdrange_type({0, 0, 0},
                                           {N[0],
                                            N[1],
                                            N[2]}),
                              KOKKOS_LAMBDA(const int i,
                                            const int j,
                                            const int k,
                                            double& tlSum,
                                            double& tlMax)
        {
        
            Vector<int, 3> iVec = {i, j, k};
            Vector<double, 3> kVec;
            double Dr = 0.0;
            for(size_t d = 0; d < Dim; ++d) {
                kVec[d] = 2 * pi / Len[d] * (iVec[d] - (N[d] / 2));
                Dr += kVec[d] * kVec[d];
            }

            Kokkos::complex<double> Ek = {0.0, 0.0}; 
            bool isNotZero = (Dr != 0.0);
            double factor = isNotZero * (1.0 / (Dr + ((!isNotZero) * 1.0))); 
            Ek = -(imag * kVec[2] * rhoview(i+nghost,j+nghost,k+nghost) * factor);
            double myVal = Ek.real() * Ek.real() + Ek.imag() * Ek.imag();

            tlSum += myVal;

            double myValMax = std::sqrt(myVal);

            if(myValMax > tlMax) tlMax = myValMax;

        }, Kokkos::Sum<double>(fieldEnergy), Kokkos::Max<double>(EzAmp));
        

        Kokkos::fence();
        double volume = (rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2]);
        fieldEnergy *= volume;


        if(rankSpace == 0) {
            std::stringstream fname;
            fname << "data/FieldBumponTail_rank_";
            fname << rankTime;
            fname << "_nc_";
            fname << nc;
            fname << "_iter_";
            fname << iter;
            fname << ".csv";


            Inform csvout(NULL, fname.str().c_str(), Inform::APPEND, Ippl::Comm->rank());
            csvout.precision(10);
            csvout.setf(std::ios::scientific, std::ios::floatfield);


            csvout << time_m << " "
                   << fieldEnergy << " "
                   << EzAmp << endl;
        }
    }

    void dumpEnergy(const unsigned int& nc, const unsigned int& iter, ParticleAttrib<Vector_t>& Ptemp,
                    int rankTime, int rankSpace, const MPI_Comm& spaceComm = MPI_COMM_WORLD) {

        double potentialEnergy, kineticEnergy;
        double temp = 0.0;

        auto rhoview = rhoPIF_m.getView();
        const int nghost = rhoPIF_m.getNghost();
        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<Dim>>;
      
        const FieldLayout_t& layout = rhoPIF_m.getLayout(); 
        const Mesh_t& mesh = rhoPIF_m.get_mesh();
        const Vector<double, Dim>& dx = mesh.getMeshSpacing();
        const auto& domain = layout.getDomain();
        Vector<double, Dim> Len;
        Vector<int, Dim> N;

        for (unsigned d=0; d < Dim; ++d) {
            N[d] = domain[d].length();
            Len[d] = dx[d] * N[d];
        }


        Kokkos::complex<double> imag = {0.0, 1.0};
        double pi = std::acos(-1.0);
        Kokkos::parallel_reduce("Potential energy",
                              mdrange_type({0, 0, 0},
                                           {N[0],
                                            N[1],
                                            N[2]}),
                              KOKKOS_LAMBDA(const int i,
                                            const int j,
                                            const int k,
                                            double& valL)
        {
        
            Vector<int, 3> iVec = {i, j, k};
            Vector<double, 3> kVec;
            double Dr = 0.0;
            for(size_t d = 0; d < Dim; ++d) {
                kVec[d] = 2 * pi / Len[d] * (iVec[d] - (N[d] / 2));
                Dr += kVec[d] * kVec[d];
            }

            Kokkos::complex<double> Ek = {0.0, 0.0}; 
            double myVal = 0.0;
            auto rho = rhoview(i+nghost,j+nghost,k+nghost);
            for(size_t d = 0; d < Dim; ++d) {
                bool isNotZero = (Dr != 0.0);
                double factor = isNotZero * (1.0 / (Dr + ((!isNotZero) * 1.0))); 
                Ek = -(imag * kVec[d] * rho * factor);
                myVal += Ek.real() * Ek.real() + Ek.imag() * Ek.imag();
            }

            valL += myVal;

        }, Kokkos::Sum<double>(temp));

        double volume = (rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2]);
        potentialEnergy = 0.5 * temp * volume;

        auto Pview = Ptemp.getView();
        auto qView = q.getView();

        temp = 0.0;

        Kokkos::parallel_reduce("Kinetic Energy", this->getLocalNum(),
                                KOKKOS_LAMBDA(const int i, double& valL){
                                    double myVal = dot(Pview(i), Pview(i)).apply();
                                    myVal *= -qView(i); //q/(q/m) where q/m=-1
                                    valL += myVal;
                                }, Kokkos::Sum<double>(temp));

        temp *= 0.5;
        double globaltemp = 0.0;
        MPI_Allreduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, spaceComm);

        kineticEnergy = globaltemp;

        auto rhoPIFhalfview = rhoPIFhalf_m.getView();
        const int nghostHalf = rhoPIFhalf_m.getNghost();
      
        const FieldLayout_t& layoutHalf = rhoPIFhalf_m.getLayout(); 
        const auto& domainHalf = layoutHalf.getDomain();

        Vector<int, Dim> Nhalf;
        for (unsigned d=0; d < Dim; ++d) {
            Nhalf[d] = domainHalf[d].length();
        }

        //Heffte needs FFTshifted field whereas the field from cuFINUFFT
        //is not shifted. Hence, here we do the shift. 
        Kokkos::parallel_for("Transfer complex rho to half domain",
                              mdrange_type({0, 0, 0},
                                           {Nhalf[0],
                                            Nhalf[1],
                                            Nhalf[2]}),
                              KOKKOS_LAMBDA(const int i,
                                            const int j,
                                            const int k)
        {
            Vector<int, 3> iVec = {i, j, k};
            int shift;
            for(size_t d = 0; d < Dim; ++d) {
                bool isLessThanHalf = (iVec[d] < (Nhalf[d]/2));
                shift = ((int)isLessThanHalf * 2) - 1;
                iVec[d] = (iVec[d] + shift * (Nhalf[d]/2)) + nghostHalf;
            }
            rhoPIFhalfview(Nhalf[0]-1-i+nghostHalf, iVec[1], iVec[2]) = 
            rhoview(i+nghostHalf,j+nghostHalf,k+nghostHalf);
        });


        rhoPIFreal_m = 0.0;
        fft_mp->transform(-1, rhoPIFreal_m, rhoPIFhalf_m);

        rhoPIFreal_m = (1.0/(N[0]*N[1]*N[2])) * volume * rhoPIFreal_m;
        auto rhoPIFrealview = rhoPIFreal_m.getView();
        temp = 0.0;
        Kokkos::parallel_reduce("Rho real sum",
                              mdrange_type({0, 0, 0},
                                           {N[0],
                                            N[1],
                                            N[2]}),
                              KOKKOS_LAMBDA(const int i,
                                            const int j,
                                            const int k,
                                            double& valL)
        {
            valL += rhoPIFrealview(i+nghost, j+nghost, k+nghost);
        }, Kokkos::Sum<double>(temp));

        double chargeTotal = temp;

        Vector_t totalMomentum = 0.0;
        
        for(size_t d = 0; d < Dim; ++d) {
             double tempD = 0.0;
             Kokkos::parallel_reduce("Total Momentum", this->getLocalNum(),
                                KOKKOS_LAMBDA(const int i, double& valL){
                                    valL  += (-qView(i)) * Pview(i)[d];
                                }, Kokkos::Sum<double>(tempD));
             totalMomentum[d] = tempD;
        }
        
        Vector_t globalMom;

        double magMomentum = 0.0;
        for(size_t d = 0; d < Dim; ++d) {
            MPI_Allreduce(&totalMomentum[d], &globalMom[d], 1, MPI_DOUBLE, MPI_SUM, spaceComm);
            magMomentum += globalMom[d] * globalMom[d];
        }

        magMomentum  = std::sqrt(magMomentum);

        if(rankSpace == 0) {
            std::stringstream fname;
            fname << "data/Energy_rank_";
            fname << rankTime;
            fname << "_nc_";
            fname << nc;
            fname << "_iter_";
            fname << iter;
            fname << ".csv";


            Inform csvout(NULL, fname.str().c_str(), Inform::APPEND, Ippl::Comm->rank());
            csvout.precision(17);
            csvout.setf(std::ios::scientific, std::ios::floatfield);

            //csvout << "time, Potential energy, Kinetic energy, Total energy" << endl;

            csvout << time_m << " "
                   << potentialEnergy << " "
                   << kineticEnergy << " "
                   << potentialEnergy + kineticEnergy << " " 
                   << chargeTotal << " " 
                   << magMomentum << endl;
        }

    }
    
    void writelocalError(double Rerror, double Perror, unsigned int nc, unsigned int iter, int rankTime, int rankSpace) {
        
        if(rankSpace == 0) {
            std::stringstream fname;
            fname << "data/localError_rank_";
            fname << rankTime;
            fname << "_nc_";
            fname << nc;
            fname << ".csv";

            Inform csvout(NULL, fname.str().c_str(), Inform::APPEND, Ippl::Comm->rank());
            csvout.precision(17);
            csvout.setf(std::ios::scientific, std::ios::floatfield);

            if(iter == 1) {
                csvout << "Iter, Rerror, Perror" << endl;
            }

            csvout << iter << " "
                   << Rerror << " "
                   << Perror << endl;
        }

    }

    void initializeShapeFunctionPIF() {

        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        auto Skview = Sk_m.getView();
        auto N = nm_m;
        const int nghost = Sk_m.getNghost();
        const Mesh_t& mesh = rhoPIF_m.get_mesh();
        const Vector_t& dx = mesh.getMeshSpacing();
        const Vector_t& Len = rmax_m - rmin_m;
        const double pi = std::acos(-1.0);
        int order = shapedegree_m + 1;
        
        if(shapetype_m == "Gaussian") {

            throw IpplException("initializeShapeFunctionPIF",
                                "Gaussian shape function not implemented yet");

        }
        else if(shapetype_m == "B-spline") {

            Kokkos::parallel_for("B-spline shape functions",
                                mdrange_type({0, 0, 0},
                                             {N[0], N[1], N[2]}),
                                KOKKOS_LAMBDA(const int i,
                                              const int j,
                                              const int k)
            {
                
                Vector<int, 3> iVec = {i, j, k};
                Vector<double, 3> kVec;
                double Sk = 1.0;
                for(size_t d = 0; d < Dim; ++d) {
                    kVec[d] = 2 * pi / Len[d] * (iVec[d] - (N[d] / 2));
                    double khbytwo = kVec[d] * dx[d] / 2;
                    bool isNotZero = (khbytwo != 0.0);
                    double factor = (1.0 / (khbytwo + ((!isNotZero) * 1.0)));
                    double arg = isNotZero * (Kokkos::sin(khbytwo) * factor) + 
                                 (!isNotZero) * 1.0;
                    //Fourier transform of CIC
                    Sk *= std::pow(arg, order);
                }
                    Skview(i+nghost, j+nghost, k+nghost) = Sk;
            });
        }
        else {
            throw IpplException("initializeShapeFunctionPIF",
                                "Unrecognized shape function type");
        }

    }

    void LeapFrogPIC(ParticleAttrib<Vector_t>& Rtemp, 
                     ParticleAttrib<Vector_t>& Ptemp, const unsigned int nt, 
                     const double dt, const double& tStartMySlice, MPI_Comm& spaceComm) {
    
        static IpplTimings::TimerRef fieldSolvePIC = IpplTimings::getTimer("fieldSolvePIC");
        PLayout& PL = this->getLayout();
        rhoPIC_m = 0.0;
        scatter(q, rhoPIC_m, Rtemp, spaceComm);
    
        rhoPIC_m = rhoPIC_m / (hr_m[0] * hr_m[1] * hr_m[2]);
        rhoPIC_m = rhoPIC_m - (Q_m/((rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2])));
    
        //Field solve
        solver_mp->solve();
    
        // gather E field
        gather(E, EfieldPIC_m, Rtemp);
    
        time_m = tStartMySlice;

        for (unsigned int it=0; it<nt; it++) {
            
            // kick
            Ptemp = Ptemp - 0.5 * dt * E;
    
            //drift
            Rtemp = Rtemp + dt * Ptemp;
    
            //Apply particle BC
            PL.applyBC(Rtemp, PL.getRegionLayout().getDomain());
    
            //scatter the charge onto the underlying grid
            rhoPIC_m = 0.0;
            scatter(q, rhoPIC_m, Rtemp, spaceComm);
    
            rhoPIC_m = rhoPIC_m / (hr_m[0] * hr_m[1] * hr_m[2]);
            rhoPIC_m = rhoPIC_m - (Q_m/((rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2])));
    
            //Field solve
            IpplTimings::startTimer(fieldSolvePIC);
            solver_mp->solve();
            IpplTimings::stopTimer(fieldSolvePIC);
    
            // gather E field
            gather(E, EfieldPIC_m, Rtemp);
    
            //kick
            Ptemp = Ptemp - 0.5 * dt * E;
            
            time_m += dt;
        }
    
    }

    void BorisPIC(ParticleAttrib<Vector_t>& Rtemp, ParticleAttrib<Vector_t>& Ptemp, const unsigned int nt, 
                  const double dt, const double& tStartMySlice, const double& Bext, MPI_Comm& spaceComm) {
    
        static IpplTimings::TimerRef fieldSolvePIC = IpplTimings::getTimer("fieldSolvePIC");
        PLayout& PL = this->getLayout();
        rhoPIC_m = 0.0;
        scatter(q, rhoPIC_m, Rtemp, spaceComm);
    
        rhoPIC_m = rhoPIC_m / (hr_m[0] * hr_m[1] * hr_m[2]);
        rhoPIC_m = rhoPIC_m - (Q_m/((rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2])));
    
        //Field solve
        solver_mp->solve();
    
        // gather E field
        gather(E, EfieldPIC_m, Rtemp);
    
        time_m = tStartMySlice;

        double alpha = -0.5 * dt;
        double DrInv = 1.0 / (1 + (std::pow((alpha * Bext), 2)));
        Vector_t rmax = rmax_m;

        for (unsigned int it=0; it<nt; it++) {
            
            // Staggered Leap frog or Boris algorithm as per 
            // https://www.sciencedirect.com/science/article/pii/S2590055219300526
            // eqns 4(a)-4(c). Note we don't use the Boris trick here and do
            // the analytical matrix inversion which is not complex in this case.
            // Here, we assume a constant charge-to-mass ratio of -1 for
            // all the particles hence eliminating the need to store mass as
            // an attribute
            // kick
            auto Rview = Rtemp.getView();
            auto Pview = Ptemp.getView();
            auto Eview = E.getView();
            double V0 = 30*rmax[2];
            Kokkos::parallel_for("Kick1", this->getLocalNum(),
                                  KOKKOS_LAMBDA(const size_t j){
                double Eext_x = -(Rview(j)[0] - 0.5*rmax[0]) * (V0/(2*std::pow(rmax[2],2)));
                double Eext_y = -(Rview(j)[1] - 0.5*rmax[1]) * (V0/(2*std::pow(rmax[2],2)));
                double Eext_z =  (Rview(j)[2] - 0.5*rmax[2]) * (V0/(std::pow(rmax[2],2)));

                Eext_x += Eview(j)[0];
                Eext_y += Eview(j)[1];
                Eext_z += Eview(j)[2];
                
                Pview(j)[0] += alpha * (Eext_x  + Pview(j)[1] * Bext);
                Pview(j)[1] += alpha * (Eext_y  - Pview(j)[0] * Bext);
                Pview(j)[2] += alpha * Eext_z;
            });
    
            //drift
            Rtemp = Rtemp + dt * Ptemp;
    
            //Apply particle BC
            PL.applyBC(Rtemp, PL.getRegionLayout().getDomain());
    
            //scatter the charge onto the underlying grid
            rhoPIC_m = 0.0;
            scatter(q, rhoPIC_m, Rtemp, spaceComm);
    
            rhoPIC_m = rhoPIC_m / (hr_m[0] * hr_m[1] * hr_m[2]);
            rhoPIC_m = rhoPIC_m - (Q_m/((rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2])));
    
            //Field solve
            IpplTimings::startTimer(fieldSolvePIC);
            solver_mp->solve();
            IpplTimings::stopTimer(fieldSolvePIC);
    
            // gather E field
            gather(E, EfieldPIC_m, Rtemp);
    
            //kick
            auto R2view = Rtemp.getView();
            auto P2view = Ptemp.getView();
            auto E2view = E.getView();
            Kokkos::parallel_for("Kick2", this->getLocalNum(),
                                  KOKKOS_LAMBDA(const size_t j){
                double Eext_x = -(R2view(j)[0] - 0.5*rmax[0]) * (V0/(2*std::pow(rmax[2],2)));
                double Eext_y = -(R2view(j)[1] - 0.5*rmax[1]) * (V0/(2*std::pow(rmax[2],2)));
                double Eext_z =  (R2view(j)[2] - 0.5*rmax[2]) * (V0/(std::pow(rmax[2],2)));
         
                Eext_x += E2view(j)[0];
                Eext_y += E2view(j)[1];
                Eext_z += E2view(j)[2];
                
                P2view(j)[0]  = DrInv * ( P2view(j)[0] + alpha * (Eext_x 
                                + P2view(j)[1] * Bext + alpha * Bext * Eext_y) );
                P2view(j)[1]  = DrInv * ( P2view(j)[1] + alpha * (Eext_y 
                                - P2view(j)[0] * Bext - alpha * Bext * Eext_x) );
                P2view(j)[2] += alpha * Eext_z;
            });
            
            time_m += dt;
        }
    
    }

    void LeapFrogPIF(ParticleAttrib<Vector_t>& Rtemp,
                     ParticleAttrib<Vector_t>& Ptemp, const unsigned int& nt, 
                     const double& dt, const double& tStartMySlice, const unsigned& /*nc*/, 
                     const unsigned int& /*iter*/, int /*rankTime*/, int /*rankSpace*/,
                     const std::string& propagator, MPI_Comm& spaceComm) {
    
        static IpplTimings::TimerRef dumpData = IpplTimings::getTimer("dumpData");
        PLayout& PL = this->getLayout();
        rhoPIF_m = {0.0, 0.0};
        if(propagator == "Coarse") {
            scatterPIFNUFFT(q, rhoPIF_m, Sk_m, Rtemp, nufftType1Coarse_mp.get(), spaceComm);
        }
        else if(propagator == "Fine") {
            scatterPIFNUFFT(q, rhoPIF_m, Sk_m, Rtemp, nufftType1Fine_mp.get(), spaceComm);
        }
    
        rhoPIF_m = rhoPIF_m / ((rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2]));
    
        // Solve for and gather E field
        if(propagator == "Coarse") {
            gatherPIFNUFFT(E, rhoPIF_m, Sk_m, Rtemp, nufftType2Coarse_mp.get(), q);
        }
        else if(propagator == "Fine") {
            gatherPIFNUFFT(E, rhoPIF_m, Sk_m, Rtemp, nufftType2Fine_mp.get(), q);
        }

        //Reset the value of q here as we used it as a temporary object in gather to 
        //save memory
        q = Q_m / Np_m;
    
        time_m = tStartMySlice;

        if((time_m == 0.0) && (propagator == "Fine")) {
            IpplTimings::startTimer(dumpData);
            //dumpFieldEnergy(nc, iter, rankTime, rankSpace);         
            //dumpEnergy(nc, iter, Ptemp, rankTime, rankSpace, spaceComm);
            IpplTimings::stopTimer(dumpData);
        }
        for (unsigned int it=0; it<nt; it++) {
    
            // kick
            Ptemp = Ptemp - 0.5 * dt * E;
    
            //drift
            Rtemp = Rtemp + dt * Ptemp;
    
            //Apply particle BC
            PL.applyBC(Rtemp, PL.getRegionLayout().getDomain());
    
            //scatter the charge onto the underlying grid
            rhoPIF_m = {0.0, 0.0};
            if(propagator == "Coarse") {
                scatterPIFNUFFT(q, rhoPIF_m, Sk_m, Rtemp, nufftType1Coarse_mp.get(), spaceComm);
            }
            else if(propagator == "Fine") {
                scatterPIFNUFFT(q, rhoPIF_m, Sk_m, Rtemp, nufftType1Fine_mp.get(), spaceComm);
            }
    
            rhoPIF_m = rhoPIF_m / ((rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2]));
    
            // Solve for and gather E field
            if(propagator == "Coarse") {
                gatherPIFNUFFT(E, rhoPIF_m, Sk_m, Rtemp, nufftType2Coarse_mp.get(), q);
            }
            else if(propagator == "Fine") {
                gatherPIFNUFFT(E, rhoPIF_m, Sk_m, Rtemp, nufftType2Fine_mp.get(), q);
            }

            q = Q_m / Np_m;

            //kick
            Ptemp = Ptemp - 0.5 * dt * E;
    
            time_m += dt;
            
            if(propagator == "Fine") {
                IpplTimings::startTimer(dumpData);
                //dumpFieldEnergy(nc, iter, rankTime, rankSpace);         
                //dumpEnergy(nc, iter, Ptemp, rankTime, rankSpace, spaceComm);         
                IpplTimings::stopTimer(dumpData);
            }
        }
    }


    void BorisPIF(ParticleAttrib<Vector_t>& Rtemp,
                     ParticleAttrib<Vector_t>& Ptemp, const unsigned int& nt, 
                     const double& dt, const double& tStartMySlice, const unsigned& /*nc*/, 
                     const unsigned int& /*iter*/, const double& Bext,
                     int /*rankTime*/, int /*rankSpace*/,
                     const std::string& propagator, MPI_Comm& spaceComm) {
    
        static IpplTimings::TimerRef dumpData = IpplTimings::getTimer("dumpData");
        PLayout& PL = this->getLayout();
        rhoPIF_m = {0.0, 0.0};
        if(propagator == "Coarse") {
            scatterPIFNUFFT(q, rhoPIF_m, Sk_m, Rtemp, nufftType1Coarse_mp.get(), spaceComm);
        }
        else if(propagator == "Fine") {
            scatterPIFNUFFT(q, rhoPIF_m, Sk_m, Rtemp, nufftType1Fine_mp.get(), spaceComm);
        }
    
        rhoPIF_m = rhoPIF_m / ((rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2]));
    
        // Solve for and gather E field
        if(propagator == "Coarse") {
            gatherPIFNUFFT(E, rhoPIF_m, Sk_m, Rtemp, nufftType2Coarse_mp.get(), q);
        }
        else if(propagator == "Fine") {
            gatherPIFNUFFT(E, rhoPIF_m, Sk_m, Rtemp, nufftType2Fine_mp.get(), q);
        }

        q = Q_m / Np_m;

        time_m = tStartMySlice;

        if((time_m == 0.0) && (propagator == "Fine")) {
            IpplTimings::startTimer(dumpData);
            //dumpEnergy(nc, iter, Ptemp, rankTime, rankSpace, spaceComm);
            IpplTimings::stopTimer(dumpData);
        }
        double alpha = -0.5 * dt;
        double DrInv = 1.0 / (1 + (std::pow((alpha * Bext), 2)));
        Vector_t rmax = rmax_m;
        for (unsigned int it=0; it<nt; it++) {
    
            // Staggered Leap frog or Boris algorithm as per 
            // https://www.sciencedirect.com/science/article/pii/S2590055219300526
            // eqns 4(a)-4(c). Note we don't use the Boris trick here and do
            // the analytical matrix inversion which is not complex in this case.
            // Here, we assume a constant charge-to-mass ratio of -1 for
            // all the particles hence eliminating the need to store mass as
            // an attribute
            // kick
            auto Rview = Rtemp.getView();
            auto Pview = Ptemp.getView();
            auto Eview = E.getView();
            double V0 = 30*rmax[2];
            Kokkos::parallel_for("Kick1", this->getLocalNum(),
                                  KOKKOS_LAMBDA(const size_t j){
                double Eext_x = -(Rview(j)[0] - 0.5*rmax[0]) * (V0/(2*std::pow(rmax[2],2)));
                double Eext_y = -(Rview(j)[1] - 0.5*rmax[1]) * (V0/(2*std::pow(rmax[2],2)));
                double Eext_z =  (Rview(j)[2] - 0.5*rmax[2]) * (V0/(std::pow(rmax[2],2)));

                Eext_x += Eview(j)[0];
                Eext_y += Eview(j)[1];
                Eext_z += Eview(j)[2];
                
                Pview(j)[0] += alpha * (Eext_x  + Pview(j)[1] * Bext);
                Pview(j)[1] += alpha * (Eext_y  - Pview(j)[0] * Bext);
                Pview(j)[2] += alpha * Eext_z;
            });
    
            //drift
            Rtemp = Rtemp + dt * Ptemp;
    
            //Apply particle BC
            PL.applyBC(Rtemp, PL.getRegionLayout().getDomain());
    
            //scatter the charge onto the underlying grid
            rhoPIF_m = {0.0, 0.0};
            if(propagator == "Coarse") {
                scatterPIFNUFFT(q, rhoPIF_m, Sk_m, Rtemp, nufftType1Coarse_mp.get(), spaceComm);
            }
            else if(propagator == "Fine") {
                scatterPIFNUFFT(q, rhoPIF_m, Sk_m, Rtemp, nufftType1Fine_mp.get(), spaceComm);
            }
    
            rhoPIF_m = rhoPIF_m / ((rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2]));
    
            // Solve for and gather E field
            if(propagator == "Coarse") {
                gatherPIFNUFFT(E, rhoPIF_m, Sk_m, Rtemp, nufftType2Coarse_mp.get(), q);
            }
            else if(propagator == "Fine") {
                gatherPIFNUFFT(E, rhoPIF_m, Sk_m, Rtemp, nufftType2Fine_mp.get(), q);
            }
    
            q = Q_m / Np_m;
            //kick
            auto R2view = Rtemp.getView();
            auto P2view = Ptemp.getView();
            auto E2view = E.getView();
            Kokkos::parallel_for("Kick2", this->getLocalNum(),
                                  KOKKOS_LAMBDA(const size_t j){
                double Eext_x = -(R2view(j)[0] - 0.5*rmax[0]) * (V0/(2*std::pow(rmax[2],2)));
                double Eext_y = -(R2view(j)[1] - 0.5*rmax[1]) * (V0/(2*std::pow(rmax[2],2)));
                double Eext_z =  (R2view(j)[2] - 0.5*rmax[2]) * (V0/(std::pow(rmax[2],2)));

                Eext_x += E2view(j)[0];
                Eext_y += E2view(j)[1];
                Eext_z += E2view(j)[2];
                
                P2view(j)[0]  = DrInv * ( P2view(j)[0] + alpha * (Eext_x 
                                + P2view(j)[1] * Bext + alpha * Bext * Eext_y) );
                P2view(j)[1]  = DrInv * ( P2view(j)[1] + alpha * (Eext_y 
                                - P2view(j)[0] * Bext - alpha * Bext * Eext_x) );
                P2view(j)[2] += alpha * Eext_z;
            });

            time_m += dt;
            
            if(propagator == "Fine") {
                IpplTimings::startTimer(dumpData);
                //dumpEnergy(nc, iter, Ptemp, rankTime, rankSpace, spaceComm);
                IpplTimings::stopTimer(dumpData);
            }
        }
    }

private:
    void setBCAllPeriodic() {
        this->setParticleBC(ippl::BC::PERIODIC);
    }

};
