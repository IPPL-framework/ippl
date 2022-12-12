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

const double pi = std::acos(-1.0);

// Test programs have to define this variable for VTK dump purposes
extern const char* TestName;

template<class PLayout>
class ChargedParticlesPinT : public ippl::ParticleBase<PLayout> {
public:
    CxField_t rhoPIF_m;
    Field_t rhoPIC_m;
    VField_t EfieldPIC_m;

    Vector<int, Dim> nr_m;

    ippl::e_dim_tag decomp_m[Dim];

    Vector_t hr_m;
    Vector_t rmin_m;
    Vector_t rmax_m;

    double Q_m;

    std::shared_ptr<Solver_t> solver_mp;
    
    double time_m;


public:
    ParticleAttrib<double>     q; // charge
    typename ippl::ParticleBase<PLayout>::particle_position_type P;  // G(P^(k)_n)
    typename ippl::ParticleBase<PLayout>::particle_position_type E;  // electric field at particle position

    typename ippl::ParticleBase<PLayout>::particle_position_type R0;  // Initial particle positions at t=0
    typename ippl::ParticleBase<PLayout>::particle_position_type P0;  // Initial particle velocities at t=0

    typename ippl::ParticleBase<PLayout>::particle_position_type RprevIter;  // G(R^(k-1)_n)
    typename ippl::ParticleBase<PLayout>::particle_position_type PprevIter;  // G(P^(k-1)_n)

    /*
      This constructor is mandatory for all derived classes from
      ParticleBase as the bunch buffer uses this
    */
    ChargedParticlesPinT(PLayout& pl)
    : ippl::ParticleBase<PLayout>(pl)
    {
        // register the particle attributes
        this->addAttribute(q);
        this->addAttribute(P);
        this->addAttribute(E);
        this->addAttribute(R0);
        this->addAttribute(P0);
        this->addAttribute(RprevIter);
        this->addAttribute(PprevIter);
    }
    
    ChargedParticlesPinT(PLayout& pl,
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


    void dumpLandau(size_type /*totalP*/, const unsigned int& iter) {
       

        double fieldEnergy = 0.0; 
        double ExAmp = 0.0;
        //auto Eview = E.getView();
        //double temp = 0.0;

        //Kokkos::parallel_reduce("Ex energy", this->getLocalNum(),
        //                        KOKKOS_LAMBDA(const int i, double& valL){
        //                            double myVal = Eview(i)[0] * Eview(i)[0];
        //                            valL += myVal;
        //                        }, Kokkos::Sum<double>(temp));

        ////double globaltemp = 0.0;
        //double globaltemp = temp;
        ////MPI_Reduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());
        //double volume = (rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2]);
        //fieldEnergy = globaltemp * volume / totalP ;

        //double tempMax = 0.0;
        //Kokkos::parallel_reduce("Ex max norm", this->getLocalNum(),
        //                        KOKKOS_LAMBDA(const size_t i, double& valL)
        //                        {
        //                            double myVal = std::fabs(Eview(i)[0]);
        //                            if(myVal > valL) valL = myVal;
        //                        }, Kokkos::Max<double>(tempMax));
        ////ExAmp = 0.0;
        //ExAmp = tempMax;
        ////MPI_Reduce(&tempMax, &ExAmp, 1, MPI_DOUBLE, MPI_MAX, 0, Ippl::getComm());

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
        Kokkos::parallel_reduce("Ex energy and Max",
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
                bool shift = (iVec[d] > (N[d]/2));
                kVec[d] = 2 * pi / Len[d] * (iVec[d] - shift * N[d]);
                Dr += kVec[d] * kVec[d];
            }

            Kokkos::complex<double> Ek = {0.0, 0.0}; 
            if(Dr != 0.0) {
                Ek = -(imag * kVec[0] * rhoview(i+nghost,j+nghost,k+nghost) / Dr);
            }
            double myVal = Ek.real() * Ek.real() + Ek.imag() * Ek.imag();

            tlSum += myVal;

            double myValMax = std::sqrt(myVal);

            if(myValMax > tlMax) tlMax = myValMax;

        }, Kokkos::Sum<double>(fieldEnergy), Kokkos::Max<double>(ExAmp));
        

        Kokkos::fence();
        double volume = (rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2]);
        fieldEnergy *= volume;


        std::stringstream fname;
        fname << "data/FieldLandau_";
        fname << Ippl::Comm->rank();
        fname << "_iter_";
        fname << iter;
        fname << ".csv";


        Inform csvout(NULL, fname.str().c_str(), Inform::APPEND, Ippl::Comm->rank());
        csvout.precision(10);
        csvout.setf(std::ios::scientific, std::ios::floatfield);

        //if(time_m == 0.0) {
        //    csvout << "time, Ex_field_energy, Ex_max_norm" << endl;
        //}

        csvout << time_m << " "
               << fieldEnergy << " "
               << ExAmp << endl;
    }


    void dumpEnergy(size_type /*totalP*/, const unsigned int& iter, ParticleAttrib<Vector_t>& Ptemp) {
       

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
                bool shift = (iVec[d] > (N[d]/2));
                kVec[d] = 2 * pi / Len[d] * (iVec[d] - shift * N[d]);
                //kVec[d] = 2 * pi / Len[d] * iVec[d];
                Dr += kVec[d] * kVec[d];
            }

            Kokkos::complex<double> Ek = {0.0, 0.0}; 
            double myVal = 0.0;
            for(size_t d = 0; d < Dim; ++d) {
                if(Dr != 0.0) {
                    Ek = -(imag * kVec[d] * rhoview(i+nghost,j+nghost,k+nghost) / Dr);
                }
                myVal += Ek.real() * Ek.real() + Ek.imag() * Ek.imag();
            }

            //double myVal = rhoview(i,j,k).real() * rhoview(i,j,k).real() + 
            //               rhoview(i,j,k).imag() * rhoview(i,j,k).imag();
            //if(Dr != 0.0) {
            //    myVal /= Dr;
            //}
            //else {
            //    myVal = 0.0;
            //}
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
                                    myVal *= -qView(i);
                                    valL += myVal;
                                }, Kokkos::Sum<double>(temp));

        temp *= 0.5;
        //globaltemp = 0.0;
        double globaltemp = temp;
        //MPI_Reduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());

        kineticEnergy = globaltemp;

        std::stringstream fname;
        fname << "data/Energy_";
        fname << Ippl::Comm->rank();
        fname << "_iter_";
        fname << iter;
        fname << ".csv";


        Inform csvout(NULL, fname.str().c_str(), Inform::APPEND, Ippl::Comm->rank());
        csvout.precision(10);
        csvout.setf(std::ios::scientific, std::ios::floatfield);

        //csvout << "time, Potential energy, Kinetic energy, Total energy" << endl;

        csvout << time_m << " "
               << potentialEnergy << " "
               << kineticEnergy << " "
               << potentialEnergy + kineticEnergy << endl;

    }

    void writeError(double Rerror, double Perror, unsigned int iter) {
        
        if(Ippl::Comm->rank() == 0) {
            std::stringstream fname;
            fname << "data/Error_Vs_Iter.csv";

            Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
            csvout.precision(10);
            csvout.setf(std::ios::scientific, std::ios::floatfield);

            if(iter == 1) {
                csvout << "Iter, Rerror, Perror" << endl;
            }

            csvout << iter << " "
                   << Rerror << " "
                   << Perror << endl;

        }
    
        Ippl::Comm->barrier();

    }

    void checkBounds(ParticleAttrib<Vector_t>& R) {

        auto Rview = R.getView();
        double xMin = 0.0;
        double yMin = 0.0;
        double zMin = 0.0;
        double xMax = 0.0;
        double yMax = 0.0;
        double zMax = 0.0;
        Kokkos::parallel_reduce("Bounds calculation", R.size(),
                                KOKKOS_LAMBDA(const int i, 
                                              double& xlMin, 
                                              double& ylMin, 
                                              double& zlMin, 
                                              double& xlMax, 
                                              double& ylMax, 
                                              double& zlMax){

                                    if(Rview(i)[0] < xlMin) xlMin = Rview(i)[0];
                                    if(Rview(i)[1] < ylMin) ylMin = Rview(i)[1];
                                    if(Rview(i)[2] < zlMin) zlMin = Rview(i)[2];

                                    if(Rview(i)[0] > xlMax) xlMax = Rview(i)[0];
                                    if(Rview(i)[1] > ylMax) ylMax = Rview(i)[1];
                                    if(Rview(i)[2] > zlMax) zlMax = Rview(i)[2];
                                
                                }, Kokkos::Min<double>(xMin), Kokkos::Min<double>(yMin), Kokkos::Min<double>(zMin),
                                   Kokkos::Max<double>(xMax), Kokkos::Max<double>(yMax), Kokkos::Max<double>(zMax));

        Kokkos::fence();

        Vector_t Rmin = {xMin, yMin, zMin};
        Vector_t Rmax = {xMax, yMax, zMax};

        for (unsigned d = 0; d < 3; ++d) {
            if(Rmin[d] < rmin_m[d]) {
                std::cout << "Invalid particles with min. in rank: " << Ippl::Comm->rank() << " Rmin: " << Rmin << std::endl;
            }
            if(Rmax[d] > rmax_m[d]) {
                std::cout << "Invalid particles with max. in rank: " << Ippl::Comm->rank() << " Rmax: " << Rmax << std::endl;
            }
        }
    }

    void LeapFrogPIC(ParticleAttrib<Vector_t>& Rtemp, 
                     ParticleAttrib<Vector_t>& Ptemp, const unsigned int nt, 
                     const double dt, const double& tStartMySlice) {
    
        PLayout& PL = this->getLayout();
        PL.applyBC(Rtemp, PL.getRegionLayout().getDomain());
        //checkBounds(Rtemp);
        rhoPIC_m = 0.0;
        scatter(q, rhoPIC_m, Rtemp);
    
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
            //checkBounds(Rtemp);
    
            //scatter the charge onto the underlying grid
            rhoPIC_m = 0.0;
            scatter(q, rhoPIC_m, Rtemp);
    
    
            rhoPIC_m = rhoPIC_m / (hr_m[0] * hr_m[1] * hr_m[2]);
            rhoPIC_m = rhoPIC_m - (Q_m/((rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2])));
    
            //Field solve
            solver_mp->solve();
    
            // gather E field
            gather(E, EfieldPIC_m, Rtemp);
    
            //kick
            Ptemp = Ptemp - 0.5 * dt * E;
            
            time_m += dt;
        }
    
    }

    void LeapFrogPIF(ParticleAttrib<Vector_t>& Rtemp,
                     ParticleAttrib<Vector_t>& Ptemp, const unsigned int& nt, 
                     const double& dt, const bool& /*isConverged*/, 
                     const double& tStartMySlice, const unsigned int& iter) {
    
        PLayout& PL = this->getLayout();
        PL.applyBC(Rtemp, PL.getRegionLayout().getDomain());
        //checkBounds(Rtemp);
        rhoPIF_m = {0.0, 0.0};
        scatterPIF(q, rhoPIF_m, Rtemp);
    
        rhoPIF_m = rhoPIF_m / ((rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2]));
    
        // Solve for and gather E field
        gatherPIF(E, rhoPIF_m, Rtemp);
    
        time_m = tStartMySlice;

        if((time_m == 0.0)) {
            dumpLandau(this->getLocalNum(), iter);         
            dumpEnergy(this->getLocalNum(), iter, Ptemp);
        }
        for (unsigned int it=0; it<nt; it++) {
    
            // kick
    
            Ptemp = Ptemp - 0.5 * dt * E;
    
            //drift
            Rtemp = Rtemp + dt * Ptemp;
    
            //Apply particle BC
            PL.applyBC(Rtemp, PL.getRegionLayout().getDomain());
            //checkBounds(Rtemp);
    
            //scatter the charge onto the underlying grid
            rhoPIF_m = {0.0, 0.0};
            scatterPIF(q, rhoPIF_m, Rtemp);
    
            rhoPIF_m = rhoPIF_m / ((rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2]));
    
            // Solve for and gather E field
            gatherPIF(E, rhoPIF_m, Rtemp);
    
            //kick
            Ptemp = Ptemp - 0.5 * dt * E;
    
            time_m += dt;
            
            dumpLandau(this->getLocalNum(), iter);         
            dumpEnergy(this->getLocalNum(), iter, Ptemp);         
    
        }
    }

private:
    void setBCAllPeriodic() {

        this->setParticleBC(ippl::BC::PERIODIC);
    }

};
