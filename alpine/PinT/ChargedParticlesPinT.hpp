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
    Field_t Sk_m;
    Field_t rhoPIC_m;
    VField_t EfieldPIC_m;
    //VField_t EfieldPICprevIter_m;

    Vector<int, Dim> nr_m;
    Vector<int, Dim> nm_m;

    ippl::e_dim_tag decomp_m[Dim];

    Vector_t hr_m;
    Vector_t rmin_m;
    Vector_t rmax_m;

    double Q_m;

    std::shared_ptr<Solver_t> solver_mp;
    
    double time_m;

    std::string shapetype_m;

    int shapedegree_m;

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

     void dumpLandauPIC() {

        const int nghostE = EfieldPIC_m.getNghost();
        auto Eview = EfieldPIC_m.getView();
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
        double globaltemp = temp;
        //MPI_Reduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());
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
        ExAmp = tempMax;
        //MPI_Reduce(&tempMax, &ExAmp, 1, MPI_DOUBLE, MPI_MAX, 0, Ippl::getComm());


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
        
        //Ippl::Comm->barrier();
     }


    
    void dumpLandau(const unsigned int& iter) {
       

        double fieldEnergy = 0.0; 
        double ExAmp = 0.0;

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
            bool isNotZero = (Dr != 0.0);
            double factor = isNotZero * (1.0 / (Dr + ((!isNotZero) * 1.0))); 
            Ek = -(imag * kVec[0] * rhoview(i+nghost,j+nghost,k+nghost) * factor);
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


        csvout << time_m << " "
               << fieldEnergy << " "
               << ExAmp << endl;
    }

    void dumpBumponTail(const unsigned int& iter) {
       

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
                bool shift = (iVec[d] > (N[d]/2));
                kVec[d] = 2 * pi / Len[d] * (iVec[d] - shift * N[d]);
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


        std::stringstream fname;
        fname << "data/FieldBumponTail_";
        fname << Ippl::Comm->rank();
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
            auto rho = rhoview(i+nghost,j+nghost,k+nghost);
            for(size_t d = 0; d < Dim; ++d) {
                bool isNotZero = (Dr != 0.0);
                double factor = isNotZero * (1.0 / (Dr + ((!isNotZero) * 1.0))); 
                Ek = -(imag * kVec[d] * rho * factor);
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


     void dumpParticleData(const unsigned int& iter, ParticleAttrib<Vector_t>& Rtemp, ParticleAttrib<Vector_t>& Ptemp, const char* fname) {

        typename ParticleAttrib<Vector_t>::HostMirror R_host = Rtemp.getHostMirror();
        typename ParticleAttrib<Vector_t>::HostMirror P_host = Ptemp.getHostMirror();
        Kokkos::deep_copy(R_host, Rtemp.getView());
        Kokkos::deep_copy(P_host, Ptemp.getView());
        std::stringstream pname;
        pname << "data/";
        pname << fname;
        pname << "_rank_";
        pname << Ippl::Comm->rank();
        pname << "_iter_";
        pname << iter;
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
     }

    void writelocalError(double Rerror, double Perror, unsigned int iter) {
        
            std::stringstream fname;
            fname << "data/localError_";
            fname << Ippl::Comm->rank();
            fname << ".csv";

            Inform csvout(NULL, fname.str().c_str(), Inform::APPEND, Ippl::Comm->rank());
            csvout.precision(10);
            csvout.setf(std::ios::scientific, std::ios::floatfield);

            if(iter == 1) {
                csvout << "Iter, Rerror, Perror" << endl;
            }

            csvout << iter << " "
                   << Rerror << " "
                   << Perror << endl;

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
                    bool shift = (iVec[d] > (N[d]/2));
                    kVec[d] = 2 * pi / Len[d] * (iVec[d] - shift * N[d]);
                    double kh = kVec[d] * dx[d];
                    bool isNotZero = (kh != 0.0);
                    double factor = (1.0 / (kh + ((!isNotZero) * 1.0)));
                    double arg = isNotZero * (Kokkos::Experimental::sin(kh) * factor) + 
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
                     const double dt, const double& tStartMySlice) {
    
        static IpplTimings::TimerRef fieldSolvePIC = IpplTimings::getTimer("fieldSolvePIC");
        PLayout& PL = this->getLayout();
        //PL.applyBC(Rtemp, PL.getRegionLayout().getDomain());
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

        //dumpLandauPIC();         

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
            IpplTimings::startTimer(fieldSolvePIC);
            solver_mp->solve();
            IpplTimings::stopTimer(fieldSolvePIC);
    
            // gather E field
            gather(E, EfieldPIC_m, Rtemp);
    
            //kick
            Ptemp = Ptemp - 0.5 * dt * E;
            
            time_m += dt;
            //dumpLandauPIC();         
        }
    
    }

    void BorisPIC(ParticleAttrib<Vector_t>& Rtemp, 
                     ParticleAttrib<Vector_t>& Ptemp, const unsigned int nt, 
                     const double dt, const double& tStartMySlice, const double& Bext) {
    
        static IpplTimings::TimerRef fieldSolvePIC = IpplTimings::getTimer("fieldSolvePIC");
        PLayout& PL = this->getLayout();
        //PL.applyBC(Rtemp, PL.getRegionLayout().getDomain());
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

        //dumpLandauPIC();         
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

                Eview(j)[0] += Eext_x;
                Eview(j)[1] += Eext_y;
                Eview(j)[2] += Eext_z;
                
                Pview(j)[0] += alpha * (Eview(j)[0]  + Pview(j)[1] * Bext);
                Pview(j)[1] += alpha * (Eview(j)[1]  - Pview(j)[0] * Bext);
                Pview(j)[2] += alpha * Eview(j)[2];
            });
    
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

                E2view(j)[0] += Eext_x;
                E2view(j)[1] += Eext_y;
                E2view(j)[2] += Eext_z;
                P2view(j)[0]  = DrInv * ( P2view(j)[0] + alpha * (E2view(j)[0] 
                                + P2view(j)[1] * Bext + alpha * Bext * E2view(j)[1]) );
                P2view(j)[1]  = DrInv * ( P2view(j)[1] + alpha * (E2view(j)[1] 
                                - P2view(j)[0] * Bext - alpha * Bext * E2view(j)[0]) );
                P2view(j)[2] += alpha * E2view(j)[2];
            });
            
            time_m += dt;
            //dumpLandauPIC();         
        }
    
    }



    void LeapFrogPIF(ParticleAttrib<Vector_t>& Rtemp,
                     ParticleAttrib<Vector_t>& Ptemp, const unsigned int& nt, 
                     const double& dt, const bool& /*isConverged*/, 
                     const double& tStartMySlice, const unsigned int& iter) {
    
        static IpplTimings::TimerRef dumpData = IpplTimings::getTimer("dumpData");
        PLayout& PL = this->getLayout();
        //PL.applyBC(Rtemp, PL.getRegionLayout().getDomain());
        //checkBounds(Rtemp);
        rhoPIF_m = {0.0, 0.0};
        scatterPIF(q, rhoPIF_m, Sk_m, Rtemp);
    
        rhoPIF_m = rhoPIF_m / ((rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2]));
    
        // Solve for and gather E field
        gatherPIF(E, rhoPIF_m, Sk_m, Rtemp);
    
        time_m = tStartMySlice;

        if((time_m == 0.0)) {
            IpplTimings::startTimer(dumpData);
            //dumpLandau(iter);         
            dumpBumponTail(iter);         
            dumpEnergy(this->getLocalNum(), iter, Ptemp);
            IpplTimings::stopTimer(dumpData);
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
            scatterPIF(q, rhoPIF_m, Sk_m, Rtemp);
    
            rhoPIF_m = rhoPIF_m / ((rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2]));
    
            // Solve for and gather E field
            gatherPIF(E, rhoPIF_m, Sk_m, Rtemp);
    
            //kick
            Ptemp = Ptemp - 0.5 * dt * E;
    
            time_m += dt;
            
            IpplTimings::startTimer(dumpData);
            //dumpLandau(iter);         
            dumpBumponTail(iter);         
            dumpEnergy(this->getLocalNum(), iter, Ptemp);         
            IpplTimings::stopTimer(dumpData);
    
        }
    }


    void BorisPIF(ParticleAttrib<Vector_t>& Rtemp,
                     ParticleAttrib<Vector_t>& Ptemp, const unsigned int& nt, 
                     const double& dt, const bool& /*isConverged*/, 
                     const double& tStartMySlice, const unsigned int& iter, const double& Bext) {
    
        static IpplTimings::TimerRef dumpData = IpplTimings::getTimer("dumpData");
        PLayout& PL = this->getLayout();
        //PL.applyBC(Rtemp, PL.getRegionLayout().getDomain());
        //checkBounds(Rtemp);
        rhoPIF_m = {0.0, 0.0};
        scatterPIF(q, rhoPIF_m, Sk_m, Rtemp);
    
        rhoPIF_m = rhoPIF_m / ((rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2]));
    
        // Solve for and gather E field
        gatherPIF(E, rhoPIF_m, Sk_m, Rtemp);
    
        time_m = tStartMySlice;

        if((time_m == 0.0)) {
            IpplTimings::startTimer(dumpData);
            dumpEnergy(this->getLocalNum(), iter, Ptemp);
            IpplTimings::stopTimer(dumpData);
        }
        double alpha = -0.5 * dt;
        double DrInv = 1.0 / (1 + (std::pow((alpha * Bext), 2)));
        Vector_t rmax = rmax_m;
        for (unsigned int it=0; it<nt; it++) {
    
            // kick
    
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

                Eview(j)[0] += Eext_x;
                Eview(j)[1] += Eext_y;
                Eview(j)[2] += Eext_z;
                
                Pview(j)[0] += alpha * (Eview(j)[0]  + Pview(j)[1] * Bext);
                Pview(j)[1] += alpha * (Eview(j)[1]  - Pview(j)[0] * Bext);
                Pview(j)[2] += alpha * Eview(j)[2];
            });
    
            //drift
            Rtemp = Rtemp + dt * Ptemp;
    
            //Apply particle BC
            PL.applyBC(Rtemp, PL.getRegionLayout().getDomain());
            //checkBounds(Rtemp);
    
            //scatter the charge onto the underlying grid
            rhoPIF_m = {0.0, 0.0};
            scatterPIF(q, rhoPIF_m, Sk_m, Rtemp);
    
            rhoPIF_m = rhoPIF_m / ((rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2]));
    
            // Solve for and gather E field
            gatherPIF(E, rhoPIF_m, Sk_m, Rtemp);
    
            //kick
            auto R2view = Rtemp.getView();
            auto P2view = Ptemp.getView();
            auto E2view = E.getView();
            Kokkos::parallel_for("Kick2", this->getLocalNum(),
                                  KOKKOS_LAMBDA(const size_t j){
                double Eext_x = -(R2view(j)[0] - 0.5*rmax[0]) * (V0/(2*std::pow(rmax[2],2)));
                double Eext_y = -(R2view(j)[1] - 0.5*rmax[1]) * (V0/(2*std::pow(rmax[2],2)));
                double Eext_z =  (R2view(j)[2] - 0.5*rmax[2]) * (V0/(std::pow(rmax[2],2)));

                E2view(j)[0] += Eext_x;
                E2view(j)[1] += Eext_y;
                E2view(j)[2] += Eext_z;
                P2view(j)[0]  = DrInv * ( P2view(j)[0] + alpha * (E2view(j)[0] 
                                + P2view(j)[1] * Bext + alpha * Bext * E2view(j)[1]) );
                P2view(j)[1]  = DrInv * ( P2view(j)[1] + alpha * (E2view(j)[1] 
                                - P2view(j)[0] * Bext - alpha * Bext * E2view(j)[0]) );
                P2view(j)[2] += alpha * E2view(j)[2];
            });
    
            time_m += dt;
            
            IpplTimings::startTimer(dumpData);
            dumpEnergy(this->getLocalNum(), iter, Ptemp);         
            IpplTimings::stopTimer(dumpData);
    
        }
    }

private:
    void setBCAllPeriodic() {

        this->setParticleBC(ippl::BC::PERIODIC);
    }

};
