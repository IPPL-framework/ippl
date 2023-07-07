// ChargedParticlesPIF header file
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


const double pi = std::acos(-1.0);

// Test programs have to define this variable for VTK dump purposes
extern const char* TestName;

template<class PLayout>
class ChargedParticlesPIF : public ippl::ParticleBase<PLayout> {
public:
    CxField_t rho_m;
    CxField_t rhoDFT_m;
    Field_t Sk_m;

    Vector<int, Dim> nr_m;

    ippl::e_dim_tag decomp_m[Dim];

    Vector_t hr_m;
    Vector_t rmin_m;
    Vector_t rmax_m;

    double Q_m;

    size_type Np_m;

    double time_m;

    double rhoNorm_m;

    std::string shapetype_m;

    int shapedegree_m;


public:
    ParticleAttrib<double>     q; // charge
    typename ippl::ParticleBase<PLayout>::particle_position_type P;  // particle velocity
    typename ippl::ParticleBase<PLayout>::particle_position_type E;  // electric field at particle position


    /*
      This constructor is mandatory for all derived classes from
      ParticleBase as the bunch buffer uses this
    */
    ChargedParticlesPIF(PLayout& pl)
    : ippl::ParticleBase<PLayout>(pl)
    {
        // register the particle attributes
        this->addAttribute(q);
        this->addAttribute(P);
        this->addAttribute(E);
    }

    ChargedParticlesPIF(PLayout& pl,
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
        setupBCs();
        for (unsigned int i = 0; i < Dim; i++)
            decomp_m[i]=decomp[i];
    }

    ~ChargedParticlesPIF(){ }

    void setupBCs() {
        setBCAllPeriodic();
    }

    void initNUFFT(FieldLayout_t& FL) {
        ippl::ParameterList fftParams;

        fftParams.add("gpu_method", 1);
        fftParams.add("gpu_sort", 0);
        fftParams.add("gpu_kerevalmeth", 1);
        fftParams.add("tolerance", 1e-6);

        fftParams.add("use_cufinufft_defaults", false);
        //fftParams.add("use_cufinufft_defaults", true);

        q.initializeNUFFT(FL, 1, fftParams);
        E.initializeNUFFT(FL, 2, fftParams);
    }

    void gather() {

        gatherPIFNUFFT(this->E, rho_m, Sk_m, this->R, this->q);
        //gatherPIFNUDFT(this->E, rho_m, Sk_m, this->R);

        //Set the charge back to original as we used this view as a 
        //temporary buffer during gather
        this->q = Q_m / Np_m; 

    }

    void scatter() {
        
        Inform m("scatter ");
        rho_m = {0.0, 0.0};
        scatterPIFNUFFT(q, rho_m, Sk_m, this->R);
        //rhoDFT_m = {0.0, 0.0};
        //scatterPIFNUDFT(q, rho_m, Sk_m, this->R);

        //dumpFieldData();

        rho_m = rho_m / ((rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2]));
    }


    void dumpLandau() {
       
       
       double fieldEnergy = 0.0; 
       double ExAmp = 0.0;

       auto rhoview = rho_m.getView();
       const int nghost = rho_m.getNghost();
       using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<Dim>>;
     
       const FieldLayout_t& layout = rho_m.getLayout(); 
       const Mesh_t& mesh = rho_m.get_mesh();
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
               //bool shift = (iVec[d] > (N[d]/2));
               //kVec[d] = 2 * pi / Len[d] * (iVec[d] - shift * N[d]);
               kVec[d] = 2 * pi / Len[d] * (iVec[d] - (N[d] / 2));
               Dr += kVec[d] * kVec[d];
           }

           Kokkos::complex<double> Ek = {0.0, 0.0}; 
           auto rho = rhoview(i+nghost,j+nghost,k+nghost);
           bool isNotZero = (Dr != 0.0);
           double factor = isNotZero * (1.0 / (Dr + ((!isNotZero) * 1.0))); 
           Ek = -(imag * kVec[0] * rho * factor);
           double myVal = Ek.real() * Ek.real() + Ek.imag() * Ek.imag();

           tlSum += myVal;

           double myValMax = std::sqrt(myVal);

           if(myValMax > tlMax) tlMax = myValMax;

       }, Kokkos::Sum<double>(fieldEnergy), Kokkos::Max<double>(ExAmp));
       

       Kokkos::fence();
       double volume = (rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2]);
       fieldEnergy *= volume;
        
       //auto Eview = E.getView();

       //double fieldEnergy, ExAmp;
       //double temp = 0.0;

       //Kokkos::parallel_reduce("Ex energy", this->getLocalNum(),
       //                        KOKKOS_LAMBDA(const int i, double& valL){
       //                            double myVal = Eview(i)[0] * Eview(i)[0];
       //                            valL += myVal;
       //                        }, Kokkos::Sum<double>(temp));

       //double globaltemp = 0.0;
       //MPI_Reduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());
       //double volume = (rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2]);
       //fieldEnergy = globaltemp * volume / totalP ;

       //double tempMax = 0.0;
       //Kokkos::parallel_reduce("Ex max norm", this->getLocalNum(),
       //                        KOKKOS_LAMBDA(const size_t i, double& valL)
       //                        {
       //                            double myVal = std::fabs(Eview(i)[0]);
       //                            if(myVal > valL) valL = myVal;
       //                        }, Kokkos::Max<double>(tempMax));
       //ExAmp = 0.0;
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
       
       Ippl::Comm->barrier();
    }


    void dumpBumponTail() {
       
       
       double fieldEnergy = 0.0; 
       double ExAmp = 0.0;

       auto rhoview = rho_m.getView();
       const int nghost = rho_m.getNghost();
       using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<Dim>>;
     
       const FieldLayout_t& layout = rho_m.getLayout(); 
       const Mesh_t& mesh = rho_m.get_mesh();
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
               Ek = -(imag * kVec[2] * rhoview(i+nghost,j+nghost,k+nghost) / Dr);
           }
           double myVal = Ek.real() * Ek.real() + Ek.imag() * Ek.imag();

           tlSum += myVal;

           double myValMax = std::sqrt(myVal);

           if(myValMax > tlMax) tlMax = myValMax;

       }, Kokkos::Sum<double>(fieldEnergy), Kokkos::Max<double>(ExAmp));
       

       Kokkos::fence();
       double volume = (rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2]);
       fieldEnergy *= volume;

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
                  << ExAmp << endl;

       }
       
       Ippl::Comm->barrier();
    }


    void dumpEnergy() {
       

       double potentialEnergy, kineticEnergy;
       double temp = 0.0;

       //auto Eview = E.getView();
       //Kokkos::parallel_reduce("Potential energy", this->getLocalNum(),
       //                        KOKKOS_LAMBDA(const int i, double& valL){
       //                            double myVal = dot(Eview(i), Eview(i)).apply();
       //                            valL += myVal;
       //                        }, Kokkos::Sum<double>(temp));



       auto rhoview = rho_m.getView();
       const int nghost = rho_m.getNghost();
       using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<Dim>>;
      
       const FieldLayout_t& layout = rho_m.getLayout(); 
       const Mesh_t& mesh = rho_m.get_mesh();
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
               //bool shift = (iVec[d] > (N[d]/2));
               //kVec[d] = 2 * pi / Len[d] * (iVec[d] - shift * N[d]);
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

       auto Pview = P.getView();
       auto qView = q.getView();

       temp = 0.0;

       Kokkos::parallel_reduce("Kinetic Energy", this->getLocalNum(),
                               KOKKOS_LAMBDA(const int i, double& valL){
                                   double myVal = dot(Pview(i), Pview(i)).apply();
                                   myVal *= -qView(i);
                                   valL += myVal;
                               }, Kokkos::Sum<double>(temp));

       temp *= 0.5;
       double globaltemp = 0.0;
       MPI_Reduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());

       kineticEnergy = globaltemp;

       if (Ippl::Comm->rank() == 0) {
           std::stringstream fname;
           fname << "data/Energy_";
           fname << Ippl::Comm->size();
           fname << ".csv";


           Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
           csvout.precision(10);
           csvout.setf(std::ios::scientific, std::ios::floatfield);

           if(time_m == 0.0) {
               csvout << "time, Potential energy, Kinetic energy, Total energy" << endl;
           }

           csvout << time_m << " "
                  << potentialEnergy << " "
                  << kineticEnergy << " "
                  << potentialEnergy + kineticEnergy << endl;

       }
       
       Ippl::Comm->barrier();
    }


    void initializeShapeFunctionPIF() {

        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        auto Skview = Sk_m.getView();
        auto N = nr_m;
        const int nghost = Sk_m.getNghost();
        const Mesh_t& mesh = rho_m.get_mesh();
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
                    //bool shift = (iVec[d] > (N[d]/2));
                    //kVec[d] = 2 * pi / Len[d] * (iVec[d] - shift * N[d]);
                    kVec[d] = 2 * pi / Len[d] * (iVec[d] - (N[d] / 2));
                    double kh = kVec[d] * dx[d];
                    bool isNotZero = (kh != 0.0);
                    double factor = (1.0 / (kh + ((!isNotZero) * 1.0)));
                    double arg = isNotZero * (Kokkos::sin(kh) * factor) + 
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
    
    //void dumpBumponTail() {

    //   const int nghostE = E_m.getNghost();
    //   auto Eview = E_m.getView();
    //   double fieldEnergy, EzAmp;
    //   using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

    //   double temp = 0.0;
    //   Kokkos::parallel_reduce("Ex inner product",
    //                           mdrange_type({nghostE, nghostE, nghostE},
    //                                        {Eview.extent(0) - nghostE,
    //                                         Eview.extent(1) - nghostE,
    //                                         Eview.extent(2) - nghostE}),
    //                           KOKKOS_LAMBDA(const size_t i, const size_t j,
    //                                         const size_t k, double& valL)
    //                           {
    //                               double myVal = std::pow(Eview(i, j, k)[2], 2);
    //                               valL += myVal;
    //                           }, Kokkos::Sum<double>(temp));
    //   double globaltemp = 0.0;
    //   MPI_Reduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());
    //   fieldEnergy = globaltemp * hr_m[0] * hr_m[1] * hr_m[2];

    //   double tempMax = 0.0;
    //   Kokkos::parallel_reduce("Ex max norm",
    //                           mdrange_type({nghostE, nghostE, nghostE},
    //                                        {Eview.extent(0) - nghostE,
    //                                         Eview.extent(1) - nghostE,
    //                                         Eview.extent(2) - nghostE}),
    //                           KOKKOS_LAMBDA(const size_t i, const size_t j,
    //                                         const size_t k, double& valL)
    //                           {
    //                               double myVal = std::fabs(Eview(i, j, k)[2]);
    //                               if(myVal > valL) valL = myVal;
    //                           }, Kokkos::Max<double>(tempMax));
    //   EzAmp = 0.0;
    //   MPI_Reduce(&tempMax, &EzAmp, 1, MPI_DOUBLE, MPI_MAX, 0, Ippl::getComm());


    //   if (Ippl::Comm->rank() == 0) {
    //       std::stringstream fname;
    //       fname << "data/FieldBumponTail_";
    //       fname << Ippl::Comm->size();
    //       fname << ".csv";


    //       Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
    //       csvout.precision(10);
    //       csvout.setf(std::ios::scientific, std::ios::floatfield);

    //       if(time_m == 0.0) {
    //           csvout << "time, Ez_field_energy, Ez_max_norm" << endl;
    //       }

    //       csvout << time_m << " "
    //              << fieldEnergy << " "
    //              << EzAmp << endl;

    //   }
    //   
    //   Ippl::Comm->barrier();
    //}

    void dumpFieldData() {

       typename CxField_t::HostMirror rhoNUFFT_host = rho_m.getHostMirror();
       typename CxField_t::HostMirror rhoNUDFT_host = rhoDFT_m.getHostMirror();
       Kokkos::deep_copy(rhoNUFFT_host, rho_m.getView());
       Kokkos::deep_copy(rhoNUDFT_host, rhoDFT_m.getView());
       const int nghost = rho_m.getNghost();
       std::stringstream pname;
       pname << "data/FieldFFT_";
       pname << Ippl::Comm->rank();
       pname << ".csv";
       Inform pcsvout(NULL, pname.str().c_str(), Inform::OVERWRITE, Ippl::Comm->rank());
       pcsvout.precision(10);
       pcsvout.setf(std::ios::scientific, std::ios::floatfield);
       pcsvout << "rho" << endl;
       for (int i = 0; i< nr_m[0]; i++) {
            for (int j = 0; j< nr_m[1]; j++) {
                for (int k = 0; k< nr_m[2]; k++) {
                    pcsvout << rhoNUFFT_host(i+nghost,j+nghost, k+nghost) << endl;
                }
            }
       }
       std::stringstream pname2;
       pname2 << "data/FieldDFT_";
       pname2 << Ippl::Comm->rank();
       pname2 << ".csv";
       Inform pcsvout2(NULL, pname2.str().c_str(), Inform::OVERWRITE, Ippl::Comm->rank());
       pcsvout2.precision(10);
       pcsvout2.setf(std::ios::scientific, std::ios::floatfield);
       pcsvout2 << "rho" << endl;
       for (int i = 0; i< nr_m[0]; i++) {
            for (int j = 0; j< nr_m[1]; j++) {
                for (int k = 0; k< nr_m[2]; k++) {
                    pcsvout2 << rhoNUDFT_host(i+nghost,j+nghost, k+nghost) << endl;
                }
            }
       }
       Ippl::Comm->barrier();
    }


    //void dumpParticleData() {

    //   typename ParticleAttrib<Vector_t>::HostMirror R_host = this->R.getHostMirror();
    //   typename ParticleAttrib<Vector_t>::HostMirror P_host = this->P.getHostMirror();
    //   Kokkos::deep_copy(R_host, this->R.getView());
    //   Kokkos::deep_copy(P_host, P.getView());
    //   std::stringstream pname;
    //   pname << "data/ParticleIC_";
    //   pname << Ippl::Comm->rank();
    //   pname << ".csv";
    //   Inform pcsvout(NULL, pname.str().c_str(), Inform::OVERWRITE, Ippl::Comm->rank());
    //   pcsvout.precision(10);
    //   pcsvout.setf(std::ios::scientific, std::ios::floatfield);
    //   pcsvout << "R_x, R_y, R_z, V_x, V_y, V_z" << endl;
    //   for (size_type i = 0; i< this->getLocalNum(); i++) {
    //       pcsvout << R_host(i)[0] << " "
    //               << R_host(i)[1] << " "
    //               << R_host(i)[2] << " "
    //               << P_host(i)[0] << " "
    //               << P_host(i)[1] << " "
    //               << P_host(i)[2] << endl;
    //   }
    //   Ippl::Comm->barrier();
    //}

private:
    void setBCAllPeriodic() {

        this->setParticleBC(ippl::BC::PERIODIC);
    }

};
