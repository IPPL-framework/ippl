// Test PIC3d
//   This test program sets up a simple sine-wave electric field in 3D,
//   creates a population of particles with random positions and and velocities,
//   and then tracks their motions in the static
//   electric field using cloud-in-cell interpolation and periodic particle BCs.
//
//   This test also provides a base for load-balancing using a domain-decomposition
//   based on an ORB.
//
//   Usage:
//     srun ./PIC3d 128 128 128 10000 10 --info 10
//
// Copyright (c) 2020, Sriramkrishnan Muralikrishnan,
// Paul Scherrer Institut, Villigen PSI, Switzerland
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
#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include <set>

#include <random>
#include "Utility/IpplTimings.h"

// dimension of our positions
constexpr unsigned Dim = 3;

// some typedefs
typedef ippl::ParticleSpatialLayout<double,Dim>   PLayout_t;
typedef ippl::UniformCartesian<double, Dim>        Mesh_t;
typedef ippl::FieldLayout<Dim> FieldLayout_t;
typedef ippl::OrthogonalRecursiveBisection<double, Dim, Mesh_t> ORB;

template<typename T, unsigned Dim>
using Vector = ippl::Vector<T, Dim>;

template<typename T, unsigned Dim>
using Field = ippl::Field<T, Dim>;

template<typename T>
using ParticleAttrib = ippl::ParticleAttrib<T>;

typedef Vector<double, Dim>  Vector_t;
typedef Field<double, Dim>   Field_t;
typedef Field<Vector_t, Dim> VField_t;

double pi = acos(-1.0);

template<class PLayout>
class ChargedParticles : public ippl::ParticleBase<PLayout> {
public:
    Field<Vector<double, Dim>, Dim> EFD_m;
    Field<double,Dim> EFDMag_m;

    // ORB
    ORB orb;

    Vector<int, Dim> nr_m;

    ippl::e_dim_tag decomp_m[Dim];

    Vector_t hr_m;
    Vector_t rmin_m;
    Vector_t rmax_m;

    double Q_m;


public:
    ParticleAttrib<double>     qm; // charge-to-mass ratio
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
        this->addAttribute(qm);
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
        this->addAttribute(qm);
        this->addAttribute(P);
        this->addAttribute(E);
        setupBCs();
        for (unsigned int i = 0; i < Dim; i++)
            decomp_m[i]=decomp[i];
    }

    void setupBCs() {
        setBCAllPeriodic();
    }


    void updateLayout(FieldLayout_t& fl, Mesh_t& mesh, ChargedParticles<PLayout>& buffer) {
        // Update local fields
        static IpplTimings::TimerRef tupdateLayout = IpplTimings::getTimer("updateLayout");
        IpplTimings::startTimer(tupdateLayout);
        this->EFD_m.updateLayout(fl);
        this->EFDMag_m.updateLayout(fl);

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

    ~ChargedParticles() {}

    void repartition(FieldLayout_t& fl, Mesh_t& mesh, ChargedParticles<PLayout>& buffer) {
        // Repartition the domains
        bool res = orb.binaryRepartition(this->R, fl);

        if (res != true) {
           std::cout << "Could not repartition!" << std::endl;
           return;
        }
        // Update
        this->updateLayout(fl, mesh, buffer);
    }

    bool balance(unsigned int totalP){//, int timestep = 1) {
        int local = 0;
        std::vector<int> res(Ippl::Comm->size());
        double threshold = 0.0;
        double equalPart = (double) totalP / Ippl::Comm->size();
        double dev = std::abs((double)this->getLocalNum() - equalPart) / totalP;
        if (dev > threshold)
            local = 1;
        MPI_Allgather(&local, 1, MPI_INT, res.data(), 1, MPI_INT, Ippl::getComm());

        /***PRINT***/
        /*
        std::ofstream file;
        file.open("imbalance.txt", std::ios_base::app);
        file << std::to_string(timestep) << " " << Ippl::Comm->rank() << " " << dev << "\n";
        file.close();
        */
        for (unsigned int i = 0; i < res.size(); i++) {
            if (res[i] == 1)
                return true;
        }
        return false;
    }

    void gatherStatistics(unsigned int totalP) {

        Ippl::Comm->barrier();
        std::cout << "Rank " << Ippl::Comm->rank() << " has "
                  << (double)this->getLocalNum()/totalP*100.0
                  << " percent of the total particles " << std::endl;
        Ippl::Comm->barrier();
    }

    void gatherCIC() {

        //static IpplTimings::TimerRef gatherTimer = IpplTimings::getTimer("gather");
        //IpplTimings::startTimer(gatherTimer);
        gather(this->E, EFD_m, this->R);
        //IpplTimings::stopTimer(gatherTimer);

    }

    void scatterCIC(unsigned int totalP, int iteration) {

         //static IpplTimings::TimerRef scatterTimer = IpplTimings::getTimer("scatter");
         //IpplTimings::startTimer(scatterTimer);
         Inform m("scatter ");
         EFDMag_m = 0.0;
         scatter(qm, EFDMag_m, this->R);
         //IpplTimings::stopTimer(scatterTimer);

         static IpplTimings::TimerRef sumTimer = IpplTimings::getTimer("CheckCharge");
         IpplTimings::startTimer(sumTimer);
         double Q_grid = EFDMag_m.sum();

         unsigned int Total_particles = 0;
         unsigned int local_particles = this->getLocalNum();

         MPI_Reduce(&local_particles, &Total_particles, 1,
                       MPI_UNSIGNED, MPI_SUM, 0, Ippl::getComm());

         double rel_error = std::fabs((Q_m-Q_grid)/Q_m);
         m << "Rel. error in charge conservation = " << rel_error << endl;
         
         if(Ippl::Comm->rank() == 0) {
             if(Total_particles != totalP || rel_error > 1e-10) {
                 std::cout << "Total particles in the sim. " << totalP 
                           << " " << "after update: " 
                           << Total_particles << std::endl;
                 std::cout << "Total particles not matched in iteration: "
                           << iteration << std::endl;
                 std::cout << "Q grid: "
                           << Q_grid << "Q particles: " << Q_m << std::endl;
                 std::cout << "Rel. error in charge conservation: " 
                           << rel_error << std::endl;
                 exit(1);
             }
         }
         
         IpplTimings::stopTimer(sumTimer);                                                    
    }

     void writePerRank() {
        double lq = 0.0, lqm = 0.0;
        Field_t::view_type viewRho = this->EFDMag_m.getView();
        ParticleAttrib<double>::view_type viewqm = this->qm.getView();
        int nghost = this->EFDMag_m.getNghost();

        using mdrange_t = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_reduce("Particle Charge", mdrange_t({nghost,nghost,nghost},{viewRho.extent(0)-nghost,viewRho.extent(1)-nghost,viewRho.extent(2)-nghost})
                                               , KOKKOS_LAMBDA(const int i, const int j, const int k, double& val){
           val += viewRho(i, j, k);
        }, lq);
        Kokkos::parallel_reduce("Particle QM", viewqm.extent(0)
                                               , KOKKOS_LAMBDA(const int i, double& val){
           val += viewqm(i);
        }, lqm);

        double lQ = lq / this->EFDMag_m.sum();

        /***PRINT***/
        /*
        std::ofstream fcharge;
        fcharge.open("charge.txt", std::ios_base::app);
        fcharge << std::to_string(step) << " " << Ippl::Comm->rank() << " " << lQ << "\n";
        fcharge.close();

        std::ofstream fqm;
        fqm.open("qm.txt", std::ios_base::app);
        fqm << std::to_string(step) << " " << Ippl::Comm->rank() << " " << lqm << "\n";
        fqm.close();
        */
     }

     void initFields() {
         static IpplTimings::TimerRef initFieldsTimer = IpplTimings::getTimer("initFields");
         IpplTimings::startTimer(initFieldsTimer);
         Inform m("initFields ");

         ippl::NDIndex<Dim> domain = EFD_m.getDomain();

         for (unsigned int i=0; i<Dim; i++)
             nr_m[i] = domain[i].length();


         double phi0 = 0.1;
         double pi = acos(-1.0);
         // scale_fact so that particles move more
         double scale_fact = 1e5; // 1e6

         Vector_t hr = hr_m;


         typename VField_t::view_type& view = EFD_m.getView();
         const FieldLayout_t& layout = EFD_m.getLayout();
         const ippl::NDIndex<Dim>& lDom = layout.getLocalNDIndex();
         const int nghost = EFD_m.getNghost();
         using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;



         Kokkos::parallel_for("Assign EFD_m[0]",
                              mdrange_type({nghost, nghost, nghost},
                                            {view.extent(0) - nghost,
                                             view.extent(1) - nghost,
                                             view.extent(2) - nghost}),
                              KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k){

                                //local to global index conversion
                                const size_t ig = i + lDom[0].first() - nghost;
                                const size_t jg = j + lDom[1].first() - nghost;
                                const size_t kg = k + lDom[2].first() - nghost;

                                view(i, j, k)[0] = -scale_fact*2.0*pi*phi0 *
                                                    cos(2.0*pi*(ig+0.5)*hr[0]) *
                                                    cos(4.0*pi*(jg+0.5)*hr[1]) *
                                                    cos(pi*(kg+0.5)*hr[2]);

                              });

         Kokkos::parallel_for("Assign EFD_m[1]",
                              mdrange_type({nghost, nghost, nghost},
                                            {view.extent(0) - nghost,
                                             view.extent(1) - nghost,
                                             view.extent(2) - nghost}),
                              KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k){

                                //local to global index conversion
                                const size_t ig = i + lDom[0].first() - nghost;
                                const size_t jg = j + lDom[1].first() - nghost;

                                view(i, j, k)[1] = scale_fact*4.0*pi*phi0 *
                                                   sin(2.0*pi*(ig+0.5)*hr[0]) *
                                                   sin(4.0*pi*(jg+0.5)*hr[1]);

                              });

         Kokkos::parallel_for("Assign EFD_m[2]",
                              mdrange_type({nghost, nghost, nghost},
                                            {view.extent(0) - nghost,
                                             view.extent(1) - nghost,
                                             view.extent(2) - nghost}),
                              KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k){

                                //local to global index conversion
                                const size_t ig = i + lDom[0].first() - nghost;
                                const size_t jg = j + lDom[1].first() - nghost;

                                view(i, j, k)[2] = scale_fact*4.0*pi*phi0 *
                                                   sin(2.0*pi*(ig+0.5)*hr[0]) *
                                                   sin(4.0*pi*(jg+0.5)*hr[1]);

                              });

         EFDMag_m = dot(EFD_m, EFD_m);
         EFDMag_m = sqrt(EFDMag_m);
         IpplTimings::stopTimer(initFieldsTimer);

     }


     void dumpParticleData(int iteration) {

        ParticleAttrib<Vector_t>::view_type& view = P.getView();

        double Energy = 0.0;

        Kokkos::parallel_reduce("Particle Energy", view.extent(0),
                                KOKKOS_LAMBDA(const int i, double& valL){
                                    double myVal = dot(view(i), view(i)).apply();
                                    valL += myVal;
                                }, Kokkos::Sum<double>(Energy));

        Energy *= 0.5;
        double gEnergy = 0.0;

        MPI_Reduce(&Energy, &gEnergy, 1,
                    MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());

        if(Ippl::Comm->rank() == 0) {
            std::ofstream csvout;
            csvout.precision(10);
            csvout.setf(std::ios::scientific, std::ios::floatfield);

            std::stringstream fname;
            fname << "data/energy.csv";
            csvout.open(fname.str().c_str(), std::ios::out | std::ofstream::app);

            csvout << iteration << " "
                   << gEnergy << std::endl;

            csvout.close();
        }

        Ippl::Comm->barrier();


     }

    // @param tag
    //        2 -> uniform(0,1)
    //        1 -> normal(0,1)
    //        0 -> gridpoints
    void initPositions(FieldLayout_t& fl, Vector_t& hr, unsigned int nloc, int tag = 2) {
        Inform m("initPositions ");
        typename ippl::ParticleBase<PLayout>::particle_position_type::HostMirror R_host = this->R.getHostMirror();

        std::mt19937_64 eng[Dim];
        for (unsigned i = 0; i < Dim; ++i) {
            eng[i].seed(42 + i * Dim);
            eng[i].discard( nloc * Ippl::Comm->rank());
        }

        std::mt19937_64 engN[4*Dim];
        for (unsigned i = 0; i < 4*Dim; ++i) {
           engN[i].seed(42 + i * Dim);
           engN[i].discard(nloc * Ippl::Comm->rank());
        }

        auto dom = fl.getDomain();
        unsigned int gridpoints = dom[0].length() * dom[1].length() * dom[2].length();
        if (tag == 0 && nloc * Ippl::Comm->size() != gridpoints) {
            if (Ippl::Comm->rank() == 0) {
                std::cerr << "Particle count must match gridpoint count to use gridpoint locations. Switching to uniform distribution." << std::endl;
            }
            tag = 2;
        }

        if (tag == 0) {
           m << "Positions are set on grid points" << endl;
           int N = fl.getDomain()[0].length();   // this only works for boxes
           const ippl::NDIndex<Dim>& lDom = fl.getLocalNDIndex();
           using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
           int size = Ippl::Comm->size();
           // Loops over particles
           Kokkos::parallel_for("initPositions", mdrange_type({0,0,0},{N/size,N,N}),
                                                 KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
              const size_t ig = i + lDom[0].first(); // index i doesn't sweep through the required size
              const size_t jg = j + lDom[1].first();
              const size_t kg = k + lDom[2].first();

              // int l = i + j * N + k * N * N;
              int l = i + j * N / size + k * N * N / size;
              R_host(l)[0] = (ig+0.5)*hr[0];
              R_host(l)[1] = (jg+0.5)*hr[1];
              R_host(l)[2] = (kg+0.5)*hr[2];
           });

        } else if (tag == 1) {
           m << "Positions follow normal distribution" << endl;
           std::vector<double> mu(Dim);
           std::vector<double> sd(Dim);
           std::vector<double> states(Dim);

           Vector_t length = {1.0, 1.0, 1.0};

           mu[0] = 0.5;
           sd[0] = 0.75;
           mu[1] = 0.6;
           sd[1] = 0.3;
           mu[2] = 0.2;
           sd[2] = 0.2;

           std::uniform_real_distribution<double> dist_uniform(0.0, 1.0);

           double sum_coord=0.0;
           for (unsigned long long int i = 0; i< nloc; i++) {
              for (unsigned d = 0; d<Dim; d++) {
                 double u1 = dist_uniform(engN[d*2]);
                 double u2 = dist_uniform(engN[d*2+1]);
                 states[d] = sd[d] * std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * pi * u2) + mu[d];
                 R_host(i)[d] =  std::fabs(std::fmod(states[d],length[d]));
                 sum_coord += R_host(i)[d];
              }
           }
        } else {
           double rmin = 0.0, rmax = 1.0;
           m << "Positions follow uniform distribution U(" << rmin << "," << rmax << ")" << endl;
           std::uniform_real_distribution<double> unif(rmin, rmax);
           for (unsigned long int i = 0; i < nloc; i++)
              for (int d = 0; d<3; d++)
                 R_host(i)[d] = unif(eng[d]);
        }

        // Copy to device
        Kokkos::deep_copy(this->R.getView(), R_host);
     }


private:
    void setBCAllPeriodic() {

        this->setParticleBC(ippl::BC::PERIODIC);
    }

};


int main(int argc, char *argv[]) {
    Ippl ippl(argc, argv);
    Inform msg("PIC3d");
    Inform msg2all(argv[0],INFORM_ALL_NODES);

    Ippl::Comm->setDefaultOverallocation(3);

    ippl::Vector<int,Dim> nr = {
        std::atoi(argv[1]),
        std::atoi(argv[2]),
        std::atoi(argv[3])
    };

    // Each rank must have a minimal volume of 8
    if (nr[0]*nr[1]*nr[2] < 8 * Ippl::Comm->size())
       msg << "!!! Ranks have not enough volume for proper working !!! (Minimal volume per rank: 8)" << endl;

    static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("mainTimer");
    IpplTimings::startTimer(mainTimer);
    const unsigned int totalP = std::atoi(argv[4]);
    const unsigned int nt     = std::atoi(argv[5]);

    msg << "Particle test PIC3d "
        << endl
        << "nt " << nt << " Np= "
        << totalP << " grid = " << nr
        << endl;


    using bunch_type = ChargedParticles<PLayout_t>;

    std::unique_ptr<bunch_type>  P;

    ippl::NDIndex<Dim> domain;
    for (unsigned i = 0; i< Dim; i++) {
        domain[i] = ippl::Index(nr[i]);
    }

    ippl::e_dim_tag decomp[Dim];
    for (unsigned d = 0; d < Dim; ++d) {
        decomp[d] = ippl::PARALLEL;
    }

    // create mesh and layout objects for this problem domain
    Vector_t rmin(0.0);
    Vector_t rmax(1.0);
    double dx = rmax[0] / nr[0];
    double dy = rmax[1] / nr[1];
    double dz = rmax[2] / nr[2];

    Vector_t hr = {dx, dy, dz};
    Vector_t origin = {rmin[0], rmin[1], rmin[2]};

    // const double dt = 0.5 * dx; // size of timestep

    const bool isAllPeriodic=true;
    Mesh_t mesh(domain, hr, origin);
    FieldLayout_t FL(domain, decomp, isAllPeriodic);
    PLayout_t PL(FL, mesh);

    /**PRINT**/
    msg << "FIELD LAYOUT (INITIAL)" << endl;
    msg << FL << endl;

    double Q=1.0;
    P = std::make_unique<bunch_type>(PL,hr,rmin,rmax,decomp,Q);


    unsigned long int nloc = totalP / Ippl::Comm->size();

    int rest = (int) (totalP - nloc * Ippl::Comm->size());

    if ( Ippl::Comm->rank() < rest )
        ++nloc;

    static IpplTimings::TimerRef particleCreation = IpplTimings::getTimer("particlesCreation");
    IpplTimings::startTimer(particleCreation);
    P->create(nloc);
    // Verifying that particles are created
    double totalParticles = 0.0;
    double localParticles = P->getLocalNum();
    MPI_Reduce(&localParticles, &totalParticles, 1, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());
    msg << "Total particles: " << totalParticles << endl;
    P->initPositions(FL, hr, nloc, 1);

    P->qm = P->Q_m/totalP;
    P->P = 0.0;
    IpplTimings::stopTimer(particleCreation);

    bunch_type bunchBuffer(PL);

    static IpplTimings::TimerRef UpdateTimer = IpplTimings::getTimer("ParticleUpdate");
    IpplTimings::startTimer(UpdateTimer);
    PL.update(*P, bunchBuffer);
    IpplTimings::stopTimer(UpdateTimer);

    msg << "particles created and initial conditions assigned " << endl;

    P->EFD_m.initialize(mesh, FL);
    P->EFDMag_m.initialize(mesh, FL);
    P->initializeORB(FL, mesh);

    // Mass conservation
    // P->writePerRank();

    static IpplTimings::TimerRef domainDecomposition0 = IpplTimings::getTimer("domainDecomp0");
    IpplTimings::startTimer(domainDecomposition0);
    if (P->balance(totalP)) {
        P->repartition(FL, mesh, bunchBuffer);
    }
    IpplTimings::stopTimer(domainDecomposition0);
    msg << "Balancing finished" << endl;

    // Mass conservation
    // P->writePerRank();

    P->scatterCIC(totalP, 0);
    msg << "scatter done" << endl;

    P->initFields();
    msg << "P->initField() done" << endl;

    // Moving particles one grid cell
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    double dr = 0.0;
    P->P = 1.0;

    msg << "Starting iterations ..." << endl;
    for (unsigned int it=0; it<nt; it++) {
        dr = dis(gen) * hr[0];
        static IpplTimings::TimerRef RTimer = IpplTimings::getTimer("positionUpdate");
        IpplTimings::startTimer(RTimer);
        P->R = P->R + dr * P->P;
        IpplTimings::stopTimer(RTimer);

        IpplTimings::startTimer(UpdateTimer);
        PL.update(*P, bunchBuffer);
        IpplTimings::stopTimer(UpdateTimer);

        // Domain Decomposition
        if (P->balance(totalP)) {
           msg << "Starting repartition" << endl;
           IpplTimings::startTimer(domainDecomposition0);
           P->repartition(FL, mesh, bunchBuffer);
           IpplTimings::stopTimer(domainDecomposition0);
           // Conservations
           // P->writePerRank();
        }

        //scatter the charge onto the underlying grid
        msg << "Starting scatterCIC" << endl;
        P->scatterCIC(totalP, it+1);

        // gather the local value of the E field
        P->gatherCIC();

        // advance the particle velocities
        // static IpplTimings::TimerRef PTimer = IpplTimings::getTimer("velocityUpdate");
        // IpplTimings::startTimer(PTimer);
        // P->P = P->P + dt * P->qm * P->E;
        // IpplTimings::stopTimer(PTimer);

        msg << "Finished iteration " << it << endl;

        P->gatherStatistics(totalP);
    }

    msg << "Particle test PIC3d: End." << endl;
    IpplTimings::stopTimer(mainTimer);
    IpplTimings::print();
    IpplTimings::print(std::string("timing" + std::to_string(Ippl::Comm->size()) + "r_" + std::to_string(nr[0]) + "c.dat"));

    return 0;
}
