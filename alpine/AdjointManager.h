#ifndef IPPL_ADJOINT_MANAGER_H
#define IPPL_ADJOINT_MANAGER_H

#include <memory>

#include "FieldContainer.hpp"
#include "FieldSolver.hpp"
#include "LoadBalancer.hpp"
#include "AlpineManager.h"
#include "Manager/BaseManager.h"
#include "ParticleContainer.hpp"
#include "Random/Distribution.h"
#include "Random/InverseTransformSampling.h"
#include "Random/NormalDistribution.h"
#include "Random/Randn.h"

using view_type = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;

using Matrix_t = ippl::Vector< ippl::Vector<double, Dim>, Dim>;

// phi_ext = p0*x0 + p1*x1 + p2*x2 + p3*x0*x1 + p4*x0*x2 + p5*x1*x2 + p6*x0^2 + p7*x1^2 + p8*x2^2 /// + p9*x0^4 + p10*x1^4 + p11*x2^4
// dphi_ext/dx0 = p0 + p3*x1 + p4*x2 + 2*p6*x0 ///+ 4*p9*x0**3
// dphi_ext/dx1 = p1 + p3*x0 + p5*x2 + 2*p7*x1 ///+ 4*p10*x1**3
// dphi_ext/dx2 = p2 + p4*x0 + p5*x1 + 2*p8*x2 ///+ 4*p11*x2**3
// dphi_ext/dpi = [x0, x1, x2, x0*x1, x0*x2, x1*x2, x0^2, x1^2, x2^2] ///, x0^4, x1^4, x2^4]

template <unsigned Dim>
struct dphi_dparam{
       KOKKOS_INLINE_FUNCTION double operator()(auto x, unsigned int idp) const {
           if(idp==0)
                return x[0];
           else if(idp==1)
                return x[1];
           else if(idp==2)
                return x[2];
           else if(idp==3)
                return x[0]*x[1];
           else if(idp==4)
                return x[0]*x[2];
           else if(idp==5)
                return x[1]*x[2];
           else if(idp==6)
                return x[0]*x[0];
           else if(idp==7)
                return x[1]*x[1];
           else if(idp==8)
                return x[2]*x[2];
           else if(idp==9)
                return Kokkos::pow(x[0], 4);
           else if(idp==10)
                return Kokkos::pow(x[1], 4);
           else //if(idp==11)
               	return Kokkos::pow(x[2], 4);
      }
};

template <unsigned Dim>
struct ExternalForceField{
       KOKKOS_INLINE_FUNCTION double operator()(auto x, unsigned int d, auto p) const {
           if(d==0)
		return p[0] + p[3]*x[1] + p[4]*x[2] + 2.*p[6]*x[0];// + 4.*p[9]*Kokkos::pow(x[0],3);
           else if(d==1)
		return p[1] + p[3]*x[0] + p[5]*x[2] + 2.*p[7]*x[1];// + 4.*p[10]*Kokkos::pow(x[1],3);
           else if(d==2)
		return p[2] + p[4]*x[0] + p[5]*x[1] + 2.*p[8]*x[2];// + 4.*p[11]*Kokkos::pow(x[2],3);
           else
                return 0.0;
      }
};

// define functions used in sampling particles
struct CustomDistributionFunctions {
  struct CDF{
       KOKKOS_INLINE_FUNCTION double operator()(double x, unsigned int d, const double *params_p) const {
           return x + (params_p[d * 2 + 0] / params_p[d * 2 + 1]) * Kokkos::sin(params_p[d * 2 + 1] * x);
       }
  };

  struct PDF{
       KOKKOS_INLINE_FUNCTION double operator()(double x, unsigned int d, double const *params_p) const {
           return 1.0 + params_p[d * 2 + 0] * Kokkos::cos(params_p[d * 2 + 1] * x);
       }
  };

  struct Estimate{
        KOKKOS_INLINE_FUNCTION double operator()(double u, unsigned int d, double const *params_p) const {
            return u + params_p[d] * 0.;
	}
  };
};

template <typename T, unsigned Dim>
class AdjointManager : public AlpineManager<T, Dim> {
public:
    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t = FieldContainer<T, Dim>;
    using FieldSolver_t= FieldSolver<T, Dim>;
    using LoadBalancer_t= LoadBalancer<T, Dim>;

    AdjointManager(size_type totalP_, int nt_, Vector_t<int, Dim> &nr_,
                       double lbt_, std::string& solver_, std::string& stepMethod_)
        : AlpineManager<T, Dim>(totalP_, nt_, nr_, lbt_, solver_, stepMethod_){}

    ~AdjointManager(){}

    int optit;
    double fD;
    int nparams_m;
    Vector_t<T, 9> params_m;
    Vector_t<T, 9> dparams_m;

    void pre_run() override {
        Inform m("Pre Run");

        if (this->solver_m == "OPEN") {
            throw IpplException("LandauDamping", "Open boundaries solver incompatible with this simulation!");
        }

        for (unsigned i = 0; i < Dim; i++) {
            this->domain_m[i] = ippl::Index(this->nr_m[i]);
        }

        this->decomp_m.fill(true);
        this->kw_m    = 0.5;
        this->alpha_m = 0.05;
        //this->rmin_m  =  0.0;//-2 * pi / this->kw_m;
        //this->rmax_m  =   2 * pi / this->kw_m;
        this->rmin_m  = - pi / this->kw_m;
        this->rmax_m  =   pi / this->kw_m;

        this->hr_m = (this->rmax_m-this->rmin_m) / this->nr_m;
        // Q = -\int\int f dx dv
        this->Q_m =  std::reduce(this->rmax_m.begin(), this->rmax_m.end(), -1., std::multiplies<double>());
        this->Q_m -= std::reduce(this->rmin_m.begin(), this->rmin_m.end(), -1., std::multiplies<double>());
        this->origin_m = this->rmin_m;
        this->dt_m     = std::min(.05, 0.5 * *std::min_element(this->hr_m.begin(), this->hr_m.end()));
        this->it_m     = 0;
        this->time_m   = 0.0;

        m << "Discretization:" << endl
          << "nt " << this->nt_m << " Np= " << this->totalP_m << " grid = " << this->nr_m << endl;

        this->isAllPeriodic_m = true;

        this->setFieldContainer( std::make_shared<FieldContainer_t>( this->hr_m, this->rmin_m, this->rmax_m, this->decomp_m, this->domain_m, this->origin_m, this->isAllPeriodic_m) );

        this->setParticleContainer( std::make_shared<ParticleContainer_t>( this->fcontainer_m->getMesh(), this->fcontainer_m->getFL()) );

        this->fcontainer_m->initializeFields(this->solver_m);

        this->setFieldSolver( std::make_shared<FieldSolver_t>( this->solver_m, &this->fcontainer_m->getRho(), &this->fcontainer_m->getE(), &this->fcontainer_m->getPhi()) );

        this->fsolver_m->initSolver();

        this->setLoadBalancer( std::make_shared<LoadBalancer_t>( this->lbt_m, this->fcontainer_m, this->pcontainer_m, this->fsolver_m) );

        initializeParticles();

        static IpplTimings::TimerRef DummySolveTimer  = IpplTimings::getTimer("solveWarmup");
        IpplTimings::startTimer(DummySolveTimer);

        this->fcontainer_m->getRho() = 0.0;

        this->fsolver_m->runSolver();

        IpplTimings::stopTimer(DummySolveTimer);

        this->par2grid();

        static IpplTimings::TimerRef SolveTimer = IpplTimings::getTimer("solve");
        IpplTimings::startTimer(SolveTimer);

        this->fsolver_m->runSolver();

        IpplTimings::stopTimer(SolveTimer);

        this->grid2par();

        this->externalForceField();

        //this->fcontainer_m->Et_m[0] = this->fcontainer_m->getE();

        m << "Done";
    }

    void initializeParticles(){
        Inform m("Initialize Particles");
        static IpplTimings::TimerRef particleCreation = IpplTimings::getTimer("particlesCreation");
        IpplTimings::startTimer(particleCreation);

        size_type totalP = this->totalP_m;
        int seed           = 42;
        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(seed + 100 * ippl::Comm->rank()));

        size_type nlocal = this->totalP_m;
        this->pcontainer_m->create(nlocal);

        view_type* R = &(this->pcontainer_m->R.getView());

        view_type* P = &(this->pcontainer_m->P.getView());

        double mu[Dim];
        double sd[Dim];
        for(unsigned int i=0; i<Dim; i++){
            mu[i] = 0.0;
            sd[i] = 1.0;
        }

        //// also sample gaussian for R
        Kokkos::parallel_for(nlocal, ippl::random::randn<double, Dim>(*R, rand_pool64, mu, sd));
        Kokkos::fence();
        ippl::Comm->barrier();

        Kokkos::parallel_for(nlocal, ippl::random::randn<double, Dim>(*P, rand_pool64, mu, sd));
        Kokkos::fence();
        ippl::Comm->barrier();

        IpplTimings::stopTimer(particleCreation);

        this->pcontainer_m->q = this->Q_m/totalP;
        m << "particles created and initial conditions assigned " << endl;
    }
    void externalForceField(){

        auto &F = this->pcontainer_m->F.getView();
        auto &R = this->pcontainer_m->R.getView();

        Vector_t params = this->params_m;
        Vector_t dparams = this->dparams_m;

        // compute F given external force field
        Kokkos::parallel_for(this->pcontainer_m->getLocalNum(), KOKKOS_LAMBDA(const size_t i) {
            for(unsigned int d=0; d<Dim; d++){
                F(i)[d] = ExternalForceField<Dim>()(R(i), d, params);
            }
	});

	Kokkos::fence();
        ippl::Comm->barrier();
    }

    void advance() override {
        if (this->stepMethod_m == "LeapFrog") {
            LeapFrogStep();
        }
	else{
            throw IpplException(TestName, "Step method is not set/recognized!");
        }
    }

    void LeapFrogStep(){
        // LeapFrog time stepping https://en.wikipedia.org/wiki/Leapfrog_integration
        // Here, we assume a constant charge-to-mass ratio of -1 for
        // all the particles hence eliminating the need to store mass as
        // an attribute
        static IpplTimings::TimerRef PTimer           = IpplTimings::getTimer("pushVelocity");
        static IpplTimings::TimerRef RTimer           = IpplTimings::getTimer("pushPosition");
        static IpplTimings::TimerRef updateTimer      = IpplTimings::getTimer("update");
        static IpplTimings::TimerRef domainDecomposition = IpplTimings::getTimer("loadBalance");
        static IpplTimings::TimerRef SolveTimer       = IpplTimings::getTimer("solve");

        double dt                               = this->dt_m;
        int it                  = this->it_m;
        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        std::shared_ptr<FieldContainer_t> fc    = this->fcontainer_m;

        // external force on particle
        externalForceField();

        IpplTimings::startTimer(PTimer);

	//if(this->dt_m<0){
        //    fc->setE( fc->Et_m[it] );
        //    this->grid2par();
        //    std::cout << "read  E at it: " << it-0.5 << " indexed " << it << std::endl << std::endl;
        //}

        pc->P = pc->P - 0.5 * dt * ( pc->E + pc->F );
        IpplTimings::stopTimer(PTimer);

        // drift
        IpplTimings::startTimer(RTimer);
        pc->R = pc->R + dt * pc->P;
        IpplTimings::stopTimer(RTimer);

        // Since the particles have moved spatially update them to correct processors
        IpplTimings::startTimer(updateTimer);
        pc->update();
        IpplTimings::stopTimer(updateTimer);

        size_type totalP        = this->totalP_m;
        bool isFirstRepartition = false;
        if (this->loadbalancer_m->balance(totalP, it + 1)) {
                IpplTimings::startTimer(domainDecomposition);
                auto* mesh = &fc->getRho().get_mesh();
                auto* FL = &fc->getFL();
                this->loadbalancer_m->repartition(FL, mesh, isFirstRepartition);
                IpplTimings::stopTimer(domainDecomposition);
        }

	// scatter the charge onto the underlying grid
        this->par2grid();

        delFdelparm();

        // Field solve
        IpplTimings::startTimer(SolveTimer);
        //if(this->dt_m>0){
            this->fsolver_m->runSolver();
        //    fc->Et_m[it+1] = fc->getE();
        //    std::cout << "store E at it: " << it+0.5 << " indexed "<< it+1 << std::endl << std::endl;
        //}
        //else{
        //    fc->setE( fc->Et_m[it-1] );
        //    std::cout << "read  E at it: " << it-1 << " indexed " << it-1 << std::endl << std::endl;
        //}
        IpplTimings::stopTimer(SolveTimer);

        // gather E field
        this->grid2par();

        // recomputed external force on particle since they have moved
        this->externalForceField();

        // kick
        IpplTimings::startTimer(PTimer);
        pc->P = pc->P - 0.5 * dt * ( pc->E + pc->F );
        IpplTimings::stopTimer(PTimer);
    }

    // TODO: Implement the perturbation of particles at the final time according to the merit, to be called at the end of forward run
    void PerturbParticles(double sgn){
        auto &P = this->pcontainer_m->P.getView();
        auto &R = this->pcontainer_m->R.getView();

        // Following perturbs particles given moments
        double gD = costFunction();
        Kokkos::parallel_for(this->pcontainer_m->getLocalNum(), KOKKOS_LAMBDA(const size_t i) {
            for (unsigned int j = 0; j < 3; ++j) {
                double tempR = R(i)[j];
                double tempP = P(i)[j];
                P(i)[j] += - sgn * 2.0 * gD * tempR * 1.e-4;
                R(i)[j] +=  sgn * 2.0 * gD * tempP * 1.e-4;
            }
        });

        /*
        // Following uses gradient flow to perturb final time distribution
        // compute mean of R and P
        Vector_t<T, 3> Rmean = 0.0;
        Vector_t<T, 3> Pmean = 0.0;

        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(1 + 100 * ippl::Comm->rank()));

        double dt = 1.e-9;
        double Rtemp, Ptemp;
        for (unsigned int iter = 0; iter < 10; ++iter) {

          for (unsigned int j = 0; j < 3; ++j){
            Rtemp = 0.0;
            Ptemp = 0.0;
            Kokkos::parallel_reduce(
                "Ptemp", this->pcontainer_m->getLocalNum(),
                KOKKOS_LAMBDA(const int i, double& valL) {
                    valL        = P(i)[j];
                 },
                 Kokkos::Sum<double>(Ptemp));

            Kokkos::parallel_reduce(
                "Rtemp", this->pcontainer_m->getLocalNum(),
                KOKKOS_LAMBDA(const int i, double& valL) {
                    valL        = R(i)[j];
                 },
                 Kokkos::Sum<double>(Rtemp));

            ippl::Comm->reduce(Ptemp, Pmean[j], 1, std::plus<double>());
            ippl::Comm->reduce(Rtemp, Rmean[j], 1, std::plus<double>());
            Rmean[j] = Rmean[j]/this->totalP_m;
            Pmean[j] = Pmean[j]/this->totalP_m;
          }

          Kokkos::parallel_for(this->pcontainer_m->getLocalNum(), KOKKOS_LAMBDA(const size_t i) {
            for (unsigned int j = 0; j < 3; ++j) {
                auto generator = rand_pool64.get_state();
                double dW = generator.drand(0., 1.);
                rand_pool64.free_state(generator);
                P(i)[j] += -sgn*( P(i)[j]-Pmean[j] )*dt + Kokkos::sqrt(2.*dt)*dW;

                dW = generator.drand(0., 1.);
                rand_pool64.free_state(generator);
                R(i)[j] += -sgn*( R(i)[j]-Rmean[j] )*dt + Kokkos::sqrt(2.*dt)*dW;
            }
          });
        }
        */
    }

    // computation of dC/dalpha
    void delFdelparm(){
        //Inform m("delF: ");

        Vector_t params = this->params_m;
        unsigned int nparams = this->nparams_m;

        // Following computes dF/dparam using grid
        auto *FL = &this->fcontainer_m->getFL();

        Field_t<Dim> *rho     = &this->fcontainer_m->getRho();

        auto rhoview         = this->fcontainer_m->getRho().getView();

        Vector_t<double, Dim> hr     = this->hr_m;
        Vector_t<double, Dim> origin = this->origin_m;

        const int nghost = (*rho).getNghost();
        const ippl::NDIndex<Dim>& lDom = FL->getLocalNDIndex();
        using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
        for (unsigned idparams = 0; idparams < nparams; ++idparams) {
            T temp = 0.0;
            ippl::parallel_reduce(
                "rho*dphi_dalpha0 reduce", (*rho).getFieldRangePolicy(),
                KOKKOS_LAMBDA(const index_array_type& args, T& valL) {
                    Vector_t<double, Dim> xvec =
                        (args + lDom.first() - nghost + 0.5) * hr + origin;
                    T myVal = ippl::apply(rhoview, args) * dphi_dparam<Dim>()(xvec, idparams);
                    valL += myVal;
              }, Kokkos::Sum<T>(temp));
            Kokkos::fence();

            T globaltemp          = 0.0;
            ippl::Comm->reduce(temp, globaltemp, 1, std::plus<double>());
            ippl::Comm->barrier();

            this->dparams_m[idparams] += this->dt_m*globaltemp*this->hr_m[0]*this->hr_m[1]*this->hr_m[2];
        }

        /*
        // Following computes dC/dparam using particles, it's converging
        auto R = this->pcontainer_m->R.getView();
        for (unsigned idparams = 0; idparams < nparams; ++idparams) {
             T temp = 0.0;
             Kokkos::parallel_reduce(
                "Dl", this->pcontainer_m->getLocalNum(),
                KOKKOS_LAMBDA(const int i, double& valL) {
                   valL += dphi_dparam<Dim>()( R(i), idparams );
                },
                Kokkos::Sum<double>(temp));
             this->dparams_m[idparams] += -this->dt_m*temp;
        }
        */
    }

    void post_step() override {

        int sign = (this->dt_m >= 0) ? 1 : -1;

        this->time_m += this->dt_m;

        this->it_m += sign;

        this->dump();

        Inform m("Post-step:");
        m << "Finished time step: " << this->it_m << " time: " << this->time_m << endl;
    }

   double costFunction(){
        auto Pview = this->pcontainer_m->P.getView();
        auto Rview = this->pcontainer_m->R.getView();
        double D   = 0.0;
        double gD  = 0.0;

        Kokkos::parallel_reduce(
            "Dl", this->pcontainer_m->getLocalNum(),
            KOKKOS_LAMBDA(const int i, double& valL) {
                double myVal = dot(Pview(i), Pview(i)).apply();
                myVal       += dot(Rview(i), Rview(i)).apply();
                valL        += myVal;
            },
            Kokkos::Sum<double>(D));
        ippl::Comm->reduce(D, gD, 1, std::plus<double>());
        gD = gD/this->totalP_m;
        gD = (gD - this->fD);// global D - final time D
        return gD;
   }

   void runOptimization(int optIt){
        Inform m("Optimization");

        this->fD = 6.;// second moment of 6 variables [R, P], where each is a normal distribution
        this->nparams_m = 9;
        this->params_m = 1.e-16;
        this->dparams_m = 0.0;

        Vector_t<T, 9> params0;
        Vector_t<T, 9> params1, params2;

        this->pre_run();

        double dt0 = this->dt_m;
        double gD;
        //double L1, L;
        for(optit = 0; optit < optIt; optit++) {
             m << "iter " << optit << endl << endl;

             params0 = this->params_m;

             ///// first run
             this->dparams_m = 0.0;
             this->pre_run();
             // Forward run, umpurturbed potential
             this->dt_m = dt0;
             this->dump();
             this->run(this->nt_m); // compute unpertubed terms of df/dalpha during forward simulation
             gD = costFunction();
             PerturbParticles(1.);
             // Backward
             this->dt_m = -dt0;
             this->dump();
             this->run(this->nt_m); // compute pertubed terms of df/dalpha during backward simulation
             this->params_m = params0 - 0.1*this->dparams_m; // 0.001

             m << this->params_m << endl;
             /*
             ///// second run
             this->dparams_m = 0.0;
             this->pre_run();
             // Forward run, umpurturbed potential
             this->dt_m = dt0;
             this->dump();
             this->run(this->nt_m); // compute unpertubed terms of df/dalpha during forward simulation
             PerturbParticles(-1.);
             // Backward
             this->dt_m = -this->dt_m;
             this->dump();
             this->run(this->nt_m); // compute pertubed terms of df/dalpha during backward simulation
             params2 = params0 + 0.001*this->dparams_m;
             */
             // test each run
             //this->params_m = params1;
             //this->pre_run();
             // Forward run, umpurturbed potential
             //this->dt_m = dt0;
             //this->run(this->nt_m); // compute unpertubed terms of df/dalpha during forward simulation
             //L1 = fabs( costFunction() );

             //this->params_m = params2;
             //this->pre_run();
             // Forward run, umpurturbed potential
             //this->dt_m = dt0;
             //this->run(this->nt_m); // compute unpertubed terms of df/dalpha during forward simulation
             //L2 = fabs( costFunction() );
             //L = L2;
             //if(L1<L2){
             //    this->params_m = params1;
             //    L = L1;
             //}

             std::stringstream fname;
             fname << "data/L.csv";
             Inform tcsvout(NULL, fname.str().c_str(), Inform::APPEND);
             tcsvout.precision(10);
             tcsvout.setf(std::ios::scientific, std::ios::floatfield);
             tcsvout << gD << endl;
             ippl::Comm->barrier();
         }
   }

    void dump() override {
        static IpplTimings::TimerRef dumpDataTimer = IpplTimings::getTimer("dumpData");
        IpplTimings::startTimer(dumpDataTimer);

        //if(this->it_m==this->nt_m && ippl::Comm->rank() == 0 && this->dt_m > 0){
        std::cout << "---------       it: " << this->it_m << "  time: " << this->time_m << std::endl << std::endl;
        if( (this->it_m==this->nt_m || this->it_m==0 || this->it_m==1) && ippl::Comm->rank() == 0 ){
            auto &P = this->pcontainer_m->P.getView();
            auto &R = this->pcontainer_m->R.getView();

            auto hostPView = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), P);
            auto hostRView = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), R);

            std::stringstream fname;
            fname << "data/particles_";
            fname << optit;
            fname << "_" << this->it_m;
            fname << "_" << this->dt_m;
            fname << ".csv";
            Inform csvout(NULL, fname.str().c_str(), Inform::OVERWRITE);
            csvout.precision(4);
            csvout.setf(std::ios::scientific, std::ios::floatfield);

            for (size_t i = 0; i < this->totalP_m; ++i) {
               for (size_t j = 0; j < Dim; ++j){
                   csvout << hostRView(i)[j] << " ";
              }
               for (size_t j = 0; j < Dim; ++j){
                   csvout << hostPView(i)[j] << " ";
              }
              csvout << endl;
            }
            ippl::Comm->barrier();
        }
/*
        if( ippl::Comm->rank() == 0 && this->dt_m > 0){
          auto Pview = this->pcontainer_m->P.getView();
          auto Rview = this->pcontainer_m->R.getView();
          double D   = 0.0;
          double gD  = 0.0;

          Kokkos::parallel_reduce(
            "Dl", this->pcontainer_m->getLocalNum(),
            KOKKOS_LAMBDA(const int i, double& valL) {
                double myVal = dot(Pview(i), Pview(i)).apply();
                myVal       += dot(Rview(i), Rview(i)).apply();
                valL        += myVal;
            },
            Kokkos::Sum<double>(D));
          ippl::Comm->reduce(D, gD, 1, std::plus<double>());
          gD = gD/this->totalP_m;

          std::stringstream fname;
          fname << "data/D_";
          fname << optit;
          fname << ".csv";
          Inform tcsvout(NULL, fname.str().c_str(), Inform::APPEND);
          tcsvout.precision(10);
          tcsvout.setf(std::ios::scientific, std::ios::floatfield);
          tcsvout << gD << endl;
          ippl::Comm->barrier();

          gD = (gD - this->fD);// global D - final time D

          Inform m("Estimate moment:");
          m << gD << " " << D/this->totalP_m << endl;

          if(this->it_m==this->nt_m){
            Inform csvout(NULL, "data/D.csv", Inform::APPEND);
            csvout.precision(10);
            csvout.setf(std::ios::scientific, std::ios::floatfield);
            csvout << gD << " " << D/this->totalP_m << " " << this->fD << endl;
            ippl::Comm->barrier();

            Inform m("Estimate moment:");
            m << " -------    res: " << gD << " " << D/this->totalP_m << endl;
          }
        }
*/
        IpplTimings::stopTimer(dumpDataTimer);
   }
};

#endif
