#ifndef IPPL_VORTEX_IN_CELL_MANAGER_H
#define IPPL_VORTEX_IN_CELL_MANAGER_H

#include <memory>

#include "AlvineManager.h"
#include "FieldContainer.hpp"
#include "FieldSolver.hpp"
#include "LoadBalancer.hpp"
#include "Manager/BaseManager.h"
#include "ParticleContainer.hpp"
#include "Random/Distribution.h"
#include "Random/InverseTransformSampling.h"
#include "Random/NormalDistribution.h"
#include "Random/Randn.h"

using view_type = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;


template <typename T, unsigned Dim>
class VortexInCellManager : public AlvineManager<T, Dim> {
public:
    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t    = FieldContainer<T, Dim>;
    using FieldSolver_t       = FieldSolver<T, Dim>; 
    using LoadBalancer_t      = LoadBalancer<T, Dim>;

    VortexInCellManager(unsigned nt_, Vector_t<int, Dim>& nr_, 
        Vector_t<double, Dim> rmin_ = 0.0,
        Vector_t<double, Dim> rmax_ = 10.0,
        Vector_t<double, Dim> origin_ = 0.0)
        : AlvineManager<T, Dim>(nt_, nr_) {
            this->rmin_m = rmin_;
            this->rmax_m = rmax_;
            this->origin_m = origin_;
        }

    ~VortexInCellManager() {}

    void pre_run() override {

      Inform csvout(NULL, "particles.csv", Inform::OVERWRITE);
      csvout.precision(16);
      csvout.setf(std::ios::scientific, std::ios::floatfield);

      if constexpr (Dim == 2) {
          csvout << "time,index,pos_x,pos_y" << endl;
      } else {
          csvout << "time,index,pos_x,pos_y,pos_z" << endl;
      }

      for (unsigned i = 0; i < Dim; i++) {
          this->domain_m[i] = ippl::Index(this->nr_m[i]);
      }


      Vector_t<double, Dim> dr = this->rmax_m - this->rmin_m;

      this->hr_m = dr / this->nr_m;

      this->dt_m = std::min(0.05, 0.5 * ( *std::min_element(this->hr_m.begin(), this->hr_m.end()) ) );

      this->it_m = 0;
      this->time_m = 0.0;

      this->np_m = 1;//this->nr_m[0];

      this->decomp_m.fill(true);
      this->isAllPeriodic_m = true;

      this->setFieldContainer(std::make_shared<FieldContainer_t>(
            this->hr_m, this->rmin_m, this->rmax_m, this->decomp_m, this->domain_m, this->origin_m,
            this->isAllPeriodic_m));

      this->setParticleContainer(std::make_shared<ParticleContainer_t>(
            this->fcontainer_m->getMesh(), this->fcontainer_m->getFL()));
        
      this->fcontainer_m->initializeFields();

      initializeParticles();


    }

    void initializeParticles() {

      std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;

      this->pcontainer_m->create(this->np_m);

      //BEGIN TODO: Make proper distribution ideally by templating the class with a distribution struct
      std::mt19937_64 eng;
      std::uniform_real_distribution<double> unif(0, 1);
      typename ParticleContainer_t::particle_position_type::HostMirror P_host = pc->P.getHostMirror();
      typename ParticleContainer_t::particle_position_type::HostMirror R_host = pc->R.getHostMirror();

      for (unsigned i = 0; i < this->np_m; i++) {
        ippl::Vector<double, Dim> p; 
        ippl::Vector<double, Dim> r;

        for (unsigned d = 0; d < Dim; d++) {
          p(d) = unif(eng) - 0.5;
          r(d) = unif(eng);
        }

        P_host(i) = p * 3;
        R_host(i) = r * (this->rmax_m - this->rmin_m) + this->origin_m;
      }
      Kokkos::deep_copy(pc->P.getView(), P_host);
      Kokkos::deep_copy(pc->R.getView(), R_host);
      //END TODO


    }
  

    void advance() override {

      std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
      pc->R = pc->R + pc->P * this->dt_m;
      pc->update();

      
    }

    void LeapFrogStep() {
    }

    void dump() override {
      std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;

      Inform csvout(NULL, "particles.csv", Inform::APPEND);

      for (unsigned i = 0; i < this->np_m; i++) {
        csvout << this->it_m << "," << i; 
        for (unsigned d = 0; d < Dim; d++) {
          csvout << "," << pc->R(i)[d];
        }
        csvout << endl;
      }
       
    }

};
#endif
