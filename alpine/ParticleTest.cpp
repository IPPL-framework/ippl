constexpr unsigned Dim = 2;
using T                = double;
const char* TestName   = "ParticleTest";

#include <random>

#include "Ippl.h"

#include "datatypes.h"

#include "VICParticleContainer.hpp"
#include "FieldContainer.hpp"


using view_type = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;


int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
      using ParticleContainer_t = VICParticleContainer<T, Dim>;
      using FieldContainer_t = FieldContainer<T, Dim>;

      Vector_t<int, Dim> nr = 128;
      Vector_t<double, Dim> kw_m = 0.5;
      Vector_t<double, Dim> rmin_m = 0.0;
      Vector_t<double, Dim> rmax_m = 2 * pi / kw_m;
      Vector_t<double, Dim> hr_m;
      Vector_t<double, Dim> origin_m;
      origin_m = rmin_m;
      bool isAllPeriodic_m = true;

      ippl::NDIndex<Dim> domain_m;

      for (unsigned i = 0; i < Dim; i++) {
          domain_m[i] = ippl::Index(nr[i]);
      }

      std::array<bool, Dim> decomp_m;
      decomp_m.fill(true);

      hr_m = rmax_m / nr;

      FieldContainer_t fc(hr_m, rmin_m, rmax_m, decomp_m, domain_m, origin_m, isAllPeriodic_m);
      ParticleContainer_t pc(fc.getMesh(), fc.getFL());

      int n = 100;
      pc.create(n);
      pc.R.print();


      std::mt19937_64 eng;
      std::uniform_real_distribution<double> unif(0, 1);
      typename ParticleContainer_t::particle_position_type::HostMirror P_host = pc.P.getHostMirror();
      typename ParticleContainer_t::particle_position_type::HostMirror R_host = pc.R.getHostMirror();

      for (int i = 0; i < n; i++) {
        ippl::Vector<double, 2> p = {unif(eng) - 0.5, unif(eng) - 0.5};
        ippl::Vector<double, 2> r = {unif(eng) - 0.5, unif(eng) - 0.5};
        P_host(i) = p;
        R_host(i) = r * 12;
      }
      Kokkos::deep_copy(pc.P.getView(), P_host);
      Kokkos::deep_copy(pc.R.getView(), R_host);

      pc.update();
      unsigned T = 100;
      double dt = 0.1;
      


      Inform csvout(NULL, "test.csv", Inform::APPEND);
      csvout.precision(16);
      csvout.setf(std::ios::scientific, std::ios::floatfield);
      csvout << "time,index,pos_x,pos_y" << endl;

      for (unsigned t = 0; t < T; t++) {
        pc.R = pc.R + pc.P * dt; 
        pc.update();

        for (unsigned i = 0; i < pc.R.size(); i++){
          csvout << t << "," << i << "," << pc.R(i)[0] << "," << pc.R(i)[1] << endl;
        }
      } 



//      FieldContainer_t fc()





    }
    ippl::finalize();

    return 0;
}
