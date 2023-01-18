#ifndef RBH_H
#define RBH_H

#include "ChargedParticles.hpp"
#include <string>
#include <vector>
#include <iostream>
#include <cmath>
#include <set>
#include <chrono>
#include <string>
#include <Kokkos_Random.hpp>
#include <random>

const char* TestName = "RosenbluthTest";

// ./rosenbluth N VTH VMAX
// the rosenbluth potentials are clearly defined in the case the prbability distribution in
// velocity space is defined as the maxwellian

int main(int argc, char *argv[]){
  Ippl ippl(argc, argv);

  Inform msg("rosenbluth tests ");

  const double   n = std::atof(argv[1]);
  const double vth = std::atof(argv[2]);
  const double fct = std::atof(argv[3]);

  const double vth2 = vth*vth;
  const double w2 = sqrt(2.0);

  std::string ALGO = "HOCKNEY";    // VICO

  double VMAX = fct*vth;
  std::string path = "./rb-alt/" + std::to_string(int(n)) + std::to_string(int(vth)) + std::to_string(int(fct)) +".csv";

  std::ofstream fout;
  fout.open(path);
  fout
    << "NV" << ","
    << "Herror_avg2" << "," 
    << "Gerror_avg2" << "," 
    << "HerrorL1" << "," 
    << "GerrorL1" << ","  
    << "HerrorL2" << "," 
    << "GerrorL2" << ","  
    << std::endl;

  for(int NV = 2; NV <=256; NV*=2){

    Vector<int, Dim> nv_mv   = {NV,NV,NV};
    Vector_t         hv_mv   = {2*VMAX/NV,  2*VMAX/NV,  2*VMAX/NV};
    const Vector_t   vmin_mv = {-VMAX, -VMAX, -VMAX};
    const Vector_t   vmax_mv = { VMAX,  VMAX,  VMAX};

    ippl::NDIndex<Dim> domain_v;
    for (unsigned i = 0; i< Dim; i++) {
        domain_v[i] = ippl::Index(NV);
    }
    
    ippl::e_dim_tag decomp_v[Dim];
    for (unsigned d = 0; d < Dim; ++d) {
        decomp_v[d] = ippl::PARALLEL;
    }

    bool isVallPeriodic = false;

    Mesh_t        mesh_v(domain_v, hv_mv, Vector_t({0.0, 0.0, 0.0}));
    FieldLayout_t FL_v(domain_v, decomp_v, isVallPeriodic);
    PLayout_t     PL_v(FL_v, mesh_v);

    Field_t fv1;
    Field_t fv2;
    
    Field_t H;
    Field_t H_sol;

    Field_t G;
    Field_t G_sol;
    
    fv1.initialize(mesh_v, FL_v);
    fv2.initialize(mesh_v, FL_v);
    H.initialize(mesh_v, FL_v);
    H_sol.initialize(mesh_v, FL_v);
    G.initialize(mesh_v, FL_v);
    G_sol.initialize(mesh_v, FL_v);

    std::shared_ptr<VSolver_t> solver_RB1;
    std::shared_ptr<VSolver_t> solver_RB2;
    ippl::ParameterList sp;

    sp.add("use_pencils", true);
    sp.add("comm", ippl::p2p_pl);  
    sp.add("use_reorder", false); 
    sp.add("use_heffte_defaults", false);  
    sp.add("use_gpu_aware", true);  
    sp.add("r2c_direction", 0); 
    sp.add("output_type", VSolver_t::SOL);

    solver_RB1 = std::make_shared<VSolver_t>(fv1, sp, ALGO);
    solver_RB2 = std::make_shared<VSolver_t>(fv2, sp, ALGO);

    mesh_v.setOrigin(vmin_mv);

    auto fvView1    = fv1.getView();
    auto fvView2    = fv2.getView();
    auto HsolView   = H_sol.getView();
    auto GsolView   = G_sol.getView();
    auto HView      = H.getView();
    auto GView      = G.getView();

    const ippl::NDIndex<3>& lDom   = FL_v.getLocalNDIndex();
    const int               nghost = fv1.getNghost();

    Kokkos::parallel_for("set values1",
			 Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
								{fvView1.extent(0),
								 fvView1.extent(1),
								 fvView1.extent(2)}),
			 KOKKOS_LAMBDA(const int i, const int j, const int k){

			   //local to global index conversion
			   const int ig = i + lDom[0].first() - nghost;
			   const int jg = j + lDom[1].first() - nghost;
			   const int kg = k + lDom[2].first() - nghost;

			   double vx = vmin_mv[0] + (ig + 0.5)*hv_mv[0];
			   double vy = vmin_mv[1] + (jg + 0.5)*hv_mv[1];
			   double vz = vmin_mv[2] + (kg + 0.5)*hv_mv[2];
			   double vnorm2  = vx*vx+vy*vy+vz*vz;

			   fvView1(i,j,k)   = n / pow(2*M_PI*vth2, 1.5)* exp(-vnorm2/(2*vth2));
			 }
			 );
    Kokkos::fence();

    Kokkos::parallel_for("set values2",
			 Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                                            {   HsolView.extent(0),
								HsolView.extent(1),
								HsolView.extent(2)}),
			 KOKKOS_LAMBDA(const int i, const int j, const int k){

			   //local to global index conversion
			   const int ig = i + lDom[0].first() - nghost;
			   const int jg = j + lDom[1].first() - nghost;
			   const int kg = k + lDom[2].first() - nghost;

			   double vx = vmin_mv[0] + (ig + 0.5)*hv_mv[0];
			   double vy = vmin_mv[1] + (jg + 0.5)*hv_mv[1];
			   double vz = vmin_mv[2] + (kg + 0.5)*hv_mv[2];
			   double vnorm2  = vx*vx+vy*vy+vz*vz;
			   double vnorm = sqrt(vnorm2); 

			   fvView2(i,j,k) = 2*n / vnorm * erf(vnorm/(w2*vth));
			   HsolView(i,j,k) = n / vnorm * erf(vnorm/(w2*vth));
			 }
			 );
    Kokkos::fence();

    Kokkos::parallel_for("set values3",
			 Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
								{GsolView.extent(0),
								 GsolView.extent(1),
								 GsolView.extent(2)}),
			 KOKKOS_LAMBDA(const int i, const int j, const int k){

			   //local to global index conversion
			   const int ig = i + lDom[0].first() - nghost;
			   const int jg = j + lDom[1].first() - nghost;
			   const int kg = k + lDom[2].first() - nghost;

			   double vx = vmin_mv[0] + (ig + 0.5)*hv_mv[0];
			   double vy = vmin_mv[1] + (jg + 0.5)*hv_mv[1];
			   double vz = vmin_mv[2] + (kg + 0.5)*hv_mv[2];
			   double vnorm2  = vx*vx+vy*vy+vz*vz;
			   double vnorm = sqrt(vnorm2); 
            
			   GsolView(i,j,k) = w2 * n * vth *( exp(-vnorm2/(2.0*vth2))/sqrt(M_PI)  +  erf(vnorm/(w2*vth))  * (vth/(w2*vnorm)+vnorm/(w2*vth)));

                    }
    );
    Kokkos::fence();
    
   //////////////////////////////////////////////////////////////////////////////

    fv1 = -4 * M_PI * fv1;
    mesh_v.setOrigin({0, 0, 0});
    solver_RB1->solve();
    mesh_v.setOrigin(vmin_mv);
    Kokkos::deep_copy(HView, fvView1); // H = fv;

    // fv2 = 2*fv1; //works


    mesh_v.setOrigin({0,0,0});
    solver_RB2->solve();
    mesh_v.setOrigin(vmin_mv);
    Kokkos::deep_copy(GView, fvView2);

    double Gerror_avg2, Herror_avg2;
    double GerrorL1, HerrorL1;
    double GerrorL2, HerrorL2;

    H = H - H_sol;
    G = G - G_sol;
    
    HerrorL1 = norm(H, 1)/norm(H_sol, 1);
    GerrorL1 = norm(G, 1)/norm(G_sol, 1);
    HerrorL2 = norm(H)/norm(H_sol);
    GerrorL2 = norm(G)/norm(G_sol);
    
    H = H/H_sol;
    G = G/G_sol;

    Herror_avg2 = pow(norm(H), 2) /(NV*NV*NV);
    Gerror_avg2 = pow(norm(G), 2) /(NV*NV*NV);

    std::cout << path+" - MESH NV: " << NV << std::endl;

    fout    << NV << ","
            << Herror_avg2 << "," 
            << Gerror_avg2 << "," 
            << HerrorL1 << "," 
            << GerrorL1 << ","  
            << HerrorL2 << "," 
            << GerrorL2 << ","  
            << std::endl;

    
  }
    return 0;
}
#endif


