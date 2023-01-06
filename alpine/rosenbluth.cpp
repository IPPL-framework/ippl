#ifndef RBH_H
#define RBH_H


// ./rosenbluth N VTH VMAX

#include "ChargedParticles.hpp"
#include <string>
#include <vector>
#include <iostream>
#include <cmath>
#include <set>
#include <chrono>

#include <string>

#include<Kokkos_Random.hpp>

#include <random>


const char* TestName = "RosenbluthTest";


// the rosenbluth potentials are clearly defined in the case the prbability distribution in
// velocity space is defined as the maxwellian

int main(int argc, char *argv[]){
    Ippl ippl(argc, argv);

    Inform msg("Langevin");

    const double n = std::atof(argv[1]);
    const double vth = std::atof(argv[2]);
    double VMAX = std::atof(argv[3]);

for(int NV = 2; NV <=128; NV*=2){

    Vector<int, Dim> nv_mv = {NV,NV,NV};
    Vector_t hv_mv { 2*VMAX/NV,  2*VMAX/NV,  2*VMAX/NV};
    const Vector_t vmin_mv = {-VMAX, -VMAX, -VMAX};
    const Vector_t vmax_mv = { VMAX,  VMAX,  VMAX};


    ippl::NDIndex<Dim> domain_v;
    for (unsigned i = 0; i< Dim; i++) {
        domain_v[i] = ippl::Index(NV);
    }
    ippl::e_dim_tag decomp_v[Dim];
    for (unsigned d = 0; d < Dim; ++d) {
        decomp_v[d] = ippl::PARALLEL;
    }
    bool isVallPeriodic = false;

    Mesh_t          mesh_v(domain_v, hv_mv, Vector_t({0.0, 0.0, 0.0}));
    FieldLayout_t   FL_v(domain_v, decomp_v, isVallPeriodic);
    PLayout_t       PL_v(FL_v, mesh_v);
    Field_t fv;

    fv.initialize(mesh_v, FL_v);


    std::shared_ptr<VSolver_t> solver_RB;
    ippl::ParameterList sp;
    sp.add("use_pencils", true);
    sp.add("comm", ippl::p2p_pl);  
    sp.add("use_reorder", false); 
    sp.add("use_heffte_defaults", false);  
    sp.add("use_gpu_aware", true);  
    sp.add("r2c_direction", 0); 
    sp.add("output_type", VSolver_t::SOL);
    solver_RB = std::make_shared<VSolver_t>(fv, sp, "HOCKNEY");

    mesh_v.setOrigin(vmin_mv);

    Field_t H;
    Field_t H_sol;
    Field_t G;
    Field_t G_sol;
    H.initialize(mesh_v, FL_v);
    H_sol.initialize(mesh_v, FL_v);
    G.initialize(mesh_v, FL_v);
    G_sol.initialize(mesh_v, FL_v);

    auto fvView     =   fv.getView();
    auto HsolView   = H_sol.getView();
    auto GsolView   = G_sol.getView();
    auto HView      =    H.getView();
    auto GView      =    G.getView();

    const double vth2 = vth*vth;
    const double w2 = sqrt(2);

    const ippl::NDIndex<3>& lDom = FL_v.getLocalNDIndex();
    const int nghost = fv.getNghost();
    Kokkos::parallel_for("set values",
                    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                                            {   fvView.extent(0),
							                                    fvView.extent(1),
							                                    fvView.extent(2)}),
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
            
                        fvView(i,j,k)   = n / pow(2*M_PI*vth2, 1.5)* exp(-vnorm2/(2*vth2));
                        HsolView(i,j,k) = n / vnorm * std::erf(vnorm/(w2*vth));
                        GsolView(i,j,k) = w2 * n * vth *( exp(-vnorm2/(2*vth2))/sqrt(M_PI)  +  std::erf(vnorm/(w2*vth))  * (vth/(w2*vnorm)+vnorm/(w2*vth))    );

                    }
    );
    Kokkos::fence();
   //////////////////////////////////////////////////////////////////////////////

    fv = -4 * M_PI * fv;
    mesh_v.setOrigin({0, 0, 0});
    solver_RB->solve();
    mesh_v.setOrigin(vmin_mv);
    Kokkos::deep_copy(HView, fvView);  // H = fv
    fv = 2*fv;
    mesh_v.setOrigin({0,0,0});
    solver_RB->solve();
    mesh_v.setOrigin(vmin_mv);
    Kokkos::deep_copy(GView, fvView);  // G = fv

    double Gerror, Herror;

    H = H - H_sol;
    G = G - G_sol;

    H = H/H_sol;
    G = G/G_sol;

    Herror = norm(H);// / norm(H_sol);
    Gerror = norm(G);// / norm(G_sol);
    
    Herror = Herror / (NV*NV*NV);
    Gerror = Gerror / (NV*NV*NV);

    std::cout << "MESH NV " << NV << std::endl;
    std::cout << "avg relative H error: "  << Herror << std::endl;
    std::cout << "avg relative G error: "  << Gerror << std::endl;
}
    return 0;
}


#endif







	// MPI_Allreduce(&Herror, &globH, 1 , MPI_DOUBLE, MPI_SUM, Ippl::getComm());
	// MPI_Allreduce(&Gerror, &globG, 1 , MPI_DOUBLE, MPI_SUM, Ippl::getComm());


 // std::cout << "FVIEW" << std::endl;
    // for(int i = 0; i<NV; ++i){
    // for(int j = 0; j<NV; ++j){
    // for(int k = 0; k<NV; ++k){
    //     std::cout << fvView(i,j,k) << " ";
    // } std::cout << "  ";
    // } std::cout << std::endl;
    // }

    // std::cout << "HVIEW" << std::endl;
//  for(int i = 0; i<NV; ++i){
//     for(int j = 0; j<NV; ++j){
//     for(int k = 0; k<NV; ++k){
//         std::cout << HView(i,j,k) << " ";
//     } std::cout << "  ";
//     } std::cout << std::endl;
//     }
// std::cout << "GVIEW" << std::endl;
//     for(int i = 0; i<NV; ++i){
//     for(int j = 0; j<NV; ++j){
//     for(int k = 0; k<NV; ++k){
//         std::cout << GView(i,j,k) << " ";
//     } std::cout << "  ";
//     } std::cout << std::endl;
//     }
// std::cout << "HSOL" << std::endl;
//     for(int i = 0; i<NV; ++i){
//     for(int j = 0; j<NV; ++j){
//     for(int k = 0; k<NV; ++k){
//         std::cout << HsolView(i,j,k) << " ";
//     } std::cout << "  ";
//     } std::cout << std::endl;
//     }
// std::cout << "GSOL" << std::endl;
//     for(int i = 0; i<NV; ++i){
//     for(int j = 0; j<NV; ++j){
//     for(int k = 0; k<NV; ++k){
//         std::cout << GsolView(i,j,k) << " ";
//     } std::cout << "  ";
//     } std::cout << std::endl;
//     }




    // Kokkos::parallel_reduce("H Error",
    //                 Kokkos::MDRangePolicy<Kokkos::Rank<3>>({nghost, nghost, nghost},
    //                                                         {   fvView.extent(0) - nghost,
	// 						                                    fvView.extent(1) - nghost,
	// 						                                    fvView.extent(2) - nghost}),
	// 	            KOKKOS_LAMBDA(const int i, const int j, const int k, double& error){
            
    //                     error += std::abs(HView(i,j,k) - HsolView(i,j,k)/HsolView(i,j,k) ); 
    //                     // error += std::abs(0 - HsolView(i,j,k)/HsolView(i,j,k) ); 
    //                 },
	// 				  Kokkos::Sum<double>(Herror)
    // );
    // Kokkos::fence();

    // Kokkos::parallel_reduce("H Error",
    //                 Kokkos::MDRangePolicy<Kokkos::Rank<3>>({nghost, nghost, nghost},
    //                                                         {   fvView.extent(0) - nghost,
	// 						                                    fvView.extent(1) - nghost,
	// 						                                    fvView.extent(2) - nghost}),
	// 	            KOKKOS_LAMBDA(const int i, const int j, const int k, double& error){
            
    //                     error += std::abs( (GView(i,j,k) - GsolView(i,j,k))/GsolView(i,j,k)); 
    //                     // error += std::abs( (0 - GsolView(i,j,k))/GsolView(i,j,k)); 
    //                 },
	// 				Kokkos::Sum<double>(Gerror)
    // );
    // Kokkos::fence();

