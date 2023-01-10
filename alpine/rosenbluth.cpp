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
#include<Kokkos_Random.hpp>
#include <random>

const char* TestName = "RosenbluthTest";

// ./rosenbluth N VTH VMAX
// the rosenbluth potentials are clearly defined in the case the prbability distribution in
// velocity space is defined as the maxwellian

int main(int argc, char *argv[]){
    Ippl ippl(argc, argv);

    Inform msg("Langevin");

    const double n = std::atof(argv[1]);
    const double vth = std::atof(argv[2]);
    const double fct = std::atof(argv[3]);
    // const double fct = 5.0;
    const double vth2 = vth*vth;
    const double w2 = sqrt(2.0);

    // std::string ALGO = argv[4];
    std::string ALGO = "HOCKNEY"; //VICO

    double VMAX = fct*vth;
    std::string path = "./rb/" + std::to_string(int(n)) + std::to_string(int(vth)) + std::to_string(int(fct)) +".csv";

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
            << "HerrorL4" << "," 
            << "GerrorL4" << ","  
            << std::endl;

for(int NV = 2; NV <=256; NV*=2){

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
    Field_t H;
    Field_t H_sol;
    Field_t G;
    Field_t G_sol;
    fv.initialize(mesh_v, FL_v);
    H.initialize(mesh_v, FL_v);
    H_sol.initialize(mesh_v, FL_v);
    G.initialize(mesh_v, FL_v);
    G_sol.initialize(mesh_v, FL_v);

    std::shared_ptr<VSolver_t> solver_RB;
    ippl::ParameterList sp;
    sp.add("use_pencils", true);
    sp.add("comm", ippl::p2p_pl);  
    sp.add("use_reorder", false); 
    sp.add("use_heffte_defaults", false);  
    sp.add("use_gpu_aware", true);  
    sp.add("r2c_direction", 0); 
    sp.add("output_type", VSolver_t::SOL);
    solver_RB = std::make_shared<VSolver_t>(fv, sp, ALGO);

    mesh_v.setOrigin(vmin_mv);

    auto fvView     =   fv.getView();
    auto HsolView   = H_sol.getView();
    auto GsolView   = G_sol.getView();
    auto HView      =    H.getView();
    auto GView      =    G.getView();

    const ippl::NDIndex<3>& lDom = FL_v.getLocalNDIndex();
    const int nghost = fv.getNghost();

    Kokkos::parallel_for("set values1",
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
                        // double vnorm = sqrt(vnorm2); 
            
                        fvView(i,j,k)   = n / pow(2*M_PI*vth2, 1.5)* exp(-vnorm2/(2*vth2));
                        // HsolView(i,j,k) = n / vnorm * std::erf(vnorm/(w2*vth));
                        // GsolView(i,j,k) = w2 * n * vth *( exp(-vnorm2/(2*vth2))/sqrt(M_PI)  +  std::erf(vnorm/(w2*vth))  * (vth/(w2*vnorm)+vnorm/(w2*vth))    );

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
            
                        // fvView(i,j,k)   = n / pow(2*M_PI*vth2, 1.5)* exp(-vnorm2/(2*vth2));
                        HsolView(i,j,k) = n / vnorm * erf(vnorm/(w2*vth));
                        // GsolView(i,j,k) = w2 * n * vth *( exp(-vnorm2/(2*vth2))/sqrt(M_PI)  +  std::erf(vnorm/(w2*vth))  * (vth/(w2*vnorm)+vnorm/(w2*vth))    );

                    }
    );
    Kokkos::fence();

        Kokkos::parallel_for("set values3",
                    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                                            {   GsolView.extent(0),
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
            
                        // fvView(i,j,k)   = n / pow(2*M_PI*vth2, 1.5)* exp(-vnorm2/(2*vth2));
                        // HsolView(i,j,k) = n / vnorm * std::erf(vnorm/(w2*vth));
                        GsolView(i,j,k) = 
                        w2 * n * vth *( exp(-vnorm2/(2.0*vth2))/sqrt(M_PI)  +  erf(vnorm/(w2*vth))  * (vth/(w2*vnorm)+vnorm/(w2*vth))    );

                    }
    );
    Kokkos::fence();
   //////////////////////////////////////////////////////////////////////////////

    fv = -4 * M_PI * fv;
    mesh_v.setOrigin({0, 0, 0});
    solver_RB->solve();
    mesh_v.setOrigin(vmin_mv);
    Kokkos::deep_copy(HView, fvView); // H = fv;

 // if u want to use true f value fopr the next step... 
    //     Kokkos::parallel_for("set values1",
    //                 Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
    //                                                         {   fvView.extent(0),
	// 						                                    fvView.extent(1),
	// 						                                    fvView.extent(2)}),
	// 	            KOKKOS_LAMBDA(const int i, const int j, const int k){

    //                     //local to global index conversion
    //                     const int ig = i + lDom[0].first() - nghost;
    //                     const int jg = j + lDom[1].first() - nghost;
    //                     const int kg = k + lDom[2].first() - nghost;

    //                     double vx = vmin_mv[0] + (ig + 0.5)*hv_mv[0];
    //                     double vy = vmin_mv[1] + (jg + 0.5)*hv_mv[1];
    //                     double vz = vmin_mv[2] + (kg + 0.5)*hv_mv[2];
    //                     double vnorm2  = vx*vx+vy*vy+vz*vz;
    //                     double vnorm = sqrt(vnorm2); 
            
    //                     fvView(i,j,k) = n / vnorm * std::erf(vnorm/(w2*vth));
                        
    //                 }
    // );
    // Kokkos::fence();


    fv = 2*fv;
    mesh_v.setOrigin({0,0,0});
    solver_RB->solve();
    mesh_v.setOrigin(vmin_mv);
    Kokkos::deep_copy(GView, fvView);



    double Gerror_avg2, Herror_avg2;
    double GerrorL1, HerrorL1;
    double GerrorL2, HerrorL2;
    double GerrorL4, HerrorL4;

    H = H - H_sol;
    G = G - G_sol;
    
    HerrorL1 = norm(H, 1)/norm(H_sol, 1);
    GerrorL1 = norm(G, 1)/norm(G_sol, 1);
    HerrorL2 = norm(H)/norm(H_sol);
    GerrorL2 = norm(G)/norm(G_sol);
    HerrorL4 = norm(H, 4)/norm(H_sol, 4);
    GerrorL4 = norm(G, 4)/norm(G_sol, 4);
    
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
            << HerrorL4 << "," 
            << GerrorL4 << ","  
            << std::endl;

    
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







    // H = fv*2.21;
    // std::cout <<  norm(H)/norm(H_sol)<<std::endl;
 



    // H = 2.5*fv - H_sol;
    // std::cout << norm(H)/norm(H_sol) << std::endl;
    // H = 2.45*fv - H_sol;
    // std::cout << norm(H)/norm(H_sol) << std::endl;
    // H = 2.4*fv - H_sol;
    // std::cout << norm(H)/norm(H_sol) << std::endl;
    // H = 2.35*fv - H_sol;
    // std::cout << norm(H)/norm(H_sol) << std::endl;
    // H = 2.3*fv - H_sol;
    // std::cout << norm(H)/norm(H_sol) << std::endl;
    // H = 2.25*fv - H_sol;
    // std::cout << norm(H)/norm(H_sol) << std::endl;
    // H = 2.2*fv - H_sol;
    // std::cout << norm(H)/norm(H_sol) << std::endl;
    // H = 2.15*fv - H_sol;
    // std::cout << norm(H)/norm(H_sol) << std::endl;
    // H = 2.1*fv - H_sol;
    // std::cout << norm(H)/norm(H_sol) << std::endl;
    
