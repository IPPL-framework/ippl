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

const char* TestName = "Rosenbluth_Relation_test";


int main(int argc, char *argv[]){
    int nnn = argc;
    nnn++;
    Ippl ippl(argc, argv);
    // int rank = Ippl::Comm->rank();
    // if(rank == 0)

    Inform msg("Langevin");

    const double n = std::atof(argv[1]);
    const double vth = std::atof(argv[2]);
    const double fct = std::atof(argv[3]);
        
    const double vth2 = vth*vth;
    const double w2 = sqrt(2.0);

    // std::string ALGO = argv[4];
    std::string ALGO = "HOCKNEY"; //VICO
    bool isVallPeriodic = false;
    double VMAX = fct*vth;
    std::string path0 = "./rb-rel/AN" + std::to_string(int(n)) + std::to_string(int(vth)) + std::to_string(int(fct)) +".csv";
    std::string path1 = "./rb-rel/CO" + std::to_string(int(n)) + std::to_string(int(vth)) + std::to_string(int(fct)) +".csv";

    std::ofstream fout0, fout1;
    fout0.open(path0);
    fout1.open(path1);

    fout0   << "NV" << ","
            << "hRel_L2" << "," 
            << "aRel_avg2_err[0]" << "," 
            << "aRel_avg2_err[1]" << "," 
            << "aRel_avg2_err[2]" << "," 
            << std::endl;
    fout1   << "NV" << ","
            << "hRel_L2" << "," 
            << "aRel_avg2_err[0]" << "," 
            << "aRel_avg2_err[1]" << "," 
            << "aRel_avg2_err[2]" << "," 
            << std::endl;

for(int NV = 8; NV <=128; NV*=2){

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
    Mesh_t          mesh_v(domain_v, hv_mv, Vector_t({0.0, 0.0, 0.0}));
    FieldLayout_t   FL_v(domain_v, decomp_v, isVallPeriodic);
    PLayout_t       PL_v(FL_v, mesh_v);
    Field_t fv;
    Field_t H;
    Field_t H2;
    Field_t G;

    VField_t Fd;
    VField_t a;
    Field_t a0;
    Field_t a1;
    Field_t a2;

    MField_t D;
    VField_t D0;
    VField_t D1;
    VField_t D2;


    fv.initialize(mesh_v, FL_v);
    H.initialize(mesh_v, FL_v);
    H2.initialize(mesh_v, FL_v);
    G.initialize(mesh_v, FL_v);
    Fd.initialize(mesh_v, FL_v);
    a.initialize(mesh_v, FL_v);
    a0.initialize(mesh_v, FL_v);
    a1.initialize(mesh_v, FL_v);
    a2.initialize(mesh_v, FL_v);
    D.initialize(mesh_v, FL_v);
    D0.initialize(mesh_v, FL_v);
    D1.initialize(mesh_v, FL_v);
    D2.initialize(mesh_v, FL_v);

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
    auto HView      =    H.getView();
    auto H2View     =   H2.getView();
    auto GView      =    G.getView();

    auto aView       =     a.getView();
    auto a0View      =    a0.getView();
    auto a1View      =    a1.getView();
    auto a2View      =    a2.getView();
    auto FdView      =    Fd.getView();

    auto DView       =    D.getView();
    auto D0View      =    D0.getView();
    auto D1View      =    D1.getView();
    auto D2View      =    D2.getView();

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
                        double vnorm = sqrt(vnorm2); 
            
                        fvView(i,j,k)   = n / pow(2*M_PI*vth2, 1.5)* exp(-vnorm2/(2*vth2));
                        HView(i,j,k) = (2.0) * n / vnorm * std::erf(vnorm/(w2*vth));
                        GView(i,j,k) = w2 * n * vth *( exp(-vnorm2/(2*vth2))/sqrt(M_PI)  +  std::erf(vnorm/(w2*vth))  * (vth/(w2*vnorm)+vnorm/(w2*vth))    );

                    }
    );
    Kokkos::fence();


    for(int kk = 0; kk<2; ++kk){
        
        Fd = grad(H);
        D  = hess(G);

        Kokkos::parallel_for("dunno",
                        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                                                {   DView.extent(0),
    							                                    DView.extent(1),
    							                                    DView.extent(2)}),
    		            KOKKOS_LAMBDA(const int i, const int j, const int k){

                            H2View(i,j,k) = D(i,j,k)[0][0] + D(i,j,k)[1][1] + D(i,j,k)[2][2];
                            //since D is symmetric i assume its not important if we take the divergence of
                            // the columns or the rows

                            D0View(i,j,k) = DView(i,j,k)[0];
                            D1View(i,j,k) = DView(i,j,k)[1];
                            D2View(i,j,k) = DView(i,j,k)[2];
                        }
        );
        Kokkos::fence();

        a0 = div(D0);
        a1 = div(D1);
        a2 = div(D2);

        Kokkos::parallel_for("dunno2",
                        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                                                {   aView.extent(0),
    							                                    aView.extent(1),
    							                                    aView.extent(2)}),
    		            KOKKOS_LAMBDA(const int i, const int j, const int k){

                            aView(i,j,k)[0] = a0View(i,j,k);
                            aView(i,j,k)[1] = a1View(i,j,k);
                            aView(i,j,k)[2] = a2View(i,j,k);
                        }
        );
        Kokkos::fence();

        double hRel_L2 = 0;
        H2 = H2 - H;
        hRel_L2 = norm(H2)/norm(H);

        ippl::Vector<double,3> aRel_L2_err {0.0, 0.0, 0.0};
        a = a - Fd;
        for (size_t d=0; d<3; ++d) {
                        // nghoste 3 array instead of 0 3 array
                        double temp = 0.0;                                                                                        
                        Kokkos::parallel_reduce("Vector errorNr reduce", 
                        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, { aView.extent(0) - nghost, 
                                                                            aView.extent(1) - nghost,
                                                                            aView.extent(2) - nghost}),          
                                KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k, double& valL) {                                
                                    double myVal = pow(aView(i, j, k)[d], 2);                                              
                                    valL += myVal;                                                                      
                        }, Kokkos::Sum<double>(temp));

                        double globaltemp = 0.0;                                                                                  
                        MPI_Allreduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, Ippl::getComm());                                             
                        double errorNr = std::sqrt(globaltemp);                                                                                   

                        temp = 0.0;                                                                                        
                        Kokkos::parallel_reduce("Vector errorDr reduce",                                                                       
                        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, { FdView.extent(0) - nghost,
                                                                            FdView.extent(1) - nghost,
                                                                            FdView.extent(2) - nghost}),          
                                KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k, double& valL) {                                
                                    double myVal = pow(FdView(i, j, k)[d], 2);                                              
                                    valL += myVal;                                                                      
                        }, Kokkos::Sum<double>(temp));  

                        globaltemp = 0.0;                                                                                  
                        MPI_Allreduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
                        double errorDr = std::sqrt(globaltemp);

                        aRel_L2_err[d] = errorNr/errorDr;
        } 


        if(kk == 0) std::cout << "MESH NV: " << NV << std::endl; 
        std::cout   << (kk==0 ? "ANALYTIC:  " : "COMPUTED:  ") 
                    << hRel_L2 << "," 
                    << aRel_L2_err[0] << "," 
                    << aRel_L2_err[1] << "," 
                    << aRel_L2_err[2] << ",   " << M_PI
                    << std::endl;


        (kk==0 ? fout0 : fout1)
                << NV << ","
                << hRel_L2 << "," 
                << aRel_L2_err[0] << "," 
                << aRel_L2_err[1] << "," 
                << aRel_L2_err[2] << "," 
                << std::endl;




        if(kk == 1) break;
        
        fv = -8.0 * M_PI * fv;
        mesh_v.setOrigin({0, 0, 0});
        solver_RB->solve();
        mesh_v.setOrigin(vmin_mv);
        Kokkos::deep_copy(HView, fvView);

        mesh_v.setOrigin({0,0,0});
        solver_RB->solve();
        mesh_v.setOrigin(vmin_mv);
        Kokkos::deep_copy(GView, fvView);
    }




    


}
    return 0;
}

//resetting of origin doesnt seem to be relevant

#endif
