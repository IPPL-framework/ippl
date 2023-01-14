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

    Ippl ippl(argc, argv);
    // int rank = Ippl::Comm->rank();
    // if(rank == 0)

    Inform msg("Langevin");

    // int arg = 1;
    // std::cout << argc;

    const double fct = (argc>=2 ? std::atof(argv[1]) : 5) ;
    const int vers   = (argc>=3 ? std::atoi(argv[2]) : 1) ;
    const int L      = (argc>=4 ? std::atoi(argv[3]) : 2) ;
    const double n   = (argc>=5 ? std::atof(argv[4]) : 1) ;
    const double vth = (argc>=6 ? std::atof(argv[5]) : 1) ;
    std::string ALGO = (argc>=7 ?           argv[6]  : "HOCKNEY") ; //VICO

        
    const double vth2 = vth*vth;
    const double w2 = sqrt(2.0);

    // std::string ALGO = argv[4];
    bool isVallPeriodic = false;
    double VMAX = fct*vth;
    std::string path0 = "./rb-rel/AN" + std::to_string(int(n)) + std::to_string(int(vth)) + std::to_string(int(fct)) + "L" + std::to_string(L) +".csv";
    std::string path1 = "./rb-rel/CO" + std::to_string(int(n)) + std::to_string(int(vth)) + std::to_string(int(fct)) + "L" + std::to_string(L) +".csv";
    std::string path2 = "./rb-rel/ZZ" + std::to_string(int(n)) + std::to_string(int(vth)) + std::to_string(int(fct)) + "L" + std::to_string(L) +".csv";
    

    std::ofstream fout0, fout1, tabout;
    fout0.open(path0);
    fout1.open(path1);
    tabout.open(path2);
    fout0   << "NV" << ","
            << "lapH_f_err" << "," 
            << "lapG_h_err" << "," 
            << "TrD_h_err" << ","    
            << "divD_gradH_err[0]" << "," 
            << "divD_gradH_err[1]" << "," 
            << "divD_gradH_err[2]" << "," 
            << std::endl;
    fout1   << "NV" << ","
            << "lapH_f_err" << "," 
            << "lapG_h_err" << "," 
            << "TrD_h_err" << ","   
            << "divD_gradH_err[0]" << "," 
            << "divD_gradH_err[1]" << "," 
            << "divD_gradH_err[2]" << "," 
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
    Mesh_t          mesh_v(domain_v, hv_mv, Vector_t({0.0, 0.0, 0.0}));
    FieldLayout_t   FL_v(domain_v, decomp_v, isVallPeriodic);
    PLayout_t       PL_v(FL_v, mesh_v);

    Field_t fv;
    
    Field_t f;
    Field_t H;
    Field_t G;

    Field_t lapH;
    Field_t lapG;
    Field_t TrD;

    VField_t Fd;
    Field_t Fd0;

    VField_t a;
    Field_t a0;
    Field_t a1;
    Field_t a2;

    MField_t D;
    VField_t D0;
    VField_t D1;
    VField_t D2;


    fv.initialize(mesh_v, FL_v);

    f.initialize(mesh_v, FL_v);
    H.initialize(mesh_v, FL_v);
    G.initialize(mesh_v, FL_v);

    lapH.initialize(mesh_v, FL_v);
    lapG.initialize(mesh_v, FL_v);
    
    TrD.initialize(mesh_v, FL_v);

    Fd.initialize(mesh_v, FL_v);
    Fd0.initialize(mesh_v, FL_v);

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

    auto fView      =    f.getView();
    auto HView      =    H.getView();
    auto GView      =    G.getView();

    auto TrDView     =   TrD.getView();
    auto lapHView    =    lapH.getView();
    auto lapGView    =    lapG.getView();

    auto aView       =     a.getView();
    auto a0View      =    a0.getView();
    auto a1View      =    a1.getView();
    auto a2View      =    a2.getView();

    auto FdView      =    Fd.getView();
    auto Fd0View      =   Fd0.getView();

    auto DView       =    D.getView();
    auto D0View      =    D0.getView();
    auto D1View      =    D1.getView();
    auto D2View      =    D2.getView();

    const ippl::NDIndex<3>& lDom = FL_v.getLocalNDIndex();
    const int nghost = fv.getNghost();

    switch(vers)
{
    case 1:
        Kokkos::parallel_for("set values1",
                        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                                                {   fView.extent(0),
	    						                                    fView.extent(1),
	    						                                    fView.extent(2)}),
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

                            fView(i,j,k)  = n / pow(2*M_PI*vth2, 1.5)* exp(-vnorm2/(2*vth2));
                            HView(i,j,k) = (2.0) * n / vnorm * std::erf(vnorm/(w2*vth));
                            GView(i,j,k) = w2 * n * vth *( exp(-vnorm2/(2*vth2))/sqrt(M_PI)  +  std::erf(vnorm/(w2*vth))  * (vth/(w2*vnorm)+vnorm/(w2*vth))    );

                        }
        );
        Kokkos::fence();
        break;


}

    fv = 1.0*f;   

    for(int kk = 0; kk<2; ++kk){

        lapH = laplace(H);
        lapH = lapH/(-8.0*M_PI);
        lapG = laplace(G);
        Fd = grad(H);
        D  = hess(G);

        Kokkos::parallel_for("dunno",
                        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                                                {   DView.extent(0),
    							                                    DView.extent(1),
    							                                    DView.extent(2)}),
    		            KOKKOS_LAMBDA(const int i, const int j, const int k){

                            TrDView(i,j,k) = D(i,j,k)[0][0] + D(i,j,k)[1][1] + D(i,j,k)[2][2];
                            //since D is symmetric i assume its not important if we take the divergence of
                            // the columns or the rows (cause i dont know the proper definition....)

                            D0View(i,j,k) = DView(i,j,k)[0];
                            D1View(i,j,k) = DView(i,j,k)[1];
                            D2View(i,j,k) = DView(i,j,k)[2];
                        }
        );
        Kokkos::fence();

        a0 = div(D0);
        a1 = div(D1);
        a2 = div(D2);
        
        Kokkos::fence();
        Kokkos::parallel_for("dunno2",
                        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                                                {   aView.extent(0),
    							                                    aView.extent(1),
    							                                    aView.extent(2)}),
    		            KOKKOS_LAMBDA(const int i, const int j, const int k){

                            aView(i,j,k)[0] = a0View(i,j,k);
                            aView(i,j,k)[1] = a1View(i,j,k);
                            aView(i,j,k)[2] = a2View(i,j,k);

                            Fd0View(i,j,k) = FdView(i,j,k)[0];
                            
                            a0View(i,j,k) = FdView(i,j,k)[0] - a0View(i,j,k);
                            a1View(i,j,k) = FdView(i,j,k)[1] - a1View(i,j,k);
                            a2View(i,j,k) = FdView(i,j,k)[2] - a2View(i,j,k);

                        }
        );
        Kokkos::fence();

        if(kk==0)
        if(NV<= 16){
            for(int d = 0; d<3; ++d){

                tabout << "Fd" << d << std::endl;
                for(int i = 1; i<NV/2 +1; ++i){
                for(int j = 1; j<NV/2 +1; ++j){
                for(int k = 1; k<NV/2 +1; ++k){
                    tabout << FdView(i,j,k)[d] << ",";
                } tabout << "  ,";
                } tabout << std::endl;
                }

                tabout << "a" << d << std::endl;
                for(int i = 1; i<NV/2 +1; ++i){
                for(int j = 1; j<NV/2 +1; ++j){
                for(int k = 1; k<NV/2 +1; ++k){
                    tabout << aView(i,j,k)[d] << ",";
                } tabout << "  ,";
                } tabout << std::endl;
                }
   
            } 
        }

        double lapH_f_err = 0;
        lapH = lapH - f;
        lapH_f_err = norm(lapH, L)/norm(f, L);

        double lapG_h_err = 0;
        lapG = lapG - H;
        lapG_h_err = norm(lapG, L)/norm(H, L);
        
        double TrD_h_err = 0;
        TrD = TrD - H;
        TrD_h_err = norm(TrD, L)/norm(H, L);

        ippl::Vector<double,3> divD_gradH_err {0.0, 0.0, 0.0};

        divD_gradH_err[0] = norm(a0, L) / norm(Fd0, L);

        a = a - Fd;
        // for (size_t d=0; d<3; ++d) {
        for (size_t d=1; d<3; ++d) {
                        // nghoste 3 array instead of 0 3 array
                        double temp = 0.0;                                                                                        
                        Kokkos::parallel_reduce("Vector errorNr reduce", 
                        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, { aView.extent(0) - nghost, 
                                                                            aView.extent(1) - nghost,
                                                                            aView.extent(2) - nghost}),          
                                KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k, double& valL) {                                
                                    double myVal = pow(fabs(aView(i, j, k)[d]), L);                                              
                                    valL += myVal;                                                                      
                        }, Kokkos::Sum<double>(temp));

                        double globaltemp = 0.0;                                                                                  
                        MPI_Allreduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, Ippl::getComm());                                             
                        double errorNr = pow(globaltemp, (1/double(L)));                                                                                   

                        temp = 0.0;                                                                                        
                        Kokkos::parallel_reduce("Vector errorDr reduce",                                                                       
                        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, { FdView.extent(0) - nghost,
                                                                            FdView.extent(1) - nghost,
                                                                            FdView.extent(2) - nghost}),          
                                KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k, double& valL) {                                
                                    double myVal = pow(fabs(FdView(i, j, k)[d]), L);                                              
                                    valL += myVal;                                                                      
                        }, Kokkos::Sum<double>(temp));  

                        globaltemp = 0.0;                                                                                  
                        MPI_Allreduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
                        double errorDr = pow(globaltemp, (1/double(L)));;

                        divD_gradH_err[d] = errorNr/errorDr;
        } 


        // if(kk == 0) std::cout << "MESH NV: " << NV << std::endl; 

        (kk==0 ? fout0 : fout1)
                << NV << ","
                << lapH_f_err << "," 
                << lapG_h_err << "," 
                << TrD_h_err << "," 
                << divD_gradH_err[0] << "," 
                << divD_gradH_err[1] << "," 
                << divD_gradH_err[2] << "," 
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

//resetting of origin relevant???

#endif
