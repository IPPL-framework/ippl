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





KOKKOS_INLINE_FUNCTION
double hessGexact( const int first, const int second, double vv[3], double vth)
{   
    double v1, v2, v3;

    if(first != second){
        v1 = vv[first];
        v2 = vv[second];
        v3 = vv[3-first - second];
    }
    else{
        v1 = vv[first];
        v2 = vv[(1+first)%3];
        v3 = vv[(2+first)%3];
    }

    const double vnorm2  = v1*v1+v2*v2+v3*v3;
    const double vnorm = sqrt(vnorm2); 
    const double vnorm4 = vnorm2*vnorm2;
    const double vth2 = vth*vth;
    const double vth4 = vth2*vth2;
    const double ERF = std::erf(vnorm/(sqrt(2)*vth));
    const double EXP = exp(vnorm2/(2*vth2));
    const double sqrt_pi = sqrt(M_PI);
    double hessG;

    if(first==second) //doesnt
    {
        const double v1_2= v1*v1;
        const double v2_2= v2*v2;
        const double v3_2= v3*v3;

        hessG =     (-vth2 + v1_2)/(EXP*sqrt_pi*vth4)
                +   (2*v1_2*(-vth2 + vnorm2))/(EXP*sqrt_pi*vth2*vnorm4)
                +   ((vth2 + vnorm2)*(-pow(v1_2,2) + vth2*(v2_2 + v3_2) -v1_2*(v2_2 + v3_2)))/(EXP*sqrt_pi*vth4*vnorm4)
                +   ((vth2*(2*v1_2 - v2_2 - v3_2) + (v2_2 + v3_2)*(vnorm2))*ERF)/(sqrt(2)*vth*vnorm4*vnorm);
      
    //     hessG =    (-2*pow(v1,2) + pow(v2,2) + pow(v3,2))/
    //  (exp((pow(v1,2) + pow(v2,2) + pow(v3,2))/(2.*pow(vth,2)))*
    //    sqrt(M_PI)*pow(pow(v1,2) + pow(v2,2) + pow(v3,2),2)) + 
    // ((pow(vth,2)*(2*pow(v1,2) - pow(v2,2) - pow(v3,2)) + 
    //      (pow(v2,2) + pow(v3,2))*(pow(v1,2) + pow(v2,2) + pow(v3,2))
    //      )*erf(sqrt(pow(v1,2) + pow(v2,2) + pow(v3,2))/(sqrt(2)*vth)))/
    //  (sqrt(2)*vth*pow(pow(v1,2) + pow(v2,2) + pow(v3,2),2.5));

    }
    else{//works
            hessG=    (v1*v2)/(EXP*sqrt_pi*vth4) 
        -   (v1*v2*(3*vth4 + vnorm4 ))/(EXP*sqrt_pi*vth4*vnorm4)
        -   (v1*v2*(-3*vth2 + vnorm2)*ERF)/(sqrt(2)*vth*vnorm4*vnorm);
        //simplyfying the hess_G file might yield better results...
    }

    return hessG;
}


           // ???same??'FsolView(i,j,k)[0] = n*vx/2*(-1/(vnorm2*vnorm)*ERF + 1/vnorm2*sqrt(2/M_PI)/vth*EXP);??? same
KOKKOS_INLINE_FUNCTION
double gradHexact( int first, double vv[3], double vth)
{   
    double gradH;

    double v1 = vv[first];
    double v2 = vv[(++first)%3];
    double v3 = vv[(++first)%3];

    const double vnorm2  = v1*v1+v2*v2+v3*v3;    
    const double vnorm = sqrt(vnorm2); 
    const double vth2 = vth*vth;
    const double ERF = std::erf(vnorm/(sqrt(2)*vth));
    const double EXP = exp(vnorm2/(2*vth2));
    gradH = (sqrt(2/M_PI)*v1)/ (EXP* vth*(vnorm2)) - (v1*ERF)/ pow(vnorm2,1.5);

    return gradH;
}

const char* TestName = "RosenbluthTest";

// ./rosenbluth N VTH VMAX
// the rosenbluth potentials are clearly defined in the case the prbability distribution in
// velocity space is defined as the maxwellian

int main(int argc, char *argv[]){
    int nnn = argc;
    nnn++;
    Ippl ippl(argc, argv);
    int rank = Ippl::Comm->rank();
    
 
    Inform msg2all("Langevin ",INFORM_ALL_NODES);
    Inform msg("Langevin");

    const double n = std::atof(argv[1]);
    const double vth = std::atof(argv[2]);
    const double fct = std::atof(argv[3]);
    const int compare_opt  = std::atoi(argv[4]);
    const char print_opt = *(argv[5]); // H h G g F f D d
    // std::string ALGO = (argc>=5 ? argv[4] : "HOCKNEY") ; //VICO

    double L =2;
    // const double cst = std::atof(argv[4]);
    // const double fct = 5.0;
    // const double cst = 2.44;
    
    
    const double vth2 = vth*vth;
    const double w2 = sqrt(2.0);

    std::string ALGO = "HOCKNEY";
    bool isVallPeriodic = false;
    double VMAX = fct*vth;
    std::string path = "./rb/" + std::to_string(int(n)) + std::to_string(int(vth)) + std::to_string(int(fct))+ std::to_string(int(compare_opt))+".csv";
    std::string pathA = "./rb/curve" + std::to_string(int(n)) + std::to_string(int(vth)) + std::to_string(int(fct)) + std::to_string(int(compare_opt))+print_opt+".csv";

    std::ofstream fout, aout;
    fout.open(path);
    aout.open(pathA);
    if(rank == 0){
        fout
            << "NV" << "," 
            << "HerrorL2" << "," 
            << "GerrorL2" << ","  
            << "Derror" << ",,,,,"
            << "Ferror" << ",,"
            << std::endl;
    }

for(int NV = 4; NV <=128 /*256*/; NV*=2){

    Vector<int, Dim> nv_mv = {NV,NV,NV};
    double hv = 2*VMAX/NV;
    Vector_t hv_mv { hv, hv, hv};
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
    Field_t H_sol;
    Field_t G;
    Field_t G_sol;

    MField_t D, D_sol;
    VField_t F, F_sol;

    fv.initialize(mesh_v, FL_v);
    H.initialize(mesh_v, FL_v);
    H_sol.initialize(mesh_v, FL_v);
    G.initialize(mesh_v, FL_v);
    G_sol.initialize(mesh_v, FL_v);
    D.initialize(mesh_v, FL_v);
    D_sol.initialize(mesh_v, FL_v);
    F.initialize(mesh_v, FL_v);
    F_sol.initialize(mesh_v, FL_v);

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
    auto GView      =    G.getView();
    auto DView      =    D.getView();
    auto FView      =    F.getView();
    auto HsolView   =H_sol.getView();
    auto GsolView   =G_sol.getView();
    auto DsolView   =D_sol.getView();
    auto FsolView   =F_sol.getView();

    const ippl::NDIndex<3>& lDom = FL_v.getLocalNDIndex();

///////!!!!!!!!!!
    int nghost = fv.getNghost();
    // nghost = 2*fv.getNghost();

    Kokkos::parallel_for("set values1", // here are ghost included?? at the start ill leave them??? border regions are critical...
                    // Kokkos::MDRangePolicy<Kokkos::Rank<3>>({nghost, nghost, nghost},
                    //                                         {   fvView.extent(0)-nghost,
					// 		                                    fvView.extent(1)-nghost,
					// 		                                    fvView.extent(2)-nghost}),
                    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                                            {   fvView.extent(0),
							                                    fvView.extent(1),
							                                    fvView.extent(2)}),


		            KOKKOS_LAMBDA(const int i, const int j, const int k){

                        //local to global index conversion
                        const int ig = i + lDom[0].first() - nghost;
                        const int jg = j + lDom[1].first() - nghost;
                        const int kg = k + lDom[2].first() - nghost;

                        const double vx = vmin_mv[0] + (ig + 0.5)*hv_mv[0];
                        const double vy = vmin_mv[1] + (jg + 0.5)*hv_mv[1];
                        const double vz = vmin_mv[2] + (kg + 0.5)*hv_mv[2];
                        const double vnorm2  = vx*vx+vy*vy+vz*vz;
                        const double vnorm = sqrt(vnorm2); 
                        const double ERF = std::erf(vnorm/(w2*vth));
                        const double EXP = exp(-vnorm2/(2*vth2));
            
                        fvView(i,j,k)   = n / pow(2*M_PI*vth2, 1.5)* EXP;
                        HsolView(i,j,k) = 2.0 * n / vnorm * ERF;
                        GsolView(i,j,k) = w2 * n * vth *( EXP/sqrt(M_PI)  + ERF  * (vth/(w2*vnorm)+vnorm/(w2*vth))    );


                        double vv[3] = {vx, vy, vz};

                        for (int di=0; di<3; ++di) {
                            FsolView(i,j,k)[di] = 2.0*n*gradHexact(di, vv, vth);
                        }
                        
                        for (int di=0; di<3; ++di) {
                        for (int dj=0; dj<3; ++dj) {
                            DsolView(i,j,k)[dj][di] = w2*n*vth*hessGexact(di, dj, vv, vth);
                        }}
                    }
    );
    Kokkos::fence();


    fv = -8.0 * M_PI * fv;
    mesh_v.setOrigin({0, 0, 0});
    solver_RB->solve();
    mesh_v.setOrigin(vmin_mv);
    Kokkos::deep_copy(HView, fvView);
    Kokkos::fence();

    mesh_v.setOrigin({0,0,0});
    solver_RB->solve();
    mesh_v.setOrigin(vmin_mv);
    Kokkos::deep_copy(GView, fvView);
    Kokkos::fence();


 

    
    // G = G + 2.44*vth*n*fct;

    //this is the better sequence of shift buti dont know why...?
    double offset = G.sum();
    offset = offset/pow(2*VMAX, 3);
    offset = offset*pow(hv, 3);
    G = G - offset;// + n*VMAX;
    G = G+n*VMAX;


    switch(compare_opt)
    {
        case 2: // compare analyt(hess(G)) with hess(analyt(G))
            D = hess(G_sol);
            F = grad(H_sol);
            break;
        case 1: // compare hess(analyt(G)) with hess(num(G))
            D_sol = hess(G_sol);
            F_sol = grad(H_sol);
            [[fallthrough]];
        case 0: // compare analyt(hess(G)) with hess(num(G)
            D = hess(G);
            F = grad(H);
    }



    if(NV==128 && rank == 0){
        // int ix[2] = {30, 60};
        // int iy[3] = {20, 40, 60};
        // int ix[] = {5,20, 40, 60};
        int ix[] = {60};
        int iy[] = {4,14,24,34,44,54,64};
        for(int i : ix){
        for(int j : iy){
            switch(print_opt)
            {       case 'h':[[fallthrough]];
                    case 'H':
                        aout << "HsolView("+std::to_string(i)+"_"+std::to_string(j)+")" << ","
                             << "HView("+std::to_string(i)+"_"+std::to_string(j)+")" << ",";
                    break;
                    case 'g':[[fallthrough]];
                    case 'G':
                        aout << "GsolView("+std::to_string(i)+"_"+std::to_string(j)+")" << ","
                             << "GView("+std::to_string(i)+"_"+std::to_string(j)+")" << ",";
                    break;
                    case 'd': [[fallthrough]];
                    case 'D':
                        aout << "DsolView(i,j,k)[0][0]" +std::to_string(i)+"_"+std::to_string(j)+")" << ","
                             << "DView(i,j,k)[0][0]" +std::to_string(i)+"_"+std::to_string(j)+")" << ",";
                    break;
                    case 'f':[[fallthrough]];
                    case 'F':
                        aout << "FsolView(i,j,k)[0]" +std::to_string(i)+"_"+std::to_string(j)+")" << ","
                             <<"FView(i,j,k)[0]" +std::to_string(i)+"_"+std::to_string(j)+")" << ",";
            }
            
        }
        }aout << std::endl;

        for(int k = 1; k<NV; ++k){
            aout << k << ",";
        for(int i : ix){
        for(int j : iy){
         switch(print_opt)
            {       case 'h':[[fallthrough]];
                    case 'H':
                        aout << HsolView(i,j,k) << ","
                             << HView(i,j,k) << ",";
                    break;
                    case 'g':[[fallthrough]];
                    case 'G':
                        aout << GsolView(i,j,k) << ","
                             << GView(i,j,k) << ",";
                    break;
                    case 'd': [[fallthrough]];
                    case 'D':
                        aout << DsolView(i,j,k)[0][0] << ","
                             << DView(i,j,k)[0][0] << ",";
                    break;
                    case 'f':[[fallthrough]];
                    case 'F':
                        aout << FsolView(i,j,k)[1]<< ","
                             << FView(i,j,k)[1] << ",";
            }
        }
        } aout<< std::endl;
        }
    }

    double GerrorL, HerrorL;

    Matrix_t  Derror;
    double Derror_tot=0;

    Vector_t  Ferror;
    double Ferror_tot=0;


        D = D - D_sol;
        F = F - F_sol;

    // if(NV<=16)
    // {
    // std::cout << "F diff is" << std::endl;
    // for(int i = 0; i<NV/2 +1; ++i){
    // for(int j = 0; j<NV/2 +1; ++j){
    // for(int k = 0; k<NV/2 +1; ++k){
    //     std::cout << FView(i,j,k)[2] << " ";
    // } std::cout << "  ";
    // } std::cout << std::endl;
    // }
    // }


        // {0, 0, 0} {nghost, nghost, nghost}???
        for (size_t di=0; di<3; ++di) {
        for (size_t dj=0; dj<3; ++dj) {
                        double temp = 0.0;                                                                                        
                        Kokkos::parallel_reduce("Vector errorNr reduce", 
                        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({nghost, nghost, nghost}, {  DView.extent(0) - nghost, 
                                                                                            DView.extent(1) - nghost,
                                                                                            DView.extent(2) - nghost}),          
                                KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k, double& valL) {                                
                                    double myVal = pow(fabs(DView(i, j, k)[di][dj]), L);                                              
                                    valL += myVal;                                                                      
                        }, Kokkos::Sum<double>(temp));
                        Kokkos::fence();

                        double globaltemp = 0.0;                                                                                  
                        MPI_Allreduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, Ippl::getComm());                                             
                        double errorNr = pow(globaltemp, (1/double(L)));                                                                                   

                        temp = 0.0;                                                                                        
                        Kokkos::parallel_reduce("Vector errorDr reduce",                                                                       
                        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({nghost, nghost, nghost}, { DsolView.extent(0) - nghost,
                                                                                            DsolView.extent(1) - nghost,
                                                                                            DsolView.extent(2) - nghost}),          
                                KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k, double& valL) {                                
                                    double myVal = pow(fabs(DsolView(i, j, k)[di][dj]), L);                                              
                                    valL += myVal;                                                                      
                        }, Kokkos::Sum<double>(temp));  

                        globaltemp = 0.0;                                                                                  
                        MPI_Allreduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
                        double errorDr = pow(globaltemp, (1/double(L)));;

                        Derror[di][dj] = errorNr/errorDr;
                        Derror_tot += errorNr/errorDr;
        }
        }


        for (size_t dj=0; dj<3; ++dj) {
                        double temp = 0.0;                                                                                        
                        Kokkos::parallel_reduce("Vector errorNr reduce", 
                        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({nghost, nghost, nghost}, { FView.extent(0) - nghost, 
                                                                                            FView.extent(1) - nghost,
                                                                                            FView.extent(2) - nghost}),          
                                KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k, double& valL) {                                
                                    double myVal = pow(fabs(FView(i, j, k)[dj]), L);                                              
                                    valL += myVal;                                                                      
                        }, Kokkos::Sum<double>(temp));
                        double globaltemp = 0.0;                                                                                  
                        MPI_Allreduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, Ippl::getComm());                                             
                        double errorNr = pow(globaltemp, (1/double(L)));                                                                                   
                        temp = 0.0;                                                                                        
                        Kokkos::parallel_reduce("Vector errorDr reduce",                                                                       
                        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({nghost, nghost, nghost}, { FsolView.extent(0) - nghost,
                                                                                            FsolView.extent(1) - nghost,
                                                                                            FsolView.extent(2) - nghost}),          
                                KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k, double& valL) {                                
                                    double myVal = pow(fabs(FsolView(i, j, k)[dj]), L);                                              
                                    valL += myVal;                                                                      
                        }, Kokkos::Sum<double>(temp));  
                        globaltemp = 0.0;                                                                                  
                        MPI_Allreduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
                        double errorDr = pow(globaltemp, (1/double(L)));;
                        Ferror[dj] = errorNr/errorDr;
                        Ferror_tot += errorNr/errorDr;
        } 



 
 

    Ferror_tot /= 3;
    Derror_tot /= 9;
    H = H - H_sol;
    G = G - G_sol;

    HerrorL = norm(H, L)/norm(H_sol, L);
    GerrorL = norm(G, L)/norm(G_sol, L);

    if(rank == 0){
    std::cout << path+" - MESH NV: " << NV << std::endl;
    fout    << NV << ","
            << HerrorL << "," 
            << GerrorL << "," 
            << Derror_tot << ","
            << Derror[0][0] << ","
            << Derror[1][1] << ","
            << Derror[0][1] << ","
            << Derror[1][2] << ","
            << Ferror_tot << ","
            << Ferror[0] << ","
            << std::endl;
    }

}
    return 0;
}

//resetting of origin doesnt seem to be relevant

#endif




// /rb/1150.csv - MESH NV: 4
// F diff is
// 0.0953026 0.10237 0.108115 0.111414   0.10237 0.111415 0.119239 0.124042   0.108115 0.119239 0.129779 0.13703   0.111414 0.124042 0.13703 0.146958   
// -0.000597123 -0.000904619 -0.00122128 -0.00142004   -0.000904619 -0.00142007 -0.0018529 -0.0019632   -0.00122128 -0.0018529 -0.00208312 -0.0019747   -0.00142004 -0.0019632 -0.0019747 -0.00326933   
// -0.000647064 -0.00115723 -0.00186756 -0.00246544   -0.00115723 -0.00246991 -0.00495641 -0.00780592   -0.00186756 -0.00495641 -0.0151542 -0.0347315   -0.00246544 -0.00780592 -0.0347315 -0.100894   
// -0.000295564 -0.000580188 -0.00103514 -0.0014662   -0.000580188 -0.00147016 -0.00366111 -0.00673389   -0.00103514 -0.00366111 -0.0152704 -0.0399873   -0.0014662 -0.00673389 -0.0399873 -0.124356   
// ./rb/1150.csv - MESH NV: 8
// ./rb/1150.csv - MESH NV: 16
// ./rb/1150.csv - MESH NV: 32
// ./rb/1150.csv - MESH NV: 64
// [klappr_s@merlin-l-001 alpine]$ ./rb/1150.csv - MESH NV: 128





















    // int y = NV/2 + 1;    
    // std::cout <<"H_sol" << 
    //         " 3 corner: " << HView(1,1,1) <<
    //         " 2 corner: " << HView(1,1,y) <<
    //         " 1 corner: " << HView(1,y,y) <<
    //         "  center: "  << HView(y,y,y) <<  
    // std::endl;
    // std::cout <<"H    " << 
    //         " 3 corner: " << HsolView(1,1,1) <<
    //         " 2 corner: " << HsolView(1,1,y) <<
    //         " 1 corner: " << HsolView(1,y,y) <<
    //         "  center: "  << HsolView(y,y,y) <<  
    // std::endl;




    // HerrorL1 = norm(H, 1)/norm(H_sol, 1);
    // GerrorL1 = norm(G, 1)/norm(G_sol, 1);
    // H = H/H_sol;
    // G = G/G_sol;
    // Herror_avg2 = pow(norm(H), 2) /(NV*NV*NV);
    // Gerror_avg2 = pow(norm(G), 2) /(NV*NV*NV);

// std::cout << "G is" << std::endl;
//     for(int i = 1; i<NV/2 +1; ++i){
//     for(int j = 1; j<NV/2 +1; ++j){
//     for(int k = 1; k<NV/2 +1; ++k){
//         std::cout << fvView(i,j,k) << " ";
//     } std::cout << "  ";
//     } std::cout << std::endl;
//     }


//===============================================


  
    // G = G + 4.0*M_PI*vth*n*fct/5.0;
    // G = G + exp(1)*vth*n*fct;
    // G = G + 2.6*vth*n*fct;
    // G = G + cst*vth*n*fct;
    // Kokkos::parallel_for("set values1",
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
            
    //                     GView(i,j,k)  = GView(i,j,k) +  vnorm/(2*VMAX);
                        
    //                 }
    // );
    // Kokkos::fence();
//===============================================





// MPI_Allreduce(&Herror, &globH, 1 , MPI_DOUBLE, MPI_SUM, Ippl::getComm());
// MPI_Allreduce(&Gerror, &globG, 1 , MPI_DOUBLE, MPI_SUM, Ippl::getComm());


//    Kokkos::parallel_for("set values1", // here are ghost included?? at the start ill leave them??? border regions are critical...
//                     // Kokkos::MDRangePolicy<Kokkos::Rank<3>>({nghost, nghost, nghost},
//                     //                                         {   fvView.extent(0)-nghost,
// 					// 		                                    fvView.extent(1)-nghost,
// 					// 		                                    fvView.extent(2)-nghost}),
//                     Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
//                                                             {   fvView.extent(0),
// 							                                    fvView.extent(1),
// 							                                    fvView.extent(2)}),


// 		            KOKKOS_LAMBDA(const int i, const int j, const int k){

//                         const int ig = i + lDom[0].first() - nghost;
//                         const int jg = j + lDom[1].first() - nghost;
//                         const int kg = k + lDom[2].first() - nghost;

//                         const double vx = vmin_mv[0] + (ig + 0.5)*hv_mv[0];
//                         const double vy = vmin_mv[1] + (jg + 0.5)*hv_mv[1];
//                         const double vz = vmin_mv[2] + (kg + 0.5)*hv_mv[2];
//                         const double vnorm2  = vx*vx+vy*vy+vz*vz;
//                         const double vnorm = sqrt(vnorm2);
            
//                         GView(i,j,k) = GView(i,j,k) + n*vnorm;
//                     }
//     );
    // Kokkos::fence();