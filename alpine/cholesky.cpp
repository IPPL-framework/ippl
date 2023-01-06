#ifndef CHOLESKY_H
#define CHOLESKY_H

// #include "../Langevin.cpp"
// #include "Langevin.cpp"

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


const char* TestName = "choleskyTest";

//rowmajor...
template<typename V>
Matrix_t choleskyb( V& d0, V& d1, V& d2){
        
    // d0 = d1 = d2;

        Matrix_t LL;
        V* D[] = {&d0, &d1, &d2}; 
        double epszero = 1e-20; // ??
        
        assert(     (fabs((*D[0])(1)-(*D[1])(0))) < epszero &&
                    (fabs((*D[0])(2)-(*D[2])(0))) < epszero &&
                    (fabs((*D[1])(2)-(*D[2])(1))) < epszero 
        );

        auto finish_LL = [&](const unsigned i0, const unsigned i1, const unsigned i2){
                // LL(0)(0) = sqrt(*D[i0])(i0));
                // LL(0)(1) = *D[i0])(i1)/LL(i0)(i0);
                LL(i0)(i2) = (*D[i0])(i2)/LL(i0)(i0);
                LL(i1)(i0) = 0.0;
                // LL(1)(1) = sqrt(*D[i1])(i1)- pow(LL(i0)(i1), 2));
                LL(i1)(i2) = ((*D[i1])(i2) - LL(i0)(i1)*LL(i0)(i2))/LL(i1)(i1);
                LL(i2)(i0) = 0.0;
                LL(i2)(i1) = 0.0;
                LL(i2)(i2) = sqrt( (*D[i2])(i2) - pow(LL(i0)(i2), 2) - pow(LL(i1)(i2), 2)) ;
        };

        auto get_2_diag = [&](const unsigned i0, const unsigned i1){//, const unsigned i2){
            return sqrt((*D[i1])(i1)- pow( LL(i0)(i1)=(*D[i0])(i1)/LL(i0)(i0), 2));
        };



            if     (( epszero <=(LL(0)(0)=sqrt(d0(0)))     )&&(    epszero <= ( LL(1)(1) = get_2_diag(0,1/*,2*/) )       ))finish_LL(0, 1, 2);
            // else if(( epszero <= LL(0)(0)                  )&&(    epszero <= ( LL(2)(2) = get_2_diag(0,2/*,1*/) )       ))finish_LL(0, 2, 1);
            // else if(( epszero <=(LL(1)(1)=sqrt(d1(1)))     )&&(    epszero <= ( LL(0)(0) = get_2_diag(1,0/*,2*/) )       ))finish_LL(1, 0, 2);
            // else if(( epszero <= LL(1)(1)                  )&&(    epszero <= ( LL(2)(2) = get_2_diag(1,2/*,0*/) )       ))finish_LL(1, 2, 0);
            // else if(( epszero <=(LL(2)(2)=sqrt(d2(2)))     )&&(    epszero <= ( LL(1)(1) = get_2_diag(2,1/*,0*/) )       ))finish_LL(2, 1, 0);
            // else if(( epszero <= LL(2)(2)                  )&&(    epszero <= ( LL(0)(0) = get_2_diag(2,0/*,1*/) )       ))finish_LL(2, 0, 1);
            else{
                // how often
                // assert(false && "no cholesky decomposition possible for at least one particle");
                for(unsigned di = 0; di<Dim; ++di)
                for(unsigned dj = 0; dj<Dim; ++dj){
                    LL(di)(dj)=0.0;
                }
            }

        return LL;
}


int main(){

    Vector_t D0, D1, D2;
    Matrix_t L, L_sol;
    double avgerror=0, error=0;

    std::ifstream fin;
    // std::ofstream fout;

    fin.open("./cholesky_test_data/cholesky.csv");
    // fout.open("./cholesky_test_data/results.csv");

    std::string tmp;


    for(int i = 0; i < 200; ++i){

        for(int d = 0; d < 3 ; ++d){            
            fin >> tmp;
            D0[d]= std::stof(tmp); 
            fin >> tmp;
            D1[d]= std::stof(tmp); 
            fin >> tmp;
            D2[d]= std::stof(tmp); 
        }

        for(int d = 0; d < 3 ; ++d){            
            fin >> tmp;
            L_sol[0][d]= std::stof(tmp);
            fin >> tmp;
            L_sol[1][d]= std::stof(tmp);
            fin >> tmp;
            L_sol[2][d]= std::stof(tmp);
        }
        
        L = choleskyb(D0, D1, D2);
        
        error = 0.0;
        for(int d = 0; d < 3 ; ++d){
            for(int c = d; c<=d; ++c)
                error += std::abs( (L[c][d] - L_sol[c][d]) /L_sol[c][d] );
                // fout << L[0][d] <<"=?"<<  L_sol[0][d] << "   ";
                // fout << L[1][d] <<"=?"<<  L_sol[1][d] << "   ";
                // fout << L[2][d] <<"=?"<<  L_sol[2][d] << "   ";
        }
        // fout << std::endl;
        // fout << error << std::endl;
        // fout << std::endl;

        avgerror += error;

    }

    avgerror  /= 200;
    std::cout << "average error(sum of relative errors over all elements of L) over 200stof tests is:  " << std::endl;
    std::cout << avgerror << std::endl;
    std::cout << "but the given solution from python is in single precision soo ... " <<  std::endl;
    
    fin.close();

    return 0;
}







#endif