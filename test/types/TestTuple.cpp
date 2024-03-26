#include "Types/Tuple.h"
#include <Kokkos_Macros.hpp>
#include <iostream>
#include <memory>
#include <Ippl.h>
//#include <Kokkos_Core.hpp>
#define alwaysAssert(X) do{if(!(X)){std::cerr<<"Assertion " << #X << " failed: " << __FILE__ << ": " << __LINE__ << '\n'; exit(-1);}}while(false)
int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {    
        //ippl::Tuple<int, std::unique_ptr<int>> tup(17, std::make_unique<int>(5));
        //tup = ippl::makeTuple(1.0, 9.5);
        //ippl::Tuple<int, std::unique_ptr<int>> tup2{5, nullptr};
        //alwaysAssert(ippl::get<0>(tup2) == 5);

        ippl::Tuple<int, float> arith{1, 5.0f};
        alwaysAssert(ippl::get<0>(arith + arith) == 2);
        alwaysAssert(ippl::get<1>(arith + arith) == 10.0f);
        alwaysAssert(arith == arith);
        ippl::Tuple<int, float> arith2 = arith;
        ippl::get<1>(arith2) -= 1.0f;
        alwaysAssert(arith2 < arith);
        alwaysAssert(ippl::get<1>(arith2) == 4.0f);
        //tup2 = std::move(tup);
        //alwaysAssert(ippl::get<1>(tup) == nullptr);
        //alwaysAssert(*ippl::get<1>(tup2) == 5);
        Kokkos::View<ippl::Tuple<int, float, double>*> view("tview", 100);
        Kokkos::parallel_for(100, KOKKOS_LAMBDA(size_t idx){
            view(idx) = ippl::makeTuple(5.5, 1.25, 800.0f);
        });
        ippl::Tuple<int, float, double> red;
        Kokkos::parallel_reduce(100, KOKKOS_LAMBDA(size_t idx, ippl::Tuple<int, float, double>& ref)->void{
            ref += view(idx);
        }, red);
        alwaysAssert(ippl::get<0>(red) == 500);
        alwaysAssert(ippl::get<1>(red) == 125);
        alwaysAssert(ippl::get<2>(red) == 80000);
    }
    ippl::finalize();
    return 0;
}
