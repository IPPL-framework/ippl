#include <iostream>
#include <vector>
#include <memory>

#include "ParameterList.h"


#include <variant>

// int floor(double a) {
//     return a;
// }
//
// template <typename T>
// T sum(T a, T b) {
//     return a + b;
// }
//
// template <class T, class U>
// double dummy(T a, U b) {
//     return a + b - 1.0;
// }
//
//
// template <typename F, typename ...Args>
// class GarbageSolver : public ippl::Solver<F, Args...>
// {
// public:
//     void solve(F&& f, Args&... args) override {
//         auto what = f(args...);
//         std::cout << *this->lhs_mp << " " << *this->rhs_mp << " " << what << std::endl;
//     }
//
// };

int main()
{
    ippl::ParameterList p1;

    p1.add<double>("tolerance", 1.0e-8);
    p1.add<bool>("boolean", false);

    ippl::ParameterList p2;

    p2.add<int>("int", 5);
    p2.add<double>("tolerance", 1.0e-12);

    p2.add<ippl::ParameterList>("test", p1);

//     int i = p2.get<int>("int");

//     std::cout << i << std::endl;


//     std::cout << p1 << std::endl;
//     std::cout << p2 << std::endl;

    p1.update(p2);

//     std::cout << p1 << std::endl;

    p1.merge(p2);

//     std::cout << p1 << std::endl;

    ippl::ParameterList p3;

    p3.add<double>("param 1", 1.0);
    p3.add<double>("param 2", 2.0);
    p3.add<ippl::ParameterList>("param 3", p2);
    p3.add<double>("param 4", 3.0);
    p3.add<double>("param 5", 4.0);
    p3.add<ippl::ParameterList>("param 6", p1);
    p3.add<double>("param 7", 7.0);

    std::cout << p3 << std::endl;

    std::cout << p3 << std::endl;


//     GarbageSolver<double(int, float), int, float> solver;

//     double rhs = 1.0;

//     double lhs = 10.5;

//     int a = 1;
//     float b = 2.5;
//
//     solver.setLhs(lhs);
//     solver.setRhs(rhs);

//     solver.solve(dummy, a, b);


    return 0;
}
