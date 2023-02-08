// solves the Hamiltonian Maxwell Equations based on the method described in Braham et al. 2022
// code structure is based on "TestLaplace.cpp" from the IPPL library
#include "Ippl.h"

#include <iostream>
#include <typeinfo>
#include <array>

int main(int argc, char *argv[]) {

    Ippl ippl(argc,argv);

    constexpr unsigned int dim = 1;
    
    int N       = 1000;         //number of grid points
    double pi = acos(-1.0);

    //primal mesh initialization
    double len  = 2 * pi;       //the grid goes from 0 to 2*pi
    double h    = len / (N-1);  //grid spacing

    ippl::Index I(N);
    ippl::Index I_m1(N - 1);
    ippl::NDIndex<dim> ndi(I);
    ippl::NDIndex<dim> ndi_m1(I_m1);


    ippl::Vector<double,dim> spacing = {h};
    ippl::Vector<double,dim> origin  = {0};
    
    
    ippl::UniformCartesian<double, dim> primal_mesh(ndi, spacing, origin); 	//this mesh includes the endpoint x=2*pi via last index N-1
    ippl::UniformCartesian<double, dim> primal_mesh_m1{ndi_m1, spacing, origin};	//this mesh does NOT include the endpoint x=2*pi
    //--------------------------
    
    //define the field:

    ippl::e_dim_tag allParallel[dim];    // Specifies SERIAL, PARALLEL dims
        for (unsigned int d=0; d<dim; d++)
            allParallel[d] = ippl::PARALLEL;



    ippl::FieldLayout<dim> layout(I_m1, allParallel);

    

//    ippl::BConds<double, dim, primal_mesh, Cell> bc;      //this sets periodic boundary conditions
//    bc[0] = new ippl::PeriodicFace<double, dim>(0);
//    bc[1] = new ippl::PeriodicFace<double, dim>(1);


//    ippl::Field<double, dim, primal_mesh, Cell> b1(layout, bc);
    //-------------------------


    return 0;
}

