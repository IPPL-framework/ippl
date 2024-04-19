/**
@page basic_mesh Basics: Mesh
*@section mesh Introduction
*The Mesh class in IPPL is a basic class used for creating and handling meshes in simulations. 
*It lets you work with meshes of different sizes and types. You can get or set where the mesh starts (its origin) and find out the size of the grid it covers. 
*It's set up so other parts of the program can be updated if the mesh changes, like if it gets bigger or needs to be rearranged.
@section example_usage Example Usage
The following example demonstrates the use of the Mesh class in IPPL:
* @code
    using Mesh_t = ippl::UniformCartesian<double, dim>;

    int pt = 25;
    ippl::Index I(pt);
    ippl::NDIndex<dim> owned(I, I, I);

    double dx = 1.0 / double(pt);
    ippl::Vector<double, dim> hx = {dx, dx, dx};
    ippl::Vector<double, dim> origin = {0.0, 0.0, 0.0};
    Mesh_t mesh(owned, hx, origin);
* @endcode
In this example, we initialize a 3-dimensional mesh using 25 points in each direction. 
We then define the mesh's origin, spacing, and size, and create a Mesh object using these parameters.
*/


