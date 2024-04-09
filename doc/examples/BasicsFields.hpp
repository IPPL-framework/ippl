
/**
* @page basics_fields Basics: Fields 
 * 
 * 
 * 
@section field_intro Introduction
This sections provides the readers with a brief background on the used classes for handling Fields in IPPL.
* ## FieldLayout 
//
FieldLayout describes how a given index space (represented by an NDIndex
object) is distributed among MPI ranks. It performs the initial
partitioning. The user may request that a particular dimension not be
partitioned by flagging that axis as 'SERIAL' (instead of 'PARALLEL').
* @code
        int pt = 25;
        ippl::Index I(pt);
        ippl::NDIndex<dim> owned(I, I, I);
        std::array<bool, dim> isParallel;
        isParallel.fill(true);  // Specifies SERIAL, PARALLEL dims
        // all parallel layout, standard domain, normal axis order
        ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);
* @endcode
In this example, we create a 3-dimensional FieldLayout with 25 points in each direction. The layout is distributed among MPI ranks.


* ## BareField 
The BareField class in IPPL is a template class that represents a numerical field in a computational domain. It is designed to handle data in up to three dimensions and supports various data types through its template parameter T.
It provides a Kokkos-based multidimensional array, or view, which is the main data structure used to store field values.
The class is integrated with IPPL's field layout and halo cell system for managing computational domains and communication between them, especially in parallel computing environments.
A field can be initialized with a specific layout, and it supports operations such as resizing, updating the layout, and operations related to ghost cells which are used for boundary conditions and domain communication.
It offers functions for calculating aggregate values over the field, such as sums, maxima, minima, and products.
Fields can be assigned values from constants or other fields through expression templates, allowing for complex computations and assignments.
The class provides range policies for iteration that can exclude or include ghost layers, giving flexibility in how the field data is traversed and manipulated.


*@code
constexpr unsigned int dim = 3;
int numPoints = 100;
ippl::Index domainIndex(numPoints);
ippl::NDIndex<dim> domain(domainIndex, domainIndex, domainIndex);

ippl::FieldLayout<dim> layout(domain);
    // Create a BareField with 3 dimensions, double data type, and 1 ghost layer.
ippl::BareField<double, dim> field(layout, 1);
*@endcode



*@section Field Field
The Field class in IPPL is an advanced version of BareField, augmented with a mesh for defining the spatial domain and equipped with customizable boundary conditions. The class is templated to be flexible for various data types (T), dimensions (Dim), mesh types (Mesh), and centering schemes (Centering), along with additional Kokkos view arguments (ViewArgs...).
It's associated with a Mesh object that dictates the structure of the simulation space.
Boundary conditions can be set and updated, affecting how the field interacts with the limits of the simulation space.
Offers methods for calculating volume integrals and averages, which are useful for analyzing the field over its entire domain.
Inherits from BareField, allowing for basic field operations and attribute access.

### Example: Creating a Field
 * The following example showcases the creation and basic manipulation of a field in IPPL:
 * 
 * @code{.cpp}
using namespace ippl;

constexpr unsigned int dim = 3;
std::array<int, dim> points = {256, 256, 256};

ippl::Index Iinput(points[0]);
ippl::Index Jinput(points[1]));
ippl::Index Kinput(points[2]);
ippl::NDIndex<dim> ownedInput(Iinput, Jinput, Kinput);

// specifies SERIAL, PARALLEL dimensions
std::array<bool, dim> isParallel;
isParallel.fill(true);

ippl::FieldLayout<dim> layoutInput(MPI_COMM_WORLD, ownedInput, isParallel);

double dx = 1.0 / double(points);
Vector<double,dim> hx = {dx, dx, dx};
Vector<double,dim> origin = {0.0, 0.0, 0.0};

UniformCartesian<double,dim> mesh(owned, hx, origin);
Field<double,dim> field(mesh, layout);
@endcode
*
* This example outlines the steps to define a three-dimensional field, specifying its layout and parallelization strategy.
*





* @subsection useful_field_functions Useful Field Functions
* @subsubsection useful_field_ops Mathematical Operations
* @code
// Computes Lp norm of the field. For Max. norm p=0. Default is 2-norm.
auto fieldNorm = norm(field,p);

// Computes dot product of vector fields (in each element) and returns a scalar field
auto field = dot(vfield, vfield)

// Innerproduct of two scalar fields and returns a scalar 
auto fSquare = innerProduct(field, field);

// Computes the sum of the field
auto fieldSum = field.sum();

// $\int f dV$ computed using midpoint rule
auto fieldVolIntegral = field.getVolumeIntegral();
* @endcode


*@subsubsection field_properties Field Properties
*@code 
// Returns the range policy for the fields excluding the ghosts layers.
// If you want to include ghost layers specify nghost as an argument
auto fieldRange = field.getFieldRangePolicy(); 

// Return the number of ghost layers in the field
auto nghost = field.getNghost();

// Returns the underyling field layout
auto layout = field.getLayout();

// Get the local n-dimension indices from the field layout
auto lDom = layout.getLocalNDIndex();

// Get the global domain indices from the field layout
auto gDom = layout.getDomain();

// Shallow copy of fields. Changing field2 changes field1 also.
auto field2 = field1;

// Deep copy of fields. Changing field2 does not change field1.
auto field2 = field1.deepCopy();
*@endcode

*@subsubsection field_distributed Parallel Helper Functions
*@code
// Return the underyling Kokkos View of the field
auto fview = field.getView();

// Return the Host Mirror corresponding to the Kokkos device view 
// of the field. It is a no-op if the field is already on the host.
auto fHostView = field.getHostMirror();

// You still have to do a deep copy to get the data from the device to host or vice-versa
Kokkos::deep_copy(fHostView, fview); // device to host
Kokkos::deep_copy(fview, fHostView); // host to device
*@endcode

* @subsubsection init_field_from_func Initializing a Field from a Function
* @code 
const ippl::NDIndex<Dim>& lDom = fieldLayout.getLocalNDIndex();
const int nghost = field.getNghost();
auto fview = field.getView();

using index_array_type = typenamep ippl::RangePolicy<Dim>::index_array_type;

ippl::parallel_for( 
    "Assign a field based on func", field.getFieldRangePolicy(),
    KOKKOS_LAMBDA(const index_array_type& args) {
        // local to global index conversion
        Vector<double, Dim> xvec = (args + lDom.first() - nghost + 0.5) * hr + origin;

        // ippl::apply accesses the view at the given indices and obtains a 
        // reference; see src/Expression/IpplOperations.h
        ippl::apply(fview, args) = func(xvec);
    });
*@endcode

- To write dimension independent kernels use the wrappers 'ippl::parallel_for' and 'ippl::parallel_reduce'.
- If you don't want dimension independence in your application then you can just use 'Kokkos::parallel_for' and 'Kokkos::parallel_reduce'.
* @subsubsection boundary_conditions_fields Boundary Conditions for Fields
Setting BCs for fields is a necessary prerequisite before applying differential operators on fields! Otherwise
you will get garbage for the points close to the boundary.
*@code 
typedef ippl::BConds<Field<t, Dim>, Dim> bc_type;
bc_type bc;
// Available BCs in Ippl see src/Field/BcTypes.h
for(unsigned i = 0; i < 2*Dim; ++i){
    bc[i]= std::make_shared<ippl::PeriodicFace<Field<T, Dim>>>(i);
}

field.setFieldBC(bc);
bc_type BCs = field.getFieldBC();
* @endcode
* @subsubsection diff_op_fields Differential Operators for Fields
* @code
using mesh_type = ippl::UniformCartesian<T, Dim >;
using centering_type = typename mesh_type::DefaultCentering ;
using sfield_type = Field<double, Dim, mesh_type, centering_type>;
using vfield_type = Field<Vector<double, Dim>, Dim, mesh_type, centering_type>;
using mfield_type = Field <Vector<Vector<double, Dim>, Dim>, Dim, mesh_type, centering_type>;

sfield_type sfield (field.get_mesh(), field.getLayout());
vfield_type vfield (field.get_mesh (), field.getLayout ());
mfield_type mfield (field.get_mesh (), field.getLayout ());

// Computes gradient of a scalar field by cell - centered finite difference and returns a vector field
vfield = grad(field);

// Computes divergence of a vector field by cell - centered finite difference and returns a scalar field
field = div(vfield);

// Computes curl of a scalar field by cell - centered finite difference and returns a vector field
// Only available in 3 D for the moment
vfield = curl(field);

// Computes Hessian of a scalar field by cell - centered finite difference and returns a matrix field
// There is no native matrix data type in ippl so the output field is vector of vectors
mfield = hess(field);

// Computes Laplacian of a scalar field by cell - centered finite difference and returns a scalar field .
// At the moment ippl does not do a copy of the original field when you specify the same scalar field
// in both rhs and lhs . So the user has to allocate a different scalar field for lhs and rhs .
sfield = laplace(field);

* @endcode



*/
