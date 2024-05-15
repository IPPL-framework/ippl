/**
@page basic_index Basics: Index and NDIndex
 * @section index Introduction
 * ## Index
 * 
 * The Index class represents a strided range of indices, and it is used to define the index extent of Field objects on construction.
 * 
 * **Constructors**:
 * - `Index()`: Creates a null interval with no elements.
 * - `Index(n)`: Instantiates an Index object representing the range of integers from 0 to n-1 inclusive, with implied stride 1.
 * - `Index(a,b)`: Instantiates an Index object representing the range of integers [a , b], with implied stride 1.
 * - `Index(a,b,s)`: Instantiates an Index object representing the range of integers [a , b], with stride s.
 *
 * **Examples**:
 * ```cpp
 * Index I(10);     // Integer set [0..9]
 * Index Low(5);    // Integer set [0..4]
 * Index High(5,9); // Integer set [5..9]
 * Index IOdd(1,9,2);   // Integer set [1..9] with stride 2
 * Index IEven(0,9,2);  // Integer set [0..9] with stride 2
 * ```
 * **Operations**:
 *
 * Given `Index I(a,n,s)` and an integer `j`, the following operations are possible:
 * ```cpp
 * I+j  : a+j+i*s     // For i in [0..n-1]
 * j-I  : j-a-i*s
 * j*I  : j*a + i*j*s
 * I/j  : a/j + i*s/j // Note: j/I is not defined due to non-uniform stride and fraction prohibition.
 * ```
 * 
 * ## NDIndex 
 * 
 * NDIndex is a class that acts as a container for multiple Index instances. It simplifies operations such as intersections across N dimensions by forwarding requests to its contained Index objects.
 *
 @section Example Usage
The following example demonstrates the use of Index and NDIndex classes.
@code
std::array<int, dim> pt = {64, 64, 64};
ippl::Index Iinput(pt[0]);
ippl::Index Jinput(pt[1]);
ippl::Index Kinput(pt[2]);
ippl::NDIndex<dim> ownedInput(Iinput, Jinput, Kinput);

std::array<bool, dim> isParallel;  // Specifies SERIAL, PARALLEL dims
isParallel.fill(true);

ippl::FieldLayout<dim> layoutInput(MPI_COMM_WORLD, ownedInput, isParallel);
@endcode
 * 
 * In this example, we define an NDIndex object `ownedInput` with three Index objects `Iinput`, `Jinput`, and `Kinput`. We then create a FieldLayout object `layoutInput` using the NDIndex object and a boolean array `isParallel` that specifies the parallel dimensions.
 *
*/