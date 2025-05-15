/**
* @page basic_particle Basics: Particles
*
* Introduction to handling particles in IPPL.
* @section particles Particles in IPPL
*
* IPPL provides a flexible and efficient framework for handling particles in parallel computing
environments.
*
* @subsection particle_concepts Key Concepts
* The core components are `ParticleBase` and `ParticleAttrib` classes, which provide the base
functionality and attributes for particles.
* - `ParticleBase` acts as the abstract base class for a set of particles, requiring a derived class
to specify data attributes (e.g., mass, charge).
* - `ParticleAttrib<T>` represents a single particle attribute, with T indicating the data type.


@subsubsection particle_base 'ParticleBase'
*   The user must define a class derived from 'ParticleBase' which describes
*   what specific data attributes the particle has (e.g., mass or charge).
*   'ParticleBase' is the abstract base class for a set of particles.
*   Each attribute is an instance of a ParticleAttribute<T> class; 'ParticleBase'
*   keeps a list of pointers to these attributes, and performs particle creation
*   and destruction.
*
*   'ParticleBase' is templated on the 'ParticleLayout' mechanism for the particles.
*   This template parameter should be a class derived from 'ParticleLayout'.
*   'ParticleLayout'-derived classes maintain the info on which particles are
*   located on which processor, and performs the specific communication
*   required between processors for the particles.  The 'ParticleLayout' is
*   templated on the type and dimension of the atom position attribute, and
*   'ParticleBase' uses the same types for these items as the given
*   'ParticleLayout'.
*
*   'ParticleBase' and all derived classes have the following common
*   characteristics:
*       - The spatial positions of the N particles are stored in the
*         particle_position_type variable R
*       - The global index of the N particles are stored in the
*         particle_index_type variable ID
*       - A pointer to an allocated layout class.  When you construct a
*         'ParticleBase', you must provide a layout instance, and 'ParticleBase'
*         will delete this instance when it (the 'ParticleBase') is deleted.
*
*   To use this class, the user defines a derived class with the same
*   structure as in this example:
* @code
*     class UserParticles :
*              public ParticleBase< ParticleSpatialLayout<double,3> > {
*     public:
*       // attributes for this class
*       ParticleAttribute<double> rad;  // radius
*       particle_position_type    vel;  // velocity, same storage type as R
*
*       // constructor: add attributes to base class
*       UserParticles(ParticleSpatialLayout<double,2>* L) : ParticleBase(L) {
*         addAttribute(rad);
*         addAttribute(vel);
*       }
*     };
*   @endcode
*   This example defines a user class with 3D position and two extra
*   attributes: a radius rad (double), and a velocity vel (a 3D Vector).

* @subsubsection particle_attrib ParticleAttrib and ParticleAttribBase
 *
 * These classes form the foundation for all particle attribute classes within the framework.
 *
 * ParticleAttrib is a templated class designed to represent a single particle attribute, such as
mass or charge.
 * It encapsulates an attribute as a data element within a particle object, stored using a
Kokkos::View for efficient parallel computation.
 * This class is essential for handling the type information of the attribute and provides a suite
of methods for creating, destroying, and performing operations on particle attributes.
 *
 * ParticleAttribBase serves as the generic base class for the templated ParticleAttrib class.
 * It provides a common interface for all particle attribute classes, including virtual methods for
creating and destroying elements of the attribute array.
 * By encapsulating data for a variable number of particles in a Kokkos::View, it facilitates
operations on this data.
 *
 * @subsubsection particle_layout ParticleLayout and ParticleSpatialLayout
 *
 * ParticleLayout manages particle distribution and serves as a base class for
ParticleSpatialLayout.
 *
 * ParticleSpatialLayout is a derivative of ParticleLayout, ParticleSpatialLayout specifically
handles particle distribution based on spatial positions relative to a predefined grid.
 * It ensures particles are processed on the same MPI rank as their corresponding spatial region,
defined by 'FieldLayout'.
 * It requires periodic updates to adjust particle locations.
 *




 * @subsection particle_example Example: Defining a Particle Bunch
 * Here's how you can define a simple particle bunch in IPPL:
 *
 * @code{.cpp}
 * using namespace ippl;
 *
 * template <class PLayout>
 * struct Bunch : public ParticleBase<PLayout> {
 *     ParticleAttrib<double> mass, charge;
 *     ParticleAttrib<Vector<double>> R, V;
 *
 *     Bunch(PLayout& layout) : ParticleBase<PLayout>(layout) {
 *         // add your own application attributes !
 *         this->addAttribute(mass);
 *         this->addAttribute(charge);
 *         this->addAttribute(R);
 *         this->addAttribute(V);
 *     }
 *    ~Bunch() = default;
 * };
 * // Compiled to single Kokkos Kernel
 * bunch->R = bunch->R + dt * bunch->V;
 * @endcode
 *
 * This example demonstrates how to extend `ParticleBase` to create a `Bunch` class that includes
common particle attributes and how to perform a single timestep.
 *
 * @subsection useful_particle_functions Useful Particle Functions
 * Consider bunch a pointer to a derived class 'Bunch' from 'ParticleBase' and has attributes 'R',
'V',
 *'mass', 'charge'
 * @subsubsection helper_particle_functions Helper Functions
 * A collection of essential functions for particle manipulation and attribute management, such as
creation, deletion, and data operations.
@code
    // Creates nParticles locally by a core or GPU ( not the total no . of particles )
    bunch->create( nParticles );

    // Get the local number of particles in that rank ( in ippl there is a one - one correspondence
between
    // MPI ranks and GPUs
    size_t localnum = bunch->getLocalNum ();

    // Particle deletion Function . invalid is a boolean View
    // marking which indices are invalid . destroyNum is the
    // total number of invalid particles
    bunch->destroy(invalid, destroyNum );

    // Device and host views and the deep copy operations between them
    // similar to fields
    Kokkos::deep_copy(bunch->R.getHostMirror(), bunch->R.getView ()); // device to host
    Kokkos::deep_copy(bunch->R.getView()      , bunch->R.getHostMirror()); // host to device

    // sum () , prod () , min () and max () functions available for
    // particle attributes similar to fields
    auto Rsum = bunch->R.sum ();
*@endcode
*@subsubsection bdry_condition_particles Boundary Conditions for Particles
* Setting up boundary conditions (e.g., periodic, reflective) for particle simulations.
*@code

// Types of BCs : PERIODIC , SINK , RFELECTIVE , NO
// Sets periodic BCs in all directions
bunch->setParticleBC(ippl::BC::PERIODIC);

// Assume a 3D problem
typedef std::array<ippl::BC, 6 > bc_container_type ;
bc_container_type bcs ;

for ( unsigned int i = 0; i < 4; ++ i ) {
    bcs[i] = ippl::BC::NO;
}

bcs[4] = ippl::BC::PERIODIC;
bcs[5] = ippl::BC::PERIODIC;
// Sets open BCs in x and y directions but periodic in z direction
bunch->setParticleBC( bcs );
*@endcode
*@subsubsection Interpolation_grid_particles Interpolation between Grid and Particles
 * Methods for data transfer between particles and grid, supporting operations like scatter and
gather for efficient particle-field interaction.
*@code
// Particles-> grid
// Can interpolate any particle attribute ( both scalar and vectors ) to the grid .
// Only linear ( cloud - in - cell ) interpolation is available at the moment .
// Interpolate a scalar particle attribute q ( e . g . charge ) onto the field rho .
scatter(bunch->q , rho , this->R);

// Grid - > particles
// Can interpolate any field ( both scalar and vectors ) to the particles .
// Only linear interpolation is available at the moment .
// Interpolate a vector field E ( e . g . electric field ) to the particles
gather (bunch->E , Efield , this->R);
*@endcode
*/
