/***************************************************************************
 *
 * The IPPL Framework
 *
 ***************************************************************************/

#ifndef FIELD_H
#define FIELD_H

// include files
#include "Field/BareField.h"
#include "Field/FieldSpec.h"
#include "Field/LField.h"
#include "Field/BCond.h"
#include "Field/ReductionLoc.h"
#include "SubField/SubField.h"
#include "DataSource/DataSource.h"
#include "Meshes/UniformCartesian.h"

// forward declarations
template <class T, unsigned D, unsigned B, class M, class C> 
class IndexedField;

/***********************************************************************
      This is the user visible Field of type T.
      It doesn't even really do expression evaluation; that is
      handled with the templates in PETE.h
***********************************************************************/

template<class T, unsigned Dim,
         class M=UniformCartesian<Dim,double>,
         class C=typename M::DefaultCentering >
class Field : public BareField<T,Dim>, public DataSource {

  friend class BareFieldIterator<T,Dim>;

public: 
  //# public typedefs
  typedef M Mesh_t;
  typedef C Centering_t;
  typedef BareField<T,Dim>                   Base_t;
  typedef FieldLayout<Dim>                   Layout_t;
  typedef BConds<T,Dim,M,C>                  bcond_container;
  typedef BCondBase<T,Dim,M,C>               bcond_value;
  typedef typename bcond_container::iterator bcond_iterator;

  // A default constructor, which should be used only if the user calls the
  // 'initialize' function before doing anything else.  There are no special
  // checks in the rest of the Field methods to check that the Field has
  // been properly initialized.
  Field();

  // Destroy the Field.
  virtual ~Field();

  // Create a new Field with a given layout and optional guard cells.
  // The default type of BCond lets you add new ones dynamically.
  // The makeMesh() global function is a way to allow for different types of
  // constructor arguments for different mesh types.
  Field(Layout_t &);
  Field(Layout_t &,const GuardCellSizes<Dim>&);
  Field(Layout_t &,const BConds<T,Dim,M,C>&);
  Field(Layout_t &,const GuardCellSizes<Dim>&,const BConds<T,Dim,M,C>&);
  Field(Layout_t &,const BConds<T,Dim,M,C>&,const GuardCellSizes<Dim>&);
  Field(FieldSpec<T,Dim,M,C>&);
  constexpr Field(Field<T,Dim,M,C>&) = default;

  // Constructors including a Mesh object as argument:
  Field(Mesh_t&, Layout_t &);
  Field(Mesh_t&, Layout_t &, const GuardCellSizes<Dim>&);
  Field(Mesh_t&, Layout_t &, const BConds<T,Dim,M,C>&);
  Field(Mesh_t&, Layout_t &, const GuardCellSizes<Dim>&,
	const BConds<T,Dim,M,C>&);
  Field(Mesh_t&, Layout_t &, const BConds<T,Dim,M,C>&,
	const GuardCellSizes<Dim>&);
  Field(Mesh_t&, FieldSpec<T,Dim,M,C>&);

  // Initialize the Field, if it was constructed from the default constructor.
  // This should NOT be called if the Field was constructed by providing
  // a FieldLayout or FieldSpec
  void initialize(Layout_t &);
  void initialize(Layout_t &, const GuardCellSizes<Dim>&);
  void initialize(Layout_t &, const BConds<T,Dim,M,C>&);
  void initialize(Layout_t &, const GuardCellSizes<Dim>&,
		  const BConds<T,Dim,M,C>&);
  void initialize(Layout_t &, const BConds<T,Dim,M,C>&,
		  const GuardCellSizes<Dim>&);
  void initialize(FieldSpec<T,Dim,M,C>&);

  // Initialize the Field, also specifying a mesh
  void initialize(Mesh_t&, Layout_t &);
  void initialize(Mesh_t&, Layout_t &, const bool); //UL: for pinned memory allocation
  void initialize(Mesh_t&, Layout_t &, const GuardCellSizes<Dim>&);
  void initialize(Mesh_t&, Layout_t &, const BConds<T,Dim,M,C>&);
  void initialize(Mesh_t&, Layout_t &, const GuardCellSizes<Dim>&,
		  const BConds<T,Dim,M,C>&);
  void initialize(Mesh_t&, Layout_t &, const BConds<T,Dim,M,C>&,
		  const GuardCellSizes<Dim>&);
  void initialize(Mesh_t&, FieldSpec<T,Dim,M,C>&);

  // Definitions for accessing boundary conditions.
  const bcond_value&     getBCond(int bc) const { return *(Bc[bc]); }
  bcond_value&           getBCond(int bc)       { return *(Bc[bc]); }
  const bcond_container& getBConds()      const { return Bc; }
  bcond_container&       getBConds()            { return Bc; }
  bcond_iterator         begin_BConds()         { return Bc.begin(); }
  bcond_iterator         end_BConds()           { return Bc.end(); }

  // Access to the mesh
  Mesh_t& get_mesh() const { return *mesh; }

  // When we apply a bracket it converts the type to IndexedField so
  // that we can check at compile time that we have the right number
  // of indexes and brackets.  There are a number of different types
  // which we can use to index a Field in order to get a reference to
  // a subset of that Field.
  IndexedField<T,Dim,1,M,C>   operator[](const Index&);
  IndexedField<T,Dim,1,M,C>   operator[](int);
  IndexedField<T,Dim,Dim,M,C> operator[](const NDIndex<Dim>&);
  SubField<T,Dim,M,C,SIndex<Dim> >  operator[](const SIndex<Dim>&);

  // Assignment from constants and other arrays.
  const Field<T,Dim,M,C>& operator=(T x) {
    assign(*this,x);
    return *this;
  }

  const Field<T,Dim,M,C>& operator=(const Field<T,Dim,M,C>& x) {
    assign(*this,x);
    return *this;
  }

  template<class X>
  const Field<T,Dim,M,C>& operator=(const BareField<X,Dim>& x) {
    assign(*this,x);
    return *this;
  }

  template<class B>
  const Field<T,Dim,M,C>& operator=(const PETE_Expr<B>& x) {
    assign(*this,x);
    return *this;
  }

  // If you make any modifications using an iterator, you must call this.
  void fillGuardCells(bool reallyFill = true) const;

  // I/O (special stuff not inherited from BareField):
  // Print out contents of Centering class
  void print_Centerings(std::ostream&);

  //
  // virtual functions for FieldLayoutUser's (and other UserList users)
  //

  // Repartition onto a new layout, or when the mesh changes
  virtual void Repartition(UserList *);

  // Tell this object that an object is being deleted
  virtual void notifyUserOfDelete(UserList *);

protected:
  // a virtual function which is called by this base class to get a
  // specific instance of DataSourceObject based on the type of data
  // and the connection method (the argument to the call).
  virtual DataSourceObject *createDataSourceObject(const char *, DataConnect *,
						   int);

private:
  // The boundary conditions.
  bcond_container Bc;

  // The Mesh object, and a flag indicating if we constructed it
  Mesh_t* mesh;
  bool WeOwnMesh;

  // store the given mesh object pointer, and the flag whether we own it or not.
  // if we own it, we must make sure to delete it when this Field is deleted.
  void store_mesh(Mesh_t*, bool);

  // delete the mesh object, if necessary; otherwise, just zero the pointer
  void delete_mesh();
};

#include "Field/Field.hpp"

#endif

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End:
