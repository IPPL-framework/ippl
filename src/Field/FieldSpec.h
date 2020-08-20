// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef FIELD_SPEC_H
#define FIELD_SPEC_H

// forward declarations
template< unsigned Dim, class T > class UniformCartesian;
template< class T, unsigned Dim , class M, class C > class BConds;
template< unsigned Dim > class GuardCellSizes;
template< unsigned Dim > class FieldLayout;

// A simple container class for the three entities used to create a Field:
// FieldLayout, Bconds, and GaurdCellSizes.
// The FieldLayout must be a reference since fields on different layouts
// will not know how to talk to each other in parallel

template<class T, unsigned Dim,
  class M=UniformCartesian<Dim,double>,class C= typename M::DefaultCentering>
class FieldSpec 
{
public:
  // Bconds and GuardCellSizes have default constructors, so
  // they can be set later. However, we can not have a default
  // constructor for FieldSpec since a reference to a layout
  // is required which is common to all FieldSpecs
  FieldSpec(const FieldLayout<Dim>& layout) : Layout(layout) { };
  
  FieldSpec(const FieldLayout<Dim>& layout, 
	    const BConds<T,Dim,M,C>& bc, 
	    const GuardCellSizes<Dim>& gc) : 
    Layout(layout), BC(bc), GC(gc) { };
  
  FieldSpec(const FieldLayout<Dim>& layout, 
	    const GuardCellSizes<Dim>& gc,
	    const BConds<T,Dim,M,C>& bc) : 
    Layout(layout), BC(bc), GC(gc) { };
  
  FieldSpec(const FieldSpec<T,Dim,M,C>& rhs ) :
    Layout(rhs.get_Layout()), BC(rhs.get_BC()), GC(rhs.get_GC() ) { };

  ~FieldSpec(void) { };
  
  const FieldLayout<Dim>& get_Layout(void) const { return Layout; }
  BConds<T,Dim,M,C> get_BC(void)  const { return BC; } 
  GuardCellSizes<Dim> get_GC(void)  const { return GC; }
  void set_BC(const BConds<T,Dim,M,C>& bc)  { BC = bc; } 
  void set_GC(const GuardCellSizes<Dim>& gc)  { GC = gc; }

  FieldSpec<T,Dim,M,C>& operator=(const FieldSpec<T,Dim,M,C>& rhs) {
    BC = rhs.get_BC();
    GC = rhs.get_GC();
    if( &Layout != &(rhs.get_Layout()) ) {
      ERRORMSG("FieldSpec::op= - FieldLayouts must be the same"<<endl);
    }
    return *this;
  }

private:
  
  const FieldLayout<Dim>& Layout;
  BConds<T,Dim,M,C> BC; 
  GuardCellSizes<Dim> GC;

};

#endif // FIELD_SPEC_H

/***************************************************************************
 * $RCSfile: FieldSpec.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:26 $
 * IPPL_VERSION_ID: $Id: FieldSpec.h,v 1.1.1.1 2003/01/23 07:40:26 adelmann Exp $ 
 ***************************************************************************/
