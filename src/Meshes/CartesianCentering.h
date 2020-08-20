// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

// CartesianCentering.h
// CenteringEnum enumeration, CartesianCentering and related classes (all
// static). These represent all types of centering of Field's on Cartesian and
// UniformCartesian meshes. See also Centering.[h,cpp] for backwards-compatible
// centering classes Cell and Vert (which are still used as defaults for some
// other classes parameterized on centering, and which are different than
// things like CommonCartesianCenterings::allCell here because they are not
// parameterized at all).

#ifndef CARTESIAN_CENTERING_H
#define CARTESIAN_CENTERING_H

#include "Meshes/Centering.h"

#include <iostream>
#include <string>

// Enumeration of basic 1D (one-component) centering types:
// May add to this when unstructured comes in, and it means something to
// simply say FACE or EDGE centering (for cartesian meshes, face and edge
// centerings are a combination of CELL and VERTEX along directions):
enum CenteringEnum {CELL=0, VERTEX=1, VERT=1};

// Primary class for canned and user-defined cartesian centerings:
template<CenteringEnum* CE, unsigned Dim, unsigned NComponents=1U>
class CartesianCentering
{
public:
  static void print_Centerings(std::ostream&);  // Print function
  static std::string CenteringName;
};
template <CenteringEnum* CE, unsigned Dim, unsigned NComponents>
void CartesianCentering<CE,Dim,NComponents>::
print_Centerings(std::ostream& out)
{
  unsigned int i,j;
  out << CenteringName << std::endl;
  out << "Dim = " << Dim << " ; NComponents = " << NComponents << std::endl;
  for (i=0;i<Dim;i++) {
    for (j=0;j<NComponents;j++) {
      out << "centering[dim=" << i << "][component=" << j << "] = "
	  << Centering::CenteringEnum_Names[CE[j+i*NComponents]] << std::endl;
    }
  }
}

// Define some common CenteringEnum arrays for (logically) cartesian meshes.
// (Use CartesianCenterings with your own CenteringEnum* for those not 
// appearing here.)
// Typically, you'll want to use the CommonCartesianCenterings class's
// specializations in your actual code where you specify a centering parameter
// for a Field you're declaring (see CommonCartesianCenterings class below)

// N.B.: the name "CCCEnums" is a shortened form of the original name for this
// class, "CommonCartesianCenteringEnums"

template<unsigned Dim, unsigned NComponents=1U, unsigned Direction=0U>
struct CCCEnums
{
  // CenteringEnum arrays Classes with simple, descriptive names
  //---------------------------------------------------------------------
  // All components of Field cell-centered in all directions:
  //  static CenteringEnum allCell[NComponents*Dim];
  // All components of Field vertex-centered in all directions:
  //  static CenteringEnum allVertex[NComponents*Dim];
  // All components of Field face-centered in specified direction (meaning
  // vertex centered in that direction, cell-centered in others):
  //  static CenteringEnum allFace[NComponents*Dim];
  // All components of Field edge-centered along specified direction (cell
  // centered in that direction, vertex-centered in others):
  //  static CenteringEnum allEdge[NComponents*Dim];
  // Each vector component of Field face-centered in the corresponding
  // direction:
  //  static CenteringEnum vectorFace[NComponents*Dim];
  // Each vector component of Field edge-centered along the corresponding
  // direction:
  //  static CenteringEnum vectorEdge[NComponents*Dim];
  //---------------------------------------------------------------------
};

//***************CommonCartesianCenteringEnum Specializations******************

//11111111111111111111111111111111111111111111111111111111111111111111111111111
// 1D fields
//11111111111111111111111111111111111111111111111111111111111111111111111111111

// 1D field of scalars (or 1D vectors, or 1D tensors, or 1D sym. tensors)
template<>
struct CCCEnums<1U,1U,0U> {
  static CenteringEnum allCell[1U*1U];
  static CenteringEnum allVertex[1U*1U];
  // Componentwise centering along/perpendicular to component direction:
  static CenteringEnum vectorFace[1U*1U];
  static CenteringEnum vectorEdge[1U*1U];
  // Face/Edge centering perpendicular to/along direction 0:
  static CenteringEnum allFace[1U*1U];
  static CenteringEnum allEdge[1U*1U];
};


//22222222222222222222222222222222222222222222222222222222222222222222222222222
// 2D fields
//22222222222222222222222222222222222222222222222222222222222222222222222222222

// 2D field of scalars (or 1D vectors, or 1D tensors, or 1D sym. tensors)
template<>
struct CCCEnums<2U,1U,0U> {
  static CenteringEnum allCell[2U*1U];
  static CenteringEnum allVertex[2U*1U];
  // Face/Edge centering perpendicular to/along direction 0:
  static CenteringEnum allFace[2U*1U];
  static CenteringEnum allEdge[2U*1U];
};
template<>
struct CCCEnums<2U,1U,1U> {
  // Face/Edge centering perpendicular to/along direction 1:
  static CenteringEnum allFace[2U*1U];
  static CenteringEnum allEdge[2U*1U];
};

// 2D field of 2D vectors:
template<>
struct CCCEnums<2U,2U,0U> {
  static CenteringEnum allCell[2U*2U];
  static CenteringEnum allVertex[2U*2U];
  // Componentwise centering along/perpendicular to component direction:
  static CenteringEnum vectorFace[2U*2U];
  static CenteringEnum vectorEdge[2U*2U];
  // Face/Edge centering perpendicular to/along direction 0:
  static CenteringEnum allFace[2U*2U];
  static CenteringEnum allEdge[2U*2U];
};
template<>
struct CCCEnums<2U,2U,1U> {
  // Face/Edge centering perpendicular to/along direction 1:
  static CenteringEnum allFace[2U*2U];
  static CenteringEnum allEdge[2U*2U];
};

// 2D field of 2D tensors:
template<>
struct CCCEnums<2U,4U,0U> {
  static CenteringEnum allCell[4U*2U];
  static CenteringEnum allVertex[4U*2U];
  // Face/Edge centering perpendicular to/along direction 0:
  static CenteringEnum allFace[2U*4U];
  static CenteringEnum allEdge[2U*4U];
};
template<>
struct CCCEnums<2U,4U,1U> {
  // Face/Edge centering perpendicular to/along direction 1:
  static CenteringEnum allFace[2U*4U];
  static CenteringEnum allEdge[2U*4U];
};

// 2D field of 2D symmetric tensors:
template<>
struct CCCEnums<2U,3U,0U> {
  static CenteringEnum allCell[2U*3U];
  static CenteringEnum allVertex[2U*3U];
  // Face/Edge centering perpendicular to/along direction 0:
  static CenteringEnum allFace[2U*3U];
  static CenteringEnum allEdge[2U*3U];
};
template<>
struct CCCEnums<2U,3U,1U> {
  // Face/Edge centering perpendicular to/along direction 1:
  static CenteringEnum allFace[2U*3U];
  static CenteringEnum allEdge[2U*3U];
};


//33333333333333333333333333333333333333333333333333333333333333333333333333333
// 3D fields
//33333333333333333333333333333333333333333333333333333333333333333333333333333

// 3D field of scalars (or 1D vectors, or 1D tensors, or 1D sym. tensors)
template<>
struct CCCEnums<3U,1U,0U> {
  static CenteringEnum allCell[3U*1U];
  static CenteringEnum allVertex[3U*1U];
  // Face/Edge centering perpendicular to/along direction 0:
  static CenteringEnum allFace[3U*1U];
  static CenteringEnum allEdge[3U*1U];
};
template<>
struct CCCEnums<3U,1U,1U> {
  // Face/Edge centering perpendicular to/along direction 1:
  static CenteringEnum allFace[3U*1U];
  static CenteringEnum allEdge[3U*1U];
};
template<>
struct CCCEnums<3U,1U,2U> {
  // Face/Edge centering perpendicular to/along direction 2:
  static CenteringEnum allFace[3U*1U];
  static CenteringEnum allEdge[3U*1U];
};

// 3D field of 2D vectors:
template<>
struct CCCEnums<3U,2U,0U> {
  static CenteringEnum allCell[3U*2U];
  static CenteringEnum allVertex[3U*2U];
  // Face/Edge centering perpendicular to/along direction 0:
  static CenteringEnum allFace[3U*2U];
  static CenteringEnum allEdge[3U*2U];
};
template<>
struct CCCEnums<3U,2U,1U> {
  // Face/Edge centering perpendicular to/along direction 1:
  static CenteringEnum allFace[3U*2U];
  static CenteringEnum allEdge[3U*2U];
};
template<>
struct CCCEnums<3U,2U,2U> {
  // Face/Edge centering perpendicular to/along direction 2:
  static CenteringEnum allFace[3U*2U];
  static CenteringEnum allEdge[3U*2U];
};

// 3D field of 3D vectors:
template<>
struct CCCEnums<3U,3U,0U> {
  static CenteringEnum allCell[3U*3U];
  static CenteringEnum allVertex[3U*3U];
  // Componentwise centering along/perpendicular to component direction:
  static CenteringEnum vectorFace[3U*3U];
  static CenteringEnum vectorEdge[3U*3U];
  // Face/Edge centering perpendicular to/along direction 0:
  static CenteringEnum allFace[3U*3U];
  static CenteringEnum allEdge[3U*3U];
};
template<>
struct CCCEnums<3U,3U,1U> {
  // Face/Edge centering perpendicular to/along direction 1:
  static CenteringEnum allFace[3U*3U];
  static CenteringEnum allEdge[3U*3U];
};
template<>
struct CCCEnums<3U,3U,2U> {
  // Face/Edge centering perpendicular to/along direction 2:
  static CenteringEnum allFace[3U*3U];
  static CenteringEnum allEdge[3U*3U];
};

// 3D field of 3D tensors:
template<>
struct CCCEnums<3U,9U,0U> {
  static CenteringEnum allCell[3U*9U];
  static CenteringEnum allVertex[3U*9U];
  // Face/Edge centering perpendicular to/along direction 0:
  static CenteringEnum allFace[3U*9U];
  static CenteringEnum allEdge[3U*9U];
};
template<>
struct CCCEnums<3U,9U,1U> {
  // Face/Edge centering perpendicular to/along direction 1:
  static CenteringEnum allFace[3U*9U];
  static CenteringEnum allEdge[3U*9U];
};
template<>
struct CCCEnums<3U,9U,2U> {
  // Face/Edge centering perpendicular to/along direction 2:
  static CenteringEnum allFace[3U*9U];
  static CenteringEnum allEdge[3U*9U];
};

// 3D field of 3D symmetric tensors:
template<>
struct CCCEnums<3U,6U,0U> {
  static CenteringEnum allCell[3U*6U];
  static CenteringEnum allVertex[3U*6U];
  // Face/Edge centering perpendicular to/along direction 0:
  static CenteringEnum allFace[3U*6U];
  static CenteringEnum allEdge[3U*6U];
};
template<>
struct CCCEnums<3U,6U,1U> {
  // Face/Edge centering perpendicular to/along direction 1:
  static CenteringEnum allFace[3U*6U];
  static CenteringEnum allEdge[3U*6U];
};
template<>
struct CCCEnums<3U,6U,2U> {
  // Face/Edge centering perpendicular to/along direction 2:
  static CenteringEnum allFace[3U*6U];
  static CenteringEnum allEdge[3U*6U];
};



//-----------------------------------------------------------------------------

// Wrapper classes that wrap CCCEnums classes into CartesianCenterings classes;
// the canned typedefs in the canned specializations of these below are what
// the user will likely use.

template<unsigned Dim, unsigned NComponents=1U, unsigned Direction=0U>
struct CommonCartesianCenterings
{
  //public:
  //  typedef CartesianCentering<CCCEnums<Dim,NComponents,
  //Direction>::allCell, Dim, NComponents> allCell;
  //typedef CartesianCentering<CCCEnums<Dim,NComponents,
  //Direction>::allVertex, Dim, NComponents> allVertex;
  //typedef CartesianCentering<CCCEnums<Dim,NComponents,
  //Direction>::allFace, Dim, NComponents> allFace;
  //typedef CartesianCentering<CCCEnums<Dim,NComponents,
  //Direction>::allEdge, Dim, NComponents> allEdge;
  //typedef CartesianCentering<CCCEnums<Dim,NComponents,
  //Direction>::vectorFace, Dim, NComponents> vectorFace;
  //typedef CartesianCentering<CCCEnums<Dim,NComponents,
  //Direction>::vectorEdge, Dim, NComponents> vectorEdge;
};

//**********CommonCartesianCententerings specializations, typedefs*************


//11111111111111111111111111111111111111111111111111111111111111111111111111111
// 1D fields
//11111111111111111111111111111111111111111111111111111111111111111111111111111

// 1D field of scalars (or 1D vectors, or 1D tensors, or 1D sym. tensors)
template<>
struct CommonCartesianCenterings<1U,1U,0U>
{
  typedef CartesianCentering<CCCEnums<1U,1U,0U>::allCell,1U,1U>   allCell;
  typedef CartesianCentering<CCCEnums<1U,1U,0U>::allVertex,1U,1U> allVertex;
  // Componentwise centering along/perpendicular to component direction:
  typedef CartesianCentering<CCCEnums<1U,1U,0U>::vectorFace,1U,1U> vectorFace;
  typedef CartesianCentering<CCCEnums<1U,1U,0U>::vectorEdge,1U,1U> vectorEdge;
  // Face/Edge centering perpendicular to/along direction 0:
  typedef CartesianCentering<CCCEnums<1U,1U,0U>::allFace,1U,1U> allFace;
  typedef CartesianCentering<CCCEnums<1U,1U,0U>::allEdge,1U,1U> allEdge;
};


//22222222222222222222222222222222222222222222222222222222222222222222222222222
// 2D fields
//22222222222222222222222222222222222222222222222222222222222222222222222222222

// 2D field of scalars (or 1D vectors, or 1D tensors, or 1D sym. tensors)
template<>
struct CommonCartesianCenterings<2U,1U,0U>
{
  typedef CartesianCentering<CCCEnums<2U,1U,0U>::allCell,2U,1U>   allCell;
  typedef CartesianCentering<CCCEnums<2U,1U,0U>::allVertex,2U,1U> allVertex;
  // Face/Edge centering perpendicular to/along direction 0:
  typedef CartesianCentering<CCCEnums<2U,1U,0U>::allFace,2U,1U> allFace;
  typedef CartesianCentering<CCCEnums<2U,1U,0U>::allEdge,2U,1U> allEdge;
};
template<>
struct CommonCartesianCenterings<2U,1U,1U> {
  // Face/Edge centering perpendicular to/along direction 1:
  typedef CartesianCentering<CCCEnums<2U,1U,1U>::allFace,2U,1U> allFace;
  typedef CartesianCentering<CCCEnums<2U,1U,1U>::allEdge,2U,1U> allEdge;
};

// 2D field of 2D vectors:
template<>
struct CommonCartesianCenterings<2U,2U,0U>
{
  typedef CartesianCentering<CCCEnums<2U,2U,0U>::allCell,2U,2U>   allCell;
  typedef CartesianCentering<CCCEnums<2U,2U,0U>::allVertex,2U,2U> allVertex;
  // Componentwise centering along/perpendicular to component direction:
  typedef CartesianCentering<CCCEnums<2U,2U,0U>::vectorFace,2U,2U> vectorFace;
  typedef CartesianCentering<CCCEnums<2U,2U,0U>::vectorEdge,2U,2U> vectorEdge;
  // Face/Edge centering perpendicular to/along direction 0:
  typedef CartesianCentering<CCCEnums<2U,2U,0U>::allFace,2U,2U> allFace;
  typedef CartesianCentering<CCCEnums<2U,2U,0U>::allEdge,2U,2U> allEdge;
};
template<>
struct CommonCartesianCenterings<2U,2U,1U>
{
  // Face/Edge centering perpendicular to/along direction 1:
  typedef CartesianCentering<CCCEnums<2U,2U,1U>::allFace,2U,2U> allFace;
  typedef CartesianCentering<CCCEnums<2U,2U,1U>::allEdge,2U,2U> allEdge;
};

// 2D field of 2D tensors:
template<>
struct CommonCartesianCenterings<2U,4U,0U>
{
  typedef CartesianCentering<CCCEnums<2U,4U,0U>::allCell,2U,4U>   allCell;
  typedef CartesianCentering<CCCEnums<2U,4U,0U>::allVertex,2U,4U> allVertex;
  // Face/Edge centering perpendicular to/along direction 0:
  typedef CartesianCentering<CCCEnums<2U,4U,0U>::allFace,2U,4U> allFace;
  typedef CartesianCentering<CCCEnums<2U,4U,0U>::allEdge,2U,4U> allEdge;
};
template<>
struct CommonCartesianCenterings<2U,4U,1U>
{
  // Face/Edge centering perpendicular to/along direction 1:
  typedef CartesianCentering<CCCEnums<2U,4U,1U>::allFace,2U,4U> allFace;
  typedef CartesianCentering<CCCEnums<2U,4U,1U>::allEdge,2U,4U> allEdge;
};

// 2D field of 2D symmetric tensors:
template<>
struct CommonCartesianCenterings<2U,3U,0U>
{
  typedef CartesianCentering<CCCEnums<2U,3U,0U>::allCell,2U,3U>   allCell;
  typedef CartesianCentering<CCCEnums<2U,3U,0U>::allVertex,2U,3U> allVertex;
  // Face/Edge centering perpendicular to/along direction 0:
  typedef CartesianCentering<CCCEnums<2U,3U,0U>::allFace,2U,3U> allFace;
  typedef CartesianCentering<CCCEnums<2U,3U,0U>::allEdge,2U,3U> allEdge;
};
template<>
struct CommonCartesianCenterings<2U,3U,1U>
{
  // Face/Edge centering perpendicular to/along direction 1:
  typedef CartesianCentering<CCCEnums<2U,3U,1U>::allFace,2U,3U> allFace;
  typedef CartesianCentering<CCCEnums<2U,3U,1U>::allEdge,2U,3U> allEdge;
};


//33333333333333333333333333333333333333333333333333333333333333333333333333333
// 3D fields
//33333333333333333333333333333333333333333333333333333333333333333333333333333

// 3D field of scalars (or 1D vectors, or 1D tensors, or 1D sym. tensors)
template<>
struct CommonCartesianCenterings<3U,1U,0U>
{
  typedef CartesianCentering<CCCEnums<3U,1U,0U>::allCell,3U,1U>   allCell;
  typedef CartesianCentering<CCCEnums<3U,1U,0U>::allVertex,3U,1U> allVertex;
  // Face/Edge centering perpendicular to/along direction 0:
  typedef CartesianCentering<CCCEnums<3U,1U,0U>::allFace,3U,1U> allFace;
  typedef CartesianCentering<CCCEnums<3U,1U,0U>::allEdge,3U,1U> allEdge;
};
template<>
struct CommonCartesianCenterings<3U,1U,1U>
{
  // Face/Edge centering perpendicular to/along direction 1:
  typedef CartesianCentering<CCCEnums<3U,1U,1U>::allFace,3U,1U> allFace;
  typedef CartesianCentering<CCCEnums<3U,1U,1U>::allEdge,3U,1U> allEdge;
};
template<>
struct CommonCartesianCenterings<3U,1U,2U>
{
  // Face/Edge centering perpendicular to/along direction 2:
  typedef CartesianCentering<CCCEnums<3U,1U,2U>::allFace,3U,1U> allFace;
  typedef CartesianCentering<CCCEnums<3U,1U,2U>::allEdge,3U,1U> allEdge;
};

// 3D field of 2D vectors:
template<>
struct CommonCartesianCenterings<3U,2U,0U>
{
  typedef CartesianCentering<CCCEnums<3U,2U,0U>::allCell,3U,2U> allCell;
  typedef CartesianCentering<CCCEnums<3U,2U,0U>::allVertex,3U,2U> allVertex;
  // Face/Edge centering perpendicular to/along direction 0:
  typedef CartesianCentering<CCCEnums<3U,2U,0U>::allFace,3U,2U> allFace;
  typedef CartesianCentering<CCCEnums<3U,2U,0U>::allEdge,3U,2U> allEdge;
};
template<>
struct CommonCartesianCenterings<3U,2U,1U>
{
  // Face/Edge centering perpendicular to/along direction 1:
  typedef CartesianCentering<CCCEnums<3U,2U,1U>::allFace,3U,2U> allFace;
  typedef CartesianCentering<CCCEnums<3U,2U,1U>::allEdge,3U,2U> allEdge;
};
template<>
struct CommonCartesianCenterings<3U,2U,2U>
{
  // Face/Edge centering perpendicular to/along direction 2:
  typedef CartesianCentering<CCCEnums<3U,2U,2U>::allFace,3U,2U> allFace;
  typedef CartesianCentering<CCCEnums<3U,2U,2U>::allEdge,3U,2U> allEdge;
};

// 3D field of 3D vectors:
template<>
struct CommonCartesianCenterings<3U,3U,0U>
{
  typedef CartesianCentering<CCCEnums<3U,3U,0U>::allCell,3U,3U> allCell;
  typedef CartesianCentering<CCCEnums<3U,3U,0U>::allVertex,3U,3U> allVertex;
  // Componentwise centering along/perpendicular to component direction:
  typedef CartesianCentering<CCCEnums<3U,3U,0U>::vectorFace,3U,3U> vectorFace;
  typedef CartesianCentering<CCCEnums<3U,3U,0U>::vectorEdge,3U,3U> vectorEdge;
  // Face/Edge centering perpendicular to/along direction 0:
  typedef CartesianCentering<CCCEnums<3U,3U,0U>::allFace,3U,3U> allFace;
  typedef CartesianCentering<CCCEnums<3U,3U,0U>::allEdge,3U,3U> allEdge;
};
template<>
struct CommonCartesianCenterings<3U,3U,1U>
{
  // Face/Edge centering perpendicular to/along direction 1:
  typedef CartesianCentering<CCCEnums<3U,3U,1U>::allFace,3U,3U> allFace;
  typedef CartesianCentering<CCCEnums<3U,3U,1U>::allEdge,3U,3U> allEdge;
};
template<>
struct CommonCartesianCenterings<3U,3U,2U>
{
  // Face/Edge centering perpendicular to/along direction 2:
  typedef CartesianCentering<CCCEnums<3U,3U,2U>::allFace,3U,3U> allFace;
  typedef CartesianCentering<CCCEnums<3U,3U,2U>::allEdge,3U,3U> allEdge;
};


// 3D field of 3D tensors:
template<>
struct CommonCartesianCenterings<3U,9U,0U>
{
  typedef CartesianCentering<CCCEnums<3U,9U,0U>::allCell,3U,9U>  allCell;
  typedef CartesianCentering<CCCEnums<3U,9U,0U>::allVertex,3U,9U> allVertex;
  // Face/Edge centering perpendicular to/along direction 0:
  typedef CartesianCentering<CCCEnums<3U,9U,0U>::allFace,3U,9U> allFace;
  typedef CartesianCentering<CCCEnums<3U,9U,0U>::allEdge,3U,9U> allEdge;
};
template<>
struct CommonCartesianCenterings<3U,9U,1U>
{
  // Face/Edge centering perpendicular to/along direction 1:
  typedef CartesianCentering<CCCEnums<3U,9U,1U>::allFace,3U,9U> allFace;
  typedef CartesianCentering<CCCEnums<3U,9U,1U>::allEdge,3U,9U> allEdge;
};
template<>
struct CommonCartesianCenterings<3U,9U,2U>
{
  // Face/Edge centering perpendicular to/along direction 2:
  typedef CartesianCentering<CCCEnums<3U,9U,2U>::allFace,3U,9U> allFace;
  typedef CartesianCentering<CCCEnums<3U,9U,2U>::allEdge,3U,9U> allEdge;
};

// 3D field of 3D symmetric tensors:
template<>
struct CommonCartesianCenterings<3U,6U,0U>
{
  typedef CartesianCentering<CCCEnums<3U,6U,0U>::allCell,3U,6U> allCell;
  typedef CartesianCentering<CCCEnums<3U,6U,0U>::allVertex,3U,6U> allVertex;
  // Face/Edge centering perpendicular to/along direction 0:
  typedef CartesianCentering<CCCEnums<3U,6U,0U>::allFace,3U,6U> allFace;
  typedef CartesianCentering<CCCEnums<3U,6U,0U>::allEdge,3U,6U> allEdge;
};
template<>
struct CommonCartesianCenterings<3U,6U,1U>
{
  // Face/Edge centering perpendicular to/along direction 1:
  typedef CartesianCentering<CCCEnums<3U,6U,1U>::allFace,3U,6U> allFace;
  typedef CartesianCentering<CCCEnums<3U,6U,1U>::allEdge,3U,6U> allEdge;
};
template<>
struct CommonCartesianCenterings<3U,6U,2U>
{
  // Face/Edge centering perpendicular to/along direction 2:
  typedef CartesianCentering<CCCEnums<3U,6U,2U>::allFace,3U,6U> allFace;
  typedef CartesianCentering<CCCEnums<3U,6U,2U>::allEdge,3U,6U> allEdge;
};

#include "Meshes/CartesianCentering.hpp"

#endif // CARTESIAN_CENTERING_H

/***************************************************************************
 * $RCSfile: CartesianCentering.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:28 $
 * IPPL_VERSION_ID: $Id: CartesianCentering.h,v 1.1.1.1 2003/01/23 07:40:28 adelmann Exp $ 
 ***************************************************************************/
