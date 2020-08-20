// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

// Centering.h
// Vert and Cell classes used to indicate Field centering method; see also
// the file CartesianCentering.h for more centering classes.

#ifndef CENTERING_H
#define CENTERING_H

#include <iostream>

// Cell-centered, all components, for all dimensions in cartesian-mesh case.
// Recommendation: use CommonCartesianCenterings<D,NComponents,0U>::allCell
// instead of this for cartesian meshes.
// Keep this class around for backwards compatibility, and possibly for use
// with non-cartesian meshes.
class Centering
{
public:
    static const char* CenteringEnum_Names[3];
};

class Cell
{
public:
  static const char* CenteringName;
  static void print_Centerings(std::ostream&);
};
// Vertex-centered, all components, for all dimensions in cartesian-mesh case.
// Recommendation: use CommonCartesianCenterings<D,NComponents,0U>::allVert
// instead of this for cartesian meshes.
// Keep this class around for backwards compatibility, and possibly for use
// with non-cartesian meshes.
class Vert
{
public:
  static const char* CenteringName;
  static void print_Centerings(std::ostream&);
};

class Edge
{
public:
    static const char* CenteringName;
    static void print_Centerings(std::ostream&);
};

#endif // CENTERING_H

/***************************************************************************
 * $RCSfile: Centering.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:28 $
 * IPPL_VERSION_ID: $Id: Centering.h,v 1.1.1.1 2003/01/23 07:40:28 adelmann Exp $
 ***************************************************************************/