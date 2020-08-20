// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef MAKE_DATA_SOURCE_H
#define MAKE_DATA_SOURCE_H

/***********************************************************************
 * 
 * make_DataSourceObject is essentially a factory function
 * for producing different specific types of DataSourceObjects.
 * It asks the IpplInfo object which type of data source to
 * use, and then makes the appropriate DataSourceObject subclass.
 *
 ***********************************************************************/

// forward declarations
template<class T, unsigned Dim, class M, class C> class Field;
template<class T> class ParticleAttrib;
template<class PLayout> class IpplParticleBase;
class DataSourceObject;
class DataConnect;


// a version of make_DataSourceObject for Field's.
// arguments: name, connection type, transfer metohd, Field
template<class T, unsigned Dim, class M, class C>
DataSourceObject *
make_DataSourceObject(const char *, DataConnect *, int, Field<T,Dim,M,C>&);


// a version of make_DataSourceObject for ParticleAttrib's.
// arguments: name, connection type, transfer method, ParticleAttrib
template<class T>
DataSourceObject *
make_DataSourceObject(const char *, DataConnect *, int, ParticleAttrib<T>&);


// a version of make_DataSourceObject for ParticleAttrib's.
// arguments: name, connection type, transfer method, IpplParticleBase
template<class PLayout>
DataSourceObject *
make_DataSourceObject(const char *,DataConnect *,int,IpplParticleBase<PLayout>&);

#include "DataSource/MakeDataSource.hpp"

#endif // MAKE_DATA_SOURCE_H

/***************************************************************************
 * $RCSfile: MakeDataSource.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:25 $
 * IPPL_VERSION_ID: $Id: MakeDataSource.h,v 1.1.1.1 2003/01/23 07:40:25 adelmann Exp $ 
 ***************************************************************************/
