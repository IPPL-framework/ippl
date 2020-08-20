// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef FIELD_DATA_SOURCE_H
#define FIELD_DATA_SOURCE_H

/***********************************************************************
 *
 * class FieldDataSource
 *
 * FieldDataSource is a specific version of DataSourceObject which takes
 * the data for a given Field and formats it properly for use by other
 * agencies.  This initial version collects all the data onto a parent node
 * and then formats it for the desired agency.  This is done by calling a
 * virtual method 'insert_data' for each LField worth of data collected onto
 * the master node.  Future versions will properly redistribute the data
 * based on the needs of the recipient.
 *
 * Subclasses must provide versions of the DataSourceObject virtual functions,
 * as well as the insert_data virtual function.
 *
 * This is a rewrite of some sections of FieldView.
 *
 ***********************************************************************/

// include files
#include "DataSource/DataSourceObject.h"


// forward declarations
template<class T, unsigned Dim, class Mesh, class Centering> class Field;
template<class T, unsigned Dim> class CompressedBrickIterator;
template<unsigned Dim> class NDIndex;


template<class T, unsigned Dim, class M, class C>
class FieldDataSource : public DataSourceObject {

public:
  // constructor: the name, the connection, the transfer method,
  // the field to connect, and the parent node
  FieldDataSource(const char *, DataConnect *, int, Field<T,Dim,M,C>&);

  // destructor
  virtual ~FieldDataSource();

protected:
  // the field to connect
  Field<T,Dim,M,C>& MyField;

  // the function which performs the work to gather data onto one node
  void gather_data();

  // copy the data out of the given LField iterator (which is occupying the
  // given domain) and into the library-specific structure
  virtual void insert_data(const NDIndex<Dim>&,
			   CompressedBrickIterator<T,Dim>) = 0;
};

#include "DataSource/FieldDataSource.hpp"

#endif // FIELD_DATA_SOURCE_H

/***************************************************************************
 * $RCSfile: FieldDataSource.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:25 $
 * IPPL_VERSION_ID: $Id: FieldDataSource.h,v 1.1.1.1 2003/01/23 07:40:25 adelmann Exp $ 
 ***************************************************************************/
