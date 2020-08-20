// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef FIELD_PRINT_H
#define FIELD_PRINT_H

// forward declarations
template<class T, unsigned Dim> class BareField;
template<unsigned Dim> class NDIndex;

// class FieldPrint
//----------------------------------------------------------------------
template<class T, unsigned Dim> 
class FieldPrint {

public:

  // attach a 2D Field to a FieldPrint
  FieldPrint(BareField<T,Dim>& f, unsigned parent = 0, int indexWidth = 3,
	     int dataWidth = 8, int dataPrecision = 4, 
	     int carReturn = -1, bool scientific = false) :
    MyField(f), Parent(parent),IndexWidth(indexWidth), DataWidth(dataWidth), 
    DataPrecision(dataPrecision), CarReturn(carReturn), Scientific(scientific) { };


  ~FieldPrint() { };
  void print(NDIndex<Dim>& view);

  // reset values
  void set_IndexWidth( unsigned in )    { IndexWidth = in; }
  void set_DataWidth( unsigned in )     { DataWidth = in; }
  void set_DataPrecision( unsigned in ) { DataPrecision = in; }
  void set_CarReturn( int in )          { CarReturn = in; }
  void set_Scientific( bool in )        { Scientific = in; }
  
private:
  BareField<T,Dim>& MyField;

  unsigned Parent;

  // formatting information
  unsigned IndexWidth; // the width for the Index fields
  unsigned DataWidth;  // the width for the Data fields
  unsigned DataPrecision; // the precision of the data
  int CarReturn; // how long before a carriage return
  bool Scientific;

};
//----------------------------------------------------------------------

#include "Utility/FieldPrint.hpp"

#endif // FIELD_PRINT_H

/***************************************************************************
 * $RCSfile: FieldPrint.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: FieldPrint.h,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $ 
 ***************************************************************************/
