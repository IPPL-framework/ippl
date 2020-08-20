// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef FIELD_DEBUG_PRINT_H
#define FIELD_DEBUG_PRINT_H

// forward declarations
template<class T, unsigned Dim> class BareField;
template<unsigned Dim> class NDIndex;
class Inform;


template<class T, unsigned Dim> 
class FieldDebugPrint {

public:
  // constructor: set up to print with the given format options, and if we
  // print the values of boundary cells
  // if they would be included in the specified domain in the print function.
  FieldDebugPrint(bool printbc = false,
		  unsigned int dataWidth = 0, unsigned int dataPrecision = 0, 
		  unsigned int carReturn = 0, bool scientific = true) :
    DataWidth(dataWidth), DataPrecision(dataPrecision),
    CarReturn(carReturn), Scientific(scientific), PrintBC(printbc) { }

  // Destructor: nothing to do
  ~FieldDebugPrint() { }

  // print out all the values in the field, or just a subset.  If the
  // final boolean argument is true, execute the routine as if it were
  // being run on all the nodes at once.  If the argument is false, assume
  // it is being run by nodes separately, and do not do any communication.
  void print(BareField<T,Dim>&, const NDIndex<Dim>&, Inform&, bool = true);
  void print(BareField<T,Dim>&, Inform&, bool = true);
  void print(BareField<T,Dim>&, const NDIndex<Dim>&, bool = true);
  void print(BareField<T,Dim>&, bool = true);

  // set values for how to format the data
  void set_DataWidth(unsigned int in) { DataWidth = in; }
  void set_DataPrecision(unsigned int in) { DataPrecision = in; }
  void set_CarReturn(unsigned int in) { CarReturn = in; }
  void set_Scientific(bool in) { Scientific = in; }
  void set_PrintBC(bool in) { PrintBC = in; }

  // query the current values for how to format the data
  unsigned int get_DataWidth() const { return DataWidth; }
  unsigned int get_DataPrecision() const { return DataPrecision; }
  unsigned int get_CarReturn() const { return CarReturn; }
  bool get_Scientific() const { return Scientific; }
  bool get_PrintBC() const { return PrintBC; }

private:
  // formatting information
  unsigned int DataWidth;	// the width for the Data fields
  unsigned int DataPrecision;	// the precision of the data
  unsigned int CarReturn;	// how long before a carriage return
  bool Scientific;		// print data in scientific format?
  bool PrintBC;			// print boundary condition cells?

  // print a single value to the screen
  void printelem(bool, T&, unsigned int, Inform&);
};

#include "Utility/FieldDebugPrint.hpp"

#endif

/***************************************************************************
 * $RCSfile: FieldDebugPrint.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: FieldDebugPrint.h,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $ 
 ***************************************************************************/
