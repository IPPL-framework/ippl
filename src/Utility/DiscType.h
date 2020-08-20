// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef DISC_TYPE_H
#define DISC_TYPE_H

/***************************************************************************
 * DiscType<T> is a simple class that returns a string via the static method
 * "str()" based on the type T.  It is specialized for our standard types,
 * and has a default value of "u" for "user" or "unknown" types.
 ***************************************************************************/

// include files
#include "AppTypes/Vektor.h"
#include "AppTypes/Tenzor.h"
#include "AppTypes/SymTenzor.h"
#include "AppTypes/AntiSymTenzor.h"

////////////////////////////////////////////////////////////////////////////
// a base class for all DiscType's, that provides common routines to
// parse a type string and determine the apptype, the dimension, and the
// scalar type
struct DiscTypeBase {
  // enums for scalar types and apptypes
  enum { CHAR, SHORT, INT, LONG, FLOAT, DOUBLE, DCOMPLEX, FCOMPLEX,
	 SCALAR, VEKTOR, TENZOR, SYMTENZOR, ANTISYMTENZOR, UNKNOWN };

  // determine the scalar data type for a given type string
  static int scalarType(const std::string &s) {
    if (s.length() == 0 || s.length() == 2 || s.length() > 3)
      return UNKNOWN;
    char c = static_cast<char>(tolower(s[s.length() - 1]));
    if (c == 'c')
      return CHAR;
    else if (c == 's')
      return SHORT;
    else if (c == 'i')
      return INT;
    else if (c == 'l')
      return LONG;
    else if (c == 'f')
      return FLOAT;
    else if (c == 'd')
      return DOUBLE;
    else if (c == 'y')
      return FCOMPLEX;
    else if (c == 'z')
      return DCOMPLEX;
    else
      return UNKNOWN;
  }

  // determine the dimension of the apptype.  If it is a scalar, return 0.
  // if there is an error, return -1.
  static int dim(const std::string &s) {
    if (s.length() == 0 || s.length() == 2 || s.length() > 3)
      return (-1);
    if (s.length() == 1)
      return 0;
    char c = s[1];
    if (c < '1' || c > '9')
      return (-1);
    return (c - '1' + 1);
  }

  // determine the apptype of the string
  static int appType(const std::string &s) {
    if (s.length() == 0 || s.length() == 2 || s.length() > 3)
      return UNKNOWN;
    if (s.length() == 1)
      return SCALAR;
    char c = static_cast<char>(tolower(s[0]));
    if (c == 'v')
      return VEKTOR;
    else if (c == 't')
      return TENZOR;
    else if (c == 's')
      return SYMTENZOR;
    else if (c == 'a')
      return ANTISYMTENZOR;
    else
      return UNKNOWN;
  }
};


////////////////////////////////////////////////////////////////////////////
// the general class, that is the default if the user does not specify
// one of the specialized types below
template<class T>
struct DiscType : public DiscTypeBase {
  static std::string str() { return std::string("u"); }
};


////////////////////////////////////////////////////////////////////////////
// specializations for char, short, int, float, double
// codes:
//   c = char
//   s = short
//   i = int
//   l = long
//   f = float
//   d = double
//   z = complex<double>
//   y = complex<float>
#define DEFINE_DISCTYPE_SCALAR(TYPE, STRING)		\
template<>						\
struct DiscType< TYPE > : public DiscTypeBase {		\
  static std::string str() { return std::string(STRING); }	\
};

DEFINE_DISCTYPE_SCALAR(char,   "c")
DEFINE_DISCTYPE_SCALAR(short,  "s")
DEFINE_DISCTYPE_SCALAR(int,    "i")
DEFINE_DISCTYPE_SCALAR(long,   "l")
DEFINE_DISCTYPE_SCALAR(float,  "f")
DEFINE_DISCTYPE_SCALAR(double, "d")

DEFINE_DISCTYPE_SCALAR(unsigned char,   "c")
DEFINE_DISCTYPE_SCALAR(unsigned short,  "s")
DEFINE_DISCTYPE_SCALAR(unsigned int,    "i")
DEFINE_DISCTYPE_SCALAR(unsigned long,   "l")

DEFINE_DISCTYPE_SCALAR(std::complex<float>,  "y")
DEFINE_DISCTYPE_SCALAR(std::complex<double>, "z")

////////////////////////////////////////////////////////////////////////////
// specializations for Vektor, Tenzor, SymTenzor, AntiSymTenzor
// codes:
//   vNs = Vektor of dimension N and scalar type s
//   tNs = Tenzor of dimension N and scalar type s
//   sNs = SymTenzor of dimension N and scalar type s
//   aNs = AntiSymTenzor of dimension N and scalar type s
#define DEFINE_DISCTYPE_APPTYPE(TYPE, STRING)		\
template<class T, unsigned D>				\
struct DiscType< TYPE<T, D> > : public DiscTypeBase {	\
  static std::string str() {					\
    CTAssert(D < 10);					\
    std::string retval = STRING;				\
    retval += "0";					\
    retval += DiscType<T>::str();			\
    retval[1] += D;					\
    return retval;					\
  }							\
};

DEFINE_DISCTYPE_APPTYPE(Vektor,        "v")
DEFINE_DISCTYPE_APPTYPE(Tenzor,        "t")
DEFINE_DISCTYPE_APPTYPE(SymTenzor,     "s")
DEFINE_DISCTYPE_APPTYPE(AntiSymTenzor, "a")

#endif

/***************************************************************************
 * $RCSfile: DiscType.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: DiscType.h,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $ 
 ***************************************************************************/
