/***************************************************************************
 *
 * The IPPL Framework
 *
  ***************************************************************************/

#ifndef PASSERT_H
#define PASSERT_H
#include "Utility/IpplInfo.h"

#include <exception>
#include <stdexcept>
#include <string>
//////////////////////////////////////////////////////////////////////
//
// This is a compile time assert.
// That is, if you say:
//   CTAssert<true>::test();
// it compiles just fine and inserts no code.
// If you say:
//   CTAssert<false>::test();
// you get a compile error that it can't find CTAssert<false>::test().
//
// The template argument can of course be a calculation of const bools
// that are known at compile time.
//
//////////////////////////////////////////////////////////////////////

template<bool B> struct IpplCTAssert {};

template<> struct IpplCTAssert<true> { static void test() {} };

#if defined(NOCTAssert)
#define CTAssert(c)
#else
#define CTAssert(c) IpplCTAssert<(c)>::test()
#endif

//===========================================================================//
// class assertion - exception notification class for assertions

// This class should really be derived from std::runtime_error, but
// unfortunately we don't have good implementation of the library standard
// yet, on compilers other than KCC.  So, this class will keep with the
// "what" method evidenced in the standard, but dispense with inheriting from
// classes for which we don't have implementations...
//===========================================================================//

class assertion: public std::runtime_error
{
    char *msg;
public:
    assertion( const char *cond, const char *file, int line );
    assertion( const char *m );
    assertion( const assertion& a );
    ~assertion() throw() { delete[] msg; }
    assertion& operator=( const assertion& a );

    using std::runtime_error::what;
    virtual const char* what() { return msg; };
};

//---------------------------------------------------------------------------//
// Now we define a run time assertion mechanism.  We will call it "PAssert",
// to reflect the idea that this is for use in IPPL per se, recognizing that
// there are numerous other assertion facilities in use in client codes.
//---------------------------------------------------------------------------//

// These are the functions that will be called in the assert macros.
void toss_cookies( const char *cond, const char *file, int line );
template <class S, class T>
void toss_cookies( const char *cond, const char *astr, const char *bstr, S a, T b, const char *file, int line) {

    std::string what = "Assertion '" + std::string(cond) + "' failed. \n";
    what += std::string(astr) + " = " + std::to_string(a) + ", ";
    what += std::string(bstr) + " = " + std::to_string(b) + "\n";
    what += "in \n";
    what += std::string(file) + ", line  " + std::to_string(line);

    throw std::runtime_error(what);
}

void insist( const char *cond, const char *msg, const char *file, int line );

//---------------------------------------------------------------------------//
// The PAssert macro is intended to be used for validating preconditions
// which must be true in order for following code to be correct, etc.  For
// example, PAssert( x > 0. ); y = sqrt(x);  If the assertion fails, the code
// should just bomb.  Philosophically, it should be used to feret out bugs in
// preceding code, making sure that prior results are within reasonable
// bounds before proceeding to use those results in further computation, etc.
//---------------------------------------------------------------------------//

#ifdef NOPAssert
#define PAssert(c)
#define PAssert_EQ(a, b)
#define PAssert_NE(a, b)
#define PAssert_LT(a, b)
#define PAssert_LE(a, b)
#define PAssert_GT(a, b)
#define PAssert_GE(a, b)
#else
#define PAssert(c) if (!(c)) toss_cookies( #c, __FILE__, __LINE__ );
#define PAssert_CMP(cmp, a, b) if (!(cmp)) toss_cookies(#cmp, #a, #b, a, b, __FILE__, __LINE__);
#define PAssert_EQ(a, b) PAssert_CMP(a == b, a, b)
#define PAssert_NE(a, b) PAssert_CMP(a != b, a, b)
#define PAssert_LT(a, b) PAssert_CMP(a < b, a, b)
#define PAssert_LE(a, b) PAssert_CMP(a <= b, a, b)
#define PAssert_GT(a, b) PAssert_CMP(a > b, a, b)
#define PAssert_GE(a, b) PAssert_CMP(a >= b, a, b)
#endif

//---------------------------------------------------------------------------//
// The PInsist macro is akin to the PAssert macro, but it provides the
// opportunity to specify an instructive message.  The idea here is that you
// should use Insist for checking things which are more or less under user
// control.  If the user makes a poor choice, we "insist" that it be
// corrected, providing a corrective hint.
//---------------------------------------------------------------------------//

#define PInsist(c,m) if (!(c)) insist( #c, m, __FILE__, __LINE__ );

//---------------------------------------------------------------------------//
// NOTE:  We provide a way to eliminate assertions, but not insistings.  The
// idea is that PAssert is used to perform sanity checks during program
// development, which you might want to eliminate during production runs for
// performance sake.  PInsist is used for things which really really must be
// true, such as "the file must've been opened", etc.  So, use PAssert for
// things which you want taken out of production codes (like, the check might
// inhibit inlining or something like that), but use PInsist for those things
// you want checked even in a production code.
//---------------------------------------------------------------------------//

#endif // PASSERT_H

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End:
