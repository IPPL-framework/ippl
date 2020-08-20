// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 * This program was prepared by PSI. 
 * All rights in the program are reserved by PSI.
 * Neither PSI nor the author(s)
 * makes any warranty, express or implied, or assumes any liability or
 * responsibility for the use of this software
 *
 * Visit www.amas.web.psi for more details
 *
 ***************************************************************************/

// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

//////////////////////////////////////////////////////////////////////
// Major functions and test code for Index.
// See main below for examples of use.
//////////////////////////////////////////////////////////////////////

// include files
#include "Index/Index.h"
#include "Utility/PAssert.h"


//////////////////////////////////////////////////////////////////////

std::ostream& operator<<(std::ostream& out, const Index& I) {
  
  out << '[' << I.first() << ':' << I.last() << ':' << I.stride() << ']';
  return out;
}


//////////////////////////////////////////////////////////////////////
// Calculate the least common multipple of s1 and s2.
// put the result in s.
// also calculate m1 = s/s1 and m2 = s/s2.
// This version is optimized for small s1 and s2 and 
// just uses an exhaustive search.
//////////////////////////////////////////////////////////////////////
inline 
void lcm(int s1, int s2, int &s, int &m1, int &m2)
{
  
  PAssert_GT(s1, 0);   // For simplicity, make some assumptions.
  PAssert_GT(s2, 0);
  int i1=s1;
  int i2=s2;
  int _m1 = 1;
  int _m2 = 1;
  if (i2<i1)
    while(true)
      {
	while (i2<i1)
	  {
	    i2 += s2;
	    ++_m2;
	  }
	if (i1==i2)
	  {
	    m1 = _m1;
	    m2 = _m2;
	    s  = i1;
	    return;
	  }
	i1 += s1;
	++_m1;
      }
  else
    while(true)
      {
	while (i1<i2)
	  {
	    i1 += s1;
	    ++_m1;
	  }
	if (i1==i2)
	  {
	    m1 = _m1;
	    m2 = _m2;
	    s  = i1;
	    return;
	  }
	i2 += s2;
	++_m2;
      }
}

//////////////////////////////////////////////////////////////////////

//
// Intersect, with the code for the common case of
// both strides equal to one.
//

Index
Index::intersect(const Index& rhs) const
{
  Index ret = DontInitialize() ;
  if ( (stride()==1) && (rhs.stride()==1) ) {
    int lf = first();
    int rf = rhs.first();
    int ll = last();
    int rl = rhs.last();
    int f = lf > rf ? lf : rf;
    int l = ll < rl ? ll : rl;
    ret.First = f;
    ret.Length = ( (l>=f) ? l-f+1 : 0 );
    ret.Stride = 1;
    ret.BaseFirst = BaseFirst + f - lf;
    ret.Base = Base;
  }
  else
    ret = general_intersect(rhs);
  return ret;
}

//////////////////////////////////////////////////////////////////////

static Index do_intersect(const Index &a, const Index &b)
{
  
  PAssert_GT(a.stride(), 0);		// This should be assured by the
  PAssert_GT(b.stride(), 0);		// caller of this function.

  int newStride;		// The stride for the new index is
  int a_mul,b_mul;		// a_mul=newStride/a.stride() ...
  lcm(a.stride(),b.stride(),	// The input strides...
      newStride,a_mul,b_mul);	// the lcm of the strides of a and b.
  
  // Find the offset from a.first() in units of newStride
  // that puts the ranges close together.
  int a_i = (b.first()-a.first())/a.stride();
  int a_off = a.first() + a_i*a.stride();
  if (a_off < b.first())
    {
      a_i++;
      a_off += a.stride();
    }
  PAssert_GE(a_off, b.first());	// make sure I'm understanding this right...

  // Now do an exhaustive search for the first point in common.
  // Count over all possible offsets for a.
  for (int a_m=0;(a_m<a_mul)&&(a_i<(int)a.length());a_m++,a_i++,a_off+=a.stride())
    {
      int b_off = b.first();
      // Count over all possible offsets for b.
      for (int b_m=0; (b_m<b_mul)&&(b_m<(int)b.length()); b_m++,b_off+=b.stride())
	if ( a_off == b_off )
	  {	// If the offsets are the same, we found it!
	    int am = a.max();	// Find the minimum maximum of a and b...
	    int bm = b.max();
	    int m = am < bm ? am : bm;
	    return Index(a_off, m, newStride);
	  }
    }
  return Index(0);		// If we get to here there is no intersection.
}

//////////////////////////////////////////////////////////////////////

Index Index::general_intersect(const Index& that) const
{
  
  // If they just don't overlap, return null indexes.
  if ( (min() > that.max()) || (that.min() > max()) )
    return Index(0);
  if ( (Stride==0) || (that.Stride==0) )
    return Index(0);

  // If one or the other counts -ve, reverse it and intersect result.
  if ( that.Stride < 0 )
    return intersect(that.reverse());
  if ( Stride < 0 )
    {
      Index r;
      r = reverse().intersect(that).reverse();
      int diff = (r.First-First)/Stride;
      PAssert_GE(diff, 0);
      r.BaseFirst = BaseFirst + diff;
      return r;
    }

  // Getting closer to the real thing: intersect them.
  // Pass the one that starts lower as the first argument.
  Index r;
  if ( First < that.First )
    r = do_intersect(*this,that);
  else
    r = do_intersect(that,*this);

  // Set the base so you can find what parts correspond
  // to the original interval.
  r.Base = Base;
  int diff = (r.First - First)/Stride;
  PAssert_GE(diff, 0);
  r.BaseFirst = BaseFirst + diff;
  return r;
}

//////////////////////////////////////////////////////////////////////

#ifdef DEBUG_INDEX
int main()
{
  
  const int N  = 16;		// Number of grid points.
  const int NP = 4;		// Number of processors.
  const int NL = N/NP;		// Grid points per processor.
  int p;			// processor counter.

  Index Ranges[NP];		// an index for each processor.
  for (p=0;p<NP;p++)		// On each processor
    Ranges[p] = Index(p*NL,(p+1)*NL-1); // Set the local range

  for (p=0;p<NP;p++)		// On each processor
    cout << Ranges[p] << endl;

  // work out A[Dest] = B[2*Dest];
  // Dest = [0...N/2-1]
  // Index Dest(N/2);
  // Index Src = 2*Dest;

  // Also try this:
  // Index Dest(N);
  // Index Src = N-1-Dest;

  // and this
  Index Dest(N);
  Index Src = Dest - 1;

  // another
  // Index Dest(0,N/2,2);
  // Index Src = Dest/2;

  // yet another
  // Index Dest = N-1-2*Index(N/2);
  // Index Src = N-1-Dest;

  cout << "Dest=" << Dest << endl;
  cout << "Src =" << Src  << endl;

  // Find out the gets from each processor for that operation.
  for (p=0; p<NP; p++)
    {
      cout << "On vp=" << p << ", range=" << Ranges[p] << endl;

      // Calculate what gets will be done.
      Index LDRange = Dest.intersect(Ranges[p]); // Local Destination Range for p
      Index SDRange = Src.plugBase(LDRange);     // Where that comes from.
      cout << "LDRange = " << LDRange << endl;
      cout << "SDRange = " << SDRange << endl;
      for (int pp=0; pp<NP; pp++)
	{              // Get from pp
	  Index LSDRange = SDRange.intersect(Ranges[pp]); // what comes from pp
	  if (!LSDRange.empty())
	    {
	      cout << "    from proc=" << pp << ", receive " << LSDRange << endl;
	    }
	}

      // Calculate the puts.
      Index LSRange = Src.intersect(Ranges[p]);
      Index DSRange = Dest.plugBase(LSRange);    // The destination for that.
      cout << "LSRange = " << LSRange << endl;
      cout << "DSRange = " << DSRange << endl;
      for (pp=0; pp<NP; pp++)
	{		       // Put to pp
	  Index LDSRange = LSRange.plugBase(DSRange.intersect(Ranges[pp]));
	  if (!LDSRange.empty())
	    {
	      cout << "    send to pp=" << pp << ", the range=" << LDSRange << endl;
	    }
	}
    }
}

#endif // DEBUG_INDEX

/***************************************************************************
 * $RCSfile: Index.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:27 $
 * IPPL_VERSION_ID: $Id: Index.cpp,v 1.1.1.1 2003/01/23 07:40:27 adelmann Exp $ 
 ***************************************************************************/
