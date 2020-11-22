//
// Class Index
//   Define a slice in an array.
//
//   This essentially defines a list of evenly spaced numbers.
//   Most commonly this list will be increasing (positive stride)
//   but it can also have negative stride and be decreasing.
//
//   Index()      --> A null interval with no elements.
//   Index(n)     --> make an Index on [0..n-1]
//   Index(a,b)   --> make an Index on [a..b]
//   Index(a,b,s) --> make an Index on [a..b] with stride s
//
//   Example1:
//   --------
//   Index I(10);           --> Index on [0..9]
//   Index Low(5);          --> Index on [0..4]
//   Index High(5,9);       --> Index on [5..9]
//   Index IOdd(1,9,2);     --> Index on [1..9] stride 2
//   Index IEven(0,9,2);    --> Index on [0..9] stride 2
//
//   Given an Index I(a,n,s), and an integer j you can do the following:
//
//   I+j  : a+j+i*s        for i in [0..n-1]
//   j-I  : j-a-i*s
//   j*I  : j*a + i*j*s
//   I/j  : a/j + i*s/j
//
//   j/I we don't do because it is not a uniform stride, and we don't
//   allow strides that are fractions.
//
// Copyright (c) 2020, Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//
#include "Index/Index.h"
#include "Utility/PAssert.h"

namespace ippl {

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
        if (i2<i1) {
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
        }
        else {
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
    }


    //
    // Intersect, with the code for the common case of
    // both strides equal to one.
    //
    KOKKOS_INLINE_FUNCTION
    Index Index::intersect(const Index& rhs) const {
        Index ret = DontInitialize() ;
        if ( (stride()==1) && (rhs.stride()==1) ) {
            int lf = first();
            int rf = rhs.first();
            int ll = last();
            int rl = rhs.last();
            int f = lf > rf ? lf : rf;
            int l = ll < rl ? ll : rl;
            ret.first_m = f;
            ret.length_m = ( (l>=f) ? l-f+1 : 0 );
            ret.stride_m = 1;
            ret.baseFirst_m = baseFirst_m + f - lf;
            ret.base_m = base_m;
        }
        else
            ret = general_intersect(rhs);
        return ret;
    }


    static Index do_intersect(const Index &a, const Index &b)
    {
        PAssert_GT(a.stride(), 0);        // This should be assured by the
        PAssert_GT(b.stride(), 0);        // caller of this function.

        int newStride;        // The stride for the new index is
        int a_mul,b_mul;        // a_mul=newStride/a.stride() ...
        lcm(a.stride(),b.stride(),    // The input strides...
            newStride,a_mul,b_mul);    // the lcm of the strides of a and b.

        // Find the offset from a.first() in units of newStride
        // that puts the ranges close together.
        int a_i = (b.first()-a.first())/a.stride();
        int a_off = a.first() + a_i*a.stride();
        if (a_off < b.first())
        {
            a_i++;
            a_off += a.stride();
        }

        PAssert_GE(a_off, b.first());    // make sure I'm understanding this right...

        // Now do an exhaustive search for the first point in common.
        // Count over all possible offsets for a.
        for (int a_m=0;(a_m<a_mul)&&(a_i<(int)a.length());a_m++,a_i++,a_off+=a.stride())
        {
            int b_off = b.first();
            // Count over all possible offsets for b.
            for (int b_m=0; (b_m<b_mul)&&(b_m<(int)b.length()); b_m++,b_off+=b.stride())
                if ( a_off == b_off )
                {    // If the offsets are the same, we found it!
                    int am = a.max();    // Find the minimum maximum of a and b...
                    int bm = b.max();
                    int m = am < bm ? am : bm;
                    return Index(a_off, m, newStride);
                }
            }
        return Index(0);    // If we get to here there is no intersection.
    }


    Index Index::general_intersect(const Index& that) const
    {

        // If they just don't overlap, return null indexes.
        if ( (min() > that.max()) || (that.min() > max()) )
            return Index(0);
        if ( (stride_m==0) || (that.stride_m==0) )
            return Index(0);

        // If one or the other counts -ve, reverse it and intersect result.
        if ( that.stride_m < 0 )
            return intersect(that.reverse());
        if ( stride_m < 0 )
        {
            Index r;
            r = reverse().intersect(that).reverse();
            int diff = (r.first_m-first_m)/stride_m;
            PAssert_GE(diff, 0);
            r.baseFirst_m = baseFirst_m + diff;
            return r;
        }

        // Getting closer to the real thing: intersect them.
        // Pass the one that starts lower as the first argument.
        Index r;
        if ( first_m < that.first_m )
            r = do_intersect(*this,that);
        else
            r = do_intersect(that,*this);

        // Set the base so you can find what parts correspond
        // to the original interval.
        r.base_m = base_m;
        int diff = (r.first_m - first_m)/stride_m;
        PAssert_GE(diff, 0);
        r.baseFirst_m = baseFirst_m + diff;
        return r;
    }
}