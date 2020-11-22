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
#include "Utility/PAssert.h"


namespace ippl {

    KOKKOS_FUNCTION
    Index::Index()
    : first_m(0)
    , stride_m(0)
    , length_m(0)
    , baseFirst_m(0)
    { }

    KOKKOS_FUNCTION
    Index::Index(size_t n)
    : first_m(0)
    , stride_m(1)
    , length_m(n)
    , baseFirst_m(0)
    { }


    KOKKOS_FUNCTION
    Index::Index(int f, int l)
    : first_m(f)
    , stride_m(1)
    , length_m(l-f+1)
    , baseFirst_m(0)
    {
        PAssert_GE(l - f + 1, 0);
    }


    KOKKOS_FUNCTION
    Index::Index(int f, int l, int s)
    : first_m(f)
    , stride_m(s)
    , baseFirst_m(0)
    {
        PAssert_NE(s, 0);
        if ( f==l ) {
            length_m = 1;
        }
        else if ( (l>f) ^ (s<0) ) {
            length_m = (l-f)/s + 1;
        }
        else {
            length_m = 0;
        }
    }


    KOKKOS_INLINE_FUNCTION
    Index::Index(int m, int a, const Index &b)
    : first_m(b.first_m*m+a)
    , stride_m(b.stride_m*m)
    , length_m(b.length_m)
    , baseFirst_m(b.baseFirst_m)
    { }


    KOKKOS_INLINE_FUNCTION
    Index::Index(int f, int s, const Index *b)
    : first_m(f)
    , stride_m(s)
    , length_m(b->length_m)
    , baseFirst_m(b->baseFirst_m)
    { }


    KOKKOS_INLINE_FUNCTION
    int Index::first() const noexcept {
        return first_m;
    }


    KOKKOS_INLINE_FUNCTION
    int Index::stride() const noexcept {
        return stride_m;
    }


    KOKKOS_INLINE_FUNCTION
    bool Index::empty()  const noexcept {
        return length_m == 0;
    }


    KOKKOS_INLINE_FUNCTION
    size_t Index::length() const noexcept {
        return length_m;
    }


    KOKKOS_INLINE_FUNCTION
    int Index::last() const noexcept {
        return (length_m == 0) ? first_m : first_m + stride_m * (length_m - 1);
    }


    KOKKOS_INLINE_FUNCTION
    int Index::min() const noexcept {
        return (stride_m >= 0) ? first_m : first_m + stride_m * (length_m - 1);
    }


    KOKKOS_INLINE_FUNCTION
    int Index::max() const noexcept {
        return (stride_m >= 0) ? first_m + stride_m * (length_m - 1) : first_m;
    }


    KOKKOS_INLINE_FUNCTION
    Index operator+(const Index& i, int off) {
        return Index(1,off,i);
    }


    KOKKOS_INLINE_FUNCTION
    Index operator+(int off, const Index& i) {
        return Index(1,off,i);
    }


    KOKKOS_INLINE_FUNCTION
    Index operator-(const Index& i, int off) {
        return Index(1,-off,i);
    }


    KOKKOS_INLINE_FUNCTION
    Index operator-(int off, const Index& i) {
        return Index(-1,off,i);
    }


    KOKKOS_INLINE_FUNCTION
    Index operator-(const Index& i)  {
        return Index(-1,0,i);
    }


    KOKKOS_INLINE_FUNCTION
    Index operator*(const Index& i, int m) {
        return Index(m,0,i);
    }


    KOKKOS_INLINE_FUNCTION
    Index operator*(int m, const Index& i) {
        return Index(m,0,i);
    }


    KOKKOS_INLINE_FUNCTION
    Index operator/(const Index& i, int d) {
        return Index(i.first_m/d, i.stride_m/d, &i);
    }


    KOKKOS_INLINE_FUNCTION
    Index Index::reverse() const {
        Index j;
        j.first_m = last();
        j.length_m = length_m;
        j.stride_m = -stride_m;
        j.baseFirst_m = baseFirst_m;
        return j;
    }


    KOKKOS_INLINE_FUNCTION
    bool Index::touches(const Index&a) const {
        return (min() <= a.max()) && (max() >= a.min());
    }


    KOKKOS_INLINE_FUNCTION
    bool Index::contains(const Index&a) const {
        return (min() <= a.min()) && (max() >= a.max());
    }


    KOKKOS_INLINE_FUNCTION
    bool Index::split(Index& l, Index& r) const {
        PAssert_EQ(stride_m, 1);
        PAssert_GT(length_m, 1);
        int first = first_m;
        int length = length_m;
        int mid = first + length/2 - 1;
        l = Index(first, mid);
        r = Index(mid+1,first+length-1);
        return true;
    }


    KOKKOS_INLINE_FUNCTION
    bool Index::split(Index& l, Index& r, double a) const {
        PAssert_EQ(stride_m, 1);
        PAssert_GT(length_m, 1);
        PAssert_LT(a, 1.0);
        PAssert_GT(a, 0.0);
        int first = first_m;
        int length = length_m;
        int mid = first + static_cast<int>(length*a+0.5) - 1;
        l = Index(first, mid);
        r = Index(mid+1,first+length-1);
        return true;
    }
}