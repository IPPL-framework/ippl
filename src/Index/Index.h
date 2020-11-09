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
#ifndef IPPL_INDEX_H
#define IPPL_INDEX_H



//
//
// All of this would be quite straightforward in serial, the fun
// begins in parallel. Indexes are not parallel objects, but they
// have operations defined for them for making regular section
// transfer calculations easier: intersect and plugBase.
//
// The idea is that we represent the part of A and B that reside on
// various processors with an Index. That index is the local domain
// of the array.

#include <Kokkos_Core.hpp>

#include "Expression/IpplExpressions.h"

#include <iostream>

namespace ippl {
    class Index : public ippl::detail::Expression<Index, 3 * sizeof(int) + 2 * sizeof(size_t) /*need to check*/>
    {
    public:
        class iterator
        {
        public:
            iterator()                          : Current(0)      , stride_m(0)      {}
            iterator(int current, int stride=1) : Current(current), stride_m(stride) {}

            int operator*() { return Current ; }
            iterator operator--(int)
            {
                iterator tmp = *this;
                Current -= stride_m;             // Post decrement
                return tmp;
            }
            iterator& operator--()
            {
                Current -= stride_m;
                return (*this);
            }
            iterator operator++(int)
            {
                iterator tmp = *this;
                Current += stride_m;              // Post increment
                return tmp;
            }
            iterator& operator++()
            {
                Current += stride_m;
                return (*this);
            }
            iterator& operator+=(int i)
            {
                Current += (stride_m * i);
                return *this;
            }
            iterator& operator-=(int i)
            {
                Current -= (stride_m * i);
                return *this;
            }
            iterator operator+(int i) const
            {
                return iterator(Current+i*stride_m,stride_m);
            }
            iterator operator-(int i) const
            {
                return iterator(Current-i*stride_m,stride_m);
            }
            int operator[](int i) const
            {
                return Current + i * stride_m;
            }
            bool operator==(const iterator &y) const
            {
                return (Current == y.Current) && (stride_m == y.stride_m);
            }
            bool operator<(const iterator &y) const
            {
                return (Current < y.Current)||
                ((Current==y.Current)&&(stride_m<y.stride_m));
            }
            bool operator!=(const iterator &y) const { return !((*this) == y); }
            bool operator> (const iterator &y) const { return y < (*this); }
            bool operator<=(const iterator &y) const { return !(y < (*this)); }
            bool operator>=(const iterator &y) const { return !((*this) < y); }
        private:
            int Current;
            int stride_m;
        };
 
        /*!
         * Instantiate Index without any range.
         */
        Index();

        /*!
         * Instantiate Index with range [0, ..., n-1]
         * @param n number of elements
         */
        inline Index(size_t n);

        /*!
         * Instantiate Index with user-defined lower and upper
         * bound [f, ..., l].
         * @param f first element
         * @param l last element
         */
        inline Index(int f, int l);


        inline Index(int f, int l, int s);	// First to Last using Step.

        ~Index() = default;

        int id() const { return Base; }

        /*!
         * @returns the smallest element
         */
        inline int min() const;

        /*!
         * @returns the largest element
         */
        inline int max() const;

        /*!
         * @returns the number of elements
         */
        inline size_t length() const;

        /*!
         * @returns the stride
         */
        inline int stride() const;

        /*!
         * @returns the first element
         */
        inline int first() const;

        /*!
         * @returns the last element
         */
        inline int last() const;

        /*!
         * @returns true if empty, otherwise false
         */
        inline bool empty() const;

        /*!
         * @returns the id from the base index
         */
        inline int getBase() const;

        // Additive operations.
        friend inline Index operator+(const Index&,int);
        friend inline Index operator+(int,const Index&);
        friend inline Index operator-(const Index&,int);
        friend inline Index operator-(int,const Index&);

        // Multipplicative operations.
        friend inline Index operator-(const Index&);
        friend inline Index operator*(const Index&,int);
        friend inline Index operator*(int,const Index&);
        friend inline Index operator/(const Index&,int);

        // Intersect with another Index.
        Index intersect(const Index &) const;

        // Plug the base range of one into another.
        inline Index plugBase(const Index &) const;

        // Test to see if two indexes are from the same base.
        inline bool sameBase(const Index&) const;

        // Test to see if there is any overlap between two Indexes.
        inline bool touches (const Index&a) const;
        // Test to see if one contains another (endpoints only)
        inline bool contains(const Index&a) const;
        // Test to see if one contains another (all points)
        inline bool containsAllPoints(const Index &b) const;
        // Split one into two.
        inline bool split(Index& l, Index& r) const;
        // Split index into two with a ratio between 0 and 1.
        inline bool split(Index& l, Index& r, double a) const;

        // iterator begin
        iterator begin() { return iterator(first_m,stride_m); }
        // iterator end
        iterator end() { return iterator(first_m+stride_m*length_m,stride_m); }

        // An operator< so we can impose some sort of ordering.
        bool operator<(const Index& r) const
        {
            return (   (length_m< r.length_m) ||
                        ( (length_m==r.length_m) && (  (first_m<r.first_m) ||
                                            ( (first_m==r.first_m) && (length_m>0) && (stride_m<r.stride_m) ) ) ) );
        }
        // Test for equality.
        bool operator==(const Index& r) const
        {
            return (length_m==r.length_m) && (first_m==r.first_m) && (stride_m==r.stride_m);
        }

        static void findPut(const Index&,const Index&, const Index&,Index&,Index&);

    private:
        int first_m;        /// First index element
        int stride_m;
        size_t length_m;    /// The number of elements
  
        /*! he first element of the base index.
         * This gets updated whenever we do index or set operations
         * so we can do inverses quickly and easily.
         */
        size_t baseFirst_m;

        // Keep id for the base so we can tell when two
        // indexes come from the same base.
        int Base;

        // Make an Index that interally counts the other direction.
        inline Index reverse() const;

        // Construct with a given base. This is private because
        // the interface shouldn't depend on how this is done.
        inline Index(int m, int a, const Index &b);
        inline Index(int f, int s, const Index *b);

        // Do a general intersect if the strides are not both 1.
        Index general_intersect(const Index&) const;

        // Provide a way to not initialize on construction.
        class DontInitialize {};
        Index(DontInitialize) {}
    };


    std::ostream& operator<<(std::ostream& out, const Index& I);
}

#include "Index/Index.hpp"

#endif