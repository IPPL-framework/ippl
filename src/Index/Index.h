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
#ifndef IPPL_INDEX_H
#define IPPL_INDEX_H

#include <Kokkos_Core.hpp>

#include <iostream>

namespace ippl {
    class Index {
    public:
        class iterator {
        public:
            iterator()
                : current_m(0)
                , stride_m(0) {}

            iterator(int current, int stride = 1)
                : current_m(current)
                , stride_m(stride) {}

            int operator*() { return current_m; }

            iterator operator--(int) {
                iterator tmp = *this;
                current_m -= stride_m;  // Post decrement
                return tmp;
            }

            iterator& operator--() {
                current_m -= stride_m;
                return (*this);
            }

            iterator operator++(int) {
                iterator tmp = *this;
                current_m += stride_m;  // Post increment
                return tmp;
            }

            iterator& operator++() {
                current_m += stride_m;
                return (*this);
            }

            iterator& operator+=(int i) {
                current_m += (stride_m * i);
                return *this;
            }

            iterator& operator-=(int i) {
                current_m -= (stride_m * i);
                return *this;
            }

            iterator operator+(int i) const { return iterator(current_m + i * stride_m, stride_m); }

            iterator operator-(int i) const { return iterator(current_m - i * stride_m, stride_m); }

            int operator[](int i) const { return current_m + i * stride_m; }

            bool operator==(const iterator& y) const {
                return (current_m == y.current_m) && (stride_m == y.stride_m);
            }

            bool operator<(const iterator& y) const {
                return (current_m < y.current_m)
                       || ((current_m == y.current_m) && (stride_m < y.stride_m));
            }

            bool operator!=(const iterator& y) const { return !((*this) == y); }

            bool operator>(const iterator& y) const { return y < (*this); }

            bool operator<=(const iterator& y) const { return !(y < (*this)); }

            bool operator>=(const iterator& y) const { return !((*this) < y); }

        private:
            int current_m;
            int stride_m;
        };

        /*!
         * Instantiate Index without any range.
         */
        KOKKOS_INLINE_FUNCTION Index();

        /*!
         * Instantiate Index with range [0, ..., n-1]
         * @param n number of elements
         */
        KOKKOS_INLINE_FUNCTION Index(size_t n);

        /*!
         * Instantiate Index with user-defined lower and upper
         * bound [f, ..., l].
         * @param f first element
         * @param l last element
         */
        KOKKOS_INLINE_FUNCTION Index(int f, int l);

        /*!
         * First to Last using Step.
         * @param f first element
         * @param l last element
         * @param s step
         */
        KOKKOS_INLINE_FUNCTION Index(int f, int l, int s);

        KOKKOS_DEFAULTED_FUNCTION
        ~Index() = default;

        /*!
         * @returns the smallest element
         */
        KOKKOS_INLINE_FUNCTION int min() const noexcept;

        /*!
         * @returns the largest element
         */
        KOKKOS_INLINE_FUNCTION int max() const noexcept;

        /*!
         * @returns the number of elements
         */
        KOKKOS_INLINE_FUNCTION size_t length() const noexcept;

        /*!
         * @returns the stride
         */
        KOKKOS_INLINE_FUNCTION int stride() const noexcept;

        /*!
         * @returns the first element
         */
        KOKKOS_INLINE_FUNCTION int first() const noexcept;

        /*!
         * @returns the last element
         */
        KOKKOS_INLINE_FUNCTION int last() const noexcept;

        /*!
         * @returns true if empty, otherwise false
         */
        KOKKOS_INLINE_FUNCTION bool empty() const noexcept;

        KOKKOS_INLINE_FUNCTION Index& operator+=(int);

        KOKKOS_INLINE_FUNCTION Index& operator-=(int);

        // Additive operations.
        KOKKOS_INLINE_FUNCTION friend Index operator+(const Index&, int);

        KOKKOS_INLINE_FUNCTION friend Index operator+(int, const Index&);

        KOKKOS_INLINE_FUNCTION friend Index operator-(const Index&, int);

        KOKKOS_INLINE_FUNCTION friend Index operator-(int, const Index&);

        // Multipplicative operations.
        KOKKOS_INLINE_FUNCTION friend Index operator-(const Index&);

        KOKKOS_INLINE_FUNCTION friend Index operator*(const Index&, int);

        KOKKOS_INLINE_FUNCTION friend Index operator*(int, const Index&);

        KOKKOS_INLINE_FUNCTION friend Index operator/(const Index&, int);

        // Intersect with another Index.
        KOKKOS_INLINE_FUNCTION Index intersect(const Index&) const;

        // Intersect with another Index.
        KOKKOS_INLINE_FUNCTION Index grow(int ncells) const;

        // Test to see if there is any overlap between two Indexes.
        KOKKOS_INLINE_FUNCTION bool touches(const Index& a) const;
        // Test to see if one contains another (endpoints only)
        KOKKOS_INLINE_FUNCTION bool contains(const Index& a) const;
        // Split one into two.
        KOKKOS_INLINE_FUNCTION bool split(Index& l, Index& r) const;
        // Split one into two at index i.
        KOKKOS_INLINE_FUNCTION bool split(Index& l, Index& r, int i) const;
        // Split index into two with a ratio between 0 and 1.
        KOKKOS_INLINE_FUNCTION bool split(Index& l, Index& r, double a) const;

        // iterator begin
        iterator begin() { return iterator(first_m, stride_m); }
        // iterator end
        iterator end() { return iterator(first_m + stride_m * length_m, stride_m); }

        // An operator< so we can impose some sort of ordering.
        KOKKOS_INLINE_FUNCTION bool operator<(const Index& r) const {
            return (
                (length_m < r.length_m)
                || ((length_m == r.length_m)
                    && ((first_m < r.first_m)
                        || ((first_m == r.first_m) && (length_m > 0) && (stride_m < r.stride_m)))));
        }
        // Test for equality.
        KOKKOS_INLINE_FUNCTION bool operator==(const Index& r) const noexcept {
            return (length_m == r.length_m) && (first_m == r.first_m) && (stride_m == r.stride_m);
        }

    private:
        int first_m;  /// First index element
        int stride_m;
        size_t length_m;  /// The number of elements

        // Make an Index that interally counts the other direction.
        KOKKOS_INLINE_FUNCTION Index reverse() const;

        // Construct with a given base. This is private because
        // the interface shouldn't depend on how this is done.

        KOKKOS_INLINE_FUNCTION Index(int m, int a, const Index& b);

        KOKKOS_INLINE_FUNCTION Index(int f, int s, const Index* b);

        // Do a general intersect if the strides are not both 1.
        KOKKOS_INLINE_FUNCTION Index general_intersect(const Index&) const;
    };

    inline std::ostream& operator<<(std::ostream& out, const Index& I) {
        out << '[' << I.first() << ':' << I.last() << ':' << I.stride() << ']';
        return out;
    }
}  // namespace ippl

#include "Index/Index.hpp"

#endif
