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
#include "Utility/PAssert.h"

namespace ippl {

    KOKKOS_INLINE_FUNCTION Index::Index()
        : first_m(0)
        , stride_m(0)
        , length_m(0) {}

    KOKKOS_INLINE_FUNCTION Index::Index(size_t n)
        : first_m(0)
        , stride_m(1)
        , length_m(n) {}

    KOKKOS_INLINE_FUNCTION Index::Index(int f, int l)
        : first_m(f)
        , stride_m(1)
        , length_m(l - f + 1) {
        PAssert_GE(l - f + 1, 0);
    }

    KOKKOS_INLINE_FUNCTION Index::Index(int f, int l, int s)
        : first_m(f)
        , stride_m(s) {
        PAssert_NE(s, 0);
        if (f == l) {
            length_m = 1;
        } else if ((l > f) ^ (s < 0)) {
            length_m = (l - f) / s + 1;
        } else {
            length_m = 0;
        }
    }

    KOKKOS_INLINE_FUNCTION Index::Index(int m, int a, const Index& b)
        : first_m(b.first_m * m + a)
        , stride_m(b.stride_m * m)
        , length_m(b.length_m) {}

    KOKKOS_INLINE_FUNCTION Index::Index(int f, int s, const Index* b)
        : first_m(f)
        , stride_m(s)
        , length_m(b->length_m) {}

    KOKKOS_INLINE_FUNCTION int Index::first() const noexcept {
        return first_m;
    }

    KOKKOS_INLINE_FUNCTION int Index::stride() const noexcept {
        return stride_m;
    }

    KOKKOS_INLINE_FUNCTION bool Index::empty() const noexcept {
        return length_m == 0;
    }

    KOKKOS_INLINE_FUNCTION size_t Index::length() const noexcept {
        return length_m;
    }

    KOKKOS_INLINE_FUNCTION int Index::last() const noexcept {
        return (length_m == 0) ? first_m : first_m + stride_m * (length_m - 1);
    }

    KOKKOS_INLINE_FUNCTION int Index::min() const noexcept {
        return (stride_m >= 0) ? first_m : first_m + stride_m * (length_m - 1);
    }

    KOKKOS_INLINE_FUNCTION int Index::max() const noexcept {
        return (stride_m >= 0) ? first_m + stride_m * (length_m - 1) : first_m;
    }

    KOKKOS_INLINE_FUNCTION Index& Index::operator+=(int off) {
        first_m += off;
        return *this;
    }

    KOKKOS_INLINE_FUNCTION Index& Index::operator-=(int off) {
        first_m -= off;
        return *this;
    }

    KOKKOS_INLINE_FUNCTION Index operator+(const Index& i, int off) {
        return Index(1, off, i);
    }

    KOKKOS_INLINE_FUNCTION Index operator+(int off, const Index& i) {
        return Index(1, off, i);
    }

    KOKKOS_INLINE_FUNCTION Index operator-(const Index& i, int off) {
        return Index(1, -off, i);
    }

    KOKKOS_INLINE_FUNCTION Index operator-(int off, const Index& i) {
        return Index(-1, off, i);
    }

    KOKKOS_INLINE_FUNCTION Index operator-(const Index& i) {
        return Index(-1, 0, i);
    }

    KOKKOS_INLINE_FUNCTION Index operator*(const Index& i, int m) {
        return Index(m, 0, i);
    }

    KOKKOS_INLINE_FUNCTION Index operator*(int m, const Index& i) {
        return Index(m, 0, i);
    }

    KOKKOS_INLINE_FUNCTION Index operator/(const Index& i, int d) {
        return Index(i.first_m / d, i.stride_m / d, &i);
    }

    KOKKOS_INLINE_FUNCTION Index Index::reverse() const {
        Index j;
        j.first_m  = last();
        j.length_m = length_m;
        j.stride_m = -stride_m;
        return j;
    }

    KOKKOS_INLINE_FUNCTION bool Index::touches(const Index& a) const {
        return (min() <= a.max()) && (max() >= a.min());
    }

    KOKKOS_INLINE_FUNCTION bool Index::contains(const Index& a) const {
        return (min() <= a.min()) && (max() >= a.max());
    }

    KOKKOS_INLINE_FUNCTION bool Index::split(Index& l, Index& r) const {
        PAssert_EQ(stride_m, 1);
        PAssert_GT(length_m, 1);
        auto mid = first_m + length_m / 2 - 1;
        l        = Index(first_m, mid);
        r        = Index(mid + 1, first_m + length_m - 1);
        return true;
    }

    KOKKOS_INLINE_FUNCTION bool Index::split(Index& l, Index& r, int i) const {
        PAssert_EQ(stride_m, 1);
        PAssert_GT(length_m, 1);
        if (i >= first_m + static_cast<int>(length_m)) {
            return false;
        }
        l = Index(first_m, i);
        r = Index(i + 1, first_m + length_m - 1);
        return true;
    }

    KOKKOS_INLINE_FUNCTION bool Index::split(Index& l, Index& r, double a) const {
        PAssert_EQ(stride_m, 1);
        PAssert_GT(length_m, 1);
        PAssert_LT(a, 1.0);
        PAssert_GT(a, 0.0);
        int mid = first_m + static_cast<int>(length_m * a + 0.5) - 1;
        l       = Index(first_m, mid);
        r       = Index(mid + 1, first_m + length_m - 1);
        return true;
    }

    //////////////////////////////////////////////////////////////////////
    // Calculate the lowest common multiple of s1 and s2.
    // put the result in s.
    // Also calculate m1 = s/s1 and m2 = s/s2.
    // This version is optimized for small s1 and s2 and
    // just uses an exhaustive search.
    //////////////////////////////////////////////////////////////////////
    KOKKOS_INLINE_FUNCTION void lcm(int s1, int s2, int& s, int& m1, int& m2) {
        PAssert_GT(s1, 0);  // For simplicity, make some assumptions.
        PAssert_GT(s2, 0);
        int i1  = s1;
        int i2  = s2;
        int _m1 = 1;
        int _m2 = 1;
        if (i2 < i1) {
            while (true) {
                while (i2 < i1) {
                    i2 += s2;
                    ++_m2;
                }
                if (i1 == i2) {
                    m1 = _m1;
                    m2 = _m2;
                    s  = i1;
                    return;
                }
                i1 += s1;
                ++_m1;
            }
        } else {
            while (true) {
                while (i1 < i2) {
                    i1 += s1;
                    ++_m1;
                }
                if (i1 == i2) {
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
    KOKKOS_INLINE_FUNCTION Index Index::intersect(const Index& rhs) const {
        Index ret;
        if ((stride() == 1) && (rhs.stride() == 1)) {
            int lf       = first();
            int rf       = rhs.first();
            int ll       = last();
            int rl       = rhs.last();
            int f        = lf > rf ? lf : rf;
            int l        = ll < rl ? ll : rl;
            ret.first_m  = f;
            ret.length_m = ((l >= f) ? l - f + 1 : 0);
            ret.stride_m = 1;
        } else
            ret = general_intersect(rhs);
        return ret;
    }

    KOKKOS_INLINE_FUNCTION Index Index::grow(int ncells) const {
        Index index;

        index.first_m  = this->first_m - ncells;
        index.length_m = this->length_m + 2 * ncells;
        index.stride_m = this->stride_m;

        return index;
    }

    KOKKOS_INLINE_FUNCTION static Index do_intersect(const Index& a, const Index& b) {
        PAssert_GT(a.stride(), 0);  // This should be assured by the
        PAssert_GT(b.stride(), 0);  // caller of this function.

        int newStride;                 // The stride for the new index is
        int a_mul, b_mul;              // a_mul=newStride/a.stride() ...
        lcm(a.stride(), b.stride(),    // The input strides...
            newStride, a_mul, b_mul);  // the lcm of the strides of a and b.

        // Find the offset from a.first() in units of newStride
        // that puts the ranges close together.
        int a_i   = (b.first() - a.first()) / a.stride();
        int a_off = a.first() + a_i * a.stride();
        if (a_off < b.first()) {
            a_i++;
            a_off += a.stride();
        }

        PAssert_GE(a_off, b.first());  // make sure I'm understanding this right...

        // Now do an exhaustive search for the first point in common.
        // Count over all possible offsets for a.
        for (int a_m = 0; (a_m < a_mul) && (a_i < (int)a.length());
             a_m++, a_i++, a_off += a.stride()) {
            int b_off = b.first();
            // Count over all possible offsets for b.
            for (int b_m = 0; (b_m < b_mul) && (b_m < (int)b.length()); b_m++, b_off += b.stride())
                if (a_off == b_off) {  // If the offsets are the same, we found it!
                    int am = a.max();  // Find the minimum maximum of a and b...
                    int bm = b.max();
                    int m  = am < bm ? am : bm;
                    return Index(a_off, m, newStride);
                }
        }
        return Index(0);  // If we get to here there is no intersection.
    }

    KOKKOS_INLINE_FUNCTION Index Index::general_intersect(const Index& that) const {
        // If they just don't overlap, return null indexes.
        if ((min() > that.max()) || (that.min() > max()))
            return Index(0);
        if ((stride_m == 0) || (that.stride_m == 0))
            return Index(0);

        // If one or the other counts -ve, reverse it and intersect result.
        if (that.stride_m < 0)
            return intersect(that.reverse());
        if (stride_m < 0) {
            Index r;
            r = reverse().intersect(that).reverse();
            return r;
        }

        // Getting closer to the real thing: intersect them.
        // Pass the one that starts lower as the first argument.
        Index r;
        if (first_m < that.first_m)
            r = do_intersect(*this, that);
        else
            r = do_intersect(that, *this);

        return r;
    }
}  // namespace ippl
