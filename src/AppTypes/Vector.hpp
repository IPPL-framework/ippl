#include "Utility/PAssert.h"

#include <iomanip>
#include <iostream>

namespace ippl {

    template<typename T, unsigned Dim>
    template<typename E>
    KOKKOS_FUNCTION
    Vector<T, Dim>::Vector(const Expression<E>& expr) {
        std::cout << "Vector(const Expression<E>& expr)" << std::endl;
        for (unsigned int i = 0; i < Dim; ++i) {
            data_m[i] = expr[i];
        }
    }

    template<typename T, unsigned Dim>
    KOKKOS_FUNCTION
    Vector<T, Dim>::Vector(const T& val) {
        std::cout << "Vector(const T& val)" << std::endl;
        for (unsigned i = 0; i < Dim; ++i) {
            data_m[i] = val;
        }
    }


    template<typename T, unsigned Dim>
    KOKKOS_FUNCTION
    Vector<T, Dim>::Vector(const std::initializer_list<T>& list) {
        std::cout << "Vector(const std::initializer_list<T>& list)" << std::endl;
        PAssert(list.size() == Dim);
        unsigned int i = 0;
        for (auto& l : list) {
            data_m[i] = l;
            ++i;
        }
    }


    /*
     *
     * Element access operators
     *
     */
    template<typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION
    typename Vector<T, Dim>::value_t& Vector<T, Dim>::operator[](unsigned int i) {
        PAssert(i < Dim);
        return data_m[i];
    }


    template<typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION
    typename Vector<T, Dim>::value_t Vector<T, Dim>::operator[](unsigned int i) const {
        PAssert(i < Dim);
        return data_m[i];
    }


    template<typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION
    typename Vector<T, Dim>::value_t& Vector<T, Dim>::operator()(unsigned int i) {
        PAssert(i < Dim);
        return data_m[i];
    }

    
    template<typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION
    typename Vector<T, Dim>::value_t Vector<T, Dim>::operator()(unsigned int i) const {
        PAssert(i < Dim);
        return data_m[i];
    }


    /*
     *
     * Vector Expression assignment operators
     *
     */
    template<typename T, unsigned Dim>
    template<typename E>
    KOKKOS_INLINE_FUNCTION
        Vector<T, Dim>& Vector<T, Dim>::operator=(const Expression<E>& expr) {
        for (unsigned int i = 0; i < Dim; ++i) {
            data_m[i] = expr[i];
        }
        return *this;
    }


    template<typename T, unsigned Dim>
    template<typename E>
    KOKKOS_INLINE_FUNCTION
    Vector<T, Dim>& Vector<T, Dim>::operator+=(const Expression<E>& expr) {
        for (unsigned int i = 0; i < Dim; ++i) {
            data_m[i] += expr[i];
        }
        return *this;
    }
    

    template<typename T, unsigned Dim>
    template<typename E>
    KOKKOS_INLINE_FUNCTION
    Vector<T, Dim>& Vector<T, Dim>::operator-=(const Expression<E>& expr) {
        for (unsigned int i = 0; i < Dim; ++i) {
            data_m[i] -= expr[i];
        }
        return *this;
    }


    template<typename T, unsigned Dim>
    template<typename E>
    KOKKOS_INLINE_FUNCTION
    Vector<T, Dim>& Vector<T, Dim>::operator*=(const Expression<E>& expr) {
        for (unsigned int i = 0; i < Dim; ++i) {
            data_m[i] *= expr[i];
        }
        return *this;
    }


    template<typename T, unsigned Dim>
    template<typename E>
    KOKKOS_INLINE_FUNCTION
    Vector<T, Dim>& Vector<T, Dim>::operator/=(const Expression<E>& expr) {
        for (unsigned int i = 0; i < Dim; ++i) {
            data_m[i] /= expr[i];
        }
        return *this;
    }


    template<typename T, unsigned Dim>
    inline std::ostream& operator<<(std::ostream& out, const Vector<T, Dim>& v) {
    std::streamsize sw = out.width();
        out << std::setw(1);
        if (Dim >= 1) {
            out << "( ";
            for (unsigned int i = 0; i < Dim - 1; i++)
            out << std::setw(sw) << v[i] << " , ";
            out << std::setw(sw) << v[Dim - 1] << " )";
        } else {
            out << "( " << std::setw(sw) << v[0] << " )";
        }
        return out;
    }
}

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End:
