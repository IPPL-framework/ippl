//
// Class ParticleAttrib
//   Templated class for all particle attribute classes.
//
//   This templated class is used to represent a single particle attribute.
//   An attribute is one data element within a particle object, and is
//   stored as a Kokkos::View. This class stores the type information for the
//   attribute, and provides methods to create and destroy new items, and
//   to perform operations involving this attribute with others.
//
//   ParticleAttrib is the primary element involved in expressions for
//   particles (just as Field is the primary element there).  This file
//   defines the necessary templated classes and functions to make
//   ParticleAttrib a capable expression-template participant.
//
// Copyright (c) 2020, Matthias Frey, Paul Scherrer Institut, Villigen PSI, Switzerland
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

namespace ippl {

    template<typename T, class... Properties>
    void ParticleAttrib<T, Properties...>::create(size_t n) {
        size_t current = this->size();
        this->resize(current + n);
    }


    template<typename T, class... Properties>
    void ParticleAttrib<T, Properties...>::destroy(boolean_view_type invalidIndex,
                                                   Kokkos::View<int*> newIndex, size_t n) {
        Kokkos::View<T*> temp("temp", n);
        Kokkos::parallel_for("ParticleAttrib::destroy()",
                             size(),
                             KOKKOS_CLASS_LAMBDA(const size_t i)
                             {
                                 if ( !invalidIndex(i) ) {
                                    temp(newIndex(i)) = this->operator()(i);
                                 }
                             });
        this->resize(n);
        Kokkos::deep_copy(*this, temp);
    }


    template<typename T, class... Properties>
    ParticleAttrib<T, Properties...>&
    ParticleAttrib<T, Properties...>::operator=(T x)
    {
        Kokkos::parallel_for("ParticleAttrib::operator=()",
                             this->extent(0),
                             KOKKOS_CLASS_LAMBDA(const int i) {
                                 this->operator()(i) = x;
                            });
        return *this;
    }


    template<typename T, class... Properties>
    template <typename E, size_t N>
    ParticleAttrib<T, Properties...>&
    ParticleAttrib<T, Properties...>::operator=(Expression<E, N> const& expr)
    {
        detail::CapturedExpression<E, N> expr_ = reinterpret_cast<const detail::CapturedExpression<E, N>&>(expr);

        Kokkos::parallel_for("ParticleAttrib::operator=()",
                             this->extent(0),
                             KOKKOS_CLASS_LAMBDA(const int i) {
                                 this->operator()(i) = expr_(i);
                            });
        return *this;
    }


    template<typename T, class... Properties>
    template <unsigned Dim, class M, class C, class PT>
    void ParticleAttrib<T, Properties...>::scatter(Field<T,Dim,M,C>& f,
                                                   const ParticleAttrib< Vector<PT,Dim>, Properties... >& pp)
    const
    {
        // single LField only
        typename Field<T, Dim, M, C>::LField_t::view_type view = f(0).getView();

        const M& mesh = f.get_mesh();

        using vector_type = typename M::vector_type;
        using value_type  = typename ParticleAttrib<T, Properties...>::value_type;

        const vector_type& dx = mesh.getMeshSpacing();
        const vector_type& origin = mesh.getOrigin();
        const vector_type invdx = 1.0 / dx;


        Kokkos::parallel_for("ParticleAttrib::scatter",
                             size(),
                             KOKKOS_CLASS_LAMBDA(const size_t idx)
                             {
                                 // find nearest grid point
                                 vector_type l = (pp(idx) - origin) * invdx + 0.5;
                                 Vector<int, Dim> index = l;
                                 Vector<double, Dim> whi = l - index;
                                 Vector<double, Dim> wlo = 1.0 - whi;

                                 const int i = index[0] + 1;
                                 const int j = index[1] + 1;
                                 const int k = index[2] + 1;

                                 // scatter
                                 const value_type& val = this->operator()(idx);
                                 Kokkos::atomic_add(&view(i-1, j-1, k-1), wlo[0] * wlo[1] * wlo[2] * val);
                                 Kokkos::atomic_add(&view(i-1, j-1, k  ), wlo[0] * wlo[1] * whi[2] * val);
                                 Kokkos::atomic_add(&view(i-1, j,   k-1), wlo[0] * whi[1] * wlo[2] * val);
                                 Kokkos::atomic_add(&view(i-1, j,   k  ), wlo[0] * whi[1] * whi[2] * val);
                                 Kokkos::atomic_add(&view(i,   j-1, k-1), whi[0] * wlo[1] * wlo[2] * val);
                                 Kokkos::atomic_add(&view(i,   j-1, k  ), whi[0] * wlo[1] * whi[2] * val);
                                 Kokkos::atomic_add(&view(i,   j,   k-1), whi[0] * whi[1] * wlo[2] * val);
                                 Kokkos::atomic_add(&view(i,   j,   k  ), whi[0] * whi[1] * whi[2] * val);
                             });
    }


    template<typename T, class... Properties>
    template <unsigned Dim, class M, class C, typename P2>
    void ParticleAttrib<T, Properties...>::gather(const Field<T, Dim, M, C>& f,
                                                  const ParticleAttrib<Vector<P2, Dim>, Properties...>& pp)
    {
        // single LField only
        const typename Field<T, Dim, M, C>::LField_t::view_type view = f(0).getView();

        const M& mesh = f.get_mesh();

        using vector_type = typename M::vector_type;
        using value_type  = typename ParticleAttrib<T, Properties...>::value_type;

        const vector_type& dx = mesh.getMeshSpacing();
        const vector_type& origin = mesh.getOrigin();
        const vector_type invdx = 1.0 / dx;


        Kokkos::parallel_for("ParticleAttrib::gather",
                             size(),
                             KOKKOS_CLASS_LAMBDA(const size_t idx)
                             {
                                 // find nearest grid point
                                 vector_type l = (pp(idx) - origin) * invdx + 0.5;
                                 Vector<int, Dim> index = l;
                                 Vector<double, Dim> whi = l - index;
                                 Vector<double, Dim> wlo = 1.0 - whi;

                                 const int i = index[0] + 1;
                                 const int j = index[1] + 1;
                                 const int k = index[2] + 1;

                                 // scatter
                                 value_type& val = this->operator()(idx);
                                 val = wlo[0] * wlo[1] * wlo[2] * view(i-1, j-1, k-1)
                                     + wlo[0] * wlo[1] * whi[2] * view(i-1, j-1, k  )
                                     + wlo[0] * whi[1] * wlo[2] * view(i-1, j,   k-1)
                                     + wlo[0] * whi[1] * whi[2] * view(i-1, j,   k  )
                                     + whi[0] * wlo[1] * wlo[2] * view(i,   j-1, k-1)
                                     + whi[0] * wlo[1] * whi[2] * view(i,   j-1, k  )
                                     + whi[0] * whi[1] * wlo[2] * view(i,   j,   k-1)
                                     + whi[0] * whi[1] * whi[2] * view(i,   j,   k  );
                             });
    }



    /*
     * Non-class function
     *
     */


    template<typename P1, unsigned Dim, class M, class C, typename P2, class... Properties>
    inline
    void scatter(const ParticleAttrib<P1, Properties...>& attrib, Field<P1, Dim, M, C>& f,
                 const ParticleAttrib<Vector<P2, Dim>, Properties...>& pp)
    {
        attrib.scatter(f, pp);
    }


    template<typename P1, unsigned Dim, class M, class C, typename P2, class... Properties>
    inline
    void gather(ParticleAttrib<P1, Properties...>& attrib, const Field<P1, Dim, M, C>& f,
                const ParticleAttrib<Vector<P2, Dim>, Properties...>& pp)
    {
        attrib.gather(f, pp);
    }
}