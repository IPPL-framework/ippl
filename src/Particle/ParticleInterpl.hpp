//
// File ParticleInterpl
//   Global scatter and gather functions. These functions call the appropriate
//   ParticleAttrib member functions. A user doesn't need to specify the scatter/gather
//   method, in this case the cloud-in-cell (CIC) is used.
//
//
// Copyright (c) 2021, Matthias Frey, University of St Andrews, St Andrews, Scotland
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
#ifndef IPPL_PARTICLE_INTERPL
#define IPPL_PARTICLE_INTERPL

namespace ippl {
    enum Interpl_t {
        NGP_t,
        CIC_t
    };

    /*
     * Non-member scatter and gather functions
     *
     */
    template <Interpl_t I,
              typename P1,
              unsigned Dim,
              class M,
              class C,
              typename P2,
              class... Properties,
              std::enable_if_t<I == CIC_t, bool> = true>
    inline
    void scatter(const ParticleAttrib<P1, Properties...>& attrib, Field<P1, Dim, M, C>& f,
                 const ParticleAttrib<Vector<P2, Dim>, Properties...>& pp)
    {
        attrib.scatterCIC(f, pp);
    }

    template <Interpl_t I,
              typename P1,
              unsigned Dim,
              class M,
              class C,
              typename P2,
              class... Properties,
              std::enable_if_t<I == NGP_t, bool> = true>
    inline
    void scatter(const ParticleAttrib<P1, Properties...>& attrib, Field<P1, Dim, M, C>& f,
                 const ParticleAttrib<Vector<P2, Dim>, Properties...>& pp)
    {
        attrib.scatterNGP(f, pp);
    }

    template <typename P1,
              unsigned Dim,
              class M,
              class C,
              typename P2,
              class... Properties>
    inline
    void scatter(const ParticleAttrib<P1, Properties...>& attrib, Field<P1, Dim, M, C>& f,
                 const ParticleAttrib<Vector<P2, Dim>, Properties...>& pp)
    {
        scatter<CIC_t>(attrib, f, pp);
    }


    template <Interpl_t I,
              typename P1,
              unsigned Dim,
              class M,
              class C,
              typename P2,
              class... Properties,
              std::enable_if_t<I == CIC_t, bool> = true>
    inline
    void gather(ParticleAttrib<P1, Properties...>& attrib, Field<P1, Dim, M, C>& f,
                const ParticleAttrib<Vector<P2, Dim>, Properties...>& pp)
    {
        attrib.gatherCIC(f, pp);
    }


    template <Interpl_t I,
              typename P1,
              unsigned Dim,
              class M,
              class C,
              typename P2,
              class... Properties,
              std::enable_if_t<I == NGP_t, bool> = true>
    inline
    void gather(ParticleAttrib<P1, Properties...>& attrib, Field<P1, Dim, M, C>& f,
                const ParticleAttrib<Vector<P2, Dim>, Properties...>& pp)
    {
        attrib.gatherNGP(f, pp);
    }

    template<typename P1,
             unsigned Dim,
             class M,
             class C,
             typename P2,
             class... Properties>
    inline
    void gather(ParticleAttrib<P1, Properties...>& attrib, Field<P1, Dim, M, C>& f,
                const ParticleAttrib<Vector<P2, Dim>, Properties...>& pp)
    {
        gather<CIC_t>(attrib, f, pp);
    }
}

#endif
