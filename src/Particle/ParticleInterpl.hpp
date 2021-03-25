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
