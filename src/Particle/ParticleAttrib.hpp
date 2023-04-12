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
//   particles (just as BareField is the primary element there).  This file
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
#include "Ippl.h"
#include "Communicate/DataTypes.h"
#include "Utility/IpplTimings.h"


namespace ippl {

    template<typename T, class... Properties>
    void ParticleAttrib<T, Properties...>::create(size_type n) {
        size_type required = *(this->localNum_mp) + n;
        if (this->size() < required) {
            int overalloc = Ippl::Comm->getDefaultOverallocation();
            this->realloc(required * overalloc);
        }
    }

    template<typename T, class... Properties>
    void ParticleAttrib<T, Properties...>::destroy(const Kokkos::View<int*>& deleteIndex,
                                                const Kokkos::View<int*>& keepIndex,
                                                size_type invalidCount) {
        // Replace all invalid particles in the valid region with valid
        // particles in the invalid region
        Kokkos::parallel_for("ParticleAttrib::destroy()",
                             invalidCount,
                             KOKKOS_CLASS_LAMBDA(const size_t i)
                             {
                                 dview_m(deleteIndex(i)) = dview_m(keepIndex(i));
                             });
    }

    template<typename T, class... Properties>
    void ParticleAttrib<T, Properties...>::pack(void* buffer,
                                                const Kokkos::View<int*>& hash) const
    {
        using this_type = ParticleAttrib<T, Properties...>;
        this_type* buffer_p = static_cast<this_type*>(buffer);
        auto& view = buffer_p->dview_m;
        auto size = hash.extent(0);
        if(view.extent(0) < size) {
            int overalloc = Ippl::Comm->getDefaultOverallocation();
            Kokkos::realloc(view, size * overalloc);
        }

        Kokkos::parallel_for(
            "ParticleAttrib::pack()",
            size,
            KOKKOS_CLASS_LAMBDA(const size_t i) {
                view(i) = dview_m(hash(i));
        });
        Kokkos::fence();
        
    
    }


    template <typename T, class... Properties>
    void ParticleAttrib<T, Properties...>::unpack(void* buffer, size_type nrecvs) {
        using this_type = ParticleAttrib<T, Properties...>;
        this_type* buffer_p = static_cast<this_type*>(buffer);
        auto& view = buffer_p->dview_m;
        auto size = dview_m.extent(0);
        size_type required = *(this->localNum_mp) + nrecvs;
        if(size < required) {
            int overalloc = Ippl::Comm->getDefaultOverallocation();
            this->resize(required * overalloc);
        }

        size_type count = *(this->localNum_mp);
        Kokkos::parallel_for(
            "ParticleAttrib::unpack()",
            nrecvs,
            KOKKOS_CLASS_LAMBDA(const size_t i) {
                dview_m(count + i) = view(i);
        });
        Kokkos::fence();
    
    }

    template<typename T, class... Properties>
    //KOKKOS_INLINE_FUNCTION
    ParticleAttrib<T, Properties...>&
    ParticleAttrib<T, Properties...>::operator=(T x)
    {
        Kokkos::parallel_for("ParticleAttrib::operator=()",
                             *(this->localNum_mp),
                             KOKKOS_CLASS_LAMBDA(const size_t i) {
                                 dview_m(i) = x;
                            });
        return *this;
    }


    template<typename T, class... Properties>
    template <typename E, size_t N>
    //KOKKOS_INLINE_FUNCTION
    ParticleAttrib<T, Properties...>&
    ParticleAttrib<T, Properties...>::operator=(detail::Expression<E, N> const& expr)
    {
        using capture_type = detail::CapturedExpression<E, N>;
        capture_type expr_ = reinterpret_cast<const capture_type&>(expr);

        Kokkos::parallel_for("ParticleAttrib::operator=()",
                             *(this->localNum_mp),
                             KOKKOS_CLASS_LAMBDA(const size_t i) {
                                 dview_m(i) = expr_(i);
                            });
        return *this;
    }


    template<typename T, class... Properties>
    template <unsigned Dim, class M, class C, class PT>
    void ParticleAttrib<T, Properties...>::scatter(Field<T,Dim,M,C>& f,
                                                   const ParticleAttrib< Vector<PT,Dim>, Properties... >& pp)
    const
    {
        static IpplTimings::TimerRef scatterPICTimer = IpplTimings::getTimer("ScatterPIC");           
        IpplTimings::startTimer(scatterPICTimer);                                               
        
        typename Field<T, Dim, M, C>::view_type view = f.getView();

        const M& mesh = f.get_mesh();

        using vector_type = typename M::vector_type;
        using value_type  = typename ParticleAttrib<T, Properties...>::value_type;

        const vector_type& dx = mesh.getMeshSpacing();
        const vector_type& origin = mesh.getOrigin();
        const vector_type invdx = 1.0 / dx;

        const FieldLayout<Dim>& layout = f.getLayout(); 
        const NDIndex<Dim>& lDom = layout.getLocalNDIndex();
        const int nghost = f.getNghost();

        Kokkos::parallel_for(
            "ParticleAttrib::scatter",
            *(this->localNum_mp),
            KOKKOS_CLASS_LAMBDA(const size_t idx)
            {
                // find nearest grid point
                vector_type l = (pp(idx) - origin) * invdx + 0.5;
                Vector<int, Dim> index = l;
                Vector<double, Dim> whi = l - index;
                Vector<double, Dim> wlo = 1.0 - whi;

                const int i = index[0] - lDom[0].first() + nghost;
                const int j = index[1] - lDom[1].first() + nghost;
                const int k = index[2] - lDom[2].first() + nghost;

                //if((i < 1) || (i > lDom[0].last() + 2) || (j < 1) || (j > lDom[1].last() + 2)
                //   || (k < 1) || (k > lDom[0].last() + 2)) {
                //    std::cout << "i: " << i << ", j: " << j << ", k: " << k << std::endl;
                //    std::cout << "Invalid particle co-ordinates: " << pp(idx) << std::endl;
                //}

                // scatter
                const value_type& val = dview_m(idx);
                Kokkos::atomic_add(&view(i-1, j-1, k-1), wlo[0] * wlo[1] * wlo[2] * val);
                Kokkos::atomic_add(&view(i-1, j-1, k  ), wlo[0] * wlo[1] * whi[2] * val);
                Kokkos::atomic_add(&view(i-1, j,   k-1), wlo[0] * whi[1] * wlo[2] * val);
                Kokkos::atomic_add(&view(i-1, j,   k  ), wlo[0] * whi[1] * whi[2] * val);
                Kokkos::atomic_add(&view(i,   j-1, k-1), whi[0] * wlo[1] * wlo[2] * val);
                Kokkos::atomic_add(&view(i,   j-1, k  ), whi[0] * wlo[1] * whi[2] * val);
                Kokkos::atomic_add(&view(i,   j,   k-1), whi[0] * whi[1] * wlo[2] * val);
                Kokkos::atomic_add(&view(i,   j,   k  ), whi[0] * whi[1] * whi[2] * val);
            }
        );
        IpplTimings::stopTimer(scatterPICTimer);
            
        //static IpplTimings::TimerRef accumulateHaloTimer = IpplTimings::getTimer("AccumulateHalo");           
        //IpplTimings::startTimer(accumulateHaloTimer);                                               
        f.accumulateHalo();
        //IpplTimings::stopTimer(accumulateHaloTimer);                                               
    }


    template<typename T, class... Properties>
    template <unsigned Dim, class M, class C, class FT, class ST, class PT>
    void ParticleAttrib<T, Properties...>::scatterPIFNUDFT(Field<FT,Dim,M,C>& f, Field<ST,Dim,M,C>& Sk,
                                                   const ParticleAttrib< Vector<PT,Dim>, Properties... >& pp)
    const
    {
        
        static IpplTimings::TimerRef scatterPIFNUDFTTimer = IpplTimings::getTimer("ScatterPIFNUDFT");           
        IpplTimings::startTimer(scatterPIFNUDFTTimer);
        
        using view_type = typename Field<FT, Dim, M, C>::view_type;
        using vector_type = typename M::vector_type;
        using value_type  = typename ParticleAttrib<T, Properties...>::value_type;
        view_type fview = f.getView();
        typename Field<ST, Dim, M, C>::view_type Skview = Sk.getView();
        const int nghost = f.getNghost();
        const FieldLayout<Dim>& layout = f.getLayout(); 
        const M& mesh = f.get_mesh();
        const vector_type& dx = mesh.getMeshSpacing();
        const auto& domain = layout.getDomain();
        vector_type Len;
        Vector<int, Dim> N;


        for (unsigned d=0; d < Dim; ++d) {
            N[d] = domain[d].length();
            Len[d] = dx[d] * N[d];
        }
        
        typedef Kokkos::TeamPolicy<> team_policy;
        typedef Kokkos::TeamPolicy<>::member_type member_type;


        //using view_type_temp = typename detail::ViewType<FT, 3>::view_type;

        //view_type_temp viewLocal("viewLocal",fview.extent(0),fview.extent(1),fview.extent(2));

        double pi = std::acos(-1.0);
        Kokkos::complex<double> imag = {0.0, 1.0};

        size_t Np = *(this->localNum_mp);

        size_t flatN = N[0]*N[1]*N[2];

        Kokkos::parallel_for("ParticleAttrib::scatterPIFNUDFT compute",
                team_policy(flatN, Kokkos::AUTO),
                KOKKOS_CLASS_LAMBDA(const member_type& teamMember) {
                const size_t flatIndex = teamMember.league_rank();
               
#ifdef KOKKOS_ENABLE_CUDA
                const int k = (int)(flatIndex / (N[0] * N[1]));
                const int flatIndex2D = flatIndex - (k * N[0] * N[1]);
                const int i = flatIndex2D % N[0];
                const int j = (int)(flatIndex2D / N[0]);
#else

                const int i = (int)(flatIndex / (N[0] * N[1]));
                const int flatIndex2D = flatIndex - (i * N[0] * N[1]);
                const int k = flatIndex2D % N[0];
                const int j = (int)(flatIndex2D / N[0]);
#endif
                
                FT reducedValue = 0.0;
                Vector<int, 3> iVec = {i, j, k};
                vector_type kVec;
                for(size_t d = 0; d < Dim; ++d) {
                    //bool shift = (iVec[d] > (N[d]/2));
                    //kVec[d] = 2 * pi / Len[d] * (iVec[d] - shift * N[d]);
                    kVec[d] = 2 * pi / Len[d] * (iVec[d] - (N[d] / 2));
                }
                auto Sk = Skview(i+nghost, j+nghost, k+nghost);
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, Np),
                [=](const size_t idx, FT& innerReduce)
                {
                    double arg = 0.0;
                    for(size_t d = 0; d < Dim; ++d) {
                        arg += kVec[d]*pp(idx)[d];
                    }
                    const value_type& val = dview_m(idx);

                    innerReduce += Sk * (Kokkos::numbers::cos(arg) 
                                - imag * Kokkos::numbers::sin(arg)) * val;
                }, Kokkos::Sum<FT>(reducedValue));

                if(teamMember.team_rank() == 0) {
                    //viewLocal(i+nghost,j+nghost,k+nghost) = reducedValue;
                    fview(i+nghost,j+nghost,k+nghost) = reducedValue;
                }

                }
        );

        IpplTimings::stopTimer(scatterPIFNUDFTTimer);

        //static IpplTimings::TimerRef scatterAllReduceTimer = IpplTimings::getTimer("scatterAllReduce");           
        //IpplTimings::startTimer(scatterAllReduceTimer);                                               
        //int viewSize = fview.extent(0)*fview.extent(1)*fview.extent(2);
        //MPI_Allreduce(viewLocal.data(), fview.data(), viewSize, 
        //              MPI_C_DOUBLE_COMPLEX, MPI_SUM, Ippl::getComm());  
        //IpplTimings::stopTimer(scatterAllReduceTimer);

    }


    template<typename T, class... Properties>
    template <unsigned Dim, class M, class C, typename P2>
    void ParticleAttrib<T, Properties...>::gather(Field<T, Dim, M, C>& f,
                                                  const ParticleAttrib<Vector<P2, Dim>, Properties...>& pp)
    {

        //static IpplTimings::TimerRef fillHaloTimer = IpplTimings::getTimer("FillHalo");           
        //IpplTimings::startTimer(fillHaloTimer);                                               
        f.fillHalo();
        //IpplTimings::stopTimer(fillHaloTimer);                                               

        static IpplTimings::TimerRef gatherPICTimer = IpplTimings::getTimer("GatherPIC");           
        IpplTimings::startTimer(gatherPICTimer);                                               
        
        const typename Field<T, Dim, M, C>::view_type view = f.getView();

        const M& mesh = f.get_mesh();

        using vector_type = typename M::vector_type;
        using value_type  = typename ParticleAttrib<T, Properties...>::value_type;

        const vector_type& dx = mesh.getMeshSpacing();
        const vector_type& origin = mesh.getOrigin();
        const vector_type invdx = 1.0 / dx;

        const FieldLayout<Dim>& layout = f.getLayout(); 
        const NDIndex<Dim>& lDom = layout.getLocalNDIndex();
        const int nghost = f.getNghost();

        Kokkos::parallel_for(
            "ParticleAttrib::gather",
            *(this->localNum_mp),
            KOKKOS_CLASS_LAMBDA(const size_t idx)
            {
                // find nearest grid point
                vector_type l = (pp(idx) - origin) * invdx + 0.5;
                Vector<int, Dim> index = l;
                Vector<double, Dim> whi = l - index;
                Vector<double, Dim> wlo = 1.0 - whi;

                const size_t i = index[0] - lDom[0].first() + nghost;
                const size_t j = index[1] - lDom[1].first() + nghost;
                const size_t k = index[2] - lDom[2].first() + nghost;

                // gather
                value_type& val = dview_m(idx);
                val = wlo[0] * wlo[1] * wlo[2] * view(i-1, j-1, k-1)
                    + wlo[0] * wlo[1] * whi[2] * view(i-1, j-1, k  )
                    + wlo[0] * whi[1] * wlo[2] * view(i-1, j,   k-1)
                    + wlo[0] * whi[1] * whi[2] * view(i-1, j,   k  )
                    + whi[0] * wlo[1] * wlo[2] * view(i,   j-1, k-1)
                    + whi[0] * wlo[1] * whi[2] * view(i,   j-1, k  )
                    + whi[0] * whi[1] * wlo[2] * view(i,   j,   k-1)
                    + whi[0] * whi[1] * whi[2] * view(i,   j,   k  );
            }
        );
        IpplTimings::stopTimer(gatherPICTimer);                                               
    }

    template<typename T, class... Properties>
    template <unsigned Dim, class M, class C, class FT, class ST, class PT>
    void ParticleAttrib<T, Properties...>::gatherPIFNUDFT(Field<FT,Dim,M,C>& f, Field<ST,Dim,M,C>& Sk,
                                                   const ParticleAttrib< Vector<PT,Dim>, Properties... >& pp)
    {
        static IpplTimings::TimerRef gatherPIFNUDFTTimer = IpplTimings::getTimer("GatherPIFNUDFT");           
        IpplTimings::startTimer(gatherPIFNUDFTTimer);
        
        using view_type = typename Field<FT, Dim, M, C>::view_type;
        using vector_type = typename M::vector_type;
        using value_type  = typename ParticleAttrib<T, Properties...>::value_type;
        view_type fview = f.getView();
        typename Field<ST, Dim, M, C>::view_type Skview = Sk.getView();
        const int nghost = f.getNghost();
        const FieldLayout<Dim>& layout = f.getLayout(); 
        const M& mesh = f.get_mesh();
        const vector_type& dx = mesh.getMeshSpacing();
        const auto& domain = layout.getDomain();
        vector_type Len;
        Vector<int, Dim> N;

        for (unsigned d=0; d < Dim; ++d) {
            N[d] = domain[d].length();
            Len[d] = dx[d] * N[d];
        }



        typedef Kokkos::TeamPolicy<> team_policy;
        typedef Kokkos::TeamPolicy<>::member_type member_type;

        double pi = std::acos(-1.0);
        Kokkos::complex<double> imag = {0.0, 1.0};

        size_t Np = *(this->localNum_mp);

        size_t flatN = N[0]*N[1]*N[2];

        Kokkos::parallel_for("ParticleAttrib::gatherPIFNUDFT",
                team_policy(Np, Kokkos::AUTO),
                KOKKOS_CLASS_LAMBDA(const member_type& teamMember) {
                const size_t idx = teamMember.league_rank();

                value_type reducedValue = 0.0;
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, flatN),
                [=](const size_t flatIndex, value_type& innerReduce)
                {
                    
#ifdef KOKKOS_ENABLE_CUDA
                    const int k = (int)(flatIndex / (N[0] * N[1]));
                    const int flatIndex2D = flatIndex - (k * N[0] * N[1]);
                    const int i = flatIndex2D % N[0];
                    const int j = (int)(flatIndex2D / N[0]);
#else
                    const int i = (int)(flatIndex / (N[0] * N[1]));
                    const int flatIndex2D = flatIndex - (i * N[0] * N[1]);
                    const int k = flatIndex2D % N[0];
                    const int j = (int)(flatIndex2D / N[0]);
#endif

                    Vector<int, 3> iVec = {i, j, k};
                    vector_type kVec;
                    double Dr = 0.0, arg = 0.0;
                    for(size_t d = 0; d < Dim; ++d) {
                        //bool shift = (iVec[d] > (N[d]/2));
                        //kVec[d] = 2 * pi / Len[d] * (iVec[d] - shift * N[d]);
                        //kVec[d] = 2 * pi / Len[d] * iVec[d];
                        kVec[d] = 2 * pi / Len[d] * (iVec[d] - (N[d]/2));
                        Dr += kVec[d] * kVec[d];
                        arg += kVec[d]*pp(idx)[d];
                    }
                    

                    FT Ek = 0.0;
                    value_type Ex = 0.0;
                    auto rho = fview(i+nghost,j+nghost,k+nghost);
                    auto Sk = Skview(i+nghost,j+nghost,k+nghost);
                    for(size_t d = 0; d < Dim; ++d) {
                        
                        bool isNotZero = (Dr != 0.0);
                        double factor = isNotZero * (1.0 / (Dr + ((!isNotZero) * 1.0))); 
                        Ek = -(imag * kVec[d] * rho * factor);
                        
                        //Inverse Fourier transform when the lhs is real. Use when 
                        //we choose k \in [0 K) instead of from [-K/2+1 K/2] 
                        //Ex[d] = 2.0 * (Ek.real() * Kokkos::numbers::cos(arg) 
                        //        - Ek.imag() * Kokkos::numbers::sin(arg));
                        Ek *= Sk * (Kokkos::numbers::cos(arg) 
                                + imag * Kokkos::numbers::sin(arg));
                        Ex[d] = Ek.real();
                    }
                    
                    innerReduce += Ex;
                }, Kokkos::Sum<value_type>(reducedValue));

                teamMember.team_barrier();

                if(teamMember.team_rank() == 0) {
                    dview_m(idx) = reducedValue;
                }

                }
        );

        
        IpplTimings::stopTimer(gatherPIFNUDFTTimer);

    }

#ifdef KOKKOS_ENABLE_CUDA

    template<typename T, class... Properties>
    template<unsigned Dim>
    void ParticleAttrib<T, Properties...>::initializeNUFFT(FieldLayout<Dim>& layout, int type, ParameterList& fftParams) {
        
        fftType_mp = std::make_shared<FFT<NUFFTransform, Dim, double>>(layout, type, fftParams);
    }
    
    
    
    template<typename T, class... Properties>
    template <unsigned Dim, class M, class C, class FT, class ST, class PT>
    void ParticleAttrib<T, Properties...>::scatterPIFNUFFT(Field<FT,Dim,M,C>& f, Field<ST,Dim,M,C>& Sk,
                                                   const ParticleAttrib< Vector<PT,Dim>, Properties... >& pp)
    const
    {
        
        static IpplTimings::TimerRef scatterPIFNUFFTTimer = IpplTimings::getTimer("ScatterPIFNUFFT");           
        IpplTimings::startTimer(scatterPIFNUFFTTimer);

        auto q = *this;
        
        //Field<FT,Dim,M,C> tempField;

        //FieldLayout<Dim>& layout = f.getLayout(); 
        //M& mesh = f.get_mesh();

        //tempField.initialize(mesh, layout);
        
        //fftType_mp->transform(pp, q, tempField);
        fftType_mp->transform(pp, q, f);

        
        using view_type = typename Field<FT, Dim, M, C>::view_type;
        view_type fview = f.getView();
        //view_type viewLocal = tempField.getView();
        typename Field<ST, Dim, M, C>::view_type Skview = Sk.getView();
        const int nghost = f.getNghost();
        
        IpplTimings::stopTimer(scatterPIFNUFFTTimer);

        //static IpplTimings::TimerRef scatterAllReduceTimer = IpplTimings::getTimer("scatterAllReduce");           
        //IpplTimings::startTimer(scatterAllReduceTimer);                                               
        //int viewSize = fview.extent(0)*fview.extent(1)*fview.extent(2);
        //MPI_Allreduce(viewLocal.data(), fview.data(), viewSize, 
        //              MPI_C_DOUBLE_COMPLEX, MPI_SUM, Ippl::getComm());  
        //IpplTimings::stopTimer(scatterAllReduceTimer);

        //IpplTimings::startTimer(scatterPIFNUFFTTimer);

        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_for("Multiply with shape functions",
                            mdrange_type({nghost, nghost, nghost},
                                         {fview.extent(0) - nghost, 
                                          fview.extent(1) - nghost,
                                          fview.extent(2) - nghost}),
                            KOKKOS_LAMBDA(const size_t i,
                                          const size_t j,
                                          const size_t k)
        {
            fview(i, j, k) *= Skview(i, j, k);    
        });

        IpplTimings::stopTimer(scatterPIFNUFFTTimer);
    }


    template<typename T, class... Properties>
    template <unsigned Dim, class M, class C, class FT, class ST, class PT>
    void ParticleAttrib<T, Properties...>::gatherPIFNUFFT(Field<FT,Dim,M,C>& f, Field<ST,Dim,M,C>& Sk,
                                                   const ParticleAttrib< Vector<PT,Dim>, Properties... >& pp, 
                                                   ParticleAttrib<PT, Properties... >& q)
    {
        static IpplTimings::TimerRef gatherPIFNUFFTTimer = IpplTimings::getTimer("GatherPIFNUFFT");           
        IpplTimings::startTimer(gatherPIFNUFFTTimer);

        Field<FT,Dim,M,C> tempField;

        FieldLayout<Dim>& layout = f.getLayout(); 
        M& mesh = f.get_mesh();

        tempField.initialize(mesh, layout);
        
        using view_type = typename Field<FT, Dim, M, C>::view_type;
        using vector_type = typename M::vector_type;
        view_type fview = f.getView();
        view_type tempview = tempField.getView();
        auto qview = q.getView();
        typename Field<ST, Dim, M, C>::view_type Skview = Sk.getView();
        const int nghost = f.getNghost();
        const vector_type& dx = mesh.getMeshSpacing();
        const auto& domain = layout.getDomain();
        vector_type Len;
        Vector<int, Dim> N;

        for (unsigned d=0; d < Dim; ++d) {
            N[d] = domain[d].length();
            Len[d] = dx[d] * N[d];
        }


        double pi = std::acos(-1.0);
        Kokkos::complex<double> imag = {0.0, 1.0};
        size_t Np = *(this->localNum_mp);

        
        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        
        for(size_t gd = 0; gd < Dim; ++gd) {
            Kokkos::parallel_for("Gather NUFFT",
                                mdrange_type({nghost, nghost, nghost},
                                             {fview.extent(0) - nghost,
                                              fview.extent(1) - nghost,
                                              fview.extent(2) - nghost}),
                                KOKKOS_LAMBDA(const int i,
                                              const int j,
                                              const int k)
            {
                Vector<int, 3> iVec = {i-nghost, j-nghost, k-nghost};
                Vector<double, 3> kVec;

                double Dr = 0.0;
                for(size_t d = 0; d < Dim; ++d) {
                    kVec[d] = 2 * pi / Len[d] * (iVec[d] - (N[d] / 2));
                    //kVec[d] = (iVec[d] - (N[d] / 2));
                    Dr += kVec[d] * kVec[d];
                }

                tempview(i, j, k) = fview(i, j, k);
                
                bool isNotZero = (Dr != 0.0);
                double factor = isNotZero * (1.0 / (Dr + ((!isNotZero) * 1.0))); 
                
                tempview(i, j, k) *= -Skview(i, j, k) * (imag * kVec[gd] * factor);
            });

            fftType_mp->transform(pp, q, tempField);

            Kokkos::parallel_for("Assign E gather NUFFT",
                                Np,
                                KOKKOS_CLASS_LAMBDA(const size_t i)
            {
                dview_m(i)[gd] = qview(i);
            });
        }

        
        IpplTimings::stopTimer(gatherPIFNUFFTTimer);

    }
#endif

    template<typename P1, unsigned Dim, class M, class C, typename P2, typename P3, typename P4, class... Properties>
    inline
    void scatterPIFNUFFT(const ParticleAttrib<P1, Properties...>& attrib, Field<P2, Dim, M, C>& f,
                 Field<P3, Dim, M, C>& Sk, const ParticleAttrib<Vector<P4, Dim>, Properties...>& pp)
    {
#ifdef KOKKOS_ENABLE_CUDA
        attrib.scatterPIFNUFFT(f, Sk, pp);
#else
        //throw IpplException("scatterPIFNUFFT", "The NUFFT library cuFINUFFT currently only works with CUDA and hence Kokkos needs to 
        //                     be compiled with CUDA. Otherwise use scatterPIFNUDFT.");
#endif
    }

    template<typename P1, unsigned Dim, class M, class C, typename P2, typename P3, typename P4, class... Properties>
    inline
    void gatherPIFNUFFT(ParticleAttrib<P1, Properties...>& attrib, Field<P2, Dim, M, C>& f,
                 Field<P3, Dim, M, C>& Sk, const ParticleAttrib<Vector<P4, Dim>, Properties...>& pp, 
                 ParticleAttrib<P4, Properties... >& q)
    {
#ifdef KOKKOS_ENABLE_CUDA
        attrib.gatherPIFNUFFT(f, Sk, pp, q);
#else
        //throw IpplException("gatherPIFNUFFT",
        //                    "The NUFFT library cuFINUFFT currently only works with CUDA and hence Kokkos needs to 
        //                     be compiled with CUDA. Otherwise use gatherPIFNUDFT.");
#endif
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

    template<typename P1, unsigned Dim, class M, class C, typename P2, typename P3, typename P4, class... Properties>
    inline
    void scatterPIFNUDFT(const ParticleAttrib<P1, Properties...>& attrib, Field<P2, Dim, M, C>& f,
                 Field<P3, Dim, M, C>& Sk, const ParticleAttrib<Vector<P4, Dim>, Properties...>& pp)
    {
        attrib.scatterPIFNUDFT(f, Sk, pp);
    }



    template<typename P1, unsigned Dim, class M, class C, typename P2, class... Properties>
    inline
    void gather(ParticleAttrib<P1, Properties...>& attrib, Field<P1, Dim, M, C>& f,
                const ParticleAttrib<Vector<P2, Dim>, Properties...>& pp)
    {
        attrib.gather(f, pp);
    }

    template<typename P1, unsigned Dim, class M, class C, typename P2, typename P3, typename P4, class... Properties>
    inline
    void gatherPIFNUDFT(ParticleAttrib<P1, Properties...>& attrib, Field<P2, Dim, M, C>& f,
                 Field<P3, Dim, M, C>& Sk, const ParticleAttrib<Vector<P4, Dim>, Properties...>& pp)
    {
        attrib.gatherPIFNUDFT(f, Sk, pp);
    }


    #define DefineParticleReduction(fun, name, op, MPI_Op)                                                   \
    template<typename T, class... Properties>                                                                \
    T ParticleAttrib<T, Properties...>::name() {                                                             \
        T temp = 0.0;                                                                                        \
        Kokkos::parallel_reduce("fun", *(this->localNum_mp),                                                  \
                               KOKKOS_CLASS_LAMBDA(const size_t i, T& valL) {                                \
                                    T myVal = dview_m(i);                                                    \
                                    op;                                                                      \
                               }, Kokkos::fun<T>(temp));                                                     \
        T globaltemp = 0.0;                                                                                  \
        MPI_Datatype type = get_mpi_datatype<T>(temp);                                                       \
        MPI_Allreduce(&temp, &globaltemp, 1, type, MPI_Op, Ippl::getComm());                                 \
        return globaltemp;                                                                                   \
    }

    DefineParticleReduction(Sum,  sum,  valL += myVal, MPI_SUM)
    DefineParticleReduction(Max,  max,  if(myVal > valL) valL = myVal, MPI_MAX)
    DefineParticleReduction(Min,  min,  if(myVal < valL) valL = myVal, MPI_MIN)
    DefineParticleReduction(Prod, prod, valL *= myVal, MPI_PROD)
}
