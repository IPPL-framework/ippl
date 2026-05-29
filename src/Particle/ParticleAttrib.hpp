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
#include "Ippl.h"

#include <limits>

#include "Communicate/DataTypes.h"

#include "Utility/BufferView.h"
#include "Utility/IpplException.h"
#include "Utility/IpplTimings.h"

#ifdef IPPL_ENABLE_FFT
#include "FFT/FFT.h"
#endif
#include "Interpolation/Binning.h"
#include "Interpolation/Gather/Gather.h"
#include "Interpolation/Kernels.h"
#include "Interpolation/Scatter/Scatter.h"
#include "Particle/ParticleSort.h"
#include "Particle/SortBuffer.h"

namespace ippl {

    template <typename T, class... Properties>
    void ParticleAttrib<T, Properties...>::create(size_type n, bool non_destructive) {
        size_type required = *(this->localNum_mp) + n;
        if (this->size() < required) {
            double overalloc = Comm->getDefaultOverallocation();
            const size_type target = static_cast<size_type>(required * overalloc);
            if (non_destructive) {
                // Kokkos::resize preserves existing entries when growing.
                this->resize(target);
            } else {
                // Kokkos::realloc is destructive (free + alloc, no copy).
                this->realloc(target);
            }
        }
    }

    template <typename T, class... Properties>
    void ParticleAttrib<T, Properties...>::alloc(size_type n) {
        double overalloc = Comm->getDefaultOverallocation();
        this->realloc(static_cast<size_type>(n * overalloc));
    }

    template <typename T, class... Properties>
    void ParticleAttrib<T, Properties...>::destroy(const hash_type& deleteIndex,
                                                   const hash_type& keepIndex,
                                                   size_type invalidCount) {
        // Replace all invalid particles in the valid region with valid
        // particles in the invalid region
        auto dview = dview_m;
        using policy_type = Kokkos::RangePolicy<execution_space>;
        Kokkos::parallel_for(
            "ParticleAttrib::destroy()", policy_type(0, invalidCount),
            KOKKOS_LAMBDA(const size_t i) {
                dview(deleteIndex(i)) = dview(keepIndex(i));
            });
    }

    template <typename T, class... Properties>
    void ParticleAttrib<T, Properties...>::pack(const hash_type& hash) {
        auto size = hash.extent(0);
        if (buf_m.extent(0) < size) {
            double overalloc = Comm->getDefaultOverallocation();
            Kokkos::realloc(buf_m, static_cast<size_type>(size * overalloc));
        }

        auto buf = buf_m;
        auto dview = dview_m;
        using policy_type = Kokkos::RangePolicy<execution_space>;
        Kokkos::parallel_for(
            "ParticleAttrib::pack()", policy_type(0, size),
            KOKKOS_LAMBDA(const size_t i) { buf(i) = dview(hash(i)); });
    }

    template <typename T, class... Properties>
    void ParticleAttrib<T, Properties...>::unpack(size_type nrecvs) {
        size_type required = *(this->localNum_mp) + nrecvs;
        this->resize(required);

        size_type count   = *(this->localNum_mp);
        auto buf = buf_m;
        auto dview = dview_m;
        using policy_type = Kokkos::RangePolicy<execution_space>;
        Kokkos::parallel_for(
            "ParticleAttrib::unpack()", policy_type(0, nrecvs),
            KOKKOS_LAMBDA(const size_t i) { dview(count + i) = buf(i); });
        Kokkos::fence();
    }

    template <typename T, class... Properties>
    ParticleAttrib<T, Properties...>& ParticleAttrib<T, Properties...>::operator=(T x) {
        auto dview = dview_m;
        using policy_type = Kokkos::RangePolicy<execution_space>;
        Kokkos::parallel_for(
            "ParticleAttrib::operator=()", policy_type(0, *(this->localNum_mp)),
            KOKKOS_LAMBDA(const size_t i) { dview(i) = x; });
        return *this;
    }

    template <typename T, class... Properties>
    template <typename E, size_t N>
    ParticleAttrib<T, Properties...>& ParticleAttrib<T, Properties...>::operator=(
        detail::Expression<E, N> const& expr) {
        using capture_type = detail::CapturedExpression<E, N>;
        capture_type expr_ = reinterpret_cast<const capture_type&>(expr);

        auto dview = dview_m;
        using policy_type = Kokkos::RangePolicy<execution_space>;
        Kokkos::parallel_for(
            "ParticleAttrib::operator=()", policy_type(0, *(this->localNum_mp)),
            KOKKOS_LAMBDA(const size_t i) { dview(i) = expr_(i); });
        return *this;
    }

    template <typename T, class... Properties>
    template <typename Field, class PT, typename policy_type>
    void ParticleAttrib<T, Properties...>::scatter(
        Field& f, const ParticleAttrib<Vector<PT, Field::dim>, Properties...>& pp,
        policy_type iteration_policy, hash_type hash_array) const {
        constexpr unsigned Dim = Field::dim;
        using PositionType     = typename Field::Mesh_t::value_type;

        const bool useHashView = hash_array.extent(0) > 0;
        const size_t nLocal    = static_cast<size_t>(*(this->localNum_mp));
        const size_t policyEnd = static_cast<size_t>(iteration_policy.end());

        // Default-policy, no-hash CIC case (alpine PIC examples): route
        // through the kernel-aware Interpolation::Scatter framework so the
        // same atomic / tiled / output-focused dispatch (and TileSizeCache
        // lookup) used by NUFFT applies to PIC too. The new framework's
        // dimension-specialised AtomicScatter only covers Dim 1..3, so for
        // higher-dimensional fields we keep the legacy direct path.
        if (!useHashView && policyEnd == nLocal && Dim <= 3) {
            Interpolation::LinearKernel<PositionType> cic;
            this->scatter_kernel(f, pp, cic);
            return;
        }

        // Custom range / hash-permuted scatter: legacy direct CIC kernel.
        static IpplTimings::TimerRef scatterTimer = IpplTimings::getTimer("scatter");
        IpplTimings::startTimer(scatterTimer);
        using view_type = typename Field::view_type;
        view_type view  = f.getView();

        using mesh_type       = typename Field::Mesh_t;
        const mesh_type& mesh = f.get_mesh();

        using vector_type = typename mesh_type::vector_type;
        using value_type  = typename ParticleAttrib<T, Properties...>::value_type;

        const vector_type& dx     = mesh.getMeshSpacing();
        const vector_type& origin = mesh.getOrigin();
        const vector_type invdx   = 1.0 / dx;

        const FieldLayout<Dim>& layout = f.getLayout();
        const NDIndex<Dim>& lDom       = layout.getLocalNDIndex();
        const int nghost               = f.getNghost();

        if (useHashView && (iteration_policy.end() > hash_array.extent(0))) {
            throw IpplException(
                "ParticleAttrib::scatter",
                "Hash array was passed to scatter, but size does not match iteration policy.");
        }
        auto dview = dview_m;
        auto ppview = pp.getView();
        Kokkos::parallel_for(
            "ParticleAttrib::scatter", iteration_policy,
            KOKKOS_LAMBDA(const size_t idx) {
                size_t mapped_idx = useHashView ? hash_array(idx) : idx;

                vector_type l                        = (ppview(mapped_idx) - origin) * invdx + 0.5;
                Vector<int, Field::dim> index        = l;
                Vector<PositionType, Field::dim> whi = l - index;
                Vector<PositionType, Field::dim> wlo = 1.0 - whi;

                Vector<size_t, Field::dim> args = index - lDom.first() + nghost;

                const value_type& val = dview(mapped_idx);
                detail::scatterToField(std::make_index_sequence<1 << Field::dim>{}, view, wlo, whi,
                                       args, val);
            });
        IpplTimings::stopTimer(scatterTimer);

        static IpplTimings::TimerRef accumulateHaloTimer = IpplTimings::getTimer("accumulateHalo");
        IpplTimings::startTimer(accumulateHaloTimer);
        f.accumulateHalo();
        IpplTimings::stopTimer(accumulateHaloTimer);
    }

    template <typename T, class... Properties>
    template <typename Field, typename P2>
    void ParticleAttrib<T, Properties...>::gather(
        Field& f, const ParticleAttrib<Vector<P2, Field::dim>, Properties...>& pp,
        const bool addToAttribute) {
        constexpr unsigned Dim = Field::dim;
        using PositionType     = typename Field::Mesh_t::value_type;

        // Route legacy CIC gather through the kernel-aware Interpolation
        // framework when the dimension is supported (Dim 1..3). For higher
        // dimensions, fall back to the direct CIC kernel.
        if constexpr (Dim <= 3) {
            Interpolation::LinearKernel<PositionType> cic;
            this->gather(f, pp, cic, addToAttribute);
            return;
        } else {
            static IpplTimings::TimerRef fillHaloTimer = IpplTimings::getTimer("fillHalo");
            IpplTimings::startTimer(fillHaloTimer);
            f.fillHalo();
            IpplTimings::stopTimer(fillHaloTimer);

            static IpplTimings::TimerRef gatherTimer = IpplTimings::getTimer("gather");
            IpplTimings::startTimer(gatherTimer);
            const typename Field::view_type view = f.getView();

            using mesh_type       = typename Field::Mesh_t;
            const mesh_type& mesh = f.get_mesh();

            using vector_type = typename mesh_type::vector_type;

            const vector_type& dx     = mesh.getMeshSpacing();
            const vector_type& origin = mesh.getOrigin();
            const vector_type invdx   = 1.0 / dx;

            const FieldLayout<Dim>& layout = f.getLayout();
            const NDIndex<Dim>& lDom       = layout.getLocalNDIndex();
            const int nghost               = f.getNghost();

            auto dview        = dview_m;
            auto ppview       = pp.getView();
            using policy_type = Kokkos::RangePolicy<execution_space>;
            Kokkos::parallel_for(
                "ParticleAttrib::gather", policy_type(0, *(this->localNum_mp)),
                KOKKOS_LAMBDA(const size_t idx) {
                    vector_type l                        = (ppview(idx) - origin) * invdx + 0.5;
                    Vector<int, Field::dim> index        = l;
                    Vector<PositionType, Field::dim> whi = l - index;
                    Vector<PositionType, Field::dim> wlo = 1.0 - whi;

                    Vector<size_t, Field::dim> args = index - lDom.first() + nghost;

                    value_type gathered = detail::gatherFromField(
                        std::make_index_sequence<1 << Field::dim>{}, view, wlo, whi, args);
                    if (addToAttribute) {
                        dview(idx) += gathered;
                    } else {
                        dview(idx) = gathered;
                    }
                });
            IpplTimings::stopTimer(gatherTimer);
        }
    }

    template <typename T, class... Properties>
    template <typename Field, typename P2, typename Kernel>
    void ParticleAttrib<T, Properties...>::scatter_kernel(
        Field& f, const ParticleAttrib<Vector<P2, Field::dim>, Properties...>& pp,
        const Kernel& kernel, const Interpolation::ScatterConfig<Field::dim>& config) const {
        auto scatter_impl = Scatter<Kernel, Field::dim>(kernel, config);
        scatter_impl(f, pp, *this);
    }

    template <typename T, class... Properties>
    template <typename Field, typename P2, typename Kernel>
    void ParticleAttrib<T, Properties...>::gather(
        Field& f, const ParticleAttrib<Vector<P2, Field::dim>, Properties...>& pp,
        const Kernel& kernel, bool addToAttribute,
        const Interpolation::GatherConfig<Field::dim>& config) {
        constexpr unsigned Dim       = Field::dim;
        auto modified_config         = config;
        modified_config.add_to_attribute = addToAttribute;
        auto gather_impl             = ippl::Gather<Kernel, Dim>(kernel, modified_config);
        gather_impl(f, pp, *this);
    }

    template <typename T, class... Properties>
    void ParticleAttrib<T, Properties...>::applyPermutation(const hash_type& permutation) {
        const auto view = this->getView();  // trimmed to localNum_mp
        const auto size = this->getParticleCount();

        view_type temp("copy", size);

        using policy_type = Kokkos::RangePolicy<execution_space>;
        Kokkos::parallel_for(
            "Copy to temp", policy_type(0, size),
            KOKKOS_LAMBDA(const size_type& i) { temp(permutation(i)) = view(i); });

        Kokkos::fence();

        Kokkos::deep_copy(view, temp);
    }

    template <typename T, class... Properties>
    void ParticleAttrib<T, Properties...>::internalCopy(const hash_type& indices) {
        auto copySize = indices.size();

        // Snapshot the current count BEFORE create() increments localNum_mp.
        const size_type oldSize = *(this->localNum_mp);

        create(copySize);  // localNum_mp becomes oldSize + copySize

        auto view = this->getView();
        Kokkos::parallel_for(
            "internalCopy", Kokkos::RangePolicy<execution_space>(0, copySize),
            KOKKOS_LAMBDA(const size_type& i) { view(oldSize + i) = view(indices(i)); });

        Kokkos::fence();
    }

#ifdef IPPL_ENABLE_FFT
    template <typename T, class... Properties>
    template <unsigned Dim, class M, class C, class FT, class ST, class PT>
    void ParticleAttrib<T, Properties...>::scatterPIFNUFFT(
        Field<FT, Dim, M, C>& f, Field<ST, Dim, M, C>& Sk,
        const ParticleAttrib<Vector<PT, Dim>, Properties...>& pp,
        FFT<NUFFTransform, Field<ST, Dim, M, C>>* nufft, const MPI_Comm&) const {
        static IpplTimings::TimerRef scatterPIFNUFFTTimer =
            IpplTimings::getTimer("ScatterPIFNUFFT");
        IpplTimings::startTimer(scatterPIFNUFFTTimer);

        auto q = *this;

        typename Field<FT, Dim, M, C>::uniform_type tempField;

        FieldLayout<Dim>& layout = f.getLayout();
        M& mesh                  = f.get_mesh();

        tempField.initialize(mesh, layout);
        tempField = 0.0;

        nufft->transform(pp, q, tempField);

        using view_type                                  = typename Field<FT, Dim, M, C>::view_type;
        view_type fview                                  = f.getView();
        view_type viewLocal                              = tempField.getView();
        typename Field<ST, Dim, M, C>::view_type Skview = Sk.getView();
        const int nghost                                 = f.getNghost();

        IpplTimings::stopTimer(scatterPIFNUFFTTimer);

        Kokkos::deep_copy(execution_space{}, fview, viewLocal);

        IpplTimings::startTimer(scatterPIFNUFFTTimer);

        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_for(
            "Multiply with shape functions",
            mdrange_type(
                {nghost, nghost, nghost},
                {fview.extent(0) - nghost, fview.extent(1) - nghost, fview.extent(2) - nghost}),
            KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                fview(i, j, k) *= Skview(i, j, k);
            });

        IpplTimings::stopTimer(scatterPIFNUFFTTimer);
    }

    template <typename T, class... Properties>
    template <unsigned Dim, class M, class C, class FT, class ST, class PT>
    void ParticleAttrib<T, Properties...>::gatherPIFNUFFT(
        Field<FT, Dim, M, C>& f, Field<ST, Dim, M, C>& Sk,
        const ParticleAttrib<Vector<PT, Dim>, Properties...>& pp,
        FFT<NUFFTransform, Field<ST, Dim, M, C>>* nufft, ParticleAttrib<PT, Properties...>& q) {
        static IpplTimings::TimerRef gatherPIFNUFFTTimer = IpplTimings::getTimer("GatherPIFNUFFT");
        IpplTimings::startTimer(gatherPIFNUFFTTimer);

        typename Field<FT, Dim, M, C>::uniform_type tempField;

        FieldLayout<Dim>& layout = f.getLayout();
        M& mesh                  = f.get_mesh();
        const auto& lDom         = layout.getLocalNDIndex();

        tempField.initialize(mesh, layout);

        using view_type                                  = typename Field<FT, Dim, M, C>::view_type;
        using vector_type                                = typename M::vector_type;
        view_type fview                                  = f.getView();
        view_type tempview                               = tempField.getView();
        auto qview                                       = q.getView();
        typename Field<ST, Dim, M, C>::view_type Skview  = Sk.getView();
        const int nghost                                 = f.getNghost();
        const vector_type& dx                            = mesh.getMeshSpacing();
        const auto& domain                               = layout.getDomain();
        vector_type Len;
        Vector<int, Dim> N;

        for (unsigned d = 0; d < Dim; ++d) {
            N[d]   = domain[d].length();
            Len[d] = dx[d] * N[d];
        }

        double pi                    = std::acos(-1.0);
        Kokkos::complex<double> imag = {0.0, 1.0};
        size_t Np                    = *(this->localNum_mp);

        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

        auto dview = dview_m;

        for (size_t gd = 0; gd < Dim; ++gd) {
            Kokkos::parallel_for(
                "Gather NUFFT",
                mdrange_type(
                    {nghost, nghost, nghost},
                    {fview.extent(0) - nghost, fview.extent(1) - nghost,
                     fview.extent(2) - nghost}),
                KOKKOS_LAMBDA(const int i, const int j, const int k) {
                    Vector<int, 3> iVec = {i, j, k};
                    for (unsigned d = 0; d < Dim; ++d) {
                        iVec[d] = iVec[d] - nghost + lDom[d].first();
                    }
                    Vector<double, 3> kVec;

                    double Dr = 0.0;
                    for (size_t d = 0; d < Dim; ++d) {
                        bool shift = (iVec[d] > (N[d] / 2));
                        kVec[d]    = 2 * pi / Len[d] * (iVec[d] - shift * N[d]);
                        Dr += kVec[d] * kVec[d];
                    }

                    tempview(i, j, k) = fview(i, j, k);

                    bool isNotZero = (Dr != 0.0);
                    double factor  = isNotZero * (1.0 / (Dr + ((!isNotZero) * 1.0)));

                    tempview(i, j, k) *= -Skview(i, j, k) * (imag * kVec[gd] * factor);
                });

            nufft->transform(pp, q, tempField);

            Kokkos::parallel_for(
                "Assign E gather NUFFT", Np,
                KOKKOS_LAMBDA(const size_t i) { dview(i)[gd] = qview(i); });
        }

        IpplTimings::stopTimer(gatherPIFNUFFTTimer);
    }
#endif  // IPPL_ENABLE_FFT

    /*
     * Non-class functions
     */

    /**
     * @brief Non-class interface for scattering particle attribute data onto a field.
     *
     * This overload preserves legacy functionality by providing a default iteration policy.
     * It calls the member scatter() with a default Kokkos::RangePolicy.
     *
     * @note The default behaviour is to scatter all particles without any custom index mapping.
     *
     * @tparam Attrib1 The type of the particle attribute.
     * @tparam Field The type of the field.
     * @tparam Attrib2 The type of the particle position attribute.
     * @tparam policy_type (Default: `Kokkos::RangePolicy<typename Field::execution_space>`)
     * @param attrib The particle attribute to scatter.
     * @param f The field onto which the data is scattered.
     * @param pp The ParticleAttrib representing particle positions.
     */
    template <typename Attrib1, typename Field, typename Attrib2,
              typename policy_type = Kokkos::RangePolicy<typename Field::execution_space>>
    inline void scatter(const Attrib1& attrib, Field& f, const Attrib2& pp) {
        attrib.scatter(f, pp, policy_type(0, attrib.getParticleCount()));
    }

    /**
     * @brief Non-class interface for scattering with a custom iteration policy and optional index
     * array.
     *
     * This overload allows the caller to specify a custom `Kokkos::range_policy` and an optional
     * `ippl::hash_type` array. It forwards the parameters to the member scatter() function.
     *
     * @note See ParticleAttrib::scatter() for more information on the custom iteration
     * functionality.
     *
     * @tparam Attrib1 The type of the particle attribute.
     * @tparam Field The type of the field.
     * @tparam Attrib2 The type of the particle position attribute.
     * @tparam policy_type (Default: `Kokkos::RangePolicy<typename Field::execution_space>`)
     * @param attrib The particle attribute to scatter.
     * @param f The field onto which the data is scattered.
     * @param pp The ParticleAttrib representing particle positions.
     * @param iteration_policy A custom `Kokkos::range_policy` defining the iteration range.
     * @param hash_array An optional `ippl::hash_type` array for index mapping.
     */
    template <typename Attrib1, typename Field, typename Attrib2,
              typename policy_type = Kokkos::RangePolicy<typename Field::execution_space>>
    inline void scatter(const Attrib1& attrib, Field& f, const Attrib2& pp,
                        policy_type iteration_policy,
                        typename Attrib1::hash_type hash_array = {}) {
        attrib.scatter(f, pp, iteration_policy, hash_array);
    }

    /**
     * @brief Non-class interface for gathering field data into a particle attribute.
     *
     * This interface calls the member ParticleAttrib::gather() function with the provided
     * parameters and preserving legacy behavior by assigning `addToAttribute` a default value.
     *
     * @note See ParticleAttrib::gather() for more information on the behavior of `addToAttribute`.
     *
     * @tparam Attrib1 The type of the particle attribute.
     * @tparam Field The type of the field.
     * @tparam Attrib2 The type of the particle position attribute.
     * @param attrib The particle attribute to gather data into.
     * @param f The field from which data is gathered.
     * @param pp The ParticleAttrib representing particle positions.
     * @param addToAttribute If true, the gathered field value is added to the current attribute
     * value; otherwise, the attribute value is overwritten.
     */
    template <typename Attrib1, typename Field, typename Attrib2>
    inline void gather(Attrib1& attrib, Field& f, const Attrib2& pp,
                       const bool addToAttribute = false) {
        attrib.gather(f, pp, addToAttribute);
    }

#ifdef IPPL_ENABLE_FFT
    template <typename P1, unsigned Dim, class M, class C, typename P2, typename P3, typename P4,
              class... Properties>
    inline void gatherPIFNUFFT(ParticleAttrib<P1, Properties...>& attrib, Field<P2, Dim, M, C>& f,
                               Field<P3, Dim, M, C>& Sk,
                               const ParticleAttrib<Vector<P4, Dim>, Properties...>& pp,
                               ippl::FFT<ippl::NUFFTransform, Field<P3, Dim, M, C>>* nufft,
                               ParticleAttrib<P4, Properties...>& q) {
        attrib.gatherPIFNUFFT(f, Sk, pp, nufft, q);
    }

    template <typename P1, unsigned Dim, class M, class C, typename P2, typename P3, typename P4,
              class... Properties>
    inline void scatterPIFNUFFT(const ParticleAttrib<P1, Properties...>& attrib,
                                Field<P2, Dim, M, C>& f, Field<P3, Dim, M, C>& Sk,
                                const ParticleAttrib<Vector<P4, Dim>, Properties...>& pp,
                                FFT<NUFFTransform, Field<P3, Dim, M, C>>* nufft,
                                const MPI_Comm& spaceComm = MPI_COMM_WORLD) {
        attrib.scatterPIFNUFFT(f, Sk, pp, nufft, spaceComm);
    }
#endif  // IPPL_ENABLE_FFT

    namespace detail {
        // Identity element for each particle reduction. Using `0` for max/min
        // would clobber legitimate negative/positive results on attributes
        // whose value range crosses 0; using `0` for prod would zero the
        // result of an empty reduction.
        template <typename T> struct ReductionIdentitySum  { static constexpr T value = T(0); };
        template <typename T> struct ReductionIdentityProd { static constexpr T value = T(1); };
        template <typename T> struct ReductionIdentityMax {
            static constexpr T value = std::numeric_limits<T>::lowest();
        };
        template <typename T> struct ReductionIdentityMin {
            static constexpr T value = std::numeric_limits<T>::max();
        };
    }  // namespace detail

#define DefineParticleReduction(KOp, name, op, MPI_Op, IdentityT)       \
    template <typename T, class... Properties>                          \
    T ParticleAttrib<T, Properties...>::name() {                        \
        T temp            = detail::IdentityT<T>::value;                \
        auto dview        = dview_m;                                    \
        using policy_type = Kokkos::RangePolicy<execution_space>;       \
        Kokkos::parallel_reduce(                                        \
            "ParticleAttrib::" #name, policy_type(0, *(this->localNum_mp)), \
            KOKKOS_LAMBDA(const size_t i, T& valL) {                    \
                T myVal = dview(i);                                     \
                op;                                                     \
            },                                                          \
            Kokkos::KOp<T>(temp));                                      \
        T globaltemp = detail::IdentityT<T>::value;                     \
        Comm->allreduce(temp, globaltemp, 1, MPI_Op<T>());              \
        return globaltemp;                                              \
    }

    DefineParticleReduction(Sum, sum, valL += myVal, std::plus, ReductionIdentitySum)
    DefineParticleReduction(Max, max, if (myVal > valL) valL = myVal, std::greater,
                            ReductionIdentityMax)
    DefineParticleReduction(Min, min, if (myVal < valL) valL = myVal, std::less,
                            ReductionIdentityMin)
    DefineParticleReduction(Prod, prod, valL *= myVal, std::multiplies, ReductionIdentityProd)

}  // namespace ippl
