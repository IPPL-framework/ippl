//
// High-Order Interpolation
//   Implementation of generic scatter/gather operations for ParticleAttrib and Field.
//

namespace ippl {
    namespace detail {

        /**
         * @brief Scatter particle attribute data to field using a higher-order kernel.
         *
         * This performs higher-order spreading using any kernel that provides:
         * - width(): returns the kernel width (number of grid points per dimension)
         * - operator()(T x): evaluates the kernel at position x in [-1, 1]
         *
         * @tparam Kernel Kernel type (e.g., NUFFT::ESKernel)
         * @tparam T Particle attribute value type
         * @tparam Field Field type
         * @tparam PT Position type
         * @tparam Properties... Kokkos properties
         */
        template <typename Kernel, typename T, typename Field, typename PT, class... Properties>
        void scatterHighOrder(const Kokkos::View<T*, Properties...>& attribView, Field& f,
                              const Kokkos::View<Vector<PT, Field::dim>*, Properties...>& posView,
                              const Kernel& kernel, size_t localNum) {
            constexpr unsigned Dim = Field::dim;
            using PositionType     = typename Field::Mesh_t::value_type;
            using view_type        = typename Field::view_type;
            using execution_space  = typename view_type::execution_space;

            view_type view = f.getView();

            using mesh_type       = typename Field::Mesh_t;
            const mesh_type& mesh = f.get_mesh();

            using vector_type = typename mesh_type::vector_type;

            const vector_type& dx     = mesh.getMeshSpacing();
            const vector_type& origin = mesh.getOrigin();
            const vector_type invdx   = 1.0 / dx;

            const FieldLayout<Dim>& layout = f.getLayout();
            const NDIndex<Dim>& lDom       = layout.getLocalNDIndex();
            const int nghost               = f.getNghost();

            const int w  = kernel.width();
            const int hw = w / 2;

            assert(nghost >= hw);

            using policy_type = Kokkos::RangePolicy<execution_space>;
            Kokkos::parallel_for(
                "scatterHighOrder_no_wrap", policy_type(0, localNum),
                KOKKOS_LAMBDA(const size_t idx) {
                    auto& view_ref    = view;
                    auto& kernel_ref  = kernel;
                    const auto lDom_c = lDom;
                    const int ngh     = nghost;

                    // 1) Global scaled position in cell units
                    Vector<PositionType, Dim> sx;
                    for (unsigned d = 0; d < Dim; ++d) {
                        sx[d] = (posView(idx)[d] - origin[d]) * invdx[d];
                    }

                    // 2) Starting global grid indices (no periodic wrap here)
                    Vector<int64_t, Dim> idx0;
                    for (unsigned d = 0; d < Dim; ++d) {
                        const bool odd = (w & 1);
                        idx0[d]        = odd ? static_cast<int64_t>(Kokkos::round(sx[d])) - hw
                                             : static_cast<int64_t>(sx[d]) + 1 - hw;
                    }

                    const T& val = attribView(idx);

                    if constexpr (Dim == 3) {
                        // Precompute kernel values
                        PositionType kernel_vals[3][20];  // assumes w <= 20
                        for (int i = 0; i < w; ++i) {
                            for (unsigned d = 0; d < 3; ++d) {
                                const PositionType diff =
                                    (sx[d] - static_cast<PositionType>(idx0[d] + i))
                                    * (PositionType(2.0) / w);
                                kernel_vals[d][i] = kernel_ref(diff);
                            }
                        }

                        for (int k = 0; k < w; ++k) {
                            const int64_t global_z = idx0[2] + k;
                            const size_t local_z =
                                static_cast<size_t>(global_z - lDom_c[2].first() + ngh);

                            for (int j = 0; j < w; ++j) {
                                const int64_t global_y = idx0[1] + j;
                                const size_t local_y =
                                    static_cast<size_t>(global_y - lDom_c[1].first() + ngh);

                                const PositionType weight_yz =
                                    kernel_vals[1][j] * kernel_vals[2][k];

                                for (int i = 0; i < w; ++i) {
                                    const int64_t global_x = idx0[0] + i;
                                    const size_t local_x =
                                        static_cast<size_t>(global_x - lDom_c[0].first() + ngh);

                                    const PositionType weight = kernel_vals[0][i] * weight_yz;
                                    Kokkos::atomic_add(&view_ref(local_x, local_y, local_z),
                                                       val * weight);
                                }
                            }
                        }
                    } else if constexpr (Dim == 2) {
                        PositionType kernel_vals[2][20];
                        for (int i = 0; i < w; ++i) {
                            for (unsigned d = 0; d < 2; ++d) {
                                const PositionType diff =
                                    (sx[d] - static_cast<PositionType>(idx0[d] + i))
                                    * (PositionType(2.0) / w);
                                kernel_vals[d][i] = kernel_ref(diff);
                            }
                        }

                        for (int j = 0; j < w; ++j) {
                            const int64_t global_y = idx0[1] + j;
                            const size_t local_y =
                                static_cast<size_t>(global_y - lDom_c[1].first() + ngh);

                            for (int i = 0; i < w; ++i) {
                                const int64_t global_x = idx0[0] + i;
                                const size_t local_x =
                                    static_cast<size_t>(global_x - lDom_c[0].first() + ngh);

                                const PositionType weight = kernel_vals[0][i] * kernel_vals[1][j];
                                Kokkos::atomic_add(&view_ref(local_x, local_y), val * weight);
                            }
                        }
                    } else {  // 1D
                        for (int i = 0; i < w; ++i) {
                            const int64_t global_x = idx0[0] + i;
                            const size_t local_x =
                                static_cast<size_t>(global_x - lDom_c[0].first() + ngh);

                            const PositionType diff =
                                (sx[0] - static_cast<PositionType>(idx0[0] + i))
                                * (PositionType(2.0) / w);
                            const PositionType weight = kernel_ref(diff);
                            Kokkos::atomic_add(&view_ref(local_x), val * weight);
                        }
                    }
                });

            // Accumulate contributions in halos into neighbors / periodic images
            f.accumulateHalo();
        }

        /**
         * @brief Gather field data to particle attribute using a higher-order kernel.
         *
         * This performs higher-order interpolation using any kernel that provides:
         * - width(): returns the kernel width (number of grid points per dimension)
         * - operator()(T x): evaluates the kernel at position x in [-1, 1]
         *
         * @tparam Kernel Kernel type (e.g., NUFFT::ESKernel)
         * @tparam T Particle attribute value type
         * @tparam Field Field type
         * @tparam PT Position type
         * @tparam Properties... Kokkos properties
         */
        template <typename Kernel, typename T, typename Field, typename PT, class... Properties>
        void gatherHighOrder(Kokkos::View<T*, Properties...>& attribView, Field& f,
                             const Kokkos::View<Vector<PT, Field::dim>*, Properties...>& posView,
                             const Kernel& kernel, size_t localNum, bool addToAttribute = false) {
            constexpr unsigned Dim = Field::dim;
            using PositionType     = typename Field::Mesh_t::value_type;
            using view_type        = typename Field::view_type;
            using execution_space  = typename view_type::execution_space;
            using value_type       = typename view_type::value_type;

            // Ensure halos are up-to-date before reading
            f.fillHalo();

            const view_type view = f.getView();

            using mesh_type       = typename Field::Mesh_t;
            const mesh_type& mesh = f.get_mesh();

            using vector_type = typename mesh_type::vector_type;

            const vector_type& dx     = mesh.getMeshSpacing();
            const vector_type& origin = mesh.getOrigin();
            const vector_type invdx   = 1.0 / dx;

            const FieldLayout<Dim>& layout = f.getLayout();
            const NDIndex<Dim>& lDom       = layout.getLocalNDIndex();
            const int nghost               = f.getNghost();

            const int w  = kernel.width();
            const int hw = w / 2;

            assert(nghost >= hw);

            using policy_type = Kokkos::RangePolicy<execution_space>;
            Kokkos::parallel_for(
                "gatherHighOrder_no_wrap", policy_type(0, localNum),
                KOKKOS_LAMBDA(const size_t idx) {
                    const auto& view_ref = view;
                    auto& kernel_ref     = kernel;
                    const auto lDom_c    = lDom;
                    const int ngh        = nghost;

                    // Global scaled position in cell units
                    Vector<PositionType, Dim> sx;
                    for (unsigned d = 0; d < Dim; ++d) {
                        sx[d] = (posView(idx)[d] - origin[d]) * invdx[d];
                    }

                    // Starting global index (no explicit periodic wrap)
                    Vector<int64_t, Dim> idx0;
                    for (unsigned d = 0; d < Dim; ++d) {
                        const bool odd = (w & 1);
                        idx0[d]        = odd ? static_cast<int64_t>(Kokkos::round(sx[d])) - hw
                                             : static_cast<int64_t>(sx[d]) + 1 - hw;
                    }

                    value_type result{};  // zero init

                    if constexpr (Dim == 3) {
                        PositionType kernel_vals[3][20];
                        for (int i = 0; i < w; ++i) {
                            for (unsigned d = 0; d < 3; ++d) {
                                const PositionType diff =
                                    (sx[d] - static_cast<PositionType>(idx0[d] + i))
                                    * (PositionType(2.0) / w);
                                kernel_vals[d][i] = kernel_ref(diff);
                            }
                        }

                        for (int k = 0; k < w; ++k) {
                            const int64_t global_z = idx0[2] + k;
                            const size_t local_z =
                                static_cast<size_t>(global_z - lDom_c[2].first() + ngh);

                            for (int j = 0; j < w; ++j) {
                                const int64_t global_y = idx0[1] + j;
                                const size_t local_y =
                                    static_cast<size_t>(global_y - lDom_c[1].first() + ngh);

                                const PositionType weight_yz =
                                    kernel_vals[1][j] * kernel_vals[2][k];

                                for (int i = 0; i < w; ++i) {
                                    const int64_t global_x = idx0[0] + i;
                                    const size_t local_x =
                                        static_cast<size_t>(global_x - lDom_c[0].first() + ngh);

                                    const PositionType weight = kernel_vals[0][i] * weight_yz;
                                    result += view_ref(local_x, local_y, local_z) * weight;
                                }
                            }
                        }
                    } else if constexpr (Dim == 2) {
                        PositionType kernel_vals[2][20];
                        for (int i = 0; i < w; ++i) {
                            for (unsigned d = 0; d < 2; ++d) {
                                const PositionType diff =
                                    (sx[d] - static_cast<PositionType>(idx0[d] + i))
                                    * (PositionType(2.0) / w);
                                kernel_vals[d][i] = kernel_ref(diff);
                            }
                        }

                        for (int j = 0; j < w; ++j) {
                            const int64_t global_y = idx0[1] + j;
                            const size_t local_y =
                                static_cast<size_t>(global_y - lDom_c[1].first() + ngh);

                            for (int i = 0; i < w; ++i) {
                                const int64_t global_x = idx0[0] + i;
                                const size_t local_x =
                                    static_cast<size_t>(global_x - lDom_c[0].first() + ngh);

                                const PositionType weight = kernel_vals[0][i] * kernel_vals[1][j];
                                result += view_ref(local_x, local_y) * weight;
                            }
                        }
                    } else {  // 1D
                        for (int i = 0; i < w; ++i) {
                            const int64_t global_x = idx0[0] + i;
                            const size_t local_x =
                                static_cast<size_t>(global_x - lDom_c[0].first() + ngh);

                            const PositionType diff =
                                (sx[0] - static_cast<PositionType>(idx0[0] + i))
                                * (PositionType(2.0) / w);
                            const PositionType weight = kernel_ref(diff);
                            result += view_ref(local_x) * weight;
                        }
                    }

                    if (addToAttribute) {
                        attribView(idx) += result;
                    } else {
                        attribView(idx) = result;
                    }
                });
        }

    }  // namespace detail
}  // namespace ippl
