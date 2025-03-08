#ifndef IPPL_FDTD_H
#define IPPL_FDTD_H
#include <cstddef>
using std::size_t;
#include "Types/Vector.h"

#include "FieldLayout/FieldLayout.h"
#include "MaxwellSolvers/AbsorbingBC.h"
#include "MaxwellSolvers/Maxwell.h"
#include "Meshes/UniformCartesian.h"
#include "Particle/ParticleBase.h"

constexpr double sq(double x) {
    return x * x;
}
constexpr float sq(float x) {
    return x * x;
}
#define assert_isreal(X) assert(!Kokkos::isnan(X) && !Kokkos::isinf(X))
namespace ippl {
    enum fdtd_bc {
        periodic,
        absorbing
    };

    template <typename EMField, typename FourField, fdtd_bc boundary_conditions = periodic>
    class StandardFDTDSolver : Maxwell<EMField, FourField> {
    public:
        constexpr static unsigned Dim = EMField::dim;
        using scalar                  = typename EMField::value_type::value_type;
        using Vector_t                = Vector<typename EMField::value_type::value_type, Dim>;
        using FourVector_t            = typename FourField::value_type;
        StandardFDTDSolver(FourField& source, EMField& E, EMField& B) {
            Maxwell<EMField, FourField>::setSource(source);
            Maxwell<EMField, FourField>::setEMFields(E, B);
            initialize();
        }
        virtual void solve() override {
            step();
            timeShift();
            evaluate_EB();
        }
        void setPeriodicBoundaryConditions() {
            periodic_bc = true;
            typename FourField::BConds_t vector_bcs;
            auto bcsetter_single = [&vector_bcs]<size_t Idx>(const std::index_sequence<Idx>&) {
                vector_bcs[Idx] = std::make_shared<ippl::PeriodicFace<FourField>>(Idx);
                return 0;
            };
            auto bcsetter = [bcsetter_single]<size_t... Idx>(const std::index_sequence<Idx...>&) {
                int x = (bcsetter_single(std::index_sequence<Idx>{}) ^ ...);
                (void)x;
            };
            bcsetter(std::make_index_sequence<Dim * 2>{});
            A_n.setFieldBC(vector_bcs);
            A_np1.setFieldBC(vector_bcs);
            A_nm1.setFieldBC(vector_bcs);
        }

        FourField A_n;
        FourField A_np1;
        FourField A_nm1;
        scalar dt;

    private:
        void timeShift() {
            // Look into this, maybe cyclic swap is better
            Kokkos::deep_copy(this->A_nm1.getView(), this->A_n.getView());
            Kokkos::deep_copy(this->A_n.getView(), this->A_np1.getView());
        }
        void applyBCs() {
            if constexpr (boundary_conditions == periodic) {
                A_n.getFieldBC().apply(A_n);
                A_nm1.getFieldBC().apply(A_nm1);
                A_np1.getFieldBC().apply(A_np1);
            } else {
                Vector<uint32_t, Dim> true_nr = nr_m;
                true_nr += (A_n.getNghost() * 2);
                second_order_mur_boundary_conditions bcs{};
                bcs.apply(A_n, A_nm1, A_np1, this->dt, true_nr, layout_mp->getLocalNDIndex());
            }
        }

    public:
        void step() {
            const auto& ldom       = layout_mp->getLocalNDIndex();
            const int nghost       = A_n.getNghost();
            const auto aview       = A_n.getView();
            const auto anp1view    = A_np1.getView();
            const auto anm1view    = A_nm1.getView();
            const auto source_view = Maxwell<EMField, FourField>::source_mp->getView();

            const scalar a1 =
                scalar(2) * (scalar(1) - sq(dt / hr_m[0]) - sq(dt / hr_m[1]) - sq(dt / hr_m[2]));
            const scalar a2               = sq(dt / hr_m[0]);
            const scalar a4               = sq(dt / hr_m[1]);
            const scalar a6               = sq(dt / hr_m[2]);
            const scalar a8               = sq(dt);
            Vector<uint32_t, Dim> true_nr = nr_m;
            true_nr += (nghost * 2);
            constexpr uint32_t one_if_absorbing_otherwise_0 =
                boundary_conditions == absorbing ? 1 : 0;
            Kokkos::parallel_for(
                "Four potential update", ippl::getRangePolicy(aview, nghost),
                KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                    // global indices
                    const uint32_t ig = i + ldom.first()[0];
                    const uint32_t jg = j + ldom.first()[1];
                    const uint32_t kg = k + ldom.first()[2];
                    uint32_t val =
                        uint32_t(ig == one_if_absorbing_otherwise_0)
                        + (uint32_t(jg == one_if_absorbing_otherwise_0) << 1)
                        + (uint32_t(kg == one_if_absorbing_otherwise_0) << 2)
                        + (uint32_t(ig == true_nr[0] - one_if_absorbing_otherwise_0 - 1) << 3)
                        + (uint32_t(jg == true_nr[1] - one_if_absorbing_otherwise_0 - 1) << 4)
                        + (uint32_t(kg == true_nr[2] - one_if_absorbing_otherwise_0 - 1) << 5);

                    if (val == 0) {
                        FourVector_t interior = -anm1view(i, j, k) + a1 * aview(i, j, k)
                                                + a2 * (aview(i + 1, j, k) + aview(i - 1, j, k))
                                                + a4 * (aview(i, j + 1, k) + aview(i, j - 1, k))
                                                + a6 * (aview(i, j, k + 1) + aview(i, j, k - 1))
                                                + a8 * (-source_view(i, j, k));
                        anp1view(i, j, k) = interior;
                    } else {
                        // std::cout << i << ", " << j << ", " << k << "\n";
                    }
                });
            Kokkos::fence();
            applyBCs();
            A_np1.fillHalo();
        }
        void evaluate_EB() {
            ippl::Vector<scalar, 3> inverse_2_spacing = ippl::Vector<scalar, 3>(0.5) / hr_m;
            const scalar idt                          = scalar(1.0) / dt;
            auto A_np1 = this->A_np1.getView(), A_n = this->A_n.getView(),
                 A_nm1  = this->A_nm1.getView();
            auto source = Maxwell<EMField, FourField>::source_mp->getView();
            auto Eview  = Maxwell<EMField, FourField>::En_mp->getView();
            auto Bview  = Maxwell<EMField, FourField>::Bn_mp->getView();

            Kokkos::parallel_for(
                this->A_n.getFieldRangePolicy(), KOKKOS_LAMBDA(size_t i, size_t j, size_t k) {
                    ippl::Vector<scalar, 3> dAdt =
                        (A_n(i, j, k).template tail<3>() - A_nm1(i, j, k).template tail<3>()) * idt;
                    ippl::Vector<scalar, 4> dAdx =
                        (A_n(i + 1, j, k) - A_n(i - 1, j, k)) * inverse_2_spacing[0];
                    ippl::Vector<scalar, 4> dAdy =
                        (A_n(i, j + 1, k) - A_n(i, j - 1, k)) * inverse_2_spacing[1];
                    ippl::Vector<scalar, 4> dAdz =
                        (A_n(i, j, k + 1) - A_n(i, j, k - 1)) * inverse_2_spacing[2];

                    ippl::Vector<scalar, 3> grad_phi{dAdx[0], dAdy[0], dAdz[0]};
                    ippl::Vector<scalar, 3> curlA{
                        dAdy[3] - dAdz[2],
                        dAdz[1] - dAdx[3],
                        dAdx[2] - dAdy[1],
                    };
                    Eview(i, j, k) = -dAdt - grad_phi;
                    Bview(i, j, k) = curlA;
                });
            Kokkos::fence();
        }

        bool periodic_bc;

        typename FourField::Mesh_t* mesh_mp;
        FieldLayout<Dim>* layout_mp;
        NDIndex<Dim> domain_m;
        Vector_t hr_m;

        Vector<int, Dim> nr_m;

        void initialize() {
            // get layout and mesh
            layout_mp = &(this->source_mp->getLayout());
            mesh_mp   = &(this->source_mp->get_mesh());

            // get mesh spacing, domain, and mesh size
            hr_m = mesh_mp->getMeshSpacing();
            dt   = hr_m[0] / 2;
            for (unsigned int i = 0; i < Dim; ++i) {
                dt = std::min(dt, hr_m[i] / 2);
            }
            domain_m = layout_mp->getDomain();
            for (unsigned int i = 0; i < Dim; ++i)
                nr_m[i] = domain_m[i].length();

            // initialize fields
            A_nm1.initialize(*mesh_mp, *layout_mp);
            A_n.initialize(*mesh_mp, *layout_mp);
            A_np1.initialize(*mesh_mp, *layout_mp);

            A_nm1 = 0.0;
            A_n   = 0.0;
            A_np1 = 0.0;
        };
    };
    /**
     * @brief Nonstandard Finite-Difference Time-Domain
     *
     * @tparam EMField
     * @tparam FourField
     */
    template <typename EMField, typename FourField, fdtd_bc boundary_conditions = periodic>
    class NonStandardFDTDSolver : Maxwell<EMField, FourField> {
    public:
        constexpr static unsigned Dim = EMField::dim;
        using scalar                  = typename EMField::value_type::value_type;
        using Vector_t                = Vector<typename EMField::value_type::value_type, Dim>;
        using FourVector_t            = typename FourField::value_type;
        NonStandardFDTDSolver(FourField& source, EMField& E, EMField& B) {
            auto hx = source.get_mesh().getMeshSpacing();
            if ((hx[2] / hx[0]) * (hx[2] / hx[0]) + (hx[2] / hx[1]) * (hx[2] / hx[1]) >= 1) {
                std::cerr << "Dispersion-free CFL condition not satisfiable\n";
                std::abort();
            }
            Maxwell<EMField, FourField>::setSource(source);
            Maxwell<EMField, FourField>::setEMFields(E, B);
            initialize();
        }
        virtual void solve() override {
            step();
            timeShift();
            evaluate_EB();
        }
        FourField A_n;
        FourField A_np1;
        FourField A_nm1;
        void setPeriodicBoundaryConditions() {
            periodic_bc = true;
            typename FourField::BConds_t vector_bcs;
            auto bcsetter_single = [&vector_bcs]<size_t Idx>(const std::index_sequence<Idx>&) {
                vector_bcs[Idx] = std::make_shared<ippl::PeriodicFace<FourField>>(Idx);
                return 0;
            };
            auto bcsetter = [bcsetter_single]<size_t... Idx>(const std::index_sequence<Idx...>&) {
                int x = (bcsetter_single(std::index_sequence<Idx>{}) ^ ...);
                (void)x;
            };
            bcsetter(std::make_index_sequence<Dim * 2>{});
            A_n.setFieldBC(vector_bcs);
            A_np1.setFieldBC(vector_bcs);
            A_nm1.setFieldBC(vector_bcs);
        }

    private:
        void timeShift() {
            // Look into this, maybe cyclic swap is better
            Kokkos::deep_copy(this->A_nm1.getView(), this->A_n.getView());
            Kokkos::deep_copy(this->A_n.getView(), this->A_np1.getView());
        }

        void applyBCs() {
            if constexpr (boundary_conditions == periodic) {
                A_n.getFieldBC().apply(A_n);
                A_nm1.getFieldBC().apply(A_nm1);
                A_np1.getFieldBC().apply(A_np1);
            } else {
                Vector<uint32_t, Dim> true_nr = nr_m;
                true_nr += (A_n.getNghost() * 2);
                second_order_mur_boundary_conditions bcs{};
                bcs.apply(A_n, A_nm1, A_np1, this->dt, true_nr, layout_mp->getLocalNDIndex());
            }
        }

    public:
        template <typename scalar>
        struct nondispersive {
            scalar a1;
            scalar a2;
            scalar a4;
            scalar a6;
            scalar a8;
        };
        void step() {
            const auto& ldom       = layout_mp->getLocalNDIndex();
            const int nghost       = A_n.getNghost();
            const auto aview       = A_n.getView();
            const auto anp1view    = A_np1.getView();
            const auto anm1view    = A_nm1.getView();
            const auto source_view = Maxwell<EMField, FourField>::source_mp->getView();

            const scalar calA = 0.25 * (1 + 0.02 / (sq(hr_m[2] / hr_m[0]) + sq(hr_m[2] / hr_m[1])));
            nondispersive<scalar> ndisp{
                .a1 = 2
                      * (1 - (1 - 2 * calA) * sq(dt / hr_m[0]) - (1 - 2 * calA) * sq(dt / hr_m[1])
                         - sq(dt / hr_m[2])),
                .a2 = sq(dt / hr_m[0]),
                .a4 = sq(dt / hr_m[1]),
                .a6 = sq(dt / hr_m[2]) - 2 * calA * sq(dt / hr_m[0]) - 2 * calA * sq(dt / hr_m[1]),
                .a8 = sq(dt)};
            Vector<uint32_t, Dim> true_nr = nr_m;
            true_nr += (nghost * 2);
            constexpr uint32_t one_if_absorbing_otherwise_0 =
                boundary_conditions == absorbing ? 1 : 0;
            Kokkos::parallel_for(
                ippl::getRangePolicy(aview, nghost), KOKKOS_LAMBDA(size_t i, size_t j, size_t k) {
                    uint32_t ig = i + ldom.first()[0];
                    uint32_t jg = j + ldom.first()[1];
                    uint32_t kg = k + ldom.first()[2];
                    uint32_t val =
                        uint32_t(ig == one_if_absorbing_otherwise_0)
                        + (uint32_t(jg == one_if_absorbing_otherwise_0) << 1)
                        + (uint32_t(kg == one_if_absorbing_otherwise_0) << 2)
                        + (uint32_t(ig == true_nr[0] - one_if_absorbing_otherwise_0 - 1) << 3)
                        + (uint32_t(jg == true_nr[1] - one_if_absorbing_otherwise_0 - 1) << 4)
                        + (uint32_t(kg == true_nr[2] - one_if_absorbing_otherwise_0 - 1) << 5);

                    if (!val) {
                        anp1view(i, j, k) = -anm1view(i, j, k) + ndisp.a1 * aview(i, j, k)
                                            + ndisp.a2
                                                  * (calA * aview(i + 1, j, k - 1)
                                                     + (1 - 2 * calA) * aview(i + 1, j, k)
                                                     + calA * aview(i + 1, j, k + 1))
                                            + ndisp.a2
                                                  * (calA * aview(i - 1, j, k - 1)
                                                     + (1 - 2 * calA) * aview(i - 1, j, k)
                                                     + calA * aview(i - 1, j, k + 1))
                                            + ndisp.a4
                                                  * (calA * aview(i, j + 1, k - 1)
                                                     + (1 - 2 * calA) * aview(i, j + 1, k)
                                                     + calA * aview(i, j + 1, k + 1))
                                            + ndisp.a4
                                                  * (calA * aview(i, j - 1, k - 1)
                                                     + (1 - 2 * calA) * aview(i, j - 1, k)
                                                     + calA * aview(i, j - 1, k + 1))
                                            + ndisp.a6 * aview(i, j, k + 1)
                                            + ndisp.a6 * aview(i, j, k - 1)
                                            + ndisp.a8 * source_view(i, j, k);
                    }
                });
            Kokkos::fence();
            A_np1.fillHalo();
            applyBCs();
        }
        void evaluate_EB() {
            *(Maxwell<EMField, FourField>::En_mp)     = typename EMField::value_type(0);
            *(Maxwell<EMField, FourField>::Bn_mp)     = typename EMField::value_type(0);
            ippl::Vector<scalar, 3> inverse_2_spacing = ippl::Vector<scalar, 3>(0.5) / hr_m;
            const scalar idt                          = scalar(1.0) / dt;
            auto A_np1 = this->A_np1.getView(), A_n = this->A_n.getView(),
                 A_nm1  = this->A_nm1.getView();
            auto source = Maxwell<EMField, FourField>::source_mp->getView();
            auto Eview  = Maxwell<EMField, FourField>::En_mp->getView();
            auto Bview  = Maxwell<EMField, FourField>::Bn_mp->getView();

            Kokkos::parallel_for(
                this->A_n.getFieldRangePolicy(), KOKKOS_LAMBDA(size_t i, size_t j, size_t k) {
                    ippl::Vector<scalar, 3> dAdt =
                        (A_n(i, j, k).template tail<3>() - A_nm1(i, j, k).template tail<3>()) * idt;
                    ippl::Vector<scalar, 4> dAdx =
                        (A_n(i + 1, j, k) - A_n(i - 1, j, k)) * inverse_2_spacing[0];
                    ippl::Vector<scalar, 4> dAdy =
                        (A_n(i, j + 1, k) - A_n(i, j - 1, k)) * inverse_2_spacing[1];
                    ippl::Vector<scalar, 4> dAdz =
                        (A_n(i, j, k + 1) - A_n(i, j, k - 1)) * inverse_2_spacing[2];

                    ippl::Vector<scalar, 3> grad_phi{dAdx[0], dAdy[0], dAdz[0]};
                    ippl::Vector<scalar, 3> curlA{
                        dAdy[3] - dAdz[2],
                        dAdz[1] - dAdx[3],
                        dAdx[2] - dAdy[1],
                    };
                    Eview(i, j, k) = -dAdt - grad_phi;
                    Bview(i, j, k) = curlA;
                });
            Kokkos::fence();
        }

        typename FourField::Mesh_t* mesh_mp;
        FieldLayout<Dim>* layout_mp;
        NDIndex<Dim> domain_m;
        Vector_t hr_m;
        scalar dt;
        Vector<int, Dim> nr_m;
        bool periodic_bc;

        void initialize() {
            // get layout and mesh
            layout_mp = &(this->source_mp->getLayout());
            mesh_mp   = &(this->source_mp->get_mesh());

            // get mesh spacing, domain, and mesh size
            hr_m     = mesh_mp->getMeshSpacing();
            dt       = hr_m[2];
            domain_m = layout_mp->getDomain();
            for (unsigned int i = 0; i < Dim; ++i)
                nr_m[i] = domain_m[i].length();

            // initialize fields
            A_nm1.initialize(*mesh_mp, *layout_mp);
            A_n.initialize(*mesh_mp, *layout_mp);
            A_np1.initialize(*mesh_mp, *layout_mp);

            A_nm1 = 0.0;
            A_n   = 0.0;
            A_np1 = 0.0;
        };
    };
}  // namespace ippl

#endif