//
// Unit test FFT
//   Test FFT features
//
// Copyright (c) 2023, Paul Scherrer Institut, Villigen PSI, Switzerland
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

#include <Kokkos_MathematicalConstants.hpp>
#include <random>

#include "MultirankUtils.h"
#include "gtest/gtest.h"

// Restrict testing to 2 and 3 dimensions since this is what heFFTe supports
class FFTTest : public ::testing::Test, public MultirankUtils<2, 3> {
public:
    template <unsigned Dim>
    using mesh_type = ippl::UniformCartesian<double, Dim>;

    template <unsigned Dim>
    using centering_type = typename mesh_type<Dim>::DefaultCentering;

    template <unsigned Dim>
    using field_type_complex =
        ippl::Field<Kokkos::complex<double>, Dim, mesh_type<Dim>, centering_type<Dim>>;

    template <unsigned Dim>
    using field_type_real = ippl::Field<double, Dim, mesh_type<Dim>, centering_type<Dim>>;

    template <unsigned Dim>
    using layout_type = ippl::FieldLayout<Dim>;

    template <typename Transform, unsigned Dim>
    using FFT_type =
        ippl::FFT<Transform, std::conditional_t<std::is_same_v<Transform, ippl::CCTransform>,
                                                field_type_complex<Dim>, field_type_real<Dim>>>;

    FFTTest() {
        computeGridSizes(pt);
        const double pi = Kokkos::numbers::pi_v<double>;
        for (unsigned d = 0; d < MaxDim; d++) {
            len[d] = pt[d] * pi / 16;
        }
        setup(this);
    }

    template <unsigned Idx, unsigned Dim>
    void setupDim() {
        std::array<ippl::Index, Dim> domains;

        ippl::Vector<double, Dim> hx;
        ippl::Vector<double, Dim> origin;

        ippl::e_dim_tag domDec[Dim];  // Specifies SERIAL, PARALLEL dims
        for (unsigned d = 0; d < Dim; d++) {
            domDec[d]  = ippl::PARALLEL;
            domains[d] = ippl::Index(pt[d]);
            hx[d]      = len[d] / pt[d];
            origin[d]  = 0;
        }

        auto owned   = std::make_from_tuple<ippl::NDIndex<Dim>>(domains);
        auto& layout = std::get<Idx>(layouts) = layout_type<Dim>(owned, domDec);

        auto& mesh = std::get<Idx>(meshes) = mesh_type<Dim>(owned, hx, origin);

        std::get<Idx>(realFields) = std::make_shared<field_type_real<Dim>>(mesh, layout);
        std::get<Idx>(compFields) = std::make_shared<field_type_complex<Dim>>(mesh, layout);
    }

    /*!
     * Gets the parameters used for trigonometric FFT transforms
     * (sine and cosine)
     */
    ippl::ParameterList getTrigParams() const {
        ippl::ParameterList fftParams;

        fftParams.add("use_heffte_defaults", false);
        fftParams.add("use_pencils", true);
        fftParams.add("use_reorder", false);
        fftParams.add("use_gpu_aware", true);
        fftParams.add("comm", ippl::p2p_pl);

        return fftParams;
    }

    /*!
     * Fill a real-valued field with random values
     * @tparam Dim field rank
     * @param nghost number of ghost cells
     * @param mirror the field view's host mirror
     */
    template <unsigned Dim>
    void randomizeRealField(int nghost, typename field_type_real<Dim>::HostMirror& mirror) {
        std::mt19937_64 eng(42 + ippl::Comm->rank());
        std::uniform_real_distribution<double> unif(0, 1);

        nestedViewLoop(mirror, nghost, [&]<typename... Idx>(const Idx... args) {
            mirror(args...) = unif(eng);
        });
    }

    /*!
     * Fill a complex-valued field with random values
     * @tparam Dim field rank
     * @param nghost number of ghost cells
     * @param mirror the field view's host mirror
     */
    template <unsigned Dim>
    void randomizeComplexField(int nghost, typename field_type_complex<Dim>::HostMirror& mirror) {
        std::mt19937_64 engReal(42 + ippl::Comm->rank());
        std::uniform_real_distribution<double> unifReal(0, 1);

        std::mt19937_64 engImag(43 + ippl::Comm->rank());
        std::uniform_real_distribution<double> unifImag(0, 1);

        nestedViewLoop(mirror, nghost, [&]<typename... Idx>(const Idx... args) {
            mirror(args...).real() = unifReal(engReal);
            mirror(args...).imag() = unifImag(engImag);
        });
    }

    /*!
     * Verify the contents of a computation
     * @tparam Dim view rank
     * @tparam MirrorA the type of the computed view
     * @tparam MirrorB the type of the expected view
     * @param nghost number of ghost cells
     * @param computed the computed result
     * @param expected the expected result
     */
    template <unsigned Dim, typename MirrorA, typename MirrorB>
    void verifyResult(int nghost, const MirrorA& computed, const MirrorB& expected) {
        double max_error_local = 0.0;
        nestedViewLoop(computed, nghost, [&]<typename... Idx>(const Idx... args) {
            double error = std::fabs(expected(args...) - computed(args...));

            if (error > max_error_local) {
                max_error_local = error;
            }

            ASSERT_NEAR(error, 0, 1e-13);
        });

        double max_error = 0.0;
        MPI_Reduce(&max_error_local, &max_error, 1, MPI_DOUBLE, MPI_MAX, 0, ippl::Comm->getCommunicator());
        ASSERT_NEAR(max_error, 0, 1e-13);
    }

    /*!
     * Tests the trigonometric FFT transforms
     * @tparam Transform either SineTransform or CosTransform
     * @tparam Dim the field rank
     * @param field a pointer to the field
     * @param layout a pointer to the layout
     */
    template <typename Transform, unsigned Dim>
    void testTrig(std::shared_ptr<field_type_real<Dim>>& field, const layout_type<Dim>& layout) {
        auto fft        = std::make_unique<FFT_type<Transform, Dim>>(layout, getTrigParams());
        auto& view      = field->getView();
        auto field_host = field->getHostMirror();

        const int nghost = field->getNghost();
        randomizeRealField<Dim>(nghost, field_host);

        Kokkos::deep_copy(view, field_host);

        // Forward transform
        fft->transform(1, *field);
        // Reverse transform
        fft->transform(-1, *field);

        auto field_result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);

        verifyResult<Dim>(nghost, field_result, field_host);
    }

    Collection<mesh_type> meshes;
    Collection<layout_type> layouts;
    PtrCollection<std::shared_ptr, field_type_real> realFields;
    PtrCollection<std::shared_ptr, field_type_complex> compFields;

    size_t pt[MaxDim];
    double len[MaxDim];
};

TEST_F(FFTTest, Cos) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type_real<Dim>>& field,
                                   const layout_type<Dim>& layout) {
        testTrig<ippl::CosTransform, Dim>(field, layout);
    };

    apply(check, realFields, layouts);
}

TEST_F(FFTTest, Sin) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type_real<Dim>>& field,
                                   const layout_type<Dim>& layout) {
        testTrig<ippl::SineTransform, Dim>(field, layout);
    };

    apply(check, realFields, layouts);
}

TEST_F(FFTTest, RC) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type_real<Dim>>& field,
                                   const layout_type<Dim>& layout, const mesh_type<Dim>& mesh) {
        ippl::ParameterList fftParams;
        fftParams.add("use_heffte_defaults", true);
        fftParams.add("r2c_direction", 0);

        ippl::NDIndex<Dim> ownedOutput;
        ippl::e_dim_tag allParallel[Dim];
        for (unsigned d = 0; d < Dim; d++) {
            allParallel[d] = ippl::PARALLEL;
            if ((int)d == fftParams.get<int>("r2c_direction")) {
                ownedOutput[d] = ippl::Index(pt[d] / 2 + 1);
            } else {
                ownedOutput[d] = ippl::Index(pt[d]);
            }
        }

        layout_type<Dim> layoutOutput(ownedOutput, allParallel);

        mesh_type<Dim> meshOutput(ownedOutput, mesh.getMeshSpacing(), mesh.getOrigin());
        field_type_complex<Dim> fieldOutput(meshOutput, layoutOutput);

        auto fft =
            std::make_unique<FFT_type<ippl::RCTransform, Dim>>(layout, layoutOutput, fftParams);

        auto& view      = field->getView();
        auto input_host = field->getHostMirror();

        const int nghost = field->getNghost();
        randomizeRealField<Dim>(nghost, input_host);

        Kokkos::deep_copy(view, input_host);

        fft->transform(1, *field, fieldOutput);
        fft->transform(-1, *field, fieldOutput);

        auto field_result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);

        verifyResult<Dim>(nghost, field_result, input_host);
    };

    apply(check, realFields, layouts, meshes);
}

TEST_F(FFTTest, CC) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type_complex<Dim>>& field,
                                   const layout_type<Dim>& layout) {
        ippl::ParameterList fftParams;

        fftParams.add("use_heffte_defaults", true);

        auto fft = std::make_unique<FFT_type<ippl::CCTransform, Dim>>(layout, fftParams);

        auto& view      = field->getView();
        auto field_host = field->getHostMirror();

        const int nghost = field->getNghost();
        randomizeComplexField<Dim>(nghost, field_host);

        Kokkos::deep_copy(view, field_host);

        fft->transform(1, *field);
        fft->transform(-1, *field);

        auto field_result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);

        Kokkos::complex<double> max_error_local(0, 0);
        nestedViewLoop(field_host, nghost, [&]<typename... Idx>(const Idx... args) {
            Kokkos::complex<double> error(
                std::fabs(field_host(args...).real() - field_result(args...).real()),
                std::fabs(field_host(args...).imag() - field_result(args...).imag()));

            if (error.real() > max_error_local.real()) {
                max_error_local.real() = error.real();
            }

            if (error.imag() > max_error_local.imag()) {
                max_error_local.imag() = error.imag();
            }

            ASSERT_NEAR(error.real(), 0, 1e-13);
            ASSERT_NEAR(error.imag(), 0, 1e-13);
        });

        Kokkos::complex<double> max_error(0, 0);
        MPI_Allreduce(&max_error_local, &max_error, 1, MPI_C_DOUBLE_COMPLEX, MPI_SUM,
                      ippl::Comm->getCommunicator());
        ASSERT_NEAR(max_error.real(), 0, 1e-13);
        ASSERT_NEAR(max_error.imag(), 0, 1e-13);
    };

    apply(check, compFields, layouts);
}

int main(int argc, char* argv[]) {
    int success = 1;
    ippl::initialize(argc, argv);
    {
        ::testing::InitGoogleTest(&argc, argv);
        success = RUN_ALL_TESTS();
    }
    ippl::finalize();
    return success;
}
