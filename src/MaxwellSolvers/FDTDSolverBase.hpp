//
// Class FDTDSolverBase
//  Base class for solvers for Maxwell's equations using the FDTD method
//

namespace ippl {

    /**
     * @brief Constructor for the FDTDSolverBase class.
     *
     * Initializes the solver by setting the source and electromagnetic fields.
     *
     * @param source Reference to the source field.
     * @param E Reference to the electric field.
     * @param B Reference to the magnetic field.
     */
    template <typename EMField, typename SourceField, fdtd_bc boundary_conditions>
    FDTDSolverBase<EMField, SourceField, boundary_conditions>::FDTDSolverBase(SourceField& source,
                                                                              EMField& E,
                                                                              EMField& B) {
        Maxwell<EMField, SourceField>::setSources(source);
        Maxwell<EMField, SourceField>::setEMFields(E, B);
    }

    /**
     * @brief Solves the FDTD equations.
     *
     * Advances the simulation by one time step, shifts the time for the fields, and evaluates the
     * electric and magnetic fields at the new time.
     */
    template <typename EMField, typename SourceField, fdtd_bc boundary_conditions>
    void FDTDSolverBase<EMField, SourceField, boundary_conditions>::solve() {
        step();
        timeShift();
        evaluate_EB();
    }

    /**
     * @brief Sets periodic boundary conditions.
     *
     * Configures the solver to use periodic boundary conditions by setting the appropriate boundary
     * conditions for the fields.
     */
    template <typename EMField, typename SourceField, fdtd_bc boundary_conditions>
    void
    FDTDSolverBase<EMField, SourceField, boundary_conditions>::setPeriodicBoundaryConditions() {
        typename SourceField::BConds_t vector_bcs;
        auto bcsetter_single = [&vector_bcs]<size_t Idx>(const std::index_sequence<Idx>&) {
            vector_bcs[Idx] = std::make_shared<ippl::PeriodicFace<SourceField>>(Idx);
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

    /**
     * @brief Shifts the saved fields in time.
     *
     * Copies the current field values to the previous time step field and the next time step field
     * values to the current field.
     */
    template <typename EMField, typename SourceField, fdtd_bc boundary_conditions>
    void FDTDSolverBase<EMField, SourceField, boundary_conditions>::timeShift() {
        // Look into this, maybe cyclic swap is better
        Kokkos::deep_copy(this->A_nm1.getView(), this->A_n.getView());
        Kokkos::deep_copy(this->A_n.getView(), this->A_np1.getView());
    }

    /**
     * @brief Applies the boundary conditions.
     *
     * Applies the specified boundary conditions (periodic or absorbing) to the fields.
     */
    template <typename EMField, typename SourceField, fdtd_bc boundary_conditions>
    void FDTDSolverBase<EMField, SourceField, boundary_conditions>::applyBCs() {
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

    /**
     * @brief Evaluates the electric and magnetic fields.
     *
     * Computes the electric and magnetic fields based on the current, previous, and next time step
     * field values, as well as the source field.
     */
    template <typename EMField, typename SourceField, fdtd_bc boundary_conditions>
    void FDTDSolverBase<EMField, SourceField, boundary_conditions>::evaluate_EB() {
        *(Maxwell<EMField, SourceField>::En_mp)   = typename EMField::value_type(0);
        *(Maxwell<EMField, SourceField>::Bn_mp)   = typename EMField::value_type(0);
        ippl::Vector<scalar, 3> inverse_2_spacing = ippl::Vector<scalar, 3>(0.5) / hr_m;
        const scalar idt                          = scalar(1.0) / dt;
        auto A_np1 = this->A_np1.getView(), A_n = this->A_n.getView(),
             A_nm1  = this->A_nm1.getView();
        auto source = Maxwell<EMField, SourceField>::JN_mp->getView();
        auto Eview  = Maxwell<EMField, SourceField>::En_mp->getView();
        auto Bview  = Maxwell<EMField, SourceField>::Bn_mp->getView();

        // Calculate the electric and magnetic fields
        // Curl and grad of ippl are not used here, because we have a 4-component vector and we need
        // it for the last three components
        Kokkos::parallel_for(
            this->A_n.getFieldRangePolicy(), KOKKOS_LAMBDA(size_t i, size_t j, size_t k) {
                ippl::Vector<scalar, 3> dAdt;
                dAdt[0] = (A_np1(i, j, k)[1] - A_n(i, j, k)[1]) * idt;
                dAdt[1] = (A_np1(i, j, k)[2] - A_n(i, j, k)[2]) * idt;
                dAdt[2] = (A_np1(i, j, k)[3] - A_n(i, j, k)[3]) * idt;

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

}  // namespace ippl