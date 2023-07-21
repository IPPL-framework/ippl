//
// Class FFTPoissonSolver
//   FFT-based Poisson Solver for open boundaries.
//   Solves laplace(phi) = -rho, and E = -grad(phi).
//
// Copyright (c) 2023, Sonali Mayani,
// Paul Scherrer Institut, Villigen PSI, Switzerland
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

// Communication specific functions (pack and unpack).
template <typename Tb, typename View, unsigned int Dim>
void pack(const ippl::NDIndex<Dim> intersect, View& view,
          ippl::detail::FieldBufferData<Tb>& fd, int nghost, const ippl::NDIndex<Dim> ldom,
          ippl::Communicate::size_type& nsends) {
    Kokkos::View<Tb*>& buffer = fd.buffer;

    size_t size = intersect.size();
    nsends      = size;
    if (buffer.size() < size) {
        const int overalloc = ippl::Comm->getDefaultOverallocation();
        Kokkos::realloc(buffer, size * overalloc);
    }
    using index_type = typename ippl::RangePolicy<Dim>::index_type;
    Kokkos::Array<index_type, Dim> first, last;
    
    for (unsigned d = 0; d < Dim; ++d) {
        first[d] = intersect[d].first() + nghost - ldom[d].first();
        last[d] = intersect[d].last() + nghost - ldom[d].first() + 1;
    }

    using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
    ippl::parallel_for(
        "pack()", ippl::createRangePolicy(first, last),
        KOKKOS_LAMBDA(const index_array_type& args) { 
            ippl::Vector<size_t, Dim> iVec = args;
            ippl::Vector<size_t, Dim> igVec;
            for (unsigned d = 0; d < Dim; ++d) {
                igVec[d] = iVec[d] - first[d];
            }


            int l = igVec[0];
            for (unsigned d = 1; d < Dim; ++d) {
                int factor = 1;
                for (unsigned d1 = 0; d1 < d; ++d1) {
                    factor *= intersect[d1].length();
                }
                l += igVec[d] * factor;
            }

            Kokkos::complex<Tb> val = apply(view, args);
            buffer(l) = Kokkos::real(val);
        });
    Kokkos::fence();
}

template <bool isVec, typename Tb, typename View, unsigned int Dim>
void unpack_impl(const ippl::NDIndex<Dim> intersect, const View& view,
                 ippl::detail::FieldBufferData<Tb>& fd, int nghost, const ippl::NDIndex<Dim> ldom,
                std::vector<bool> coordBool, size_t dim = 0) {
    Kokkos::View<Tb*>& buffer = fd.buffer;

    using index_type = typename ippl::RangePolicy<Dim>::index_type;
    Kokkos::Array<index_type, Dim> first, last;

    for (unsigned d = 0; d < Dim; ++d) {
        first[d] = intersect[d].first() + nghost - ldom[d].first();
        last[d] = intersect[d].last() + nghost - ldom[d].first() + 1;
    }

    using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
    ippl::parallel_for(
        "pack()", ippl::createRangePolicy(first, last),
        KOKKOS_LAMBDA(const index_array_type& args) { 
            ippl::Vector<size_t, Dim> iVec = args;
            ippl::Vector<size_t, Dim> igVec;
            for (unsigned d = 0; d < Dim; ++d) {
                igVec[d] = iVec[d] - first[d];
            
                igVec[d] = coordBool[d] * (intersect[d].length() - 2 * igVec[d] - 1) + igVec[d];
            }
            int l = igVec[0];
            for (unsigned d = 1; d < Dim; ++d) {
                int factor = 1;
                for (unsigned d1 = 0; d1 < d; ++d1) {
                    factor *= intersect[d1].length();
                }
                l += igVec[d] * factor;
            }

            ippl::detail::ViewAccess<isVec, View, Dim>::get(view, dim, args) = buffer(l);
        });
    Kokkos::fence();
}

template <typename Tb, typename View, unsigned int Dim>
void unpack(const ippl::NDIndex<Dim> intersect, const View& view,
            ippl::detail::FieldBufferData<Tb>& fd, int nghost, const ippl::NDIndex<Dim> ldom,
            std::vector<bool> coordBool, size_t dim = 0) {
    unpack_impl<false, Tb, View>(intersect, view, fd, nghost, ldom, coordBool, dim);
}

template <typename Tb, typename View, unsigned int Dim>
void unpack(const ippl::NDIndex<Dim> intersect, View& view,
            ippl::detail::FieldBufferData<Tb>& fd, int nghost,
            const ippl::NDIndex<Dim> ldom, std::vector<bool> coordBool, size_t dim) {
    unpack_impl<true, Tb, View>(intersect, view, fd, nghost, ldom, coordBool, dim);
}

namespace ippl {

    /////////////////////////////////////////////////////////////////////////
    // constructor and destructor
    template <typename FieldLHS, typename FieldRHS>
    FFTPoissonSolver<FieldLHS, FieldRHS>::FFTPoissonSolver()
        : Base()
        , mesh_mp(nullptr)
        , layout_mp(nullptr)
        , mesh2_m(nullptr)
        , layout2_m(nullptr)
        , meshComplex_m(nullptr)
        , layoutComplex_m(nullptr)
        , mesh4_m(nullptr)
        , layout4_m(nullptr)
        , isGradFD_m(false) {
        setDefaultParameters();
    }

    template <typename FieldLHS, typename FieldRHS>
    FFTPoissonSolver<FieldLHS, FieldRHS>::FFTPoissonSolver(rhs_type& rhs, ParameterList& params)
        : mesh_mp(nullptr)
        , layout_mp(nullptr)
        , mesh2_m(nullptr)
        , layout2_m(nullptr)
        , meshComplex_m(nullptr)
        , layoutComplex_m(nullptr)
        , mesh4_m(nullptr)
        , layout4_m(nullptr)
        , isGradFD_m(false) {
        using T = typename FieldLHS::value_type::value_type;
        static_assert(std::is_floating_point<T>::value, "Not a floating point type");

        setDefaultParameters();
        this->params_m.merge(params);
        this->params_m.update("output_type", Base::SOL);

        this->setRhs(rhs);
    }

    template <typename FieldLHS, typename FieldRHS>
    FFTPoissonSolver<FieldLHS, FieldRHS>::FFTPoissonSolver(lhs_type& lhs, rhs_type& rhs,
                                                           ParameterList& params)
        : mesh_mp(nullptr)
        , layout_mp(nullptr)
        , mesh2_m(nullptr)
        , layout2_m(nullptr)
        , meshComplex_m(nullptr)
        , layoutComplex_m(nullptr)
        , mesh4_m(nullptr)
        , layout4_m(nullptr)
        , isGradFD_m(false) {
        using T = typename FieldLHS::value_type::value_type;
        static_assert(std::is_floating_point<T>::value, "Not a floating point type");

        setDefaultParameters();
        this->params_m.merge(params);

        this->setLhs(lhs);
        this->setRhs(rhs);
    }

    /////////////////////////////////////////////////////////////////////////
    // override setRhs to call class-specific initialization
    template <typename FieldLHS, typename FieldRHS>
    void FFTPoissonSolver<FieldLHS, FieldRHS>::setRhs(rhs_type& rhs) {
        Base::setRhs(rhs);

        // start a timer
        static IpplTimings::TimerRef initialize = IpplTimings::getTimer("Initialize");
        IpplTimings::startTimer(initialize);

        initializeFields();

        IpplTimings::stopTimer(initialize);
    }

    /////////////////////////////////////////////////////////////////////////
    // allows user to set gradient of phi = Efield instead of spectral
    // calculation of Efield (which uses FFTs)

    template <typename FieldLHS, typename FieldRHS>
    void FFTPoissonSolver<FieldLHS, FieldRHS>::setGradFD() {
        // get the output type (sol, grad, or sol & grad)
        const int out = this->params_m.template get<int>("output_type");

        if (out != Base::SOL_AND_GRAD) {
            throw IpplException(
                "FFTPoissonSolver::setGradFD()",
                "Cannot use gradient for Efield computation unless output type is SOL_AND_GRAD");
        } else {
            isGradFD_m = true;
        }
    }

    /////////////////////////////////////////////////////////////////////////
    // initializeFields method, called in constructor

    template <typename FieldLHS, typename FieldRHS>
    void FFTPoissonSolver<FieldLHS, FieldRHS>::initializeFields() {
        const int alg = this->params_m.template get<int>("algorithm");

        // first check if valid algorithm choice
        if ((alg != Algorithm::VICO) && (alg != Algorithm::HOCKNEY)
            && (alg != Algorithm::BIHARMONIC)) {
            throw IpplException(
                "FFTPoissonSolver::initializeFields()",
                "Currently only Hockney, Vico, and Biharmonic are supported for open BCs");
        }

        // get layout and mesh
        layout_mp = &(this->rhs_mp->getLayout());
        mesh_mp   = &(this->rhs_mp->get_mesh());

        // get mesh spacing
        hr_m = mesh_mp->getMeshSpacing();

        // get origin
        vector_type origin    = mesh_mp->getOrigin();
        const scalar_type sum = std::abs(origin[0]) + std::abs(origin[1]) + std::abs(origin[2]);

        // origin should always be 0 for Green's function computation to work...
        if (sum != 0.0) {
            throw IpplException("FFTPoissonSolver::initializeFields", "Origin is not 0");
        }

        // create domain for the real fields
        domain_m = layout_mp->getDomain();

        // get the mesh spacings and sizes for each dimension
        for (unsigned int i = 0; i < Dim; ++i) {
            nr_m[i] = domain_m[i].length();

            // create the doubled domain for the FFT procedure
            domain2_m[i] = Index(2 * nr_m[i]);
        }

        // define decomposition (parallel / serial)
        e_dim_tag decomp[Dim];
        for (unsigned int d = 0; d < Dim; ++d) {
            decomp[d] = layout_mp->getRequestedDistribution(d);
        }

        // create double sized mesh and layout objects using the previously defined domain2_m
        using mesh_type = typename lhs_type::Mesh_t;
        mesh2_m         = std::unique_ptr<mesh_type>(new mesh_type(domain2_m, hr_m, origin));
        layout2_m       = std::unique_ptr<FieldLayout_t>(new FieldLayout_t(domain2_m, decomp));

        // create the domain for the transformed (complex) fields
        // since we use HeFFTe for the transforms it doesn't require permuting to the right
        // one of the dimensions has only (n/2 +1) as our original fields are fully real
        // the dimension is given by the user via r2c_direction
        unsigned int RCDirection = this->params_m.template get<int>("r2c_direction");
        for (unsigned int i = 0; i < Dim; ++i) {
            if (i == RCDirection) {
                domainComplex_m[RCDirection] = Index(nr_m[RCDirection] + 1);
            } else {
                domainComplex_m[i] = Index(2 * nr_m[i]);
            }
        }

        // create mesh and layout for the real to complex FFT transformed fields
        meshComplex_m = std::unique_ptr<mesh_type>(new mesh_type(domainComplex_m, hr_m, origin));
        layoutComplex_m =
            std::unique_ptr<FieldLayout_t>(new FieldLayout_t(domainComplex_m, decomp));

        // initialize fields
        storage_field.initialize(*mesh2_m, *layout2_m);
        rho2tr_m.initialize(*meshComplex_m, *layoutComplex_m);
        grntr_m.initialize(*meshComplex_m, *layoutComplex_m);

        int out = this->params_m.template get<int>("output_type");
        if (((out == Base::GRAD) || (out == Base::SOL_AND_GRAD)) && (!isGradFD_m)) {
            temp_m.initialize(*meshComplex_m, *layoutComplex_m);
        }

        // create the FFT object
        fft_m = std::make_unique<FFT_t>(*layout2_m, *layoutComplex_m, this->params_m);
        // if Vico, also need to create mesh and layout for 4N Fourier domain
        // on this domain, the truncated Green's function is defined
        // also need to create the 4N complex grid, on which precomputation step done
        if ((alg == Algorithm::VICO) || (alg == Algorithm::BIHARMONIC)) {
            // start a timer
            static IpplTimings::TimerRef initialize_vico =
                IpplTimings::getTimer("Initialize: extra Vico");
            IpplTimings::startTimer(initialize_vico);

            for (unsigned int i = 0; i < Dim; ++i) {
                domain4_m[i] = Index(4 * nr_m[i]);
            }

            // 4N grid
            using mesh_type = typename lhs_type::Mesh_t;
            mesh4_m         = std::unique_ptr<mesh_type>(new mesh_type(domain4_m, hr_m, origin));
            layout4_m       = std::unique_ptr<FieldLayout_t>(new FieldLayout_t(domain4_m, decomp));

            // initialize fields
            grnL_m.initialize(*mesh4_m, *layout4_m);

            // create a Complex-to-Complex FFT object to transform for layout4
            fft4n_m = std::make_unique<FFT<CCTransform, CxField_gt>>(*layout4_m, this->params_m);

            IpplTimings::stopTimer(initialize_vico);
        }

        // these are fields that are used for calculating the Green's function for Hockney
        if (alg == Algorithm::HOCKNEY) {
            // start a timer
            static IpplTimings::TimerRef initialize_hockney =
                IpplTimings::getTimer("Initialize: extra Hockney");
            IpplTimings::startTimer(initialize_hockney);

            for (unsigned int d = 0; d < Dim; ++d) {
                grnIField_m[d].initialize(*mesh2_m, *layout2_m);

                // get number of ghost points and the Kokkos view to iterate over field
                auto view        = grnIField_m[d].getView();
                const int nghost = grnIField_m[d].getNghost();
                const auto& ldom = layout2_m->getLocalNDIndex();

                // the length of the physical domain
                const int size = nr_m[d];

                // Kokkos parallel for loop to initialize grnIField[d]
                using index_array_type = typename RangePolicy<Dim>::index_array_type;
                switch (d) {
                    case 0:
                        ippl::parallel_for(
                            "Helper index Green field initialization", getRangePolicy(view, nghost),
                            KOKKOS_LAMBDA(const index_array_type& args) {
                                Vector<int, Dim> iVec = args;
                                Vector<int, Dim> igVec;
                                scalar_type checkVal = 0;
                                for (unsigned d1 = 0; d1 < Dim; ++d1) {
                                    // go from local indices to global
                                    igVec[d1] = iVec[d1] + ldom[d1].first() - nghost;
                                    checkVal += igVec[d1];
                                }
                                // assign (index)^2 if 0 <= index < N, and (2N-index)^2 elsewhere
                                const bool outsideN = (igVec[d] >= size);
                                apply(view, args) =
                                    (2 * size * outsideN - igVec[d]) * (2 * size * outsideN - igVec[d]);

                                // add 1.0 if at (0,0,0) to avoid singularity
                                const bool isZero = (checkVal == 0);
                                apply(view, args) += isZero * 1.0;
                            });
                        break;
                    case 1:
                        ippl::parallel_for(
                            "Helper index Green field initialization", getRangePolicy(view, nghost),
                            KOKKOS_LAMBDA(const index_array_type& args) {
                                Vector<int, Dim> iVec = args;
                                Vector<int, Dim> igVec;
                                for (unsigned d1 = 0; d1 < Dim; ++d1) {
                                    // go from local indices to global
                                    igVec[d1] = iVec[d1] + ldom[d1].first() - nghost;
                                }
                                // assign (index)^2 if 0 <= index < N, and (2N-index)^2 elsewhere
                                const bool outsideN = (igVec[d] >= size);
                                apply(view, args) =
                                    (2 * size * outsideN - igVec[d]) * (2 * size * outsideN - igVec[d]);
                            });
                        break;
                    case 2:
                        ippl::parallel_for(
                            "Helper index Green field initialization", getRangePolicy(view, nghost),
                            KOKKOS_LAMBDA(const index_array_type& args) {
                                Vector<int, Dim> iVec = args;
                                Vector<int, Dim> igVec;
                                for (unsigned d1 = 0; d1 < Dim; ++d1) {
                                    // go from local indices to global
                                    igVec[d1] = iVec[d1] + ldom[d1].first() - nghost;
                                }
                                // assign (index)^2 if 0 <= index < N, and (2N-index)^2 elsewhere
                                const bool outsideN = (igVec[d] >= size);
                                apply(view, args) =
                                    (2 * size * outsideN - igVec[d]) * (2 * size * outsideN - igVec[d]);
                            });
                        break;
                }
            }
            IpplTimings::stopTimer(initialize_hockney);
        }

        static IpplTimings::TimerRef warmup = IpplTimings::getTimer("Warmup");
        IpplTimings::startTimer(warmup);

        fft_m->transform(+1, rho2_mr, rho2tr_m);
        if ((alg == Algorithm::VICO) || (alg == Algorithm::BIHARMONIC)) {
            fft4n_m->transform(+1, grnL_m);
        }

        IpplTimings::stopTimer(warmup);

        rho2_mr  = 0.0;
        rho2tr_m = 0.0;
        grnL_m   = 0.0;

        // call greensFunction and we will get the transformed G in the class attribute
        // this is done in initialization so that we already have the precomputed fct
        // for all timesteps (green's fct will only change if mesh size changes)
        static IpplTimings::TimerRef ginit = IpplTimings::getTimer("Green Init");
        IpplTimings::startTimer(ginit);
        greensFunction();
        IpplTimings::stopTimer(ginit);
    };

    /////////////////////////////////////////////////////////////////////////
    // compute electric potential by solving Poisson's eq given a field rho and mesh spacings hr
    template <typename FieldLHS, typename FieldRHS>
    void FFTPoissonSolver<FieldLHS, FieldRHS>::solve() {
        // start a timer
        static IpplTimings::TimerRef solve = IpplTimings::getTimer("Solve");
        IpplTimings::startTimer(solve);

        // get the output type (sol, grad, or sol & grad)
        const int out = this->params_m.template get<int>("output_type");

        // get the algorithm (hockney, vico, or biharmonic)
        const int alg = this->params_m.template get<int>("algorithm");

        // set the mesh & spacing, which may change each timestep
        mesh_mp = &(this->rhs_mp->get_mesh());

        // check whether the mesh spacing has changed with respect to the old one
        // if yes, update and set green flag to true
        bool green = false;
        for (unsigned int i = 0; i < Dim; ++i) {
            if (hr_m[i] != mesh_mp->getMeshSpacing(i)) {
                hr_m[i] = mesh_mp->getMeshSpacing(i);
                green   = true;
            }
        }

        // set mesh spacing on the other grids again
        mesh2_m->setMeshSpacing(hr_m);
        meshComplex_m->setMeshSpacing(hr_m);

        // field object on the doubled grid; zero-padded
        rho2_mr = 0.0;

        // start a timer
        static IpplTimings::TimerRef stod = IpplTimings::getTimer("Solve: Physical to double");
        IpplTimings::startTimer(stod);

        // store rho (RHS) in the lower left quadrant of the doubled grid
        // with or without communication (if only 1 rank)

        const int ranks = Comm->size();

        auto view2 = rho2_mr.getView();
        auto view1 = this->rhs_mp->getView();

        const int nghost2 = rho2_mr.getNghost();
        const int nghost1 = this->rhs_mp->getNghost();

        const auto& ldom2 = layout2_m->getLocalNDIndex();
        const auto& ldom1 = layout_mp->getLocalNDIndex();

        if (ranks > 1) {
            // COMMUNICATION
            const auto& lDomains2 = layout2_m->getHostLocalDomains();

            // send
            std::vector<MPI_Request> requests(0);

            for (int i = 0; i < ranks; ++i) {
                if (lDomains2[i].touches(ldom1)) {
                    auto intersection = lDomains2[i].intersect(ldom1);

                    requests.resize(requests.size() + 1);

                    Communicate::size_type nsends;
                    pack(intersection, view1, fd_m, nghost1, ldom1, nsends);

                    buffer_type buf =
                        Comm->getBuffer<memory_space, Trhs>(IPPL_SOLVER_SEND + i, nsends);

                    Comm->isend(i, OPEN_SOLVER_TAG, fd_m, *buf, requests.back(), nsends);
                    buf->resetWritePos();
                }
            }

            // receive
            const auto& lDomains1 = layout_mp->getHostLocalDomains();
            int myRank            = Comm->rank();

            for (int i = 0; i < ranks; ++i) {
                if (lDomains1[i].touches(ldom2)) {
                    auto intersection = lDomains1[i].intersect(ldom2);

                    Communicate::size_type nrecvs;
                    nrecvs = intersection.size();

                    buffer_type buf =
                        Comm->getBuffer<memory_space, Trhs>(IPPL_SOLVER_RECV + myRank, nrecvs);

                    Comm->recv(i, OPEN_SOLVER_TAG, fd_m, *buf, nrecvs * sizeof(Trhs), nrecvs);
                    buf->resetReadPos();

                    std::vector<bool> coordBoolVec(Dim, false);
                    unpack(intersection, view2, fd_m, nghost2, ldom2, coordBoolVec);
                }
            }

            // wait for all messages to be received
            if (requests.size() > 0) {
                MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
            }
            Comm->barrier();

        } else {
            using index_array_type = typename RangePolicy<Dim>::index_array_type;
            ippl::parallel_for(
                "Write rho on the doubled grid", getRangePolicy(view1, nghost1),
                KOKKOS_LAMBDA(const index_array_type& args) {
                    Vector<int, Dim> iVec = args;
                    Vector<int, Dim> igVec1;
                    Vector<int, Dim> igVec2;
                    scalar_type checkVal = 0;
                    for (unsigned d = 0; d < Dim; ++d) {
                        igVec1[d] = iVec[d] + ldom1[d].first() - nghost1;
                        igVec2[d] = iVec[d] + ldom2[d].first() - nghost2;
                    
                        checkVal += igVec1[d] - igVec2[d]; // will be zero if all indicies are equal
                    }
                    const bool isZero = (checkVal == 0);
                    // write physical rho on [0,N-1] of doubled field
                    apply(view2, args)         = apply(view1, args) * isZero;
                });
        }

        IpplTimings::stopTimer(stod);

        // start a timer
        static IpplTimings::TimerRef fftrho = IpplTimings::getTimer("FFT: Rho");
        IpplTimings::startTimer(fftrho);

        // forward FFT of the charge density field on doubled grid
        fft_m->transform(+1, rho2_mr, rho2tr_m);

        IpplTimings::stopTimer(fftrho);

        // call greensFunction to recompute if the mesh spacing has changed
        if (green) {
            greensFunction();
        }

        // multiply FFT(rho2)*FFT(green)
        // convolution becomes multiplication in FFT
        // minus sign since we are solving laplace(phi) = -rho
        rho2tr_m = -rho2tr_m * grntr_m;

        // if output_type is SOL or SOL_AND_GRAD, we caculate solution
        if ((out == Base::SOL) || (out == Base::SOL_AND_GRAD)) {
            // start a timer
            static IpplTimings::TimerRef fftc = IpplTimings::getTimer("FFT: Convolution");
            IpplTimings::startTimer(fftc);

            // inverse FFT of the product and store the electrostatic potential in rho2_mr
            fft_m->transform(-1, rho2_mr, rho2tr_m);

            IpplTimings::stopTimer(fftc);

            // Hockney: multiply the rho2_mr field by the total number of points to account for
            // double counting (rho and green) of normalization factor in forward transform
            // also multiply by the mesh spacing^3 (to account for discretization)
            // Vico: need to multiply by normalization factor of 1/4N^3,
            // since only backward transform was performed on the 4N grid
            for (unsigned int i = 0; i < Dim; ++i) {
                if ((alg == Algorithm::VICO) || (alg == Algorithm::BIHARMONIC))
                    rho2_mr = rho2_mr * 2.0 * (1.0 / 4.0);
                else
                    rho2_mr = rho2_mr * 2.0 * nr_m[i] * hr_m[i];
            }

            // start a timer
            static IpplTimings::TimerRef dtos = IpplTimings::getTimer("Solve: Double to physical");
            IpplTimings::startTimer(dtos);

            // get the physical part only --> physical electrostatic potential is now given in RHS
            // need communication if more than one rank

            if (ranks > 1) {
                // COMMUNICATION

                // send
                const auto& lDomains1 = layout_mp->getHostLocalDomains();

                std::vector<MPI_Request> requests(0);

                for (int i = 0; i < ranks; ++i) {
                    if (lDomains1[i].touches(ldom2)) {
                        auto intersection = lDomains1[i].intersect(ldom2);

                        requests.resize(requests.size() + 1);

                        Communicate::size_type nsends;
                        pack(intersection, view2, fd_m, nghost2, ldom2, nsends);

                        buffer_type buf =
                            Comm->getBuffer<memory_space, Trhs>(IPPL_SOLVER_SEND + i, nsends);

                        Comm->isend(i, OPEN_SOLVER_TAG, fd_m, *buf, requests.back(), nsends);
                        buf->resetWritePos();
                    }
                }

                // receive
                const auto& lDomains2 = layout2_m->getHostLocalDomains();
                int myRank            = Comm->rank();

                for (int i = 0; i < ranks; ++i) {
                    if (ldom1.touches(lDomains2[i])) {
                        auto intersection = ldom1.intersect(lDomains2[i]);

                        Communicate::size_type nrecvs;
                        nrecvs = intersection.size();

                        buffer_type buf =
                            Comm->getBuffer<memory_space, Trhs>(IPPL_SOLVER_RECV + myRank, nrecvs);

                        Comm->recv(i, OPEN_SOLVER_TAG, fd_m, *buf, nrecvs * sizeof(Trhs), nrecvs);
                        buf->resetReadPos();
                        
                        std::vector<bool> coordBoolVec(Dim, false);
                        unpack(intersection, view1, fd_m, nghost1, ldom1, coordBoolVec);
                    }
                }

                // wait for all messages to be received
                if (requests.size() > 0) {
                    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
                }
                Comm->barrier();

            } else {
                using index_array_type = typename RangePolicy<Dim>::index_array_type;
                ippl::parallel_for(
                    "Write the solution into the LHS on physical grid", getRangePolicy(view1, nghost1),
                    KOKKOS_LAMBDA(const index_array_type& args) {
                    Vector<int, Dim> iVec = args;
                    Vector<int, Dim> igVec1;
                    Vector<int, Dim> igVec2;
                    scalar_type checkVal = 0;
                    for (unsigned d = 0; d < Dim; ++d) {
                        igVec2[d] = iVec[d] + ldom2[d].first() - nghost2;
                        igVec1[d] = iVec[d] + ldom1[d].first() - nghost1;
                    
                        checkVal += igVec1[d] - igVec2[d]; // will be zero if all indicies are equal
                    }
                    // take [0,N-1] as physical solution
                    const bool isZero = (checkVal == 0);
                    apply(view1, args)         = apply(view2, args) * isZero;
                    });
            }
            IpplTimings::stopTimer(dtos);
        }

        // if we want finite differences Efield = -grad(phi)
        // instead of computing it in Fourier domain
        // this is only possible if SOL_AND_GRAD is the output type
        if (isGradFD_m && (out == Base::SOL_AND_GRAD)) {
            *(this->lhs_mp) = -grad(*this->rhs_mp);
        }

        // if output_type is GRAD or SOL_AND_GRAD, we calculate E-field (gradient in Fourier domain)
        if (((out == Base::GRAD) || (out == Base::SOL_AND_GRAD)) && (!isGradFD_m)) {
            // start a timer
            static IpplTimings::TimerRef efield = IpplTimings::getTimer("Solve: Electric field");
            IpplTimings::startTimer(efield);

            // get E field view (LHS)
            auto viewL        = this->lhs_mp->getView();
            const int nghostL = this->lhs_mp->getNghost();

            // get rho2tr_m view (as we want to multiply by ik then transform)
            auto viewR        = rho2tr_m.getView();
            const int nghostR = rho2tr_m.getNghost();
            const auto& ldomR = layoutComplex_m->getLocalNDIndex();

            // use temp_m as a temporary complex field
            auto view_g = temp_m.getView();

            // define some constants
            const scalar_type pi          = Kokkos::numbers::pi_v<scalar_type>;
            const Kokkos::complex<Trhs> I = {0.0, 1.0};

            // define some member variables in local scope for the parallel_for
            vector_type hsize  = hr_m;
            Vector<int, Dim> N = nr_m;

            // loop over each component (E = vector field)
            for (size_t gd = 0; gd < Dim; ++gd) {
                // loop over rho2tr_m to multiply by -ik (gradient in Fourier space)
                using index_array_type = typename RangePolicy<Dim>::index_array_type;
                ippl::parallel_for(
                    "Gradient - E field", getRangePolicy(viewR, nghostR),
                    KOKKOS_LAMBDA(const index_array_type& args) {
                        Vector<int, Dim> iVec = args;
                        Vector<int, Dim> igVec;
                        // global indices for 2N rhotr_m
                        for (unsigned d = 0; d < Dim; ++d) {
                            igVec[d] = iVec[d] + ldomR[d].first() - nghostR;
                        }

                        scalar_type k_gd;
                        const scalar_type Len = N[gd] * hsize[gd];
                        const bool shift      = (igVec[gd] > N[gd]);
                        const bool notMid     = (igVec[gd] != N[gd]);

                        k_gd = notMid * (pi / Len) * (igVec[gd] - shift * 2 * N[gd]);

                        apply(view_g, args) = -(I * k_gd) * apply(viewR, args);
                    });


                // start a timer
                static IpplTimings::TimerRef ffte = IpplTimings::getTimer("FFT: Efield");
                IpplTimings::startTimer(ffte);

                // transform to get E-field
                fft_m->transform(-1, rho2_mr, temp_m);

                IpplTimings::stopTimer(ffte);

                // apply proper normalization
                for (unsigned int i = 0; i < Dim; ++i) {
                    if ((alg == Algorithm::VICO) || (alg == Algorithm::BIHARMONIC))
                        rho2_mr = rho2_mr * 2.0 * (1.0 / 4.0);
                    else
                        rho2_mr = rho2_mr * 2.0 * nr_m[i] * hr_m[i];
                }

                // start a timer
                static IpplTimings::TimerRef edtos =
                    IpplTimings::getTimer("Efield: double to phys.");
                IpplTimings::startTimer(edtos);

                // restrict to physical grid (N^3) and assign to LHS (E-field)
                // communication needed if more than one rank
                if (ranks > 1) {
                    // COMMUNICATION

                    // send
                    const auto& lDomains1 = layout_mp->getHostLocalDomains();
                    std::vector<MPI_Request> requests(0);

                    for (int i = 0; i < ranks; ++i) {
                        if (lDomains1[i].touches(ldom2)) {
                            auto intersection = lDomains1[i].intersect(ldom2);

                            requests.resize(requests.size() + 1);

                            Communicate::size_type nsends;
                            pack(intersection, view2, fd_m, nghost2, ldom2, nsends);

                            buffer_type buf =
                                Comm->getBuffer<memory_space, Trhs>(IPPL_SOLVER_SEND + i, nsends);

                            Comm->isend(i, OPEN_SOLVER_TAG, fd_m, *buf, requests.back(), nsends);
                            buf->resetWritePos();
                        }
                    }

                    // receive
                    const auto& lDomains2 = layout2_m->getHostLocalDomains();
                    int myRank            = Comm->rank();

                    for (int i = 0; i < ranks; ++i) {
                        if (ldom1.touches(lDomains2[i])) {
                            auto intersection = ldom1.intersect(lDomains2[i]);

                            Communicate::size_type nrecvs;
                            nrecvs = intersection.size();

                            buffer_type buf = Comm->getBuffer<memory_space, Trhs>(
                                IPPL_SOLVER_RECV + myRank, nrecvs);

                            Comm->recv(i, OPEN_SOLVER_TAG, fd_m, *buf, nrecvs * sizeof(Trhs),
                                       nrecvs);
                            buf->resetReadPos();
                            
                            std::vector<bool> coordBoolVec(Dim, false);
                            unpack(intersection, viewL, fd_m, nghostL, ldom1, coordBoolVec, gd);
                        }
                    }

                    // wait for all messages to be received
                    if (requests.size() > 0) {
                        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
                    }
                    Comm->barrier();

                } else {
                using index_array_type = typename RangePolicy<Dim>::index_array_type;
                ippl::parallel_for(
                    "Write the E-field on physical grid", getRangePolicy(viewL, nghostL),
                    KOKKOS_LAMBDA(const index_array_type& args) {
                        Vector<int, Dim> iVec = args;
                        Vector<int, Dim> igVec1;
                        Vector<int, Dim> igVec2;
                        scalar_type checkVal = 0;
                        for (unsigned d = 0; d < Dim; ++d) {
                            igVec2[d] = iVec[d] + ldom2[d].first() - nghost2;
                            igVec1[d] = iVec[d] + ldom1[d].first() - nghostL;

                            checkVal += igVec1[d] - igVec2[d]; // will be zero if all indicies are equal
                        }
                        // take [0,N-1] as physical solution
                        const bool isZero = (checkVal == 0);
                        apply(viewL, args)[gd]         = apply(view2, args) * isZero;
                        });
                }
                IpplTimings::stopTimer(edtos);
            }
            IpplTimings::stopTimer(efield);
        }
        IpplTimings::stopTimer(solve);
    };

    ////////////////////////////////////////////////////////////////////////
    // calculate FFT of the Green's function

    template <typename FieldLHS, typename FieldRHS>
    void FFTPoissonSolver<FieldLHS, FieldRHS>::greensFunction() {
        const scalar_type pi = Kokkos::numbers::pi_v<scalar_type>;
        grn_mr               = 0.0;

        const int alg = this->params_m.template get<int>("algorithm");

        if ((alg == Algorithm::VICO) || (alg == Algorithm::BIHARMONIC)) {
            Vector_t l(hr_m * nr_m);
            Vector_t hs_m;
            double L_sum(0.0);

            // compute length of the physical domain
            // compute Fourier domain spacing
            for (unsigned int i = 0; i < Dim; ++i) {
                hs_m[i] = pi * 0.5 / l[i];
                L_sum   = L_sum + l[i] * l[i];
            }

            // define the origin of the 4N grid
            vector_type origin;

            for (unsigned int i = 0; i < Dim; ++i) {
                origin[i] = -2 * nr_m[i] * pi / l[i];
            }

            // set mesh for the 4N mesh
            mesh4_m->setMeshSpacing(hs_m);

            // size of truncation window
            L_sum = std::sqrt(L_sum);
            L_sum = 1.1 * L_sum;

            // initialize grnL_m
            typename CxField_gt::view_type view_g = grnL_m.getView();
            const int nghost_g                    = grnL_m.getNghost();
            const auto& ldom_g                    = layout4_m->getLocalNDIndex();

            Vector<int, Dim> size = nr_m;

            // Kokkos parallel for loop to assign analytic grnL_m
            using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

            if (alg == Algorithm::VICO) {
                Kokkos::parallel_for(
                    "Initialize Green's function ",
                    mdrange_type({nghost_g, nghost_g, nghost_g},
                                 {view_g.extent(0) - nghost_g, view_g.extent(1) - nghost_g,
                                  view_g.extent(2) - nghost_g}),
                    KOKKOS_LAMBDA(const int i, const int j, const int k) {
                        // go from local indices to global
                        const int ig = i + ldom_g[0].first() - nghost_g;
                        const int jg = j + ldom_g[1].first() - nghost_g;
                        const int kg = k + ldom_g[2].first() - nghost_g;

                        bool isOutside = (ig > 2 * size[0] - 1);
                        const Tg t     = ig * hs_m[0] + isOutside * origin[0];

                        isOutside  = (jg > 2 * size[1] - 1);
                        const Tg u = jg * hs_m[1] + isOutside * origin[1];

                        isOutside  = (kg > 2 * size[2] - 1);
                        const Tg v = kg * hs_m[2] + isOutside * origin[2];

                        Tg s = (t * t) + (u * u) + (v * v);
                        s    = Kokkos::sqrt(s);

                        // assign the green's function value
                        // if (0,0,0), assign L^2/2 (analytical limit of sinc)

                        const bool isOrig    = ((ig == 0 && jg == 0 && kg == 0));
                        const Tg analyticLim = -L_sum * L_sum * 0.5;
                        const Tg value = -2.0 * (Kokkos::sin(0.5 * L_sum * s) / (s + isOrig * 1.0))
                                         * (Kokkos::sin(0.5 * L_sum * s) / (s + isOrig * 1.0));

                        view_g(i, j, k) = (!isOrig) * value + isOrig * analyticLim;
                    });

            } else if (alg == Algorithm::BIHARMONIC) {
                Kokkos::parallel_for(
                    "Initialize Green's function ",
                    mdrange_type({nghost_g, nghost_g, nghost_g},
                                 {view_g.extent(0) - nghost_g, view_g.extent(1) - nghost_g,
                                  view_g.extent(2) - nghost_g}),
                    KOKKOS_LAMBDA(const int i, const int j, const int k) {
                        // go from local indices to global
                        const int ig = i + ldom_g[0].first() - nghost_g;
                        const int jg = j + ldom_g[1].first() - nghost_g;
                        const int kg = k + ldom_g[2].first() - nghost_g;

                        bool isOutside = (ig > 2 * size[0] - 1);
                        const Tg t     = ig * hs_m[0] + isOutside * origin[0];

                        isOutside  = (jg > 2 * size[1] - 1);
                        const Tg u = jg * hs_m[1] + isOutside * origin[1];

                        isOutside  = (kg > 2 * size[2] - 1);
                        const Tg v = kg * hs_m[2] + isOutside * origin[2];

                        Tg s = (t * t) + (u * u) + (v * v);
                        s    = Kokkos::sqrt(s);

                        // assign value and replace with analytic limit at origin (0,0,0)
                        const bool isOrig    = ((ig == 0 && jg == 0 && kg == 0));
                        const Tg analyticLim = -L_sum * L_sum * L_sum * L_sum / 8.0;
                        const Tg value = -((2 - (L_sum * L_sum * s * s)) * Kokkos::cos(L_sum * s)
                                           + 2 * L_sum * s * Kokkos::sin(L_sum * s) - 2)
                                         / (2 * s * s * s * s + isOrig * 1.0);

                        view_g(i, j, k) = (!isOrig) * value + isOrig * analyticLim;
                    });
            }

            // start a timer
            static IpplTimings::TimerRef fft4 = IpplTimings::getTimer("FFT: Precomputation");
            IpplTimings::startTimer(fft4);

            // inverse Fourier transform of the green's function for precomputation
            fft4n_m->transform(-1, grnL_m);

            IpplTimings::stopTimer(fft4);

            // Restrict transformed grnL_m to 2N domain after precomputation step

            // get the field data first
            typename Field_t::view_type view = grn_mr.getView();
            const int nghost                 = grn_mr.getNghost();
            const auto& ldom                 = layout2_m->getLocalNDIndex();

            // start a timer
            static IpplTimings::TimerRef ifftshift = IpplTimings::getTimer("Vico shift loop");
            IpplTimings::startTimer(ifftshift);

            // get number of ranks to see if need communication
            const int ranks = Comm->size();

            if (ranks > 1) {
                communicateVico(size, view_g, ldom_g, nghost_g, view, ldom, nghost);
            } else {
                // restrict the green's function to a (2N)^3 grid from the (4N)^3 grid
                Kokkos::parallel_for(
                    "Restrict domain of Green's function from 4N to 2N",
                    mdrange_type({nghost, nghost, nghost}, {view.extent(0) - nghost - size[0],
                                                            view.extent(1) - nghost - size[1],
                                                            view.extent(2) - nghost - size[2]}),
                    KOKKOS_LAMBDA(const int i, const int j, const int k) {
                        // go from local indices to global
                        const int ig = i + ldom[0].first() - nghost;
                        const int jg = j + ldom[1].first() - nghost;
                        const int kg = k + ldom[2].first() - nghost;

                        const int ig2 = i + ldom_g[0].first() - nghost_g;
                        const int jg2 = j + ldom_g[1].first() - nghost_g;
                        const int kg2 = k + ldom_g[2].first() - nghost_g;

                        if ((ig == ig2) && (jg == jg2) && (kg == kg2)) {
                            view(i, j, k) = real(view_g(i, j, k));
                        }

                        // Now fill the rest of the field
                        const int s = 2 * size[0] - ig - 1 - ldom_g[0].first() + nghost_g;
                        const int p = 2 * size[1] - jg - 1 - ldom_g[1].first() + nghost_g;
                        const int q = 2 * size[2] - kg - 1 - ldom_g[2].first() + nghost_g;

                        view(s, j, k) = real(view_g(i + 1, j, k));
                        view(i, p, k) = real(view_g(i, j + 1, k));
                        view(i, j, q) = real(view_g(i, j, k + 1));
                        view(s, j, q) = real(view_g(i + 1, j, k + 1));
                        view(s, p, k) = real(view_g(i + 1, j + 1, k));
                        view(i, p, q) = real(view_g(i, j + 1, k + 1));
                        view(s, p, q) = real(view_g(i + 1, j + 1, k + 1));
                    });
            }
            IpplTimings::stopTimer(ifftshift);

        } else {
            // Hockney case

            // calculate square of the mesh spacing for each dimension
            Vector_t hrsq(hr_m * hr_m);

            // use the grnIField_m helper field to compute Green's function
            for (unsigned int i = 0; i < Dim; ++i) {
                grn_mr = grn_mr + grnIField_m[i] * hrsq[i];
            }

            //  Need error for Dim != to 2 or 3 since no Green's function is implemented for other dimensions
            if (Dim == 3){ 
                grn_mr = -1.0 / (4.0 * pi * sqrt(grn_mr));
            } else if (Dim == 2){
                grn_mr = log(sqrt(grn_mr))/(2 * pi); 
            }
            
            typename Field_t::view_type view = grn_mr.getView();
            const int nghost                 = grn_mr.getNghost();
            const auto& ldom                 = layout2_m->getLocalNDIndex();

            // Kokkos parallel for loop to find (0,0,0) point and regularize
            using index_array_type = typename RangePolicy<Dim>::index_array_type;
            ippl::parallel_for(
                "Regularize Green's function ", getRangePolicy(view, nghost),
                    KOKKOS_LAMBDA(const index_array_type& args) {
                    Vector<int, Dim> iVec = args;
                    Vector<int, Dim> igVec;
                    scalar_type checkVal = 0;
                    // go from local indices to global
                    for (unsigned d = 0; d < Dim; ++d) {
                        igVec[d] = iVec[d] + ldom[d].first() - nghost;

                        checkVal += igVec[d];
                    }

                    // if (0,0,0), assign to it 1/(4*pi)
                    // assign 1/2*pi if using the 2D Green's function
                    const bool isZero = (checkVal == 0);
                    if (Dim == 3){
                        apply(view, args)     = isZero * (-1.0 / (4.0 * pi)) + (!isZero) * apply(view, args);
                    } else if (Dim == 2){
                        apply(view, args)     = isZero * (1.0 / (2.0 * pi)) + (!isZero) * apply(view, args);          
                    }
                });
        }

        // start a timer
        static IpplTimings::TimerRef fftg = IpplTimings::getTimer("FFT: Green");
        IpplTimings::startTimer(fftg);

        // perform the FFT of the Green's function for the convolution
        fft_m->transform(+1, grn_mr, grntr_m);

        IpplTimings::stopTimer(fftg);
    };

    template <typename FieldLHS, typename FieldRHS>
    void FFTPoissonSolver<FieldLHS, FieldRHS>::communicateVico(
        Vector<int, Dim> size, typename CxField_gt::view_type view_g,
        const ippl::NDIndex<Dim> ldom_g, const int nghost_g, typename Field_t::view_type view,
        const ippl::NDIndex<Dim> ldom, const int nghost) {
        const auto& lDomains2 = layout2_m->getHostLocalDomains();
        const auto& lDomains4 = layout4_m->getHostLocalDomains();

        std::vector<MPI_Request> requests(0);
        const int myRank = Comm->rank();
        const int ranks  = Comm->size();

        // 1st step: Define 8 domains corresponding to the different quadrants
        ippl::NDIndex<Dim> none;
        for (unsigned i = 0; i < Dim; i++) {
            none[i] = ippl::Index(size[i]);
        }

        ippl::NDIndex<Dim> x;
        x[0] = ippl::Index(size[0], 2 * size[0] - 1);
        x[1] = ippl::Index(size[1]);
        x[2] = ippl::Index(size[2]);

        ippl::NDIndex<Dim> y;
        y[0] = ippl::Index(size[0]);
        y[1] = ippl::Index(size[1], 2 * size[1] - 1);
        y[2] = ippl::Index(size[2]);

        ippl::NDIndex<Dim> z;
        z[0] = ippl::Index(size[0]);
        z[1] = ippl::Index(size[1]);
        z[2] = ippl::Index(size[2], 2 * size[2] - 1);

        ippl::NDIndex<Dim> xy;
        xy[0] = ippl::Index(size[0], 2 * size[0] - 1);
        xy[1] = ippl::Index(size[1], 2 * size[1] - 1);
        xy[2] = ippl::Index(size[2]);

        ippl::NDIndex<Dim> xz;
        xz[0] = ippl::Index(size[0], 2 * size[0] - 1);
        xz[1] = ippl::Index(size[1]);
        xz[2] = ippl::Index(size[2], 2 * size[2] - 1);

        ippl::NDIndex<Dim> yz;
        yz[0] = ippl::Index(size[0]);
        yz[1] = ippl::Index(size[1], 2 * size[1] - 1);
        yz[2] = ippl::Index(size[2], 2 * size[2] - 1);

        ippl::NDIndex<Dim> xyz;
        for (unsigned i = 0; i < Dim; i++) {
            xyz[i] = ippl::Index(size[i], 2 * size[i] - 1);
        }

        // 2nd step: send
        for (int i = 0; i < ranks; ++i) {
            auto domain2 = lDomains2[i];

            if (domain2.touches(none)) {
                auto intersection = domain2.intersect(none);

                if (ldom_g.touches(intersection)) {
                    intersection = intersection.intersect(ldom_g);
                    requests.resize(requests.size() + 1);

                    Communicate::size_type nsends;
                    pack(intersection, view_g, fd_m, nghost_g, ldom_g, nsends);

                    buffer_type buf =
                        Comm->getBuffer<memory_space, Trhs>(IPPL_VICO_SEND + i, nsends);

                    int tag = VICO_SOLVER_TAG;

                    Comm->isend(i, tag, fd_m, *buf, requests.back(), nsends);
                    buf->resetWritePos();
                }
            }

            if (domain2.touches(x)) {
                auto intersection = domain2.intersect(x);
                auto xdom         = ippl::Index((2 * size[0] - intersection[0].first()),
                                                (2 * size[0] - intersection[0].last()), -1);

                ippl::NDIndex<Dim> domain4;
                domain4[0] = xdom;
                domain4[1] = intersection[1];
                domain4[2] = intersection[2];

                if (ldom_g.touches(domain4)) {
                    intersection = ldom_g.intersect(domain4);

                    requests.resize(requests.size() + 1);

                    Communicate::size_type nsends;
                    pack(intersection, view_g, fd_m, nghost_g, ldom_g, nsends);

                    buffer_type buf =
                        Comm->getBuffer<memory_space, Trhs>(IPPL_VICO_SEND + 8 + i, nsends);

                    int tag = VICO_SOLVER_TAG + 1;

                    Comm->isend(i, tag, fd_m, *buf, requests.back(), nsends);
                    buf->resetWritePos();
                }
            }

            if (domain2.touches(y)) {
                auto intersection = domain2.intersect(y);
                auto ydom         = ippl::Index((2 * size[1] - intersection[1].first()),
                                                (2 * size[1] - intersection[1].last()), -1);

                ippl::NDIndex<Dim> domain4;
                domain4[0] = intersection[0];
                domain4[1] = ydom;
                domain4[2] = intersection[2];

                if (ldom_g.touches(domain4)) {
                    intersection = ldom_g.intersect(domain4);

                    requests.resize(requests.size() + 1);

                    Communicate::size_type nsends;
                    pack(intersection, view_g, fd_m, nghost_g, ldom_g, nsends);

                    buffer_type buf =
                        Comm->getBuffer<memory_space, Trhs>(IPPL_VICO_SEND + 2 * 8 + i, nsends);

                    int tag = VICO_SOLVER_TAG + 2;

                    Comm->isend(i, tag, fd_m, *buf, requests.back(), nsends);
                    buf->resetWritePos();
                }
            }

            if (domain2.touches(z)) {
                auto intersection = domain2.intersect(z);
                auto zdom         = ippl::Index((2 * size[2] - intersection[2].first()),
                                                (2 * size[2] - intersection[2].last()), -1);

                ippl::NDIndex<Dim> domain4;
                domain4[0] = intersection[0];
                domain4[1] = intersection[1];
                domain4[2] = zdom;

                if (ldom_g.touches(domain4)) {
                    intersection = ldom_g.intersect(domain4);

                    requests.resize(requests.size() + 1);

                    Communicate::size_type nsends;
                    pack(intersection, view_g, fd_m, nghost_g, ldom_g, nsends);

                    buffer_type buf =
                        Comm->getBuffer<memory_space, Trhs>(IPPL_VICO_SEND + 3 * 8 + i, nsends);

                    int tag = VICO_SOLVER_TAG + 3;

                    Comm->isend(i, tag, fd_m, *buf, requests.back(), nsends);
                    buf->resetWritePos();
                }
            }

            if (domain2.touches(xy)) {
                auto intersection = domain2.intersect(xy);
                auto xdom         = ippl::Index((2 * size[0] - intersection[0].first()),
                                                (2 * size[0] - intersection[0].last()), -1);
                auto ydom         = ippl::Index((2 * size[1] - intersection[1].first()),
                                                (2 * size[1] - intersection[1].last()), -1);

                ippl::NDIndex<Dim> domain4;
                domain4[0] = xdom;
                domain4[1] = ydom;
                domain4[2] = intersection[2];

                if (ldom_g.touches(domain4)) {
                    intersection = ldom_g.intersect(domain4);

                    requests.resize(requests.size() + 1);

                    Communicate::size_type nsends;
                    pack(intersection, view_g, fd_m, nghost_g, ldom_g, nsends);

                    buffer_type buf =
                        Comm->getBuffer<memory_space, Trhs>(IPPL_VICO_SEND + 4 * 8 + i, nsends);

                    int tag = VICO_SOLVER_TAG + 4;

                    Comm->isend(i, tag, fd_m, *buf, requests.back(), nsends);
                    buf->resetWritePos();
                }
            }

            if (domain2.touches(yz)) {
                auto intersection = domain2.intersect(yz);
                auto ydom         = ippl::Index((2 * size[1] - intersection[1].first()),
                                                (2 * size[1] - intersection[1].last()), -1);
                auto zdom         = ippl::Index((2 * size[2] - intersection[2].first()),
                                                (2 * size[2] - intersection[2].last()), -1);

                ippl::NDIndex<Dim> domain4;
                domain4[0] = intersection[0];
                domain4[1] = ydom;
                domain4[2] = zdom;

                if (ldom_g.touches(domain4)) {
                    intersection = ldom_g.intersect(domain4);

                    requests.resize(requests.size() + 1);

                    Communicate::size_type nsends;
                    pack(intersection, view_g, fd_m, nghost_g, ldom_g, nsends);

                    buffer_type buf =
                        Comm->getBuffer<memory_space, Trhs>(IPPL_VICO_SEND + 5 * 8 + i, nsends);

                    int tag = VICO_SOLVER_TAG + 5;

                    Comm->isend(i, tag, fd_m, *buf, requests.back(), nsends);
                    buf->resetWritePos();
                }
            }

            if (domain2.touches(xz)) {
                auto intersection = domain2.intersect(xz);
                auto xdom         = ippl::Index((2 * size[0] - intersection[0].first()),
                                                (2 * size[0] - intersection[0].last()), -1);
                auto zdom         = ippl::Index((2 * size[2] - intersection[2].first()),
                                                (2 * size[2] - intersection[2].last()), -1);

                ippl::NDIndex<Dim> domain4;
                domain4[0] = xdom;
                domain4[1] = intersection[1];
                domain4[2] = zdom;

                if (ldom_g.touches(domain4)) {
                    intersection = ldom_g.intersect(domain4);

                    requests.resize(requests.size() + 1);

                    Communicate::size_type nsends;
                    pack(intersection, view_g, fd_m, nghost_g, ldom_g, nsends);

                    buffer_type buf =
                        Comm->getBuffer<memory_space, Trhs>(IPPL_VICO_SEND + 6 * 8 + i, nsends);

                    int tag = VICO_SOLVER_TAG + 6;

                    Comm->isend(i, tag, fd_m, *buf, requests.back(), nsends);
                    buf->resetWritePos();
                }
            }

            if (domain2.touches(xyz)) {
                auto intersection = domain2.intersect(xyz);
                auto xdom         = ippl::Index((2 * size[0] - intersection[0].first()),
                                                (2 * size[0] - intersection[0].last()), -1);
                auto ydom         = ippl::Index((2 * size[1] - intersection[1].first()),
                                                (2 * size[1] - intersection[1].last()), -1);
                auto zdom         = ippl::Index((2 * size[2] - intersection[2].first()),
                                                (2 * size[2] - intersection[2].last()), -1);

                ippl::NDIndex<Dim> domain4;
                domain4[0] = xdom;
                domain4[1] = ydom;
                domain4[2] = zdom;

                if (ldom_g.touches(domain4)) {
                    intersection = ldom_g.intersect(domain4);

                    requests.resize(requests.size() + 1);

                    Communicate::size_type nsends;
                    pack(intersection, view_g, fd_m, nghost_g, ldom_g, nsends);

                    buffer_type buf =
                        Comm->getBuffer<memory_space, Trhs>(IPPL_VICO_SEND + 7 * 8 + i, nsends);

                    int tag = VICO_SOLVER_TAG + 7;

                    Comm->isend(i, tag, fd_m, *buf, requests.back(), nsends);
                    buf->resetWritePos();
                }
            }
        }

        // 3rd step: receive
        for (int i = 0; i < ranks; ++i) {
            if (ldom.touches(none)) {
                auto intersection = ldom.intersect(none);

                if (lDomains4[i].touches(intersection)) {
                    intersection = intersection.intersect(lDomains4[i]);

                    Communicate::size_type nrecvs;
                    nrecvs = intersection.size();

                    buffer_type buf =
                        Comm->getBuffer<memory_space, Trhs>(IPPL_VICO_RECV + myRank, nrecvs);

                    int tag = VICO_SOLVER_TAG;

                    Comm->recv(i, tag, fd_m, *buf, nrecvs * sizeof(Trhs), nrecvs);
                    buf->resetReadPos();
                    
                    std::vector<bool> coordBoolVec(Dim, false);
                    unpack(intersection, view, fd_m, nghost, ldom, coordBoolVec);
                }
            }

            if (ldom.touches(x)) {
                auto intersection = ldom.intersect(x);

                auto xdom = ippl::Index((2 * size[0] - intersection[0].first()),
                                        (2 * size[0] - intersection[0].last()), -1);

                ippl::NDIndex<Dim> domain4;
                domain4[0] = xdom;
                domain4[1] = intersection[1];
                domain4[2] = intersection[2];

                if (lDomains4[i].touches(domain4)) {
                    domain4    = lDomains4[i].intersect(domain4);
                    domain4[0] = ippl::Index(2 * size[0] - domain4[0].first(),
                                             2 * size[0] - domain4[0].last(), -1);

                    intersection = intersection.intersect(domain4);

                    Communicate::size_type nrecvs;
                    nrecvs = intersection.size();

                    buffer_type buf =
                        Comm->getBuffer<memory_space, Trhs>(IPPL_VICO_RECV + 8 + myRank, nrecvs);

                    int tag = VICO_SOLVER_TAG + 1;

                    Comm->recv(i, tag, fd_m, *buf, nrecvs * sizeof(Trhs), nrecvs);
                    buf->resetReadPos();

                    std::vector<bool> coordBoolVec = {true, false, false};
                    unpack(intersection, view, fd_m, nghost, ldom, coordBoolVec);
                }
            }

            if (ldom.touches(y)) {
                auto intersection = ldom.intersect(y);

                auto ydom = ippl::Index((2 * size[1] - intersection[1].first()),
                                        (2 * size[1] - intersection[1].last()), -1);

                ippl::NDIndex<Dim> domain4;
                domain4[0] = intersection[0];
                domain4[1] = ydom;
                domain4[2] = intersection[2];

                if (lDomains4[i].touches(domain4)) {
                    domain4    = lDomains4[i].intersect(domain4);
                    domain4[1] = ippl::Index(2 * size[1] - domain4[1].first(),
                                             2 * size[1] - domain4[1].last(), -1);

                    intersection = intersection.intersect(domain4);

                    Communicate::size_type nrecvs;
                    nrecvs = intersection.size();

                    buffer_type buf = Comm->getBuffer<memory_space, Trhs>(
                        IPPL_VICO_RECV + 8 * 2 + myRank, nrecvs);

                    int tag = VICO_SOLVER_TAG + 2;

                    Comm->recv(i, tag, fd_m, *buf, nrecvs * sizeof(Trhs), nrecvs);
                    buf->resetReadPos();

                    std::vector<bool> coordBoolVec = {false, true, false};
                    unpack(intersection, view, fd_m, nghost, ldom, coordBoolVec);
                }
            }

            if (ldom.touches(z)) {
                auto intersection = ldom.intersect(z);

                auto zdom = ippl::Index((2 * size[2] - intersection[2].first()),
                                        (2 * size[2] - intersection[2].last()), -1);

                ippl::NDIndex<Dim> domain4;
                domain4[0] = intersection[0];
                domain4[1] = intersection[1];
                domain4[2] = zdom;

                if (lDomains4[i].touches(domain4)) {
                    domain4    = lDomains4[i].intersect(domain4);
                    domain4[2] = ippl::Index(2 * size[2] - domain4[2].first(),
                                             2 * size[2] - domain4[2].last(), -1);

                    intersection = intersection.intersect(domain4);

                    Communicate::size_type nrecvs;
                    nrecvs = intersection.size();

                    buffer_type buf = Comm->getBuffer<memory_space, Trhs>(
                        IPPL_VICO_RECV + 8 * 3 + myRank, nrecvs);

                    int tag = VICO_SOLVER_TAG + 3;

                    Comm->recv(i, tag, fd_m, *buf, nrecvs * sizeof(Trhs), nrecvs);
                    buf->resetReadPos();

                    std::vector<bool> coordBoolVec = {false, false, true};
                    unpack(intersection, view, fd_m, nghost, ldom, coordBoolVec);
                }
            }

            if (ldom.touches(xy)) {
                auto intersection = ldom.intersect(xy);

                auto xdom = ippl::Index((2 * size[0] - intersection[0].first()),
                                        (2 * size[0] - intersection[0].last()), -1);
                auto ydom = ippl::Index((2 * size[1] - intersection[1].first()),
                                        (2 * size[1] - intersection[1].last()), -1);

                ippl::NDIndex<Dim> domain4;
                domain4[0] = xdom;
                domain4[1] = ydom;
                domain4[2] = intersection[2];

                if (lDomains4[i].touches(domain4)) {
                    domain4    = lDomains4[i].intersect(domain4);
                    domain4[0] = ippl::Index(2 * size[0] - domain4[0].first(),
                                             2 * size[0] - domain4[0].last(), -1);
                    domain4[1] = ippl::Index(2 * size[1] - domain4[1].first(),
                                             2 * size[1] - domain4[1].last(), -1);

                    intersection = intersection.intersect(domain4);

                    Communicate::size_type nrecvs;
                    nrecvs = intersection.size();

                    buffer_type buf = Comm->getBuffer<memory_space, Trhs>(
                        IPPL_VICO_RECV + 8 * 4 + myRank, nrecvs);

                    int tag = VICO_SOLVER_TAG + 4;

                    Comm->recv(i, tag, fd_m, *buf, nrecvs * sizeof(Trhs), nrecvs);
                    buf->resetReadPos();

                    std::vector<bool> coordBoolVec = {true, true, false};
                    unpack(intersection, view, fd_m, nghost, ldom, coordBoolVec);
                }
            }

            if (ldom.touches(yz)) {
                auto intersection = ldom.intersect(yz);

                auto ydom = ippl::Index((2 * size[1] - intersection[1].first()),
                                        (2 * size[1] - intersection[1].last()), -1);
                auto zdom = ippl::Index((2 * size[2] - intersection[2].first()),
                                        (2 * size[2] - intersection[2].last()), -1);

                ippl::NDIndex<Dim> domain4;
                domain4[0] = intersection[0];
                domain4[1] = ydom;
                domain4[2] = zdom;

                if (lDomains4[i].touches(domain4)) {
                    domain4    = lDomains4[i].intersect(domain4);
                    domain4[1] = ippl::Index(2 * size[1] - domain4[1].first(),
                                             2 * size[1] - domain4[1].last(), -1);
                    domain4[2] = ippl::Index(2 * size[2] - domain4[2].first(),
                                             2 * size[2] - domain4[2].last(), -1);

                    intersection = intersection.intersect(domain4);

                    Communicate::size_type nrecvs;
                    nrecvs = intersection.size();

                    buffer_type buf = Comm->getBuffer<memory_space, Trhs>(
                        IPPL_VICO_RECV + 8 * 5 + myRank, nrecvs);

                    int tag = VICO_SOLVER_TAG + 5;

                    Comm->recv(i, tag, fd_m, *buf, nrecvs * sizeof(Trhs), nrecvs);
                    buf->resetReadPos();

                    std::vector<bool> coordBoolVec = {false, true, true};
                    unpack(intersection, view, fd_m, nghost, ldom, coordBoolVec);
                }
            }

            if (ldom.touches(xz)) {
                auto intersection = ldom.intersect(xz);

                auto xdom = ippl::Index((2 * size[0] - intersection[0].first()),
                                        (2 * size[0] - intersection[0].last()), -1);
                auto zdom = ippl::Index((2 * size[2] - intersection[2].first()),
                                        (2 * size[2] - intersection[2].last()), -1);

                ippl::NDIndex<Dim> domain4;
                domain4[0] = xdom;
                domain4[1] = intersection[1];
                domain4[2] = zdom;

                if (lDomains4[i].touches(domain4)) {
                    domain4    = lDomains4[i].intersect(domain4);
                    domain4[0] = ippl::Index(2 * size[0] - domain4[0].first(),
                                             2 * size[0] - domain4[0].last(), -1);
                    domain4[2] = ippl::Index(2 * size[2] - domain4[2].first(),
                                             2 * size[2] - domain4[2].last(), -1);

                    intersection = intersection.intersect(domain4);

                    Communicate::size_type nrecvs;
                    nrecvs = intersection.size();

                    buffer_type buf = Comm->getBuffer<memory_space, Trhs>(
                        IPPL_VICO_RECV + 8 * 6 + myRank, nrecvs);

                    int tag = VICO_SOLVER_TAG + 6;

                    Comm->recv(i, tag, fd_m, *buf, nrecvs * sizeof(Trhs), nrecvs);
                    buf->resetReadPos();
                    
                    std::vector<bool> coordBoolVec = {true, false, true};
                    unpack(intersection, view, fd_m, nghost, ldom, coordBoolVec);
                }
            }

            if (ldom.touches(xyz)) {
                auto intersection = ldom.intersect(xyz);

                auto xdom = ippl::Index((2 * size[0] - intersection[0].first()),
                                        (2 * size[0] - intersection[0].last()), -1);
                auto ydom = ippl::Index((2 * size[1] - intersection[1].first()),
                                        (2 * size[1] - intersection[1].last()), -1);
                auto zdom = ippl::Index((2 * size[2] - intersection[2].first()),
                                        (2 * size[2] - intersection[2].last()), -1);

                ippl::NDIndex<Dim> domain4;
                domain4[0] = xdom;
                domain4[1] = ydom;
                domain4[2] = zdom;

                if (lDomains4[i].touches(domain4)) {
                    domain4    = lDomains4[i].intersect(domain4);
                    domain4[0] = ippl::Index(2 * size[0] - domain4[0].first(),
                                             2 * size[0] - domain4[0].last(), -1);
                    domain4[1] = ippl::Index(2 * size[1] - domain4[1].first(),
                                             2 * size[1] - domain4[1].last(), -1);
                    domain4[2] = ippl::Index(2 * size[2] - domain4[2].first(),
                                             2 * size[2] - domain4[2].last(), -1);

                    intersection = intersection.intersect(domain4);

                    Communicate::size_type nrecvs;
                    nrecvs = intersection.size();

                    buffer_type buf = Comm->getBuffer<memory_space, Trhs>(
                        IPPL_VICO_RECV + 8 * 7 + myRank, nrecvs);

                    int tag = VICO_SOLVER_TAG + 7;

                    Comm->recv(i, tag, fd_m, *buf, nrecvs * sizeof(Trhs), nrecvs);
                    buf->resetReadPos();

                    std::vector<bool> coordBoolVec = {true, true, true};
                    unpack(intersection, view, fd_m, nghost, ldom, coordBoolVec);
                }
            }
        }

        if (requests.size() > 0) {
            MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        }
        Comm->barrier();
    };
}  // namespace ippl
