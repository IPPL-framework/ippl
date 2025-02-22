#ifndef NSFDSOLVERWITHPARTICLES_H
#define NSFDSOLVERWITHPARTICLES_H
#include "MaxwellSolvers/FDTD.h"
namespace ippl{
    template <typename _scalar, class PLayout>
    struct Bunch : public ippl::ParticleBase<PLayout> {
        using scalar = _scalar;

        // Constructor for the Bunch class, taking a PLayout reference
        Bunch(PLayout& playout)
            : ippl::ParticleBase<PLayout>(playout) {
            // Add attributes to the particle bunch
            this->addAttribute(Q);     // Charge attribute
            this->addAttribute(mass);  // Mass attribute
            this->addAttribute(
                gamma_beta);  // Gamma-beta attribute (product of relativistiv gamma and beta)
            this->addAttribute(R_np1);     // Position attribute for the next time step
            this->addAttribute(R_nm1);     // Position attribute for the next time step
            this->addAttribute(E_gather);  // Electric field attribute for particle gathering
            this->addAttribute(B_gather);  // Magnedit field attribute for particle gathering
        }

        // Destructor for the Bunch class
        ~Bunch() {}

        // Define container types for various attributes
        using charge_container_type   = ippl::ParticleAttrib<scalar>;
        using velocity_container_type = ippl::ParticleAttrib<ippl::Vector<scalar, 3>>;
        using vector_container_type   = ippl::ParticleAttrib<ippl::Vector<scalar, 3>>;
        using vector4_container_type  = ippl::ParticleAttrib<ippl::Vector<scalar, 4>>;

        // Declare instances of the attribute containers
        charge_container_type Q;             // Charge container
        charge_container_type mass;          // Mass container
        velocity_container_type gamma_beta;  // Gamma-beta container
        typename ippl::ParticleBase<PLayout>::particle_position_type
            R_np1;  // Position container for the next time step
        typename ippl::ParticleBase<PLayout>::particle_position_type
            R_nm1;  // Position container for the previous time step
        ippl::ParticleAttrib<ippl::Vector<scalar, 3>>
            E_gather;  // Electric field container for particle gathering
        ippl::ParticleAttrib<ippl::Vector<scalar, 3>>
            B_gather;  // Magnetic field container for particle gathering
    };
    template <typename T>
    KOKKOS_INLINE_FUNCTION Kokkos::pair<ippl::Vector<int, 3>, ippl::Vector<T, 3>> gridCoordinatesOf(
        const ippl::Vector<T, 3> hr, const ippl::Vector<T, 3> origin, ippl::Vector<T, 3> pos, int nghost = 1) {

        // Declare a pair to hold the resulting grid coordinates (integer part) and fractional part
        Kokkos::pair<ippl::Vector<int, 3>, ippl::Vector<T, 3>> ret;

        // Calculate the relative position of pos with respect to the origin
        ippl::Vector<T, 3> relpos = pos - origin;

        // Convert the relative position to grid coordinates by dividing by the grid spacing (hr)
        ippl::Vector<T, 3> gridpos = relpos / hr;

        // Declare an integer vector to hold the integer part of the grid coordinates
        ippl::Vector<int, 3> ipos;

        // Cast the grid position to an integer vector, which gives us the integer part of the coordinates
        ipos = gridpos.template cast<int>();

        // Declare a vector to hold the fractional part of the grid coordinates
        ippl::Vector<T, 3> fracpos;

        // Calculate the fractional part of the grid coordinates
        for (unsigned k = 0; k < 3; k++) {
            fracpos[k] = gridpos[k] - static_cast<int>(ipos[k]);
        }

        // Add the number of ghost cells to the integer part of the coordinates
        ipos += ippl::Vector<int, 3>(nghost);

        // Set the integer part of the coordinates in the return pair
        ret.first = ipos;

        // Set the fractional part of the coordinates in the return pair
        ret.second = fracpos;

        // Return the pair containing both the integer and fractional parts of the grid coordinates
        return ret;
    }

    template <typename view_type, typename scalar>
    KOKKOS_FUNCTION void scatterToGrid(const ippl::NDIndex<3>& ldom, view_type& view,
                                       ippl::Vector<scalar, 3> hr, ippl::Vector<scalar, 3> orig,
                                       const ippl::Vector<scalar, 3>& pos, const scalar value) {
        auto [ipos, fracpos] = gridCoordinatesOf(hr, orig, pos);
        ipos -= ldom.first();
        // std::cout << pos << " 's scatter args (will have 1, or nghost in general added): " << ipos << "\n";
        if (ipos[0] < 0 || ipos[1] < 0 || ipos[2] < 0 || size_t(ipos[0]) >= view.extent(0) - 1
            || size_t(ipos[1]) >= view.extent(1) - 1 || size_t(ipos[2]) >= view.extent(2) - 1
            || fracpos[0] < 0 || fracpos[1] < 0 || fracpos[2] < 0) {
            return;
        }
        assert(fracpos[0] >= 0.0f);
        assert(fracpos[0] <= 1.0f);
        assert(fracpos[1] >= 0.0f);
        assert(fracpos[1] <= 1.0f);
        assert(fracpos[2] >= 0.0f);
        assert(fracpos[2] <= 1.0f);
        ippl::Vector<scalar, 3> one_minus_fracpos = ippl::Vector<scalar, 3>(1) - fracpos;
        assert(one_minus_fracpos[0] >= 0.0f);
        assert(one_minus_fracpos[0] <= 1.0f);
        assert(one_minus_fracpos[1] >= 0.0f);
        assert(one_minus_fracpos[1] <= 1.0f);
        assert(one_minus_fracpos[2] >= 0.0f);
        assert(one_minus_fracpos[2] <= 1.0f);
        scalar accum = 0;

        for (unsigned i = 0; i < 8; i++) {
            scalar weight               = 1;
            ippl::Vector<int, 3> ipos_l = ipos;
            for (unsigned d = 0; d < 3; d++) {
                weight *= ((i & (1 << d)) ? fracpos[d] : one_minus_fracpos[d]);
                ipos_l[d] += !!(i & (1 << d));
            }
            assert_isreal(value);
            assert_isreal(weight);
            accum += weight;
            Kokkos::atomic_add(&(view(ipos_l[0], ipos_l[1], ipos_l[2])[0]), value * weight);
        }
        assert(abs(accum - 1.0f) < 1e-6f);
    }
    template <typename view_type, typename scalar>
    KOKKOS_FUNCTION void scatterToGrid(const ippl::NDIndex<3>& ldom, view_type& view,
                                       ippl::Vector<scalar, 3> hr, ippl::Vector<scalar, 3> orig,
                                       const ippl::Vector<scalar, 3>& pos,
                                       const ippl::Vector<scalar, 3>& value) {
        auto [ipos, fracpos] = gridCoordinatesOf(hr, orig, pos);
        ipos -= ldom.first();
        if (ipos[0] < 0 || ipos[1] < 0 || ipos[2] < 0 || size_t(ipos[0]) >= view.extent(0) - 1
            || size_t(ipos[1]) >= view.extent(1) - 1 || size_t(ipos[2]) >= view.extent(2) - 1
            || fracpos[0] < 0 || fracpos[1] < 0 || fracpos[2] < 0) {
            // Out of bounds case (you'll do nothing)
            return;
        }
        assert(fracpos[0] >= 0.0f);
        assert(fracpos[0] <= 1.0f);
        assert(fracpos[1] >= 0.0f);
        assert(fracpos[1] <= 1.0f);
        assert(fracpos[2] >= 0.0f);
        assert(fracpos[2] <= 1.0f);
        ippl::Vector<scalar, 3> one_minus_fracpos = ippl::Vector<scalar, 3>(1) - fracpos;
        assert(one_minus_fracpos[0] >= 0.0f);
        assert(one_minus_fracpos[0] <= 1.0f);
        assert(one_minus_fracpos[1] >= 0.0f);
        assert(one_minus_fracpos[1] <= 1.0f);
        assert(one_minus_fracpos[2] >= 0.0f);
        assert(one_minus_fracpos[2] <= 1.0f);
        scalar accum = 0;

        for (unsigned i = 0; i < 8; i++) {
            scalar weight               = 1;
            ippl::Vector<int, 3> ipos_l = ipos;
            for (unsigned d = 0; d < 3; d++) {
                weight *= ((i & (1 << d)) ? fracpos[d] : one_minus_fracpos[d]);
                ipos_l[d] += !!(i & (1 << d));
            }
            assert_isreal(weight);
            accum += weight;
            Kokkos::atomic_add(&(view(ipos_l[0], ipos_l[1], ipos_l[2])[1]), value[0] * weight);
            Kokkos::atomic_add(&(view(ipos_l[0], ipos_l[1], ipos_l[2])[2]), value[1] * weight);
            Kokkos::atomic_add(&(view(ipos_l[0], ipos_l[1], ipos_l[2])[3]), value[2] * weight);
        }
        assert(abs(accum - 1.0f) < 1e-6f);
    }
    template <typename view_type, typename scalar>
    KOKKOS_INLINE_FUNCTION void scatterLineToGrid(const ippl::NDIndex<3>& ldom, view_type& Jview,
                                                  ippl::Vector<scalar, 3> hr,
                                                  ippl::Vector<scalar, 3> origin,
                                                  const ippl::Vector<scalar, 3>& from,
                                                  const ippl::Vector<scalar, 3>& to,
                                                  const scalar factor) {
        Kokkos::pair<ippl::Vector<int, 3>, ippl::Vector<scalar, 3>> from_grid =
            gridCoordinatesOf(hr, origin, from);
        Kokkos::pair<ippl::Vector<int, 3>, ippl::Vector<scalar, 3>> to_grid =
            gridCoordinatesOf(hr, origin, to);

        if (from_grid.first[0] == to_grid.first[0] && from_grid.first[1] == to_grid.first[1]
            && from_grid.first[2] == to_grid.first[2]) {
            scatterToGrid(ldom, Jview, hr, origin,
                          /*Scatter point, geometric average */         ippl::Vector<scalar, 3>((from + to) * scalar(0.5)),
                          /*Scatter value, factor=charge / timestep */ ippl::Vector<scalar, 3>((to - from) * factor));

            return;
        }
        ippl::Vector<scalar, 3> relay;
        const int nghost                   = 1;
        const ippl::Vector<scalar, 3> orig = origin;
        using Kokkos::max;
        using Kokkos::min;
        for (unsigned int i = 0; i < 3; i++) {
            relay[i] = min(min(from_grid.first[i] - nghost, to_grid.first[i] - nghost) * hr[i]
                               + scalar(1.0) * hr[i] + orig[i],
                           max(max(from_grid.first[i] - nghost, to_grid.first[i] - nghost) * hr[i]
                                   + scalar(0.0) * hr[i] + orig[i],
                               scalar(0.5) * (to[i] + from[i])));
        }
        scatterToGrid(ldom, Jview, hr, origin,
                      ippl::Vector<scalar, 3>((from + relay) * scalar(0.5)),
                      ippl::Vector<scalar, 3>((relay - from) * factor));
        scatterToGrid(ldom, Jview, hr, origin, ippl::Vector<scalar, 3>((relay + to) * scalar(0.5)),
                      ippl::Vector<scalar, 3>((to - relay) * factor));
    }
    template <typename scalar, fdtd_bc boundary_conditions>
    class NSFDSolverWithParticles {
    public:
        constexpr static unsigned dim = 3;
        using vector_type             = ippl::Vector<scalar, 3>;
        using vector4_type            = ippl::Vector<scalar, 4>;
        using FourField =
            ippl::Field<vector4_type, dim, ippl::UniformCartesian<scalar, dim>,
                        typename ippl::UniformCartesian<scalar, dim>::DefaultCentering>;
        using ThreeField =
            ippl::Field<vector_type, dim, ippl::UniformCartesian<scalar, dim>,
                        typename ippl::UniformCartesian<scalar, dim>::DefaultCentering>;
        using playout_type = ParticleSpatialLayout<scalar, 3>;
        using bunch_type   = Bunch<scalar, ParticleSpatialLayout<scalar, 3>>;
        using Mesh_t       = ippl::UniformCartesian<scalar, dim>;
        FieldLayout<dim>* layout_mp;
        Mesh_t* mesh_mp;
        playout_type playout;
        Bunch<scalar, ParticleSpatialLayout<scalar, 3>> particles;
        FourField J;
        ThreeField E;
        ThreeField B;
        NonStandardFDTDSolver<ThreeField, FourField, absorbing> field_solver;

        ippl::Vector<uint32_t, 3> nr_global;
        ippl::Vector<scalar, 3> hr_m;
        size_t steps_taken;
        NSFDSolverWithParticles(FieldLayout<dim>& layout, Mesh_t& mesch, size_t nparticles)
            : layout_mp(&layout)
            , mesh_mp(&mesch)
            , playout(layout, mesch)
            , particles(playout)
            , J(mesch, layout)
            , E(mesch, layout)
            , B(mesch, layout)
            , field_solver(J, E, B) {
            particles.create(nparticles);
            nr_global = ippl::Vector<uint32_t, 3>{
                uint32_t(layout.getDomain()[0].last() - layout.getDomain()[0].first() + 1),
                uint32_t(layout.getDomain()[1].last() - layout.getDomain()[1].first() + 1),
                uint32_t(layout.getDomain()[2].last() - layout.getDomain()[2].first() + 1)};
            hr_m        = mesh_mp->getMeshSpacing();
            steps_taken = 0;
        }
        template <bool space_charge = false>
        void scatterBunch() {
            auto hr_m           = mesh_mp->getMeshSpacing();
            const scalar volume = hr_m[0] * hr_m[1] * hr_m[2];
            assert_isreal(volume);
            assert_isreal((scalar(1) / volume));
            J                       = typename decltype(J)::value_type(0);
            auto Jview              = J.getView();
            auto qview              = particles.Q.getView();
            auto rview              = particles.R.getView();
            auto rm1view            = particles.R_nm1.getView();
            auto orig               = mesh_mp->getOrigin();
            auto hr                 = mesh_mp->getMeshSpacing();
            auto dt                 = field_solver.dt;
            bool sc                 = space_charge;
            ippl::NDIndex<dim> lDom = layout_mp->getLocalNDIndex();
            Kokkos::parallel_for(
                particles.getLocalNum(), KOKKOS_LAMBDA(size_t i) {
                    vector_type pos  = rview(i);
                    vector_type to   = rview(i);
                    vector_type from = rm1view(i);
                    if (sc) {
                        scatterToGrid(lDom, Jview, hr, orig, pos, qview(i) / volume);
                    }
                    scatterLineToGrid(lDom, Jview, hr, orig, from, to,
                                      scalar(qview(i)) / (volume * dt));
                });
            Kokkos::fence();
            J.accumulateHalo();
        }
        template <typename callable>
        void updateBunch(scalar time, callable external_field) {
            Kokkos::fence();
            auto gbview     = particles.gamma_beta.getView();
            auto eview      = particles.E_gather.getView();
            auto bview      = particles.B_gather.getView();
            auto qview      = particles.Q.getView();
            auto mview      = particles.mass.getView();
            auto rview      = particles.R.getView();
            auto rm1view    = particles.R_nm1.getView();
            auto rp1view    = particles.R_np1.getView();
            scalar bunch_dt = field_solver.dt / 3;
            Kokkos::deep_copy(particles.R_nm1.getView(), particles.R.getView());
            E.fillHalo();
            B.fillHalo();
            Kokkos::fence();
            for (int bts = 0; bts < 3; bts++) {
                particles.E_gather.gather(E, particles.R, /*offset = */ 0.0);
                particles.B_gather.gather(B, particles.R, /*offset = */ 0.0);
                Kokkos::fence();
                Kokkos::parallel_for(
                    particles.getLocalNum(), KOKKOS_LAMBDA(size_t i) {
                        const ippl::Vector<scalar, 3> pgammabeta = gbview(i);
                        ippl::Vector<scalar, 3> E_grid           = eview(i);
                        ippl::Vector<scalar, 3> B_grid           = bview(i);
                        ippl::Vector<scalar, 3> bunchpos = rview(i);
                        Kokkos::pair<ippl::Vector<scalar, 3>, ippl::Vector<scalar, 3>> external_eb =
                            external_field(bunchpos, time + bunch_dt * bts);

                        ippl::Vector<ippl::Vector<scalar, 3>, 2> EB{
                            ippl::Vector<scalar, 3>(E_grid + external_eb.first),
                            ippl::Vector<scalar, 3>(B_grid + external_eb.second)};

                        const scalar charge = qview(i);
                        const scalar mass   = mview(i);
                        const ippl::Vector<scalar, 3> t1 =
                            pgammabeta + charge * bunch_dt * EB[0] / (scalar(2) * mass);
                        const scalar alpha =
                            charge * bunch_dt / (scalar(2) * mass * Kokkos::sqrt(1 + t1.dot(t1)));
                        const ippl::Vector<scalar, 3> t2 = t1 + alpha * t1.cross(EB[1]);
                        const ippl::Vector<scalar, 3> t3 =
                            t1
                            + t2.cross(scalar(2) * alpha
                                       * (EB[1] / (1.0 + alpha * alpha * (EB[1].dot(EB[1])))));
                        const ippl::Vector<scalar, 3> ngammabeta =
                            t3 + charge * bunch_dt * EB[0] / (scalar(2) * mass);

                        rview(i) =
                            rview(i)
                            + bunch_dt * ngammabeta
                                  / (Kokkos::sqrt(scalar(1.0) + (ngammabeta.dot(ngammabeta))));
                        gbview(i) = ngammabeta;
                    });
                Kokkos::fence();
            }
            Kokkos::View<bool*> invalid("OOB Particcel", particles.getLocalNum());
            size_t invalid_count = 0;
            auto origo           = mesh_mp->getOrigin();
            ippl::Vector<scalar, 3> extenz;  //
            extenz[0] = nr_global[0] * hr_m[0];
            extenz[1] = nr_global[1] * hr_m[1];
            extenz[2] = nr_global[2] * hr_m[2];
            Kokkos::parallel_reduce(
                Kokkos::RangePolicy<
                    typename playout_type::RegionLayout_t::view_type::execution_space>(
                    0, particles.getLocalNum()),
                KOKKOS_LAMBDA(size_t i, size_t & ref) {
                    bool out_of_bounds             = false;
                    ippl::Vector<scalar, dim> ppos = rview(i);
                    for (size_t d = 0; d < dim; d++) {
                        out_of_bounds |= (ppos[d] <= origo[d]);
                        out_of_bounds |=
                            (ppos[d] >= origo[d] + extenz[d]);  // Check against simulation domain
                    }
                    invalid(i) = out_of_bounds;
                    ref += out_of_bounds;
                },
                invalid_count);
            particles.destroy(invalid, invalid_count);
            Kokkos::fence();
            playout.update(particles);
        }
        void solve() {
            scatterBunch();
            field_solver.solve();
            updateBunch(
                field_solver.dt * steps_taken,
                /*no external field*/ [] KOKKOS_FUNCTION(vector_type /*pos*/, scalar /*time*/) {
                    return Kokkos::pair<vector_type, vector_type>{vector_type(0), vector_type(0)};
                });
            ++steps_taken;
        }
        template <typename callable>
        void solve(callable external_field) {
            scatterBunch();
            field_solver.solve();
            // std::cout << field_solver.dt * steps_taken << "\n";
            updateBunch(field_solver.dt * steps_taken, external_field);
            ++steps_taken;
        }
    };
}
#endif