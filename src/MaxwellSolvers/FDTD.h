#ifndef IPPL_FDTD_H
#define IPPL_FDTD_H
#include <cstddef>
using std::size_t;
#include "Types/Vector.h"



#include "FieldLayout/FieldLayout.h"
#include "Meshes/UniformCartesian.h"
#include "MaxwellSolvers/Maxwell.h"
#include "Particle/ParticleBase.h"
#include "MaxwellSolvers/AbsorbingBC.h"
constexpr double sqrt_4pi = 3.54490770181103205459;
constexpr double alpha_scaling_factor = 1e30;
constexpr double unit_length_in_meters = 1.616255e-35 * alpha_scaling_factor;
constexpr double unit_charge_in_electron_charges = 11.70623710394218618969 / sqrt_4pi;
constexpr double unit_time_in_seconds = 5.391247e-44 * alpha_scaling_factor;
constexpr double electron_mass_in_kg = 9.1093837015e-31;
constexpr double unit_mass_in_kg = 2.176434e-8 / alpha_scaling_factor;
constexpr double unit_energy_in_joule = unit_mass_in_kg * unit_length_in_meters * unit_length_in_meters / (unit_time_in_seconds * unit_time_in_seconds);
constexpr double kg_in_unit_masses = 1.0 / unit_mass_in_kg;
constexpr double meter_in_unit_lengths = 1.0 / unit_length_in_meters;
constexpr double electron_charge_in_coulombs = 1.602176634e-19;
constexpr double coulomb_in_electron_charges = 1.0 / electron_charge_in_coulombs;

constexpr double electron_charge_in_unit_charges = 1.0 / unit_charge_in_electron_charges;
constexpr double second_in_unit_times = 1.0 / unit_time_in_seconds;
constexpr double electron_mass_in_unit_masses = electron_mass_in_kg * kg_in_unit_masses;
constexpr double unit_force_in_newtons = unit_mass_in_kg * unit_length_in_meters / (unit_time_in_seconds * unit_time_in_seconds);

constexpr double coulomb_in_unit_charges = coulomb_in_electron_charges * electron_charge_in_unit_charges;
constexpr double unit_voltage_in_volts = unit_energy_in_joule * coulomb_in_unit_charges;
constexpr double unit_charges_in_coulomb = 1.0 / coulomb_in_unit_charges;
constexpr double unit_current_in_amperes = unit_charges_in_coulomb / unit_time_in_seconds;
constexpr double ampere_in_unit_currents = 1.0 / unit_current_in_amperes;
constexpr double unit_current_length_in_ampere_meters = unit_current_in_amperes * unit_length_in_meters;
constexpr double unit_magnetic_fluxdensity_in_tesla = unit_voltage_in_volts * unit_time_in_seconds / (unit_length_in_meters * unit_length_in_meters);
constexpr double unit_electric_fieldstrength_in_voltpermeters = (unit_voltage_in_volts / unit_length_in_meters);
constexpr double voltpermeter_in_unit_fieldstrengths = 1.0 / unit_electric_fieldstrength_in_voltpermeters;
constexpr double unit_powerdensity_in_watt_per_square_meter = 1.389e122 / (alpha_scaling_factor * alpha_scaling_factor * alpha_scaling_factor * alpha_scaling_factor);
constexpr double volts_in_unit_voltages = 1.0 / unit_voltage_in_volts;
constexpr double epsilon0_in_si = unit_current_in_amperes * unit_time_in_seconds / (unit_voltage_in_volts * unit_length_in_meters);
constexpr double mu0_in_si = unit_force_in_newtons / (unit_current_in_amperes * unit_current_in_amperes);
constexpr double G = unit_length_in_meters * unit_length_in_meters * unit_length_in_meters / (unit_mass_in_kg * unit_time_in_seconds * unit_time_in_seconds);
constexpr double verification_gravity = unit_mass_in_kg * unit_mass_in_kg / (unit_length_in_meters * unit_length_in_meters) * G;
constexpr double verification_coulomb = (unit_charges_in_coulomb * unit_charges_in_coulomb / (unit_length_in_meters * unit_length_in_meters) * (1.0 / (epsilon0_in_si))) / unit_force_in_newtons;

constexpr double sq(double x){
    return x * x;
}
constexpr float sq(float x){
    return x * x;
}
#define assert_isreal(X) assert(!Kokkos::isnan(X) && !Kokkos::isinf(X))
namespace ippl {
    enum fdtd_bc{
        periodic, absorbing
    };

    template <typename EMField, typename FourField, fdtd_bc boundary_conditions = periodic>
    class StandardFDTDSolver : Maxwell<EMField, FourField> {
        public:
        constexpr static unsigned Dim = EMField::dim;
        using scalar = typename EMField::value_type::value_type;
        using Vector_t = Vector<typename EMField::value_type::value_type, Dim>;
        using FourVector_t = typename FourField::value_type;
        StandardFDTDSolver(FourField& source, EMField& E, EMField& B) {
            Maxwell<EMField, FourField>::setSource(source);
            Maxwell<EMField, FourField>::setEMFields(E, B);
            initialize();
        }
        virtual void solve()override{
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
            A_n  .setFieldBC(vector_bcs);
            A_np1.setFieldBC(vector_bcs);
            A_nm1.setFieldBC(vector_bcs);
        }


        FourField A_n;
        FourField A_np1;
        FourField A_nm1;
        scalar dt;

        private:
        void timeShift(){

            //Look into this, maybe cyclic swap is better
            Kokkos::deep_copy(this->A_nm1.getView(), this->A_n.getView());
            Kokkos::deep_copy(this->A_n.getView(), this->A_np1.getView());
        }
        void applyBCs(){
            if constexpr(boundary_conditions == periodic){
                A_n.getFieldBC().apply(A_n);
                A_nm1.getFieldBC().apply(A_nm1);
                A_np1.getFieldBC().apply(A_np1);
            }
            else{
                Vector<uint32_t, Dim> true_nr = nr_m;
                true_nr += (A_n.getNghost() * 2);
                second_order_mur_boundary_conditions bcs{};
                bcs.apply(A_n, A_nm1, A_np1, this->dt, true_nr, layout_mp->getLocalNDIndex());
            }
        }
        public:
        void step(){
            const auto& ldom    = layout_mp->getLocalNDIndex();
            const int nghost    = A_n.getNghost();
            const auto aview    = A_n  .getView();
            const auto anp1view = A_np1.getView();
            const auto anm1view = A_nm1.getView();
            const auto source_view = Maxwell<EMField, FourField>::source_mp->getView();

            const scalar a1 = scalar(2) * (scalar(1) - sq(dt / hr_m[0]) - sq(dt / hr_m[1]) - sq(dt / hr_m[2]));
            const scalar a2 = sq(dt / hr_m[0]);
            const scalar a4 = sq(dt / hr_m[1]);
            const scalar a6 = sq(dt / hr_m[2]);
            const scalar a8 = sq(dt);
            Vector<uint32_t, Dim> true_nr = nr_m;
            true_nr += (nghost * 2);
            constexpr uint32_t one_if_absorbing_otherwise_0 = boundary_conditions == absorbing ? 1 : 0;
            Kokkos::parallel_for(
            "Four potential update", ippl::getRangePolicy(aview, nghost),
            KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                // global indices
                const uint32_t ig = i + ldom.first()[0];
                const uint32_t jg = j + ldom.first()[1];
                const uint32_t kg = k + ldom.first()[2];
                uint32_t val = uint32_t(ig == one_if_absorbing_otherwise_0) + (uint32_t(jg == one_if_absorbing_otherwise_0) << 1) + (uint32_t(kg == one_if_absorbing_otherwise_0) << 2)
                             + (uint32_t(ig == true_nr[0] - one_if_absorbing_otherwise_0 - 1) << 3) + (uint32_t(jg == true_nr[1] - one_if_absorbing_otherwise_0 - 1) << 4) + (uint32_t(kg == true_nr[2] - one_if_absorbing_otherwise_0 - 1) << 5);
                
                if(val == 0){
                    FourVector_t interior = -anm1view(i, j, k) + a1 * aview(i, j, k)
                                      + a2 * (aview(i + 1, j, k) + aview(i - 1, j, k))
                                      + a4 * (aview(i, j + 1, k) + aview(i, j - 1, k))
                                      + a6 * (aview(i, j, k + 1) + aview(i, j, k - 1))
                                      + a8 * (-source_view(i, j, k));
                    anp1view(i, j, k) = interior;
                }
                else{
                    //std::cout << i << ", " << j << ", " << k << "\n";
                }
            });
            Kokkos::fence();
            applyBCs();
            A_np1.fillHalo();
        }
        void evaluate_EB(){
            ippl::Vector<scalar, 3> inverse_2_spacing = ippl::Vector<scalar, 3>(0.5) / hr_m;
            const scalar idt = scalar(1.0) / dt;
            auto A_np1 = this->A_np1.getView(), A_n = this->A_n.getView(), A_nm1 = this->A_nm1.getView();
            auto source = Maxwell<EMField, FourField>::source_mp->getView();
            auto Eview  = Maxwell<EMField, FourField>::En_mp->getView();
            auto Bview  = Maxwell<EMField, FourField>::Bn_mp->getView();

            Kokkos::parallel_for(this->A_n.getFieldRangePolicy(), KOKKOS_LAMBDA(size_t i, size_t j, size_t k){
                ippl::Vector<scalar, 3> dAdt = (A_n(i, j, k).template tail<3>() - A_nm1(i, j, k).template tail<3>()) * idt;
                ippl::Vector<scalar, 4> dAdx = (A_n(i + 1, j, k) - A_n(i - 1, j, k)) * inverse_2_spacing[0];
                ippl::Vector<scalar, 4> dAdy = (A_n(i, j + 1, k) - A_n(i, j - 1, k)) * inverse_2_spacing[1];
                ippl::Vector<scalar, 4> dAdz = (A_n(i, j, k + 1) - A_n(i, j, k - 1)) * inverse_2_spacing[2];

                ippl::Vector<scalar, 3> grad_phi{
                    dAdx[0], dAdy[0], dAdz[0]
                };
                ippl::Vector<scalar, 3> curlA{
                    dAdy[3] - dAdz[2],
                    dAdz[1] - dAdx[3],
                    dAdx[2] - dAdy[1],
                };
                Eview(i,j,k) = -dAdt - grad_phi;
                Bview(i,j,k) = curlA;
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
            hr_m     = mesh_mp->getMeshSpacing();
            dt = hr_m[0] / 2;
            for (unsigned int i = 0; i < Dim; ++i){
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
        using scalar = typename EMField::value_type::value_type;
        using Vector_t = Vector<typename EMField::value_type::value_type, Dim>;
        using FourVector_t = typename FourField::value_type;
        NonStandardFDTDSolver(FourField& source, EMField& E, EMField& B) {
            auto hx = source.get_mesh().getMeshSpacing();
            if((hx[2] / hx[0]) * (hx[2] / hx[0]) + (hx[2] / hx[1]) * (hx[2] / hx[1]) >= 1){
                std::cerr << "Dispersion-free CFL condition not satisfiable\n";
                std::abort();
            }
            Maxwell<EMField, FourField>::setSource(source);
            Maxwell<EMField, FourField>::setEMFields(E, B);
            initialize();
        }
        virtual void solve()override{
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
            A_n  .setFieldBC(vector_bcs);
            A_np1.setFieldBC(vector_bcs);
            A_nm1.setFieldBC(vector_bcs);
        }


        private:
        void timeShift(){

            //Look into this, maybe cyclic swap is better
            Kokkos::deep_copy(this->A_nm1.getView(), this->A_n.getView());
            Kokkos::deep_copy(this->A_n.getView(), this->A_np1.getView());
        }

        void applyBCs(){
            if constexpr(boundary_conditions == periodic){
                A_n.getFieldBC().apply(A_n);
                A_nm1.getFieldBC().apply(A_nm1);
                A_np1.getFieldBC().apply(A_np1);
            }
            else{
                Vector<uint32_t, Dim> true_nr = nr_m;
                true_nr += (A_n.getNghost() * 2);
                second_order_mur_boundary_conditions bcs{};
                bcs.apply(A_n, A_nm1, A_np1, this->dt, true_nr, layout_mp->getLocalNDIndex());
            }
        }
        template<typename scalar>
        struct nondispersive{
            scalar a1;
            scalar a2;
            scalar a4;
            scalar a6;
            scalar a8;
        };
        public:
        void step(){
            const auto& ldom    = layout_mp->getLocalNDIndex();
            const int nghost    = A_n.getNghost();
            const auto aview    = A_n  .getView();
            const auto anp1view = A_np1.getView();
            const auto anm1view = A_nm1.getView();
            const auto source_view = Maxwell<EMField, FourField>::source_mp->getView();

            const scalar calA = 0.25 * (1 + 0.02 / (sq(hr_m[2] / hr_m[0]) + sq(hr_m[2] / hr_m[1])));
            nondispersive<scalar> ndisp{
                .a1 = 2 * (1 - (1 - 2 * calA) * sq(dt / hr_m[0]) - (1 - 2*calA) * sq(dt / hr_m[1]) - sq(dt / hr_m[2])),
                .a2 = sq(dt / hr_m[0]),
                .a4 = sq(dt / hr_m[1]),
                .a6 = sq(dt / hr_m[2]) - 2 * calA * sq(dt / hr_m[0]) - 2 * calA * sq(dt / hr_m[1]),
                .a8 = sq(dt)
            };
            Vector<uint32_t, Dim> true_nr = nr_m;
            true_nr += (nghost * 2);
            constexpr uint32_t one_if_absorbing_otherwise_0 = boundary_conditions == absorbing ? 1 : 0;
            Kokkos::parallel_for(ippl::getRangePolicy(aview, nghost), KOKKOS_LAMBDA(size_t i, size_t j, size_t k){
                uint32_t ig = i + ldom.first()[0];
                uint32_t jg = j + ldom.first()[1];
                uint32_t kg = k + ldom.first()[2];
                uint32_t val = uint32_t(ig == one_if_absorbing_otherwise_0) + (uint32_t(jg == one_if_absorbing_otherwise_0) << 1) + (uint32_t(kg == one_if_absorbing_otherwise_0) << 2)
                             + (uint32_t(ig == true_nr[0] - one_if_absorbing_otherwise_0 - 1) << 3) + (uint32_t(jg == true_nr[1] - one_if_absorbing_otherwise_0 - 1) << 4) + (uint32_t(kg == true_nr[2] - one_if_absorbing_otherwise_0 - 1) << 5);
                
                if(!val){
                    anp1view(i, j, k) = -anm1view(i,j,k)
                            + ndisp.a1 * aview(i,j,k)
                            + ndisp.a2 * (calA * aview(i + 1, j, k - 1) + (1 - 2 * calA) * aview(i + 1, j, k) + calA * aview(i + 1, j, k + 1))
                            + ndisp.a2 * (calA * aview(i - 1, j, k - 1) + (1 - 2 * calA) * aview(i - 1, j, k) + calA * aview(i - 1, j, k + 1))
                            + ndisp.a4 * (calA * aview(i, j + 1, k - 1) + (1 - 2 * calA) * aview(i, j + 1, k) + calA * aview(i, j + 1, k + 1))
                            + ndisp.a4 * (calA * aview(i, j - 1, k - 1) + (1 - 2 * calA) * aview(i, j - 1, k) + calA * aview(i, j - 1, k + 1))
                            + ndisp.a6 * aview(i, j, k + 1) + ndisp.a6 * aview(i, j, k - 1) + ndisp.a8 * source_view(i, j, k);
                }
            });
            Kokkos::fence();
            A_np1.fillHalo();
            applyBCs();
        }
        void evaluate_EB(){
            *(Maxwell<EMField, FourField>::En_mp) = typename EMField::value_type(0);
            *(Maxwell<EMField, FourField>::Bn_mp) = typename EMField::value_type(0);
            ippl::Vector<scalar, 3> inverse_2_spacing = ippl::Vector<scalar, 3>(0.5) / hr_m;
            const scalar idt = scalar(1.0) / dt;
            auto A_np1 = this->A_np1.getView(), A_n = this->A_n.getView(), A_nm1 = this->A_nm1.getView();
            auto source = Maxwell<EMField, FourField>::source_mp->getView();
            auto Eview  = Maxwell<EMField, FourField>::En_mp->getView();
            auto Bview  = Maxwell<EMField, FourField>::Bn_mp->getView();

            Kokkos::parallel_for(this->A_n.getFieldRangePolicy(), KOKKOS_LAMBDA(size_t i, size_t j, size_t k){
                ippl::Vector<scalar, 3> dAdt = (A_n(i, j, k).template tail<3>() - A_nm1(i, j, k).template tail<3>()) * idt;
                ippl::Vector<scalar, 4> dAdx = (A_n(i + 1, j, k) - A_n(i - 1, j, k)) * inverse_2_spacing[0];
                ippl::Vector<scalar, 4> dAdy = (A_n(i, j + 1, k) - A_n(i, j - 1, k)) * inverse_2_spacing[1];
                ippl::Vector<scalar, 4> dAdz = (A_n(i, j, k + 1) - A_n(i, j, k - 1)) * inverse_2_spacing[2];

                ippl::Vector<scalar, 3> grad_phi{
                    dAdx[0], dAdy[0], dAdz[0]
                };
                ippl::Vector<scalar, 3> curlA{
                    dAdy[3] - dAdz[2],
                    dAdz[1] - dAdx[3],
                    dAdx[2] - dAdy[1],
                };
                Eview(i,j,k) = -dAdt - grad_phi;
                Bview(i,j,k) = curlA;
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
            dt = hr_m[2];
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
    
    template <typename _scalar, class PLayout>
    struct  Bunch : public ippl::ParticleBase<PLayout> {
        using scalar = _scalar;

        // Constructor for the Bunch class, taking a PLayout reference
        Bunch(PLayout& playout)
            : ippl::ParticleBase<PLayout>(playout) {
            // Add attributes to the particle bunch
            this->addAttribute(Q);          // Charge attribute
            this->addAttribute(mass);       // Mass attribute
            this->addAttribute(gamma_beta); // Gamma-beta attribute (product of relativistiv gamma and beta)
            this->addAttribute(R_np1);      // Position attribute for the next time step
            this->addAttribute(R_nm1);      // Position attribute for the next time step
            this->addAttribute(E_gather);   // Electric field attribute for particle gathering
            this->addAttribute(B_gather);   // Magnedit field attribute for particle gathering
        }

        // Destructor for the Bunch class
        ~Bunch() {}

        // Define container types for various attributes
        using charge_container_type   = ippl::ParticleAttrib<scalar>;
        using velocity_container_type = ippl::ParticleAttrib<ippl::Vector<scalar, 3>>;
        using vector_container_type   = ippl::ParticleAttrib<ippl::Vector<scalar, 3>>;
        using vector4_container_type   = ippl::ParticleAttrib<ippl::Vector<scalar, 4>>;

        // Declare instances of the attribute containers
        charge_container_type Q;          // Charge container
        charge_container_type mass;       // Mass container
        velocity_container_type gamma_beta; // Gamma-beta container
        typename ippl::ParticleBase<PLayout>::particle_position_type R_np1; // Position container for the next time step
        typename ippl::ParticleBase<PLayout>::particle_position_type R_nm1; // Position container for the previous time step
        ippl::ParticleAttrib<ippl::Vector<scalar, 3>> E_gather;   // Electric field container for particle gathering
        ippl::ParticleAttrib<ippl::Vector<scalar, 3>> B_gather;   // Magnetio field container for particle gathering

    };
    template<typename T>
    KOKKOS_INLINE_FUNCTION Kokkos::pair<ippl::Vector<int, 3>, ippl::Vector<T, 3>> gridCoordinatesOf(const ippl::Vector<T, 3> hr, const ippl::Vector<T, 3> origin, ippl::Vector<T, 3> pos){
        //return pear<ippl::Vector<int, 3>, ippl::Vector<T, 3>>{ippl::Vector<int, 3>{5,5,5}, ippl::Vector<T, 3>{0,0,0}};
        //printf("%.10e, %.10e, %.10e\n", (inverse_spacing * spacing)[0], (inverse_spacing * spacing)[1], (inverse_spacing * spacing)[2]);
        Kokkos::pair<ippl::Vector<int, 3>, ippl::Vector<T, 3>> ret;
        //pos -= spacing * T(0.5);
        ippl::Vector<T, 3> relpos = pos - origin;
        ippl::Vector<T, 3> gridpos = relpos / hr;
        ippl::Vector<int, 3> ipos;
        ipos = gridpos.template cast<int>();
        ippl::Vector<T, 3> fracpos;
        for(unsigned k = 0;k < 3;k++){
            fracpos[k] = gridpos[k] - (int)ipos[k];
        }
        //TODO: NGHOST!!!!!!!
        ipos += ippl::Vector<int, 3>(1);
        ret.first = ipos;
        ret.second = fracpos;
        return ret;
    }
    template<typename view_type, typename scalar>
    KOKKOS_FUNCTION void scatterToGrid(const ippl::NDIndex<3>& ldom, view_type& view, ippl::Vector<scalar, 3> hr, ippl::Vector<scalar, 3> orig, const ippl::Vector<scalar, 3>& pos, const scalar value){
        auto [ipos, fracpos] = gridCoordinatesOf(hr, orig, pos);
        ipos -= ldom.first();
        //std::cout << pos << " 's scatter args (will have 1 added): " << ipos << "\n";
        if(
            ipos[0] < 0
            ||ipos[1] < 0
            ||ipos[2] < 0
            ||size_t(ipos[0]) >= view.extent(0) - 1
            ||size_t(ipos[1]) >= view.extent(1) - 1
            ||size_t(ipos[2]) >= view.extent(2) - 1
            ||fracpos[0] < 0
            ||fracpos[1] < 0
            ||fracpos[2] < 0
        ){
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

        for(unsigned i = 0;i < 8;i++){
            scalar weight = 1;
            ippl::Vector<int, 3> ipos_l = ipos;
            for(unsigned d = 0;d < 3;d++){
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
    template<typename view_type, typename scalar>
    KOKKOS_FUNCTION void scatterToGrid(const ippl::NDIndex<3>& ldom, view_type& view, ippl::Vector<scalar, 3> hr, ippl::Vector<scalar, 3> orig, const ippl::Vector<scalar, 3>& pos, const ippl::Vector<scalar, 3>& value){
        //std::cout << "Value: " << value << "\n";
        auto [ipos, fracpos] = gridCoordinatesOf(hr, orig, pos);
        ipos -= ldom.first();
        if(
            ipos[0] < 0
            ||ipos[1] < 0
            ||ipos[2] < 0
            ||size_t(ipos[0]) >= view.extent(0) - 1
            ||size_t(ipos[1]) >= view.extent(1) - 1
            ||size_t(ipos[2]) >= view.extent(2) - 1
            ||fracpos[0] < 0
            ||fracpos[1] < 0
            ||fracpos[2] < 0
        ){
            //std::cout << "out of bounds\n";
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

        for(unsigned i = 0;i < 8;i++){
            scalar weight = 1;
            ippl::Vector<int, 3> ipos_l = ipos;
            for(unsigned d = 0;d < 3;d++){
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
    template<typename view_type, typename scalar>
    KOKKOS_INLINE_FUNCTION void scatterLineToGrid(const ippl::NDIndex<3>& ldom, view_type& Jview, ippl::Vector<scalar, 3> hr, ippl::Vector<scalar, 3> origin, const ippl::Vector<scalar, 3>& from, const ippl::Vector<scalar, 3>& to, const scalar factor){ 


        Kokkos::pair<ippl::Vector<int, 3>, ippl::Vector<scalar, 3>> from_grid = gridCoordinatesOf(hr, origin, from);
        Kokkos::pair<ippl::Vector<int, 3>, ippl::Vector<scalar, 3>> to_grid   = gridCoordinatesOf(hr, origin, to  );
        //printf("Scatterdest: %.4e, %.4e, %.4e\n", from_grid.second[0], from_grid.second[1], from_grid.second[2]);
        for(int d = 0;d < 3;d++){
            //if(abs(from_grid.first[d] - to_grid.first[d]) > 1){
            //    std::cout <<abs(from_grid.first[d] - to_grid.first[d]) << " violation " << from_grid.first << " " << to_grid.first << std::endl;
            //}
            //assert(abs(from_grid.first[d] - to_grid.first[d]) <= 1);
        }
        //const uint32_t nghost = g.nghost();
        //from_ipos += ippl::Vector<int, 3>(nghost);
        //to_ipos += ippl::Vector<int, 3>(nghost);

        if(from_grid.first[0] == to_grid.first[0] && from_grid.first[1] == to_grid.first[1] && from_grid.first[2] == to_grid.first[2]){
            scatterToGrid(ldom, Jview, hr, origin, ippl::Vector<scalar, 3>((from + to) * scalar(0.5)), ippl::Vector<scalar, 3>((to - from) * factor));

            return;
        }
        ippl::Vector<scalar, 3> relay;
        const int nghost = 1;
        const ippl::Vector<scalar, 3> orig = origin;
        using Kokkos::max;
        using Kokkos::min;
        for (unsigned int i = 0; i < 3; i++) {
            relay[i] = min(min(from_grid.first[i] - nghost, to_grid.first[i] - nghost) * hr[i] + scalar(1.0) * hr[i] + orig[i],
                           max(max(from_grid.first[i] - nghost, to_grid.first[i] - nghost) * hr[i] + scalar(0.0) * hr[i] + orig[i],
                               scalar(0.5) * (to[i] + from[i])));
        }
        scatterToGrid(ldom, Jview, hr, origin, ippl::Vector<scalar, 3>((from + relay) * scalar(0.5)), ippl::Vector<scalar, 3>((relay - from) * factor));
        scatterToGrid(ldom, Jview, hr, origin, ippl::Vector<scalar, 3>((relay + to) * scalar(0.5))  , ippl::Vector<scalar, 3>((to - relay) * factor));
    }
    template<typename scalar, fdtd_bc boundary_conditions>
    class NSFDSolverWithParticles{
        public:
        constexpr static unsigned dim = 3;
        using vector_type = ippl::Vector<scalar, 3>;
        using vector4_type = ippl::Vector<scalar, 4>;
        using FourField = ippl::Field<vector4_type, dim, ippl::UniformCartesian<scalar, dim>, typename ippl::UniformCartesian<scalar, dim>::DefaultCentering>;
        using ThreeField = ippl::Field<vector_type, dim, ippl::UniformCartesian<scalar, dim>, typename ippl::UniformCartesian<scalar, dim>::DefaultCentering>;
        using playout_type = ParticleSpatialLayout<scalar, 3>;
        using bunch_type = Bunch<scalar, ParticleSpatialLayout<scalar, 3>>;
        using Mesh_t = ippl::UniformCartesian<scalar, dim>;
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
        NSFDSolverWithParticles(FieldLayout<dim>& layout, Mesh_t& mesch, size_t nparticles) : layout_mp(&layout), mesh_mp(&mesch), playout(layout, mesch), particles(playout), J(mesch, layout), E(mesch, layout), B(mesch, layout), field_solver(J, E, B){
            particles.create(nparticles);
            nr_global = ippl::Vector<uint32_t, 3>{
                uint32_t(layout.getDomain()[0].last() - layout.getDomain()[0].first() + 1),
                uint32_t(layout.getDomain()[1].last() - layout.getDomain()[1].first() + 1),
                uint32_t(layout.getDomain()[2].last() - layout.getDomain()[2].first() + 1)
            };
            hr_m = mesh_mp->getMeshSpacing();
            steps_taken = 0;
            //field_solver.setEMFields(E, B);
        }
        template<bool space_charge = true>
        void scatterBunch(){
            //ippl::Vector<scalar, 3>* gammaBeta = this->gammaBeta;
            auto hr_m = mesh_mp->getMeshSpacing();
            const scalar volume = hr_m[0] * hr_m[1] * hr_m[2];
            assert_isreal(volume);
            assert_isreal((scalar(1) / volume));
            J = typename decltype(J)::value_type(0);
            auto Jview = J.getView();
            auto qview = particles.Q.getView();
            auto rview = particles.R.getView();
            auto rm1view = particles.R_nm1.getView();
            auto orig = mesh_mp->getOrigin();
            auto hr = mesh_mp->getMeshSpacing();
            auto dt = field_solver.dt;
            bool sc = space_charge;
            ippl::NDIndex<dim> lDom = layout_mp->getLocalNDIndex();
            Kokkos::parallel_for(particles.getLocalNum(), KOKKOS_LAMBDA(size_t i){
                vector_type pos = rview(i);
                vector_type to = rview(i);
                vector_type from = rm1view(i);
                if(sc){
                    scatterToGrid(lDom, Jview, hr, orig, pos, qview(i) / volume);
                }
                scatterLineToGrid(lDom, Jview, hr, orig, from, to , scalar(qview(i)) / (volume * dt));
            });
            Kokkos::fence();
            J.accumulateHalo();
        }
        template<typename callable>
        void updateBunch(scalar time, callable external_field){
            
            Kokkos::fence();
            auto gbview = particles.gamma_beta.getView();
            auto eview = particles.E_gather.getView();
            auto bview = particles.B_gather.getView();
            auto qview = particles.Q.getView();
            auto mview = particles.mass.getView();
            auto rview = particles.R.getView();
            auto rm1view = particles.R_nm1.getView();
            auto rp1view = particles.R_np1.getView();
            scalar bunch_dt = field_solver.dt / 3;
            Kokkos::deep_copy(particles.R_nm1.getView(), particles.R.getView());
            E.fillHalo();
            B.fillHalo();
            Kokkos::fence();
            for(int bts = 0;bts < 3;bts++){
                
                particles.E_gather.gather(E, particles.R);
                particles.B_gather.gather(B, particles.R);
                Kokkos::fence();
                Kokkos::parallel_for(particles.getLocalNum(), KOKKOS_LAMBDA(size_t i){
                    const ippl::Vector<scalar, 3> pgammabeta = gbview(i);
                    ippl::Vector<scalar, 3> E_grid = eview(i);
                    ippl::Vector<scalar, 3> B_grid = bview(i);
                    //std::cout << "E_grid: " << E_grid << "\n";
                    //std::cout << "B_grid: " << B_grid << "\n";
                    ippl::Vector<scalar, 3> bunchpos = rview(i);
                    Kokkos::pair<ippl::Vector<scalar, 3>, ippl::Vector<scalar, 3>> external_eb = external_field(bunchpos, time + bunch_dt * bts);
                    
                    ippl::Vector<ippl::Vector<scalar, 3>, 2> EB{
                        ippl::Vector<scalar, 3>(E_grid + external_eb.first), 
                        ippl::Vector<scalar, 3>(B_grid + external_eb.second)
                    };

                    const scalar charge = qview(i);
                    const scalar mass = mview(i);
                    const ippl::Vector<scalar, 3> t1 = pgammabeta + charge * bunch_dt * EB[0] / (scalar(2) * mass);
                    const scalar alpha = charge * bunch_dt / (scalar(2) * mass * Kokkos::sqrt(1 + t1.dot(t1)));
                    const ippl::Vector<scalar, 3> t2 = t1 + alpha * t1.cross(EB[1]);
                    const ippl::Vector<scalar, 3> t3 = t1 + t2.cross(scalar(2) * alpha * (EB[1] / (1.0  + alpha * alpha * (EB[1].dot(EB[1])))));
                    const ippl::Vector<scalar, 3> ngammabeta = t3 + charge * bunch_dt * EB[0] / (scalar(2) * mass);

                    rview(i) = rview(i) + bunch_dt * ngammabeta / (Kokkos::sqrt(scalar(1.0) + (ngammabeta.dot(ngammabeta))));
                    gbview(i) = ngammabeta;
                });
                Kokkos::fence();
            }
            Kokkos::View<bool*> invalid("OOB Particcel", particles.getLocalNum());
            size_t invalid_count = 0;
            auto origo = mesh_mp->getOrigin();
            ippl::Vector<scalar, 3> extenz;//
            extenz[0] = nr_global[0] * hr_m[0];
            extenz[1] = nr_global[1] * hr_m[1];
            extenz[2] = nr_global[2] * hr_m[2];
            Kokkos::parallel_reduce(
                Kokkos::RangePolicy<typename playout_type::RegionLayout_t::view_type::execution_space>(0, particles.getLocalNum()),
                KOKKOS_LAMBDA(size_t i, size_t& ref){
                    bool out_of_bounds = false;
                    ippl::Vector<scalar, dim> ppos = rview(i);
                    for(size_t d = 0;d < dim;d++){
                        out_of_bounds |= (ppos[d] <= origo[d]);
                        out_of_bounds |= (ppos[d] >= origo[d] + extenz[d]); //Check against simulation domain
                    }
                    invalid(i) = out_of_bounds;
                    ref += out_of_bounds;
                }, 
                invalid_count);
            particles.destroy(invalid, invalid_count);
            Kokkos::fence();
            playout.update(particles);
            
        }
        void solve(){
            scatterBunch();
            field_solver.solve();
            updateBunch(field_solver.dt * steps_taken, /*no external field*/[]KOKKOS_FUNCTION(vector_type /*pos*/, scalar /*time*/){return Kokkos::pair<vector_type, vector_type>{
                vector_type(0),
                vector_type(0)
            };});
            ++steps_taken;
        }
        template<typename callable>
        void solve(callable external_field){
            scatterBunch();
            field_solver.solve();
            //std::cout << field_solver.dt * steps_taken << "\n";
            updateBunch(field_solver.dt * steps_taken, external_field);
            ++steps_taken;
        }
    };
}

#endif