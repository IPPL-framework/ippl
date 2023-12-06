//
// Class Multigrid
//      Multigrid Solver for Linear System of Equations
//

#ifndef IPPL_MULTIGRID_H
#define IPPL_MULTIGRID_H

#include "SolverAlgorithm.h"
#include "Preconditioner.h"
#include "PCG.h"

namespace ippl {
    template <typename OpRet, typename FieldLHS, typename FieldRHS = FieldLHS>
    class Multigrid : public SolverAlgorithm<FieldLHS, FieldRHS> {
        using Base = SolverAlgorithm<FieldLHS, FieldRHS>;
        typedef typename Base::lhs_type::value_type T;

    public:
        using typename Base::lhs_type, typename Base::rhs_type;
        static constexpr unsigned Dim = rhs_type::dim;
        using mesh_type = typename lhs_type::Mesh_t;
        using layout_type = typename lhs_type::Layout_t;
        using vector_type = Vector<T , Dim>;
        using bc_type  = BConds<lhs_type, Dim>;

        using operator_type = std::function<OpRet(lhs_type)>;

        Multigrid(){}

        /*!
         * Sets the differential operator
         * @param op A function that returns OpRet and takes a field of the LHS type
         */
        void setOperator(operator_type op) {op_m = std::move(op); }
        /*!
         * Sets the solver at the lowest level
         * @param op A function that returns OpRet and takes a field of the LHS type
         */
        void setCG(const ParameterList& sp , operator_type op) {
            cg_params.merge(sp);
            cg_m.setOperator(op);
        }
        /*!
         * Query how many iterations were required to obtain the solution
         * the last time this solver was used
         * @return Iteration count of last solve
         */
        int getIterationCount() { return iterations_m; }

        template<typename View, typename Idx>
        T restricted_value(const View& old_view , Idx a[Dim] , Idx b[Dim] ,typename layout_type::NDIndex_t ldom_old, int nghost, unsigned counter=0) const {
            if (counter == Dim){
                //Compute local indexes
                const int i = a[0] - ldom_old[0].first() + nghost;
                const int j = a[1] - ldom_old[1].first() + nghost;
                const int k = a[2] - ldom_old[2].first() + nghost;
                return old_view(i,j,k);
            }
            Idx a1[Dim];
            Idx a2[Dim];
            for(unsigned int d = 0; d<counter; ++d){
                a1[d] = a[d];
                a2[d] = a[d];
            }
            a1[counter] = b[counter]*2+1;
            a2[counter] = b[counter]*2-1;
            a[counter] = b[counter]*2;

            return restricted_value(old_view , a , b ,ldom_old,nghost, counter+1)+0.5*(restricted_value(old_view, a1,b,ldom_old,nghost,counter+1) + restricted_value(old_view, a2,b,ldom_old,nghost,counter+1));
        }

        rhs_type restrict(rhs_type& rhs_old){
            // get layout and mesh
            layout_type layout_old = rhs_old.getLayout();
            mesh_type mesh_old     = rhs_old.get_mesh();

            // get mesh spacing, domain and origin
            vector_type h_old                        = mesh_old.getMeshSpacing();
            vector_type origin                       = mesh_old.getOrigin();
            std::cout << "Origin inside restrict : " << origin << std::endl;
            typename layout_type::NDIndex_t domain_old = layout_old.getDomain();
            typename layout_type::NDIndex_t ldom_old = layout_old.getLocalNDIndex();

            //Define the new values
            vector_type h_new;
            e_dim_tag decomp[Dim];
            typename layout_type::NDIndex_t domain_new;

            for (unsigned int i = 0; i < Dim; ++i) {

                // create the doubled domain
                domain_new[i] = Index(domain_old[i].length()/2);

                // define decomposition (parallel / serial)
                decomp[i] = layout_old.getRequestedDistribution(i);

                //define the new mesh_spacing
                h_new[i] = h_old[i] * 2.;
            }

            // create half sized mesh, layout and Field
            mesh_type mesh_new     = mesh_type(domain_new, h_new, origin);
            layout_type layout_new = layout_type(domain_new, decomp);
            lhs_type restricted_lhs(mesh_new, layout_new);

            auto& view    = restricted_lhs.getView();
            const int nghost = restricted_lhs.getNghost();
            const auto ldom = layout_new.getLocalNDIndex();
            std::cout << "Restricted LHS before kokkos loop" <<std::endl;
            ippl::detail::write<T , Dim>(view);
            // COMMUNICATION
            rhs_old.fillHalo();
            BConds <FieldRHS, Dim> &bcField = rhs_old.getFieldBC();
            bcField.apply(rhs_old);
            restricted_lhs.setFieldBC(bcField);

            Kokkos::parallel_for(
                    "Assign lhs field", restricted_lhs.getFieldRangePolicy(),
                    KOKKOS_LAMBDA(const int i, const int j, const int k) {
                // go from local to global indices
                const int ig = i + ldom[0].first() - nghost;
                const int jg = j + ldom[1].first() - nghost;
                const int kg = k + ldom[2].first() - nghost;
                int a[Dim];
                int b[Dim] = {ig,jg,kg};
                T restricted_entry = restricted_value(rhs_old.getView(),a,b,ldom_old,nghost);
                view(i , j, k) = restricted_entry;
            });
            return restricted_lhs;
        }

        template<typename View, typename Idx>
        KOKKOS_FUNCTION T prolongated_value(const View& old_view , Idx a[Dim] , Idx b[Dim] , typename layout_type::NDIndex_t ldom_old, int nghost, unsigned counter=0) const {
            if (counter == Dim){
                //Compute local indexes
                const int i = a[0] - ldom_old[0].first() + nghost;
                const int j = a[1] - ldom_old[1].first() + nghost;
                const int k = a[2] - ldom_old[2].first() + nghost;
                return old_view(i,j,k);
            }
            if (b[counter] % 2){
                //Odd case
                Idx a1[Dim];
                Idx a2[Dim];
                for(unsigned int d = 0; d<counter; ++d){
                    a1[d] = a[d];
                    a2[d] = a[d];
                }
                a1[counter] = (b[counter]+1)/2;
                a2[counter] = (b[counter]-1)/2;
                return 0.5*(prolongated_value(old_view , a1 , b ,ldom_old,nghost, counter+1)+prolongated_value(old_view , a2 , b ,ldom_old,nghost, counter+1));
            }
            a[counter] = b[counter]/2;
            return prolongated_value(old_view , a , b ,ldom_old,nghost, counter+1);
        }

        rhs_type prolongate(rhs_type& rhs_old){

            // get layout and mesh
            layout_type layout_old = rhs_old.getLayout();
            mesh_type mesh_old     = rhs_old.get_mesh();

            // get mesh spacing, domain and origin
            vector_type h_old                        = mesh_old.getMeshSpacing();
            vector_type origin                       = mesh_old.getOrigin();

            typename layout_type::NDIndex_t domain_old = layout_old.getDomain();
            typename layout_type::NDIndex_t ldom_old = layout_old.getLocalNDIndex();

            //Define the new values
            vector_type h_new;
            e_dim_tag decomp[Dim];
            typename layout_type::NDIndex_t domain_new;

            for (unsigned int i = 0; i < Dim; ++i) {

                // create the doubled domain
                domain_new[i] = Index(2*domain_old[i].length());

                // define decomposition (parallel / serial)
                decomp[i] = layout_old.getRequestedDistribution(i);

                //define the new mesh_spacing
                h_new[i] = h_old[i] / 2.;
            }

            // create double sized mesh, layout and Field
            mesh_type mesh_new             = mesh_type(domain_new, h_new, origin);
            layout_type layout_new       = layout_type(domain_new, decomp);
            lhs_type prolongated_lhs(mesh_new, layout_new);

            auto& view    = prolongated_lhs.getView();
            const int nghost = prolongated_lhs.getNghost();
            const auto& ldom = layout_new.getLocalNDIndex();

            // COMMUNICATION
            rhs_old.fillHalo();
            BConds <FieldRHS, Dim> &bcField = rhs_old.getFieldBC();
            bcField.apply(rhs_old);
            prolongated_lhs.setFieldBC(bcField);

            Kokkos::parallel_for(
                    "Assign lhs field", prolongated_lhs.getFieldRangePolicy(),
                    KOKKOS_LAMBDA(const int i, const int j, const int k) {
                // go from local to global indices
                const int ig = i + ldom[0].first() - nghost;
                const int jg = j + ldom[1].first() - nghost;
                const int kg = k + ldom[2].first() - nghost;
                int a[Dim];
                int b[Dim] = {ig,jg,kg};
                T prolongated_entry = prolongated_value(rhs_old.getView(),a,b,ldom_old,nghost);
                view(i , j, k) = prolongated_entry;
            });
            return prolongated_lhs;
        }

        //Implements one V-cycle
        void recursive_step(lhs_type& lhs, rhs_type& rhs, int level){
            if (level == 0){
                std::cout << "Before CG" << std::endl;
                cg_m(lhs , rhs, cg_params);
                std::cout << "After CG" << std::endl;
            }
            else{
                // Pre-smoothening
                std::cout << "Before Pre-smoothing" << std::endl;
                gauss_seidel_sweep(lhs,rhs);
                std::cout << "After Pre-smoothing" << std::endl;
                /*
                auto&& lhs_view = lhs.getView();
                auto&& rhs_view = rhs.getView();
                ippl::detail::write<double , Dim>(lhs_view);
                ippl::detail::write<double , Dim>(rhs_view);
                */
                //Restriction
                rhs_type residual = lhs.deepCopy();
                bc_type bc = lhs.getFieldBC();
                residual.setFieldBC(bc);
                residual = rhs - op_m(lhs);

                std::cout << "Before Restrict" << std::endl;
                auto&& res_view = residual.getView();
                ippl::detail::write<double , Dim>(res_view);
                rhs_type restricted_residual = restrict(residual);

                std::cout << "After Restrict" << std::endl;
                auto&& restricted_view = restricted_residual.getView();
                ippl::detail::write<double , Dim>(restricted_view);
                std::cout << "After view output" << std::endl;

                layout_type& restricted_layout = restricted_residual.getLayout();
                std::cout << restricted_layout << std::endl;
                mesh_type& restricted_mesh = restricted_residual.get_mesh();
                std::cout << restricted_mesh.getOrigin() << std::endl;
                bc_type& restricted_bc = restricted_residual.getFieldBC();
                restricted_bc.write(std::cout);

                std::cout << "Before init lhs" << std::endl;
                rhs_type restricted_lhs(restricted_mesh , restricted_layout);
                std::cout << "After init lhs" << std::endl;
                //restricted_lhs = 0;
                //std::cout << "lhs = 0" << std::endl;
                auto&& tmp = restricted_lhs.getView();
                ippl::detail::write<double , Dim>(tmp);//Looks wrong

                restricted_lhs.setFieldBC(restricted_bc);//Breaks
                std::cout << "After set bc" << std::endl;

                // Recursive Call
                recursive_step(restricted_lhs , restricted_residual, level-1);
                // Prolongation
                std::cout << "Before Prolongate" << std::endl;
                lhs = lhs + prolongate(restricted_lhs);
                std::cout << "After Prolongate" << std::endl;

                //Post-smoothening
                gauss_seidel_sweep(lhs,rhs);
            }
        }
        void operator()(lhs_type& lhs, rhs_type& rhs, const ParameterList& params) override {

            mesh_type mesh     = lhs.get_mesh();
            layout_type layout = lhs.getLayout();

            iterations_m            = 0;
            const int maxIterations = params.get<int>("max_iterations");
            const int levels = params.get<int>("levels");
            const T tolerance = params.get<T>("tolerance") * norm(rhs);

            lhs_type r(mesh, layout);

            bc_type lhsBCs = lhs.getFieldBC();
            bc_type bc;
            lhsBCs.write(std::cout);
            bool allFacesPeriodic = true;
            for (unsigned int i = 0; i < 2 * Dim; ++i) {
                FieldBC bcType = lhsBCs[i]->getBCType();
                if (bcType == PERIODIC_FACE) {
                    // If the LHS has periodic BCs, so does the residue
                    bc[i] = std::make_shared<PeriodicFace<lhs_type>>(i);
                } else if (bcType & CONSTANT_FACE) {
                    // If the LHS has constant BCs, the residue is zero on the BCs
                    // Bitwise AND with CONSTANT_FACE will succeed for ZeroFace or ConstantFace
                    bc[i]            = std::make_shared<ZeroFace<lhs_type>>(i);
                    allFacesPeriodic = false;
                } else {
                    throw IpplException("PCG::operator()",
                                        "Only periodic or constant BCs for LHS supported.");
                    return;
                }
            }
            r = rhs - op_m(lhs);
            r.setFieldBC(bc);
            residueNorm = norm(r); // Update residueNorm
            std::cout << "Before Iterations" << std::endl;
            while (iterations_m < maxIterations && residueNorm > tolerance) {
                recursive_step(lhs , rhs , levels); // Do a V-cycle
                r = rhs - op_m(lhs); // Update residual
                residueNorm = norm(r); // Update residueNorm
                ++iterations_m; // Update iteration count
            }

            if (allFacesPeriodic) {
                T avg = lhs.getVolumeAverage();
                lhs   = lhs - avg;
            }
            std::cout << "After Iterations" << std::endl;

        }

        T getResidue() const { return residueNorm;}

        //Computes diag{A^-1}*u
        FieldLHS Dinv(FieldLHS &u){
            Field res = u.deepCopy();
            auto&& bc = u.getFieldBC();
            res.setFieldBC(bc);
            double sum = 0.0;
            double factor = 1.0;
            mesh_type mesh = u.get_mesh();
            typename mesh_type::vector_type hvector(0);
            for (unsigned d = 0; d < Dim; ++d) {
                hvector[d] = std::pow(mesh.getMeshSpacing(d), 2);
                sum += std::pow(mesh.getMeshSpacing(d), 2) * std::pow(mesh.getMeshSpacing((d + 1) % Dim), 2);
                factor *= hvector[d];
            }
            res = res * 0.5 * factor / sum;
            return res;
        }

        void gauss_seidel_sweep(FieldLHS& x , const FieldRHS &b){
            Field r = b.deepCopy();
            Field r_inner = b.deepCopy();
            Field L = b.deepCopy();
            Field U = b.deepCopy();
            bc_type bc = x.getFieldBC();
            r_inner.setFieldBC(bc);
            for (unsigned int k=0; k<outerloops_m;++k) {
                U = -upper_laplace(x);
                r = b - U;
                for (unsigned int j = 0; j < innerloops_m; ++j) {
                    L = -lower_laplace(x);
                    r_inner = r - L;
                    x = Dinv(r_inner);
                }
                L = -lower_laplace(x);
                r = b - L;
                for (unsigned int j = 0; j < innerloops_m; ++j) {
                    U = -upper_laplace(x);
                    r_inner = r - U;
                    x = Dinv(r_inner);
                }
            }
        }


    protected:
        operator_type op_m;
        CG<OpRet, OpRet, FieldLHS, FieldRHS> cg_m;
        ParameterList cg_params;
        T residueNorm    = 0;
        int iterations_m = 0;
        unsigned int innerloops_m = 5;
        unsigned int outerloops_m = 5;

    };
} // namespace ippl

#endif //IPPL_MULTIGRID_H