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
    template <typename OpRet, typename FieldLHS, typename FieldRHS = FieldLHS , int levels = 0>
    class Multigrid : public SolverAlgorithm<FieldLHS, FieldRHS> {
        using Base = SolverAlgorithm<FieldLHS, FieldRHS>;
        typedef typename Base::lhs_type::value_type T;
        constexpr static int levels_m = levels;

    public:
        using typename Base::lhs_type, typename Base::rhs_type;
        static constexpr unsigned Dim = rhs_type::dim;
        using mesh_type = typename lhs_type::Mesh_t;
        using layout_type = typename lhs_type::Layout_t;
        using vector_type = Vector<T , Dim>;
        using bc_type  = BConds<lhs_type, Dim>;

        using operator_type = std::function<OpRet(lhs_type)>;

        Multigrid(lhs_type& lhs , unsigned int innerloops = 5, unsigned int outerloops = 1): innerloops_m(innerloops) ,  outerloops_m(outerloops){
            meshes_m[0]             = lhs.get_mesh();
            layouts_m[0]            = lhs.getLayout();
            origin_m                = meshes_m[0].getOrigin();
            h_m[0]                  = meshes_m[0].getMeshSpacing();
            domains_m[0]            = layouts_m[0].getDomain();
            bc_m                    = lhs.getFieldBC();

            for (unsigned int i = 0; i < Dim; ++i) {
                // define decomposition (parallel / serial)
                decomp_m[i] = lhs.getLayout().getRequestedDistribution(i);
            }


            for (int k=0; k<levels_m; ++k){
                for (unsigned int i = 0; i < Dim; ++i) {

                    // create the doubled domain
                    domains_m[k+1][i] = Index(domains_m[k][i].length()/2);

                    //define the new mesh_spacing
                    h_m[k+1][i] = h_m[k][i] * 2.;
                }

                // create half sized mesh, layout and Field
                meshes_m[k+1]  = mesh_type(domains_m[k+1], h_m[k+1], origin_m);
                layouts_m[k+1] = layout_type(domains_m[k+1], decomp_m);
            }
        }

        /*!
         * Sets the differential operator
         * @param op A function that returns OpRet and takes a field of the LHS type
         */
        void setOperator(operator_type op) {op_m = std::move(op); }
        /*!
         * Sets the solver at the lowest level
         * @param op A function that returns OpRet and takes a field of the LHS type
         */
        void setCG(const ParameterList sp , operator_type op) {
            cg_params.add("gauss_seidel_inner_iterations", 5);
            cg_params.add("gauss_seidel_outer_iterations", 1);
            cg_params.merge(sp);
            cg_m.setOperator(op);
            cg_m.setPreconditioner("gauss-seidel");

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
                const int i = a[0] - ldom_old[0].first();
                const int j = a[1] - ldom_old[1].first();
                const int k = a[2] - ldom_old[2].first();

                return 0.125*old_view(i,j,k);
            }
            Idx a1[Dim];
            Idx a2[Dim];
            Idx a3[Dim];
            for(unsigned int d = 0; d<counter; ++d){
                a1[d] = a[d];
                a2[d] = a[d];
                a3[d] = a[d];
            }
            a1[counter] = b[counter]*2+1;
            a2[counter] = b[counter]*2-1;
            a3[counter] = b[counter]*2;

            return restricted_value(old_view , a3, b ,ldom_old,nghost, counter+1)+0.5*(restricted_value(old_view, a1,b,ldom_old,nghost,counter+1) + restricted_value(old_view, a2,b,ldom_old,nghost,counter+1));
        }

        // level needs to be > 0
        rhs_type restrict(rhs_type& rhs_old,int level){

            lhs_type restricted_lhs(meshes_m[level], layouts_m[level]);
            restricted_lhs.setFieldBC(bc_m);

            auto& view    = restricted_lhs.getView();
            const int nghost = restricted_lhs.getNghost();
            const auto ldom = layouts_m[level].getLocalNDIndex();
            const auto ldom_old = layouts_m[level-1].getLocalNDIndex();

            // COMMUNICATION
            rhs_old.fillHalo();
            bc_m.apply(rhs_old);

            Kokkos::parallel_for(
                    "Assign lhs field", restricted_lhs.getFieldRangePolicy(),
                    KOKKOS_LAMBDA(const int i, const int j, const int k) {
                // go from local to global indices
                const int ig = i + ldom[0].first();
                const int jg = j + ldom[1].first();
                const int kg = k + ldom[2].first();
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
                const int i = a[0] - ldom_old[0].first();
                const int j = a[1] - ldom_old[1].first();
                const int k = a[2] - ldom_old[2].first();
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
            // Even case
            a[counter] = b[counter]/2;
            return prolongated_value(old_view , a , b ,ldom_old,nghost, counter+1);
        }

        rhs_type prolongate(rhs_type& rhs_old, int level){

            lhs_type prolongated_lhs(meshes_m[level], layouts_m[level]);
            prolongated_lhs.setFieldBC(bc_m);

            auto& view    = prolongated_lhs.getView();
            const int nghost = prolongated_lhs.getNghost();
            const auto ldom = layouts_m[level].getLocalNDIndex();
            const auto ldom_old = layouts_m[level+1].getLocalNDIndex();

            // COMMUNICATION
            rhs_old.fillHalo();
            bc_m.apply(rhs_old);

            Kokkos::parallel_for(
                    "Assign lhs field", prolongated_lhs.getFieldRangePolicy(),
                    KOKKOS_LAMBDA(const int i, const int j, const int k) {
                // go from local to global indices
                const int ig = i + ldom[0].first();
                const int jg = j + ldom[1].first();
                const int kg = k + ldom[2].first();
                int a[Dim];
                int b[Dim] = {ig,jg,kg};
                T prolongated_entry = prolongated_value(rhs_old.getView(),a,b,ldom_old,nghost);
                view(i , j, k) = prolongated_entry;
            });
            return prolongated_lhs;
        }

        //Implements one V-cycle
        void recursive_step(lhs_type& lhs, rhs_type& rhs, int level){
            if (level == levels_m){
                /*
                rhs_type residue = rhs.deepCopy();
                residue = rhs - op_m(lhs);
                int max_iterations = 5;
                int iter = 0;
                double tol = 1e-10 * norm(rhs);
                do{
                    gauss_seidel_sweep(lhs,rhs);
                    residue = rhs - op_m(lhs);
                    //std::cout << "gs-sweeps residue " << norm(residue) << std::endl;
                    iter++;
                } while(norm(residue) > tol && iter<max_iterations);
                */
                cg_m(lhs, rhs , cg_params);
            }
            else{
                // Pre-smoothening
                gauss_seidel_sweep(lhs,rhs);

                //Compute Residual
                rhs_type residual(meshes_m[level] , layouts_m[level]);
                residual.setFieldBC(bc_m);
                residual = rhs - op_m(lhs);

                // Restrict to coarser grid
                rhs_type restricted_residual(meshes_m[level+1] , layouts_m[level+1]);
                restricted_residual = restrict(residual,level+1);

                lhs_type restricted_lhs(meshes_m[level+1] , layouts_m[level+1]);
                restricted_lhs = 0;
                restricted_lhs.setFieldBC(bc_m);

                // Recursive Call
                recursive_step(restricted_lhs , restricted_residual, level+1);
                // Prolongation

                lhs_type prolongated_lhs(meshes_m[level] , layouts_m[level]);
                prolongated_lhs = prolongate(restricted_lhs , level);
                //auto view = prolongated_lhs.getView();
                //ippl::detail::write<T , Dim>(view);
                lhs_type res(meshes_m[level] , layouts_m[level]);
                lhs = lhs + prolongated_lhs;
                res = rhs - op_m(lhs);
                //std::cout << "Residual After prolongation : " << norm(res) << std::endl;

                //Post-smoothening
                gauss_seidel_sweep(lhs,rhs);
            }
        }

        void operator()(lhs_type& lhs, rhs_type& rhs, const ParameterList& params) override {

            mesh_type mesh     = lhs.get_mesh();
            layout_type layout = lhs.getLayout();

            iterations_m            = 0;
            const int maxIterations = params.get<int>("max_iterations");
            const T tolerance = params.get<T>("tolerance") * norm(rhs);

            lhs_type r(mesh, layout);
            bc_type bc = bc_m;
            bool allFacesPeriodic = true;
            for (unsigned int i = 0; i < 2 * Dim; ++i) {
                FieldBC bcType = bc_m[i]->getBCType();
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
            r.setFieldBC(bc_m);
            residueNorm = norm(r); // Update residueNorm
            print();
            while (iterations_m < maxIterations && residueNorm > tolerance) {
                recursive_step(lhs , rhs , 0); // Do a V-cycle
                r = rhs - op_m(lhs); // Update residual
                residueNorm = norm(r); // Update residueNorm
                std::cout << "MG residue " << residueNorm << std::endl;
                ++iterations_m; // Update iteration count
                //ippl::detail::write<T , Dim>(lhs_view);
            }
            if (allFacesPeriodic) {
                T avg = lhs.getVolumeAverage();
                lhs   = lhs - avg;
            }
        }

        T getResidue() const { return residueNorm;}

        //Computes diag{A^-1}*u
        FieldLHS Dinv(FieldLHS &u){
            Field res = u.deepCopy();
            res.setFieldBC(bc_m);
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
            r_inner.setFieldBC(bc_m);
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

        void print(){
            for (int i=0; i<levels+1; ++i){
                std::cout << "Level " << i << std::endl;
                std::cout << "Grid size : " << meshes_m[i].getGridsize() << std::endl;
                std::cout << "Mesh Spacing : " << meshes_m[i].getMeshSpacing() << std::endl;
                layouts_m[i].write(std::cout);

            }
        }
        void test_restrict([[maybe_unused]] const FieldLHS& lhs){
            lhs_type restricted_lhs(meshes_m[1] , layouts_m[1]);
            restricted_lhs = 1.0;
            restricted_lhs.setFieldBC(bc_m);
            lhs_type prolongated_lhs(meshes_m[0] , layouts_m[0]);
            prolongated_lhs = prolongate(restricted_lhs , 0);
            lhs_type error(meshes_m[1] , layouts_m[1]);
            lhs_type restricted_lhs2(meshes_m[1] , layouts_m[1]);
            restricted_lhs2 = restrict(prolongated_lhs, 1);

            error = restricted_lhs - restricted_lhs2;

            auto& res_view = restricted_lhs.getView();
            auto& pro_view = prolongated_lhs.getView();
            auto& res2_view = restricted_lhs2.getView();
            auto& error_view = error.getView();
            ippl::detail::write<T , Dim>(res_view);
            ippl::detail::write<T , Dim>(pro_view);
            ippl::detail::write<T , Dim>(res2_view);
            ippl::detail::write<T , Dim>(error_view);

            std::cout << "Error : " << norm(error) << std::endl;
        }

    protected:
        operator_type op_m;
        CG<OpRet, OpRet, FieldLHS, FieldRHS> cg_m;
        ParameterList cg_params;
        T residueNorm    = 0;
        int iterations_m = 0;
        unsigned int innerloops_m;
        unsigned int outerloops_m;

    private:
        layout_type layouts_m[levels_m+1];
        mesh_type meshes_m[levels_m+1];
        typename layout_type::NDIndex_t domains_m[levels_m+1];
        vector_type origin_m;
        vector_type h_m[levels_m+1];
        e_dim_tag decomp_m[Dim];
        bc_type bc_m;

    };
} // namespace ippl

#endif //IPPL_MULTIGRID_H