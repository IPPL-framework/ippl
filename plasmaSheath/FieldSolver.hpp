#ifndef IPPL_FIELD_SOLVER_H
#define IPPL_FIELD_SOLVER_H

#include <memory>

#include "Manager/BaseManager.h"
#include "Manager/FieldSolverBase.h"

// Define the FieldSolver class
template <typename T, unsigned Dim>
class FieldSolver : public ippl::FieldSolverBase<T, Dim> {
private:
    Field_t<Dim>* rho_m;
    VField_t<T, Dim>* E_m;
    Field<T, Dim>* phi_m;
    T phiWall_m;  // Dirichlet BC at wall
    std::vector<std::string> preconditioner_params_m;

public:
    FieldSolver(std::string solver, Field_t<Dim>* rho, VField_t<T, Dim>* E, Field<T, Dim>* phi,
                T phiWall, std::vector<std::string> preconditioner_params = {})
        : ippl::FieldSolverBase<T, Dim>(solver)
        , rho_m(rho)
        , E_m(E)
        , phi_m(phi)
        , phiWall_m(phiWall)
        , preconditioner_params_m(preconditioner_params) {
        setPotentialBCs();
    }

    ~FieldSolver() {}

    Field_t<Dim>* getRho() const { return rho_m; }
    void setRho(Field_t<Dim>* rho) { rho_m = rho; }

    VField_t<T, Dim>* getE() const { return E_m; }
    void setE(VField_t<T, Dim>* E) { E_m = E; }

    Field<T, Dim>* getPhi() const { return phi_m; }
    void setPhi(Field<T, Dim>* phi) { phi_m = phi; }

    void setPhiWall(T phiWall) { phiWall_m = phiWall; }

    void initSolver() override {
        Inform m("solver ");
        if (this->getStype() == "CG") {
            initCGSolver();
        } else if (this->getStype() == "PCG") {
            initPCGSolver();
        } else {
            m << "Only CG or PCG supported" << endl;
        }
    }

    void setPotentialBCs() {
        // we are setting Dirichlet BCs for phi
        // phi = 0 at x = 0
        // phi = phi_wall at x = L

        typedef ippl::BConds<Field<T, Dim>, Dim> bc_type;
        bc_type dirichlet;
        for (unsigned int i = 0; i < 2 * Dim; ++i) {
            if (i & 1) {
                dirichlet[i] = std::make_shared<ippl::ConstantFace<Field<T, Dim>>>(i, 0.0);
            } else {
                dirichlet[i] = std::make_shared<ippl::ConstantFace<Field<T, Dim>>>(i, phiWall_m);
            }
        }
        phi_m->setFieldBC(dirichlet);
    }

    void runSolver() override {
        if ((this->getStype() == "CG") || (this->getStype() == "PCG")) {
            CGSolver_t<T, Dim>& solver = std::get<CGSolver_t<T, Dim>>(this->getSolver());
            solver.solve();

            if (ippl::Comm->rank() == 0) {
                std::stringstream fname;
                if (this->getStype() == "CG") {
                    fname << "data/CG_";
                } else {
                    fname << "data_";
                    fname << preconditioner_params_m[0];
                    fname << "/";
                    fname << preconditioner_params_m[0];
                    fname << "_";
                }
                fname << ippl::Comm->size();
                fname << ".csv";

                Inform log(NULL, fname.str().c_str(), Inform::APPEND);
                int iterations = solver.getIterationCount();
                // Assume the dummy solve is the first call
                if (iterations == 0) {
                    log << "residue,iterations" << endl;
                }
                // Don't print the dummy solve
                if (iterations > 0) {
                    log << solver.getResidue() << "," << iterations << endl;
                }
            }
            ippl::Comm->barrier();
        } else {
            throw std::runtime_error("Unknown solver type");
        }
    }

    template <typename Solver>
    void initSolverWithParams(const ippl::ParameterList& sp) {
        this->getSolver().template emplace<Solver>();
        Solver& solver = std::get<Solver>(this->getSolver());

        solver.mergeParameters(sp);

        solver.setRhs(*rho_m);

        // The CG solver computes the potential directly and
        // uses this to get the electric field
        solver.setLhs(*phi_m);
        solver.setGradient(*E_m);
    }

    void initCGSolver() {
        ippl::ParameterList sp;
        sp.add("output_type", CGSolver_t<T, Dim>::GRAD);
        // Increase tolerance in the 1D case
        sp.add("tolerance", 1e-10);

        initSolverWithParams<CGSolver_t<T, Dim>>(sp);
    }

    void initPCGSolver() {
        ippl::ParameterList sp;
        sp.add("solver", "preconditioned");
        sp.add("output_type", CGSolver_t<T, Dim>::GRAD);
        // Increase tolerance in the 1D case
        sp.add("tolerance", 1e-10);

        int arg = 0;

        int gauss_seidel_inner_iterations;
        int gauss_seidel_outer_iterations;
        int newton_level;
        int chebyshev_degree;
        int richardson_iterations;
        int communication = 0;
        double ssor_omega;
        std::string preconditioner_type = "";

        preconditioner_type = preconditioner_params_m[arg++];
        if (preconditioner_type == "newton") {
            newton_level = std::stoi(preconditioner_params_m[arg++]);
        } else if (preconditioner_type == "chebyshev") {
            chebyshev_degree = std::stoi(preconditioner_params_m[arg++]);
        } else if (preconditioner_type == "richardson") {
            richardson_iterations = std::stoi(preconditioner_params_m[arg++]);
            communication         = std::stoi(preconditioner_params_m[arg++]);
        } else if (preconditioner_type == "gauss-seidel") {
            gauss_seidel_inner_iterations = std::stoi(preconditioner_params_m[arg++]);
            gauss_seidel_outer_iterations = std::stoi(preconditioner_params_m[arg++]);
            communication                 = std::stoi(preconditioner_params_m[arg++]);
        } else if (preconditioner_type == "ssor") {
            gauss_seidel_inner_iterations = std::stoi(preconditioner_params_m[arg++]);
            gauss_seidel_outer_iterations = std::stoi(preconditioner_params_m[arg++]);
            ssor_omega                    = std::stod(preconditioner_params_m[arg++]);
        }

        sp.add("preconditioner_type", preconditioner_type);
        sp.add("gauss_seidel_inner_iterations", gauss_seidel_inner_iterations);
        sp.add("gauss_seidel_outer_iterations", gauss_seidel_outer_iterations);
        sp.add("newton_level", newton_level);
        sp.add("chebyshev_degree", chebyshev_degree);
        sp.add("richardson_iterations", richardson_iterations);
        sp.add("communication", communication);
        sp.add("ssor_omega", ssor_omega);

        initSolverWithParams<CGSolver_t<T, Dim>>(sp);
    }
};
#endif
