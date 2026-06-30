#ifndef IPPL_FIELD_SOLVER_H
#define IPPL_FIELD_SOLVER_H

#include <algorithm>
#include <array>
#include <filesystem>
#include <memory>

#include "LinearSolvers/PCG.h"
#include "LinearSolvers/PreconditionerValidation.h"
#include "Manager/BaseManager.h"
#include "Manager/FieldSolverBase.h"

// Define the FieldSolver class
template <typename T, unsigned Dim>
class FieldSolver : public ippl::FieldSolverBase<T, Dim> {
private:
    struct ParsedPreconditionerParams {
        std::string type          = "";
        bool use_defaults         = true;
        int newton_level          = ippl::pcg_preconditioner_defaults::newton_level;
        int chebyshev_degree      = ippl::pcg_preconditioner_defaults::chebyshev_degree;
        int richardson_iterations = ippl::pcg_preconditioner_defaults::richardson_iterations;
        int gauss_seidel_inner_iterations = ippl::pcg_preconditioner_defaults::gauss_seidel_inner;
        int gauss_seidel_outer_iterations = ippl::pcg_preconditioner_defaults::gauss_seidel_outer;
        int communication                 = ippl::pcg_preconditioner_defaults::communication;
        double ssor_omega                 = ippl::pcg_preconditioner_defaults::ssor_omega;
        // Multigrid preconditioner parameters
        int mg_pre_smooth_iters  = ippl::pcg_preconditioner_defaults::mg_pre_smooth;
        int mg_post_smooth_iters = ippl::pcg_preconditioner_defaults::mg_post_smooth;
        double mg_omega          = ippl::pcg_preconditioner_defaults::mg_omega;
        unsigned mg_min_cells    = ippl::pcg_preconditioner_defaults::mg_min_cells;
    };

    Field_t<Dim>* rho_m;
    VField_t<T, Dim>* E_m;
    Field<T, Dim>* phi_m;
    std::vector<std::string> preconditioner_params_m;

    // Parse the preconditioner params (PCG and FEMPrecon solvers);
    //   [type, optional numeric args...]
    // Type validation and numeric sanitization are delegated to the shared
    // preconditioner validation helper to avoid duplication with Poisson solvers.
    // If the type is known but the numeric args are invalid or the number
    // of arguments is wrong, a warning is printed and default parameters
    // are used.
    ParsedPreconditionerParams parsePreconditionerParams(const std::string& solver_name) const {
        Inform warn("FieldSolver");
        ParsedPreconditionerParams parsed;
        if (preconditioner_params_m.empty()) {
            warn << "No preconditioner type provided for solver " << solver_name
                 << "; using default identity preconditioner." << endl;
            return parsed;
        }

        parsed.type = preconditioner_params_m[0];
        ippl::preconditioner_validation::throwIfUnknownType(
            parsed.type, "FieldSolver::parsePreconditionerParams");

        size_t expected_arg_count = 0;
        if (parsed.type == "newton" || parsed.type == "chebyshev"
            || parsed.type == "richardson_alt") {
            expected_arg_count = 1;
        } else if (parsed.type == "richardson") {
            expected_arg_count = 2;
        } else if (parsed.type == "gauss-seidel" || parsed.type == "ssor") {
            expected_arg_count = 3;
        } else if (parsed.type == "multigrid") {
            expected_arg_count = 4;
        }

        const size_t provided_arg_count = preconditioner_params_m.size() - 1;
        if (provided_arg_count == 0) {
            return parsed;
        }

        auto warnAndUseDefaults = [&](const std::string& detail) {
            warn << "Invalid parameters for preconditioner '" << parsed.type << "' in solver "
                 << solver_name << " (" << detail << "). Using default preconditioner parameters."
                 << endl;
            parsed.use_defaults = true;
        };

        if (provided_arg_count != expected_arg_count) {
            warnAndUseDefaults("expected " + std::to_string(expected_arg_count)
                               + " argument(s), got " + std::to_string(provided_arg_count));
            return parsed;
        }

        try {
            int arg = 1;
            if (parsed.type == "newton") {
                parsed.newton_level = std::stoi(preconditioner_params_m[arg++]);
            } else if (parsed.type == "chebyshev") {
                parsed.chebyshev_degree = std::stoi(preconditioner_params_m[arg++]);
            } else if (parsed.type == "richardson" || parsed.type == "richardson_alt") {
                parsed.richardson_iterations = std::stoi(preconditioner_params_m[arg++]);
                if (parsed.type == "richardson") {
                    parsed.communication = std::stoi(preconditioner_params_m[arg++]);
                }
            } else if (parsed.type == "gauss-seidel") {
                parsed.gauss_seidel_inner_iterations = std::stoi(preconditioner_params_m[arg++]);
                parsed.gauss_seidel_outer_iterations = std::stoi(preconditioner_params_m[arg++]);
                parsed.communication                 = std::stoi(preconditioner_params_m[arg++]);
            } else if (parsed.type == "ssor") {
                parsed.gauss_seidel_inner_iterations = std::stoi(preconditioner_params_m[arg++]);
                parsed.gauss_seidel_outer_iterations = std::stoi(preconditioner_params_m[arg++]);
                parsed.ssor_omega                    = std::stod(preconditioner_params_m[arg++]);
            } else if (parsed.type == "multigrid") {
                parsed.mg_pre_smooth_iters  = std::stoi(preconditioner_params_m[arg++]);
                parsed.mg_post_smooth_iters = std::stoi(preconditioner_params_m[arg++]);
                parsed.mg_omega             = std::stod(preconditioner_params_m[arg++]);
                parsed.mg_min_cells =
                    static_cast<unsigned>(std::stoul(preconditioner_params_m[arg++]));
            }
            parsed.use_defaults = false;
        } catch (const std::exception& ex) {
            warnAndUseDefaults(std::string("failed to parse numeric values: ") + ex.what());
        }

        ippl::preconditioner_validation::sanitizeParams(
            parsed.type, warn, parsed.newton_level, parsed.chebyshev_degree,
            parsed.richardson_iterations, parsed.gauss_seidel_inner_iterations,
            parsed.gauss_seidel_outer_iterations, parsed.ssor_omega, &parsed.communication,
            parsed.mg_pre_smooth_iters, parsed.mg_post_smooth_iters, parsed.mg_omega,
            parsed.mg_min_cells);

        return parsed;
    }

public:
    FieldSolver(std::string solver, Field_t<Dim>* rho, VField_t<T, Dim>* E, Field<T, Dim>* phi,
                std::vector<std::string> preconditioner_params = {})
        : ippl::FieldSolverBase<T, Dim>(solver)
        , rho_m(rho)
        , E_m(E)
        , phi_m(phi)
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

    void initSolver() override {
        Inform m("solver ");
        if (this->getStype() == "FFT") {
            initFFTSolver();
        } else if (this->getStype() == "CG") {
            initCGSolver();
        } else if (this->getStype() == "TG") {
            initTGSolver();
        } else if (this->getStype() == "PCG") {
            initPCGSolver();
        } else if (this->getStype() == "OPEN") {
            initOpenSolver();
        } else if (this->getStype() == "FEM") {
            initFEMSolver();
        } else if (this->getStype() == "FEM_PRECON") {
            initFEMPreconSolver();
        } else {
            m << "No solver matches the argument" << endl;
        }
    }

    void setPotentialBCs() {
        // CG requires explicit periodic boundary conditions while the periodic Poisson solver
        // simply assumes them
        typedef ippl::BConds<Field<T, Dim>, Dim> bc_type;
        if ((this->getStype() == "CG") || (this->getStype() == "PCG") || (this->getStype() == "FEM")
            || (this->getStype() == "FEM_PRECON")) {
            bc_type allPeriodic;
            for (unsigned int i = 0; i < 2 * Dim; ++i) {
                allPeriodic[i] = std::make_shared<ippl::PeriodicFace<Field<T, Dim>>>(i);
            }
            phi_m->setFieldBC(allPeriodic);
            if ((this->getStype() == "FEM") || (this->getStype() == "FEM_PRECON")) {
                rho_m->setFieldBC(allPeriodic);
            }
        }
    }

    void runSolver() override {
        if ((this->getStype() == "CG") || (this->getStype() == "PCG") || (this->getStype() == "FEM")
            || (this->getStype() == "FEM_PRECON")) {
            int iterations = 0;
            int residue    = 0;

            if (this->getStype() == "FEM") {
                FEMSolver_t<T, Dim>& solver = std::get<FEMSolver_t<T, Dim>>(this->getSolver());
                solver.solve();

                iterations = solver.getIterationCount();
                residue    = solver.getResidue();
            } else if (this->getStype() == "FEM_PRECON") {
                FEMPreconSolver_t<T, Dim>& solver =
                    std::get<FEMPreconSolver_t<T, Dim>>(this->getSolver());
                solver.solve();

                iterations = solver.getIterationCount();
                residue    = solver.getResidue();
            } else {
                CGSolver_t<T, Dim>& solver = std::get<CGSolver_t<T, Dim>>(this->getSolver());
                solver.solve();

                iterations = solver.getIterationCount();
                residue    = solver.getResidue();
            }

            if (ippl::Comm->rank() == 0) {
                std::filesystem::create_directory("data_CG");
                std::stringstream fname;
                if ((this->getStype() == "CG") || (this->getStype() == "FEM")
                    || (this->getStype() == "FEM_PRECON")) {
                    fname << "data_CG/CG_";
                } else {
                    fname << "data_CG/";
                    fname << preconditioner_params_m[0];
                    fname << "_";
                }
                fname << ippl::Comm->size();
                fname << ".csv";

                Inform log(NULL, fname.str().c_str(), Inform::APPEND);
                // Assume the dummy solve is the first call
                if (iterations == 0) {
                    log << "residue,iterations" << endl;
                }
                // Don't print the dummy solve
                if (iterations > 0) {
                    log << residue << "," << iterations << endl;
                }
            }
            ippl::Comm->barrier();
        } else if (this->getStype() == "FFT") {
            if constexpr (Dim == 2 || Dim == 3) {
                std::get<FFTSolver_t<T, Dim>>(this->getSolver()).solve();
            }
        } else if (this->getStype() == "TG") {
            if constexpr (Dim == 3) {
                std::get<FFTTruncatedGreenSolver_t<T, Dim>>(this->getSolver()).solve();
            }
        } else if (this->getStype() == "OPEN") {
            if constexpr (Dim == 3) {
                std::get<OpenSolver_t<T, Dim>>(this->getSolver()).solve();
            }
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

        if constexpr ((std::is_same_v<Solver, CGSolver_t<T, Dim>>)
                      || (std::is_same_v<Solver, FEMSolver_t<T, Dim>>)
                      || (std::is_same_v<Solver, FEMPreconSolver_t<T, Dim>>)) {
            // The CG solver and FEMPoissonSolver compute the potential
            // directly and use this to get the electric field
            solver.setLhs(*phi_m);
            solver.setGradient(*E_m);
        } else {
            // The periodic Poisson solver, Open boundaries solver,
            // and the TG solver compute the electric field directly
            solver.setLhs(*E_m);
        }
    }

    void initFFTSolver() {
        if constexpr (Dim == 2 || Dim == 3) {
            ippl::ParameterList sp;
            sp.add("output_type", FFTSolver_t<T, Dim>::GRAD);
            sp.add("use_heffte_defaults", false);
            sp.add("use_pencils", true);
            sp.add("use_reorder", false);
            sp.add("use_gpu_aware", true);
            sp.add("comm", ippl::p2p_pl);
            sp.add("r2c_direction", 0);

            initSolverWithParams<FFTSolver_t<T, Dim>>(sp);
        } else {
            throw std::runtime_error("Unsupported dimensionality for FFT solver");
        }
    }

    void initCGSolver() {
        ippl::ParameterList sp;
        sp.add("output_type", CGSolver_t<T, Dim>::GRAD);
        // Increase tolerance in the 1D case
        sp.add("tolerance", 1e-4);

        initSolverWithParams<CGSolver_t<T, Dim>>(sp);
    }

    void initPCGSolver() {
        ippl::ParameterList sp;
        sp.add("solver", "preconditioned");
        sp.add("output_type", CGSolver_t<T, Dim>::GRAD);
        // Increase tolerance in the 1D case
        sp.add("tolerance", 1e-4);
        const auto parsed = parsePreconditionerParams("PCG");

        sp.add("preconditioner_type", parsed.type);
        sp.add("gauss_seidel_inner_iterations", parsed.gauss_seidel_inner_iterations);
        sp.add("gauss_seidel_outer_iterations", parsed.gauss_seidel_outer_iterations);
        sp.add("newton_level", parsed.newton_level);
        sp.add("chebyshev_degree", parsed.chebyshev_degree);
        sp.add("richardson_iterations", parsed.richardson_iterations);
        sp.add("communication", parsed.communication);
        sp.add("ssor_omega", parsed.ssor_omega);
        sp.add("mg_pre_smooth_iters", parsed.mg_pre_smooth_iters);
        sp.add("mg_post_smooth_iters", parsed.mg_post_smooth_iters);
        sp.add("mg_omega", parsed.mg_omega);
        sp.add("min_cells_per_rank_per_dim", parsed.mg_min_cells);

        initSolverWithParams<CGSolver_t<T, Dim>>(sp);
    }

    void initFEMSolver() {
        ippl::ParameterList sp;
        sp.add("output_type", FEMSolver_t<T, Dim>::SOL);
        sp.add("tolerance", 1e-4);

        initSolverWithParams<FEMSolver_t<T, Dim>>(sp);
    }

    void initFEMPreconSolver() {
        ippl::ParameterList sp;
        sp.add("solver", "preconditioned");
        sp.add("output_type", FEMPreconSolver_t<T, Dim>::SOL);
        sp.add("tolerance", 1e-4);
        const auto parsed = parsePreconditionerParams("FEM_PRECON");

        sp.add("preconditioner_type", parsed.type);
        sp.add("gauss_seidel_inner_iterations", parsed.gauss_seidel_inner_iterations);
        sp.add("gauss_seidel_outer_iterations", parsed.gauss_seidel_outer_iterations);
        sp.add("newton_level", parsed.newton_level);
        sp.add("chebyshev_degree", parsed.chebyshev_degree);
        sp.add("richardson_iterations", parsed.richardson_iterations);
        sp.add("communication", parsed.communication);
        sp.add("ssor_omega", parsed.ssor_omega);

        initSolverWithParams<FEMPreconSolver_t<T, Dim>>(sp);
    }

    void initTGSolver() {
        if constexpr (Dim == 3) {
            ippl::ParameterList sp;
            sp.add("output_type", FFTTruncatedGreenSolver_t<T, Dim>::GRAD);
            sp.add("use_heffte_defaults", false);
            sp.add("use_pencils", true);
            sp.add("use_reorder", false);
            sp.add("use_gpu_aware", true);
            sp.add("comm", ippl::p2p_pl);
            sp.add("r2c_direction", 0);

            initSolverWithParams<FFTTruncatedGreenSolver_t<T, Dim>>(sp);
        } else {
            throw std::runtime_error("Unsupported dimensionality for TG solver");
        }
    }

    void initOpenSolver() {
        if constexpr (Dim == 3) {
            ippl::ParameterList sp;
            sp.add("output_type", OpenSolver_t<T, Dim>::GRAD);
            sp.add("use_heffte_defaults", false);
            sp.add("use_pencils", true);
            sp.add("use_reorder", false);
            sp.add("use_gpu_aware", true);
            sp.add("comm", ippl::p2p_pl);
            sp.add("r2c_direction", 0);
            sp.add("algorithm", OpenSolver_t<T, Dim>::HOCKNEY);

            initSolverWithParams<OpenSolver_t<T, Dim>>(sp);
        } else {
            throw std::runtime_error("Unsupported dimensionality for OPEN solver");
        }
    }

    auto& getSpace() {
        if (this->getStype() == "FEM") {
            return std::get<FEMSolver_t<T, Dim>>(this->getSolver()).getSpace();
        } else if (this->getStype() == "FEM_PRECON") {
            return std::get<FEMPreconSolver_t<T, Dim>>(this->getSolver()).getSpace();
        } else {
            throw std::runtime_error("getSpace() called on non-FEM solver");
        }
    }
};
#endif
