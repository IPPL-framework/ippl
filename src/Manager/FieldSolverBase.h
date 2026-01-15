#ifndef IPPL_FIELD_SOLVER_BASE_H
#define IPPL_FIELD_SOLVER_BASE_H

#include <memory>

#include "Manager/BaseManager.h"
#include "datatypes.h"

// Define the FieldSolverBase class
namespace ippl {
    template <typename T, unsigned Dim>
    class FieldSolverBase {
    private:
        std::string stype_m;
        Solver_t<T, Dim> solver_m;

    public:
        FieldSolverBase(std::string solver)
            : stype_m(solver) {}

        virtual void initSolver() = 0;

        virtual void runSolver() = 0;

        virtual ~FieldSolverBase() = default;

        const std::string& getStype() const { return stype_m; }

        void setStype(const std::string& solver) { stype_m = solver; }

        Solver_t<T, Dim>& getSolver() { return solver_m; }

        void setSolver(Solver_t<T, Dim>& solver) { solver_m = solver; }
    };
}  // namespace ippl
#endif
