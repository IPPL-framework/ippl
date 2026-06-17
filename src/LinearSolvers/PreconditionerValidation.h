#ifndef IPPL_PRECONDITIONER_VALIDATION_H
#define IPPL_PRECONDITIONER_VALIDATION_H

#include <cmath>
#include <string>

#include "LinearSolvers/PCG.h"
#include "Utility/Inform.h"
#include "Utility/IpplException.h"

namespace ippl::preconditioner_validation {
    inline void throwIfUnknownType(const std::string& preconditioner_type,
                                   const std::string& caller_name) {
        if (!pcg_preconditioner_defaults::is_valid_type(preconditioner_type)) {
            throw IpplException(
                caller_name.c_str(),
                ("Unknown preconditioner_type '" + preconditioner_type
                 + "'. Supported types: jacobi, newton, chebyshev, richardson, "
                   "richardson_alt, gauss-seidel, ssor")
                    .c_str());
        }
    }

    inline void sanitizeParams(const std::string& preconditioner_type, Inform& warn, int& level,
                               int& degree, int& richardson_iterations, int& inner, int& outer,
                               double& omega, int* communication = nullptr) {
        auto warnAndDefault = [&](const std::string& what, const std::string& value,
                                  const std::string& default_value) {
            warn << "Invalid " << what << "='" << value << "' for preconditioner '"
                 << preconditioner_type << "'. Using default " << what << "=" << default_value << "."
                 << endl;
        };

        if (level <= 0) {
            warnAndDefault("newton_level", std::to_string(level),
                           std::to_string(pcg_preconditioner_defaults::newton_level));
            level = pcg_preconditioner_defaults::newton_level;
        }
        if (degree <= 0) {
            warnAndDefault("chebyshev_degree", std::to_string(degree),
                           std::to_string(pcg_preconditioner_defaults::chebyshev_degree));
            degree = pcg_preconditioner_defaults::chebyshev_degree;
        }
        if (richardson_iterations <= 0) {
            warnAndDefault("richardson_iterations", std::to_string(richardson_iterations),
                           std::to_string(pcg_preconditioner_defaults::richardson_iterations));
            richardson_iterations = pcg_preconditioner_defaults::richardson_iterations;
        }
        if (inner <= 0) {
            warnAndDefault("gauss_seidel_inner_iterations", std::to_string(inner),
                           std::to_string(pcg_preconditioner_defaults::gauss_seidel_inner));
            inner = pcg_preconditioner_defaults::gauss_seidel_inner;
        }
        if (outer <= 0) {
            warnAndDefault("gauss_seidel_outer_iterations", std::to_string(outer),
                           std::to_string(pcg_preconditioner_defaults::gauss_seidel_outer));
            outer = pcg_preconditioner_defaults::gauss_seidel_outer;
        }
        if (!std::isfinite(omega)) {
            warnAndDefault("ssor_omega", std::to_string(omega),
                           std::to_string(pcg_preconditioner_defaults::ssor_omega));
            omega = pcg_preconditioner_defaults::ssor_omega;
        }
        if (communication != nullptr && !(*communication == 0 || *communication == 1)) {
            warnAndDefault("communication", std::to_string(*communication),
                           std::to_string(pcg_preconditioner_defaults::communication));
            *communication = pcg_preconditioner_defaults::communication;
        }
    }
}  // namespace ippl::preconditioner_validation

#endif
