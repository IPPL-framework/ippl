/*!
 * @file InferenceBridge.h
 * @brief Calls functions of a Python module from IPPL through an embedded interpreter.
 *
 * Tested by unit_tests/Interoperability/DLPackExport.
 */
#ifndef IPPL_INFERENCE_BRIDGE_H
#define IPPL_INFERENCE_BRIDGE_H

#include "Ippl.h"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <pybind11/pybind11.h>
#include <string>

namespace ippl {
    namespace Interoperability {

        namespace py = pybind11;

        /*!
         * @class InferenceBridge
         * @brief Loads a Python module and calls its functions from C++.
         *
         * Each rank holds its own bridge and module; the bridge itself does no communication,
         * so whether a call is collective depends only on what the Python function does
         * (e.g. torch.distributed). The embedded interpreter (e.g. a py::scoped_interpreter)
         * must be created after ippl::initialize(), outlive the bridge, and be destroyed
         * before ippl::finalize(), so that Python releases all exported Kokkos Views while
         * Kokkos is still alive.
         *
         * Any Python error or C++ exception raised inside the bridge prints the traceback
         * and aborts all ranks through ippl::Comm->abort().
         */
        class InferenceBridge {
        private:
            py::module_ mod_m;

            template <typename F>
            auto guarded(F&& f) -> decltype(f()) {
                try {
                    return f();
                } catch (py::error_already_set& e) {
                    e.restore();
                    PyErr_Print();
                    fflush(stderr);
                } catch (const std::exception& e) {
                    std::cerr << "[InferenceBridge] rank " << ippl::Comm->rank()
                              << ": C++ exception: " << e.what() << std::endl;
                }
                ippl::Comm->abort();
                std::abort();
            }

        public:
            /*!
             * Adds dir to the front of Python's sys.path and imports the module name.
             * @param dir directory containing the module
             * @param name module name, without the .py extension
             */
            InferenceBridge(const std::string& dir, const std::string& name) {
                guarded([&] {
                    py::module_::import("sys").attr("path").attr("insert")(0, dir);
                    mod_m = py::module_::import(name.c_str());
                });
            }

            /*!
             * Calls the module function fn with the given arguments.
             *
             * Arguments are passed positionally and converted to Python objects by
             * pybind11.
             * Use py::arg("name") = value to pass keyword arguments.
             * @param fn name of the module-level function
             * @param args arguments forwarded to the function
             * @returns the function's return value, to be converted with .cast<T>()
             */
            template <typename... Args>
            py::object call(const char* fn, Args&&... args) {
                return guarded([&] {
                    return mod_m.attr(fn)(std::forward<Args>(args)...);
                });
            }

            /*!
             * Calls the module's finalize() function, which the module must define, and
             * releases the module. Call it before the interpreter is destroyed.
             */
            void finalize() {
                guarded([&] {
                    mod_m.attr("finalize")();
                });
                mod_m = py::object();
            }
        };

    }  // namespace Interoperability
}  // namespace ippl

#endif