#ifndef INFERENCE_BRIDGE_H
#define INFERENCE_BRIDGE_H

#include <string>
#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <exception>

#include "Ippl.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

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
        InferenceBridge(const std::string& dir, const std::string& name) {
            guarded([&]{
                py::module_::import("sys").attr("path").attr("insert")(0, dir);
                mod_m = py::module_::import(name.c_str());
            });
        }

        //Generic variadic call function, pybind converts params to Python objects
        // and py::object::operator() is itself a variadic forwarding template
        template <typename... Args>
        py::object call(const char* fn, Args&&... args) {
            return guarded([&]{return mod_m.attr(fn)(std::forward<Args>(args)...); });
        }

        void finalize() {
            guarded([&]{ mod_m.attr("finalize")(); });
            mod_m = py::object();
        }
};

#endif