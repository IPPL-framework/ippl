// SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef CatalystAdaptor_h
#define CatalystAdaptor_h

#include "Ippl.h"

#include <catalyst.hpp>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "Utility/IpplException.h"

namespace CatalystAdaptor {
    constexpr unsigned int dim{2};
    using Mesh_t      = ippl::UniformCartesian<double, dim>;
    using Centering_t = Mesh_t::DefaultCentering;
    using Field_t     = ippl::Field<double, dim, Mesh_t, Centering_t>;

    void Initialize(int argc, char* argv[]) {
        conduit_cpp::Node node;
        for (int cc = 1; cc < argc; ++cc) {
            node["catalyst/scripts/script" + std::to_string(cc - 1)].set_string(argv[cc]);
        }
        try {
            node["catalyst_load/implementation"]        = getenv("CATALYST_IMPLEMENTATION_NAME");
            node["catalyst_load/search_paths/paraview"] = getenv("PARAVIEW_CATALYST_DIR");
        } catch (...) {
            throw IpplException("CatalystAdaptor::Initialize",
                                "no environmental variable for CATALYST_IMPLEMENTATION_NAME or "
                                "PARAVIEW_CATALYST_DIR found");
        }
        // TODO: catch catalyst error also with IpplException
        catalyst_status err = catalyst_initialize(conduit_cpp::c_node(&node));
        if (err != catalyst_status_ok) {
            std::cerr << "Failed to initialize Catalyst: " << err << std::endl;
        }
    }

    template <typename Field_t>
    void Execute(int cycle, double time, int rank,
                 Field_t& field)  // int cycle, double time) //, Grid& grid, Attributes& attribs,
                                  // Particles& particles)
    {
        // catalyst blueprint definition
        // https://docs.paraview.org/en/latest/Catalyst/blueprints.html
        //
        // conduit blueprint definition (v.8.3)
        // https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html
        conduit_cpp::Node node;

        // add time/cycle information
        auto state = node["catalyst/state"];
        state["cycle"].set(cycle);
        state["time"].set(time);
        state["domain_id"].set(rank);

        // add catalyst channel named ippl_field, as fields is reserved
        auto channel = node["catalyst/channels/ippl_field"];
        channel["type"].set_string("mesh");

        // in data channel now we adhere to conduits mesh blueprint definition
        auto mesh = channel["data"];
        mesh["coordsets/coords/type"].set("uniform");

        // number of points in specific dimension
        std::string field_node_dim{"coordsets/coords/dims/i"};
        std::string field_node_origin{"coordsets/coords/origin/x"};
        std::string field_node_spacing{"coordsets/coords/spacing/dx"};
        auto origin = field.get_mesh().getOrigin();

        for (unsigned int iDim = 0; iDim < field.get_mesh().getGridsize().dim; ++iDim) {
            // include ghost cells to the "left" and "right" + 1 point
            mesh[field_node_dim].set(
                int(field.get_mesh().getGridsize(iDim) + 2 * field.getNghost() + 1));
            // shift origin by one ghost cell
            mesh[field_node_origin].set(
                origin(iDim) - field.get_mesh().getMeshSpacing(iDim) * field.getNghost());
            mesh[field_node_spacing].set(field.get_mesh().getMeshSpacing(iDim));

            // increment last char in string
            ++field_node_dim.back();
            ++field_node_origin.back();
            ++field_node_spacing.back();
        }

        // add topology
        mesh["topologies/mesh/type"].set("uniform");
        mesh["topologies/mesh/coordset"].set("coords");
        field_node_origin = "topologies/mesh/origin/x";
        for (unsigned int iDim = 0; iDim < field.get_mesh().getGridsize().dim; ++iDim) {
            // shift origin by one ghost cell
            mesh[field_node_origin].set(
                origin(iDim) - field.get_mesh().getMeshSpacing(iDim) * field.getNghost());

            ++field_node_origin.back();
        }

        // add values and subscribe to data
        auto fields = mesh["fields"];
        fields["density/association"].set("element");
        fields["density/topology"].set("mesh");
        fields["density/volume_dependent"].set("false");
        fields["density/values"].set_external(field.getView().data(), field.getView().size());

        // print node to have visual representation
        if (cycle == 0)
            catalyst_conduit_node_print(conduit_cpp::c_node(&node));

        catalyst_status err = catalyst_execute(conduit_cpp::c_node(&node));
        if (err != catalyst_status_ok) {
            std::cerr << "Failed to execute Catalyst: " << err << std::endl;
        }
    }

    void Finalize() {
        conduit_cpp::Node node;
        catalyst_status err = catalyst_finalize(conduit_cpp::c_node(&node));
        if (err != catalyst_status_ok) {
            std::cerr << "Failed to finalize Catalyst: " << err << std::endl;
        }
    }
}  // namespace CatalystAdaptor

#endif
