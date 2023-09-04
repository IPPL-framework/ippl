// SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef CatalystAdaptor_h
#define CatalystAdaptor_h

#include <catalyst.hpp>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "Utility/IpplException.h"
#include "Ippl.h"

namespace CatalystAdaptor
{
    constexpr unsigned int dim {3};
    using Mesh_t      = ippl::UniformCartesian<double, 3>;
    using Centering_t = Mesh_t::DefaultCentering;
    typedef ippl::Field<double, dim, Mesh_t, Centering_t> field_type;

/**
 * In this example, we show how we can use Catalysts's C++
 * wrapper around conduit's C API to create Conduit nodes.
 * This is not required. A C++ adaptor can just as
 * conveniently use the Conduit C API to setup the
 * `conduit_node`. However, this example shows that one can
 * indeed use Catalyst's C++ API, if the developer so chooses.
 */
    void Initialize(int argc, char* argv[])
    {
        conduit_cpp::Node node;
        for (int cc = 1; cc < argc; ++cc)
        {
            node["catalyst/scripts/script" + std::to_string(cc - 1)].set_string(argv[cc]);
        }
        try {
            node["catalyst_load/implementation"] = getenv("CATALYST_IMPLEMENTATION_NAME");
            node["catalyst_load/search_paths/paraview"] = getenv("PARAVIEW_CATALYST_DIR");
        } catch (...){
            throw IpplException("CatalystAdaptor::Initialize", "no environmental variable for CATALYST_IMPLEMENTATION_NAME or PARAVIEW_CATALYST_DIR found");
        }
        // TODO: catch catalyst error also with IpplException
        catalyst_status err = catalyst_initialize(conduit_cpp::c_node(&node));
        if (err != catalyst_status_ok)
        {
            std::cerr << "Failed to initialize Catalyst: " << err << std::endl;
        }
    }

    void Execute(int cycle, double time, int rank, field_type &field) //int cycle, double time) //, Grid& grid, Attributes& attribs, Particles& particles)
    {
        //conduit_cpp::Node exec_params;
        conduit_cpp::Node node;

        // include information about catalyst, conduit implementation
        catalyst_about(conduit_cpp::c_node(&node));

        // add time/cycle information
        auto state = node["catalyst/state"];
        state["cycle"].set(cycle);
        state["time"].set(time);
        state["domain_id"].set(rank);

//        // Add channels.
//        // We have 2 channels here. First once is called 'grid'.
        auto channel_field = node["catalyst/channels/field"]; // /coordsets/coords"];
        channel_field["type"].set("mesh");
        
        auto field_channel_mesh = channel_field["data"];
        
        field_channel_mesh["coordsets/coords/type"].set("uniform");

        // number of points in specific dimension
        std::string field_node_dim {"coordsets/coords/dims/i"};
        std::string field_node_origin {"coordsets/coords/origin/x"};
        std::string field_node_spacing {"coordsets/coords/spacing/dx"};
        auto origin = field.get_mesh().getOrigin();

        for (unsigned int iDim = 0; iDim < field.get_mesh().getGridsize().dim; ++iDim){
            // include ghost cells to the "left" and "right"
            field_channel_mesh[field_node_dim].set(
                std::to_string(field.get_mesh().getGridsize(iDim) + 2 * field.getNghost()));
            // shift origin by one ghost cell
            field_channel_mesh[field_node_origin].set(
                std::to_string(origin(iDim) - field.get_mesh().getMeshSpacing(iDim) * field.getNghost()));
            field_channel_mesh[field_node_spacing].set(
                std::to_string(field.get_mesh().getMeshSpacing(iDim)));

            ++field_node_dim.back();
            ++field_node_origin.back();
            ++field_node_spacing.back();
        }

        auto field_channel_topology = channel_field["data"];
        field_channel_topology["type"].set("uniform");
        field_channel_topology["coordset"].set("coords");

        auto field_channel_fields = channel_field["data/fields/density"];
        field_channel_fields["association"].set("element");
        field_channel_fields["volume_dependent"].set("false");
        field_channel_fields["topology"].set("mesh");

        field_channel_fields["values"].set_external(
            field.getView().data(),
            field.getOwned().size(),
            field.getLayout().getLocalNDIndex(rank)[1].first());

        // print node to see what I write there
        if (cycle == 1) catalyst_conduit_node_print(conduit_cpp::c_node(&node));

        catalyst_status err = catalyst_execute(conduit_cpp::c_node(&node));
        if (err != catalyst_status_ok)
        {
            std::cerr << "Failed to execute Catalyst: " << err << std::endl;
        }
    }

    void Finalize()
    {
        conduit_cpp::Node node;
        catalyst_status err = catalyst_finalize(conduit_cpp::c_node(&node));
        if (err != catalyst_status_ok)
        {
            std::cerr << "Failed to finalize Catalyst: " << err << std::endl;
        }
    }
}

#endif
