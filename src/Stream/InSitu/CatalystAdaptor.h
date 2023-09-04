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
        // auto field_node = node["catalyst/channels/field"];
        auto field_node = node["catalyst/channel/field/coordsets/coords"];
        // field_node["type"].set_string("mesh");
        field_node["type"].set_string("uniform");

        // number of points in specific dimension
        std::string field_node_dim {"dims/i"};
        std::string field_node_origin {"origin/x"};
        std::string field_node_spacing {"spacing/dx"};
        auto origin = field.get_mesh().getOrigin();

        for (unsigned int iDim = 0; iDim < field.get_mesh().getGridsize().dim; ++iDim){
            // include ghost cells to the "left" and "right"
            field_node[field_node_dim].set_string(
                std::to_string(field.get_mesh().getGridsize(iDim) + 2 * field.getNghost()));
            // shift origin by one ghost cell
            field_node[field_node_origin].set_string(
                std::to_string(origin(iDim) - field.get_mesh().getMeshSpacing(iDim) * field.getNghost()));
            field_node[field_node_spacing].set_string(
                std::to_string(field.get_mesh().getMeshSpacing(iDim)));

            ++field_node_dim.back();
            ++field_node_origin.back();
            ++field_node_spacing.back();
        }

        auto field_node_topology = node["catalyst/channel/field/topologies/mesh"];
        field_node_topology["type"].set_string("uniform");
        field_node_topology["coordset"].set_string("coords");
//         field_node["dims/j"].set_string(std::to_string(field.get_mesh().getGridsize(1))); //         field_node["dims/k"].set_string(std::to_string(field.get_mesh().getGridsize(2)));

        auto field_node_fields = node["catalyst/channel/field/fields/density"];
        field_node_fields["association"].set_string("element");
        field_node_fields["volume_dependent"].set_string("false");
        field_node_fields["topology"].set_string("mesh");

        auto & layout = field.getLayout();
        auto & view = field.getView();

        field_node_fields["values"].set_external(
            field.getView().data(),
            field.getOwned().size(),
            field.getLayout().getLocalNDIndex(rank)[1].first());
            // field.getLayout().getLocalNDIndex(rank)[1].first());  // + field.getNghost());

//        std::string field_node_values {"values/x"};
//        for (unsigned int iDim = 0; iDim < field.get_mesh().getGridsize().dim; ++iDim) {
//            field_node_fields[field_node_values].set_external(
//                field.getView().data(), field.getOwned()[iDim].length(),
//                field.getLayout().getLocalNDIndex(rank)[iDim].first());  // + field.getNghost());
//            ++field_node_values.back();
//        }

        // origin

//        // Since this example is using Conduit Mesh Blueprint to define the mesh,
//        // we set the channel_grid's type to "mesh".

        // now create the mesh.
//        auto mesh_grid = channel_grid["data"];
//
//        // start with coordsets (of course, the sequence is not important, just make
//        // it easier to think in this order).
//        mesh_grid["coordsets/coords/type"].set_string("explicit");
//
//        // We don't use the conduit_cpp::Node::set(std::vector<..>) API since that deep
//        // copies. For zero-copy, we use the set_.._ptr(..) API.
//        mesh_grid["coordsets/coords/values/x"].set_external(
//                grid.GetPointsArray(), grid.GetNumberOfPoints(), /*offset=*/0, /*stride=*/3 * sizeof(double));
//        mesh_grid["coordsets/coords/values/y"].set_external(grid.GetPointsArray(),
//                                                            grid.GetNumberOfPoints(),
//                /*offset=*/sizeof(double), /*stride=*/3 * sizeof(double));
//        mesh_grid["coordsets/coords/values/z"].set_external(grid.GetPointsArray(),
//                                                            grid.GetNumberOfPoints(),
//                /*offset=*/2 * sizeof(double), /*stride=*/3 * sizeof(double));
//
//        // Next, add topology
//        mesh_grid["topologies/mesh/type"].set_string("unstructured");
//        mesh_grid["topologies/mesh/coordset"].set_string("coords");
//        mesh_grid["topologies/mesh/elements/shape"].set_string("hex");
//        mesh_grid["topologies/mesh/elements/connectivity"].set(
//                grid.GetCellPoints(0), grid.GetNumberOfCells() * 8);
//
//        // Finally, add fields.
//        auto fields_grid = mesh_grid["fields"];
//        fields_grid["velocity/association"].set_string("vertex");
//        fields_grid["velocity/topology"].set_string("mesh");
//        fields_grid["velocity/volume_dependent"].set_string("false");
//
//        // velocity is stored in non-interlaced form (unlike points).
//        fields_grid["velocity/values/x"].set_external(
//                attribs.GetVelocityArray(), grid.GetNumberOfPoints(), /*offset=*/0);
//        fields_grid["velocity/values/y"].set_external(attribs.GetVelocityArray(),
//                                                      grid.GetNumberOfPoints(),
//                /*offset=*/grid.GetNumberOfPoints() * sizeof(double));
//        fields_grid["velocity/values/z"].set_external(attribs.GetVelocityArray(),
//                                                      grid.GetNumberOfPoints(),
//                /*offset=*/grid.GetNumberOfPoints() * sizeof(double) * 2);
//
//        // pressure is cell-data.
//        fields_grid["pressure/association"].set_string("element");
//        fields_grid["pressure/topology"].set_string("mesh");
//        fields_grid["pressure/volume_dependent"].set_string("false");
//        fields_grid["pressure/values"].set_external(attribs.GetPressureArray(), grid.GetNumberOfCells());
//
//        // Now add the second channel, called "particles".
//        auto channel_particles = exec_params["catalyst/channels/particles"];
//
//        // make the particles' time update every other step of the mesh's
//        channel_particles["state/cycle"].set(cycle - (cycle % 2));
//        channel_particles["state/time"].set(time - (cycle % 2) * 0.1);
//        channel_particles["state/multiblock"].set(1);
//
//        // Since this example is using Conduit Mesh Blueprint to define the mesh,
//        // we set the channel_particles's type to "mesh".
//        channel_particles["type"].set_string("mesh");
//
//        // now create the mesh.
//        auto mesh_particles = channel_particles["data"];
//        mesh_particles["coordsets/coords/type"].set_string("explicit");
//        mesh_particles["coordsets/coords/values/x"].set_external(particles.GetPointsArray(),
//                                                                 particles.GetNumberOfPoints(), /*offset=*/0, /*stride=*/3 * sizeof(double));
//        mesh_particles["coordsets/coords/values/y"].set_external(particles.GetPointsArray(),
//                                                                 particles.GetNumberOfPoints(), /*offset=*/sizeof(double), /*stride=*/3 * sizeof(double));
//        mesh_particles["coordsets/coords/values/z"].set_external(particles.GetPointsArray(),
//                                                                 particles.GetNumberOfPoints(), /*offset=*/2 * sizeof(double), /*stride=*/3 * sizeof(double));
//
//        // now, the topology.
//        mesh_particles["topologies/mesh/type"].set_string("unstructured");
//        mesh_particles["topologies/mesh/coordset"].set_string("coords");
//        mesh_particles["topologies/mesh/elements/shape"].set_string("point");
//        std::vector<conduit_int64> connectivity(particles.GetNumberOfPoints());
//        std::iota(connectivity.begin(), connectivity.end(), 0);
//        mesh_particles["topologies/mesh/elements/connectivity"].set_external(
//                &connectivity[0], particles.GetNumberOfPoints());

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
