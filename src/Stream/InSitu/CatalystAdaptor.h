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

    using View_vector =
        Kokkos::View<ippl::Vector<double, 3>***, Kokkos::LayoutLeft, Kokkos::HostSpace>;
    void setData(conduit_cpp::Node& node, const View_vector& view) {
        node["electrostatic/association"].set("element");
        node["electrostatic/topology"].set("mesh");
        node["electrostatic/volume_dependent"].set("false");

        auto length = std::size(view);

        // offset is zero as we start without the ghost cells
        // stride is 1 as we have every index of the array
        node["electrostatic/values/x"].set_external(&view.data()[0][0], length, 0, 1);
        node["electrostatic/values/y"].set_external(&view.data()[0][1], length, 0, 1);
        node["electrostatic/values/z"].set_external(&view.data()[0][2], length, 0, 1);
    }

    using View_scalar = Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::HostSpace>;
    void setData(conduit_cpp::Node& node, const View_scalar& view) {
        node["density/association"].set("element");
        node["density/topology"].set("mesh");
        node["density/volume_dependent"].set("false");
        node["density/values"].set_external(view.data(), view.size());
    }

    void Initialize(int argc, char* argv[]) {
        conduit_cpp::Node node;
        for (int cc = 1; cc < argc; ++cc) {
            node["catalyst/scripts/script" + std::to_string(cc - 1)].set_string(argv[cc]);
        }
        try {
            node["catalyst_load/implementation"]        = getenv("CATALYST_IMPLEMENTATION_NAME");
            node["catalyst_load/search_paths/paraview"] = getenv("CATALYST_IMPLEMENTATION_PATHS");
        } catch (...) {
            throw IpplException("CatalystAdaptor::Initialize",
                                "no environmental variable for CATALYST_IMPLEMENTATION_NAME or "
                                "CATALYST_IMPLEMENTATION_PATHS found");
        }
        // TODO: catch catalyst error also with IpplException
        catalyst_status err = catalyst_initialize(conduit_cpp::c_node(&node));
        if (err != catalyst_status_ok) {
            std::cerr << "Failed to initialize Catalyst: " << err << std::endl;
        }
    }

    template <class Field>
    void Execute(int cycle, double time, int rank, Field& field) {
        static_assert(Field::dimension == 3, "CatalystAdaptor only supports 3D");
        // catalyst blueprint definition
        // https://docs.paraview.org/en/latest/Catalyst/blueprints.html
        //
        // conduit blueprint definition (v.8.3)
        // https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html

        auto nGhost = field.getNghost();

        typename Field::view_type::host_mirror_type host_view =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field.getView());

        Kokkos::View<typename Field::type***, Kokkos::LayoutLeft, Kokkos::HostSpace>
            host_view_layout_left("host_view_layout_left",
                                  field.getLayout().getLocalNDIndex()[0].length(),
                                  field.getLayout().getLocalNDIndex()[1].length(),
                                  field.getLayout().getLocalNDIndex()[2].length());

        for (size_t i = 0; i < field.getLayout().getLocalNDIndex()[0].length(); ++i) {
            for (size_t j = 0; j < field.getLayout().getLocalNDIndex()[1].length(); ++j) {
                for (size_t k = 0; k < field.getLayout().getLocalNDIndex()[2].length(); ++k) {
                    host_view_layout_left(i, j, k) = host_view(i + nGhost, j + nGhost, k + nGhost);
                }
            }
        }

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

        for (unsigned int iDim = 0; iDim < field.get_mesh().getGridsize().dim; ++iDim) {
            mesh[field_node_dim].set(field.getLayout().getLocalNDIndex()[iDim].length() + 1);

            // shift origin by one ghost cell
            mesh[field_node_origin].set(
                field.get_mesh().getOrigin()[iDim]  // global origin
                + field.getLayout().getLocalNDIndex()[iDim].first()
                      * field.get_mesh().getMeshSpacing(iDim));  // shift to local index
            mesh[field_node_spacing].set(field.get_mesh().getMeshSpacing(iDim));

            // increment last char in string
            ++field_node_dim.back();
            ++field_node_origin.back();
            ++field_node_spacing.back();
        }

        // add topology
        mesh["topologies/mesh/type"].set("uniform");
        mesh["topologies/mesh/coordset"].set("coords");
        std::string field_node_origin_topo = "topologies/mesh/origin/x";
        for (unsigned int iDim = 0; iDim < field.get_mesh().getGridsize().dim; ++iDim) {
            // shift origin
            mesh[field_node_origin_topo].set(field.get_mesh().getOrigin()[iDim]
                                             + field.getLayout().getLocalNDIndex()[iDim].first()
                                                   * field.get_mesh().getMeshSpacing(iDim));

            ++field_node_origin_topo.back();
        }

        // add values and subscribe to data
        auto fields = mesh["fields"];

        setData(fields, host_view_layout_left);

        // print node to have visual representation
        if (cycle == 0)
            catalyst_conduit_node_print(conduit_cpp::c_node(&node));

        catalyst_status err = catalyst_execute(conduit_cpp::c_node(&node));
        if (err != catalyst_status_ok) {
            std::cerr << "Failed to execute Catalyst: " << err << std::endl;
        }
    }

    template <class ChargedParticles>
    void Execute_Particle(int cycle, double time, int rank, ChargedParticles& particle) {

        auto particle_view = particle->q.getView();
        auto layout_view = particle->R.getView();

//        typename ChargedParticles::view_type::host_mirror_type host_view =
//            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), particle_view);

//        Kokkos::View<typename ChargedParticles::type***, Kokkos::LayoutLeft, Kokkos::HostSpace>
//            host_view_layout_left("host_view_layout_left",
//                                  field.getLayout().getLocalNDIndex()[0].length(),
//                                  field.getLayout().getLocalNDIndex()[1].length(),
//                                  field.getLayout().getLocalNDIndex()[2].length());
//
//        for (size_t i = 0; i < field.getLayout().getLocalNDIndex()[0].length(); ++i) {
//            for (size_t j = 0; j < field.getLayout().getLocalNDIndex()[1].length(); ++j) {
//                for (size_t k = 0; k < field.getLayout().getLocalNDIndex()[2].length(); ++k) {
//                    host_view_layout_left(i, j, k) = host_view(i + nGhost, j + nGhost, k + nGhost);
//                }
//            }
//        }

        conduit_cpp::Node node;

        // add time/cycle information
        auto state = node["catalyst/state"];
        state["cycle"].set(cycle);
        state["time"].set(time);
        state["domain_id"].set(rank);

        // add catalyst channel named ippl_field, as fields is reserved
        auto channel = node["catalyst/channels/ippl_particles"];
        channel["type"].set_string("mesh");

        // in data channel now we adhere to conduits mesh blueprint definition
        auto mesh = channel["data"];
        mesh["coordsets/coords/type"].set("explicit");

        // number of points in specific dimension
        std::string particle_node_dim{"coordsets/coords/values/x"};

        for (unsigned int iDim = 0; iDim < 3; ++iDim) {
            mesh[particle_node_dim].set_external(&layout_view.data()[0][iDim], std::size(layout_view), 0, 1);

            // increment last char in string
            ++particle_node_dim.back();
        }

        // auto test = particle->ID;
        // add topology
        mesh["topologies/mesh/type"].set("unstructured");
        mesh["topologies/mesh/coordset"].set("coords");
        mesh["topologies/mesh/elements/shape"].set("point");
        mesh["topologies/mesh/elements/connectivity"].set_external(particle->ID.getView().data());
//
//        // add values and subscribe to data
//        auto fields = mesh["fields"];
//
//        setData(fields, host_view_layout_left);

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
