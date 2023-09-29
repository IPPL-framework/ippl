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
#include <optional>

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

    void callCatalystExecute(const conduit_cpp::Node& node) {

        catalyst_conduit_node_print(conduit_cpp::c_node(&node));
        catalyst_status err = catalyst_execute(conduit_cpp::c_node(&node));
        if (err != catalyst_status_ok) {
            std::cerr << "Failed to execute Catalyst: " << err << std::endl;
        }
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


    void Initialize_Adios(int argc, char* argv[])
    {
        conduit_cpp::Node node;
        for (int cc = 1; cc < argc; ++cc)
        {
            if (strstr(argv[cc], "xml"))
            {
                node["adios/config_filepath"].set_string(argv[cc]);
            }
            else
            {
                node["catalyst/scripts/script" +std::to_string(cc - 1)].set_string(argv[cc]);
            }
        }
        node["catalyst_load/implementation"] = getenv("CATALYST_IMPLEMENTATION_NAME");
        catalyst_status err = catalyst_initialize(conduit_cpp::c_node(&node));
        if (err != catalyst_status_ok)
        {
            std::cerr << "Failed to initialize Catalyst: " << err << std::endl;
        }
    }


    template <class Field>
    std::optional<conduit_cpp::Node> Execute_Field(int cycle, double time, int rank, Field& field, std::optional<conduit_cpp::Node>& node_in) {
        static_assert(Field::dimension == 3, "CatalystAdaptor only supports 3D");
        // catalyst blueprint definition
        // https://docs.paraview.org/en/latest/Catalyst/blueprints.html
        //
        // conduit blueprint definition (v.8.3)
        // https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html
        conduit_cpp::Node node;
        if (node_in)
            node = node_in.value();

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
            // add dimension
            mesh[field_node_dim].set(field.getLayout().getLocalNDIndex()[iDim].length() + 1);

            // add origin
            mesh[field_node_origin].set(
                field.get_mesh().getOrigin()[iDim] + field.getLayout().getLocalNDIndex()[iDim].first()
                      * field.get_mesh().getMeshSpacing(iDim));

            // add spacing
            mesh[field_node_spacing].set(field.get_mesh().getMeshSpacing(iDim));

            // increment last char in string
            ++field_node_dim.back();
            ++field_node_origin.back();
            ++field_node_spacing.back();
        }

        // add topology
        mesh["topologies/mesh/type"].set("uniform");
        mesh["topologies/mesh/coordset"].set("coords");
        std::string field_node_origin_topo{"topologies/mesh/origin/x"};
        for (unsigned int iDim = 0; iDim < field.get_mesh().getGridsize().dim; ++iDim) {
            // shift origin
            mesh[field_node_origin_topo].set(field.get_mesh().getOrigin()[iDim]
                                             + field.getLayout().getLocalNDIndex()[iDim].first()
                                                   * field.get_mesh().getMeshSpacing(iDim));

            // increment last char in string
            ++field_node_origin_topo.back();
        }

        // add values and subscribe to data
        auto fields = mesh["fields"];
        setData(fields, host_view_layout_left);

        // as we have a local copy of the field, the catalyst_execute needs to be called
        // within this scope otherwise the memory location might be already overwritten
        if (node_in != std::nullopt)
        {
            callCatalystExecute(node);
            return {};
        }
        else
          return node;

    }

    template <class ChargedParticles>
    std::optional<conduit_cpp::Node> Execute_Particle(int cycle, double time, int rank, ChargedParticles& particle, std::optional<conduit_cpp::Node>& node_in) {

        auto layout_view = particle->R.getView();

        // if node is passed in, append data to it
        conduit_cpp::Node node;
        if (node_in)
            node = node_in.value();

        // add time/cycle information
        auto state = node["catalyst/state"];
        state["cycle"].set(cycle);
        state["time"].set(time);
        state["domain_id"].set(rank);

        // channel for particles
        auto channel = node["catalyst/channels/ippl_particle"];
        channel["type"].set_string("mesh");

        // in data channel now we adhere to conduits mesh blueprint definition
        auto mesh = channel["data"];
        mesh["coordsets/coords/type"].set("explicit");

        mesh["coordsets/coords/values/x"].set_external(&layout_view.data()[0][0], particle->getLocalNum(), 0, sizeof(double)*3);
        mesh["coordsets/coords/values/y"].set_external(&layout_view.data()[0][1], particle->getLocalNum(), 0, sizeof(double)*3);
        mesh["coordsets/coords/values/z"].set_external(&layout_view.data()[0][2], particle->getLocalNum(), 0, sizeof(double)*3);

        mesh["topologies/mesh/type"].set("unstructured");
        mesh["topologies/mesh/coordset"].set("coords");
        mesh["topologies/mesh/elements/shape"].set("point");
        mesh["topologies/mesh/elements/connectivity"].set_external(particle->ID.getView().data(),particle->getLocalNum());

        auto charge_view = particle->q.getView();

        // add values for scalar charge field
        auto fields = mesh["fields"];
        fields["charge/association"].set("vertex");
        fields["charge/topology"].set("mesh");
        fields["charge/volume_dependent"].set("false");

        fields["charge/values"].set_external(charge_view.data(), particle->getLocalNum());

        // add values for vector velocity field
        auto velocity_view = particle->P.getView();
        fields["velocity/association"].set("vertex");
        fields["velocity/topology"].set("mesh");
        fields["velocity/volume_dependent"].set("false");

        fields["velocity/values/x"].set_external(&velocity_view.data()[0][0], particle->getLocalNum(),0 ,sizeof(double)*3);
        fields["velocity/values/y"].set_external(&velocity_view.data()[0][1], particle->getLocalNum(),0 ,sizeof(double)*3);
        fields["velocity/values/z"].set_external(&velocity_view.data()[0][2], particle->getLocalNum(),0 ,sizeof(double)*3);

        // this node we can return as the pointer to velocity and charge is globally valid
        if (node_in == std::nullopt)
        {
            callCatalystExecute(node);
            return {};
        }
        else
            return node;
    }


    template <class Field, class ChargedParticles>
    void Execute_Field_Particle(int cycle, double time, int rank, Field& field, ChargedParticles& particle) {

        auto node = std::make_optional<conduit_cpp::Node>();
        node = CatalystAdaptor::Execute_Particle(cycle, time, rank, particle, node);
        CatalystAdaptor::Execute_Field(cycle, time, rank, field, node);

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