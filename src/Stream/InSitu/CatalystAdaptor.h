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
        // catalyst blueprint definition
        // https://docs.paraview.org/en/latest/Catalyst/blueprints.html
        //
        // conduit blueprint definition (v.8.3)
        // https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html

        auto host_view_with_ghost_cells = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field.getView());
//        typename Field::view_type::host_mirror_type h_view = Kokkos::create_mirror(field.getView());
//        Kokkos::deep_copy(h_view, field.getView());

        //auto my_host_view = Kokkos::create_mirror_view_and_copy(field.getView());
//        typename Field::view_type::host_mirror_type host_view_full = Kokkos::create_mirror(field.getView());
//        Kokkos::deep_copy(host_view, field.getView());

        //auto host_view = ippl::detail::shrinkView<Field::dimension, typename Field::type>("tempFieldf", field.getView(), field.getNghost());
// auto host_view = ippl::detail::shrinkView<Field::dimension, typename Field::type>("tempFieldf", h_view, field.getNghost());

//        using index_array_type = typename ippl::RangePolicy<Field::dimension>::index_array_type;
//        ippl::parallel_for(
//            "copy from Kokkos f field in FFT", ippl::getRangePolicy<Field::dimension>(h_view, field.getNghost()),
//            KOKKOS_LAMBDA(const index_array_type& args) {
//                ippl::apply<Field::dimension>(host_view, args - field.getNghost()) = ippl::apply<Field::dimension>(h_view, args);
//            });
//

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
            mesh[field_node_dim].set(
                int(field.getLayout().getLocalNDIndex()[iDim].length() + 1));

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
        fields["density/association"].set("element");
        fields["density/topology"].set("mesh");
        fields["density/volume_dependent"].set("false");
        fields["density/values"].set_external(host_view.data(), host_view.size());

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
