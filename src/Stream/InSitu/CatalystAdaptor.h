// SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef CatalystAdaptor_h
#define CatalystAdaptor_h

#include "Ippl.h"

#include <Kokkos_DynamicView.hpp>
#include <catalyst.hpp>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "Utility/IpplException.h"

 template <typename T, unsigned Dim = 3>
 using Vector = ippl::Vector<T, Dim>;

 template <unsigned Dim = 3>
 using Mesh_t = ippl::UniformCartesian<double, Dim>;

 template <unsigned Dim = 3>
 using Centering_t = typename Mesh_t<Dim>::DefaultCentering;

 template <typename T, unsigned Dim = 3>
 using Field = ippl::Field<T, Dim, Mesh_t<Dim>, Centering_t<Dim>>;

 template <typename T>
 using ParticleAttrib = ippl::ParticleAttrib<T>;

 template <unsigned Dim = 3>
 using Vector_t = Vector<double, Dim>;

 template <unsigned Dim = 3>
 using Field_t = Field<double, Dim>;

 template <unsigned Dim = 3>
 using VField_t = Field<Vector_t<Dim>, Dim>;


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
        static_assert(Field::dimension == 3, "CatalystAdaptor only supports 3D");
        // catalyst blueprint definition
        // https://docs.paraview.org/en/latest/Catalyst/blueprints.html
        //
        // conduit blueprint definition (v.8.3)
        // https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html

       auto nGhost = field.getNghost();

        typename VField_t<3>::view_type::host_mirror_type vhost_view = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),field.getView());

//        Kokkos::View<typename Field::type*, Kokkos::LayoutLeft, Kokkos::HostSpace> vhost_view_layout_left("vhost_view_layout_left", field.getLayout().getLocalNDIndex()[0].length()+
//            field.getLayout().getLocalNDIndex()[1].length()+
//            field.getLayout().getLocalNDIndex()[2].length());
//
//        auto y_offset = field.getLayout().getLocalNDIndex()[1].length();
//        auto z_offset = field.getLayout().getLocalNDIndex()[2].length();

//        for (size_t i = 0; i < field.getLayout().getLocalNDIndex()[0].length(); ++i)
//        {
//            for (size_t j = 0; j < field.getLayout().getLocalNDIndex()[1].length(); ++j)
//            {
//                for (size_t k = 0; k < field.getLayout().getLocalNDIndex()[2].length(); ++k)
//                {
//                    host_view_layout_left(i,j,k) = host_view(i+nGhost, j+nGhost, k+nGhost);
//                }
//            }
//        }
        //            vhost_view_layout_left(i) = vhost_view.data()[i+nGhost][0];
        //            vhost_view_layout_left(i + y_offset) = vhost_view.data()[i+nGhost][1];
        //            vhost_view_layout_left(i + y_offset + z_offset) =
        //            vhost_view.data()[i+nGhost][2];

        typename Field::view_type::host_mirror_type host_view = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),field.getView());

        Kokkos::View<typename Field::type***,  Kokkos::LayoutLeft, Kokkos::HostSpace> host_view_layout_left("host_view_layout_left", field.getLayout().getLocalNDIndex()[0].length(),
                                                                                             field.getLayout().getLocalNDIndex()[1].length(),
                                                                                             field.getLayout().getLocalNDIndex()[2].length());

        for (size_t i = 0; i < field.getLayout().getLocalNDIndex()[0].length(); ++i)
        {
            for (size_t j = 0; j < field.getLayout().getLocalNDIndex()[1].length(); ++j)
            {
                for (size_t k = 0; k < field.getLayout().getLocalNDIndex()[2].length(); ++k)
                {
                    // host_view_layout_left(i,j,k) = host_view(i+nGhost, j+nGhost, k+nGhost);
                    host_view_layout_left(i,j,k) = vhost_view(i+nGhost, j+nGhost, k+nGhost);
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
//        fields["density/association"].set("element");
//        fields["density/topology"].set("mesh");
//        fields["density/volume_dependent"].set("false");
        //fields["density/values"].set_external(host_view_layout_left.data(), host_view_layout_left.size());

        fields["electrostatic/association"].set("element");
        fields["electrostatic/topology"].set("mesh");
        fields["electrostatic/volume_dependent"].set("false");


        auto length = host_view_layout_left.size();
        // offset is zero as we start without the ghost cells
        // stried is 1 as we have every index of the array
        fields["electrostatic/values/x"].set_external(&host_view_layout_left.data()[0][0], length, 0, 1);
        fields["electrostatic/values/y"].set_external(&host_view_layout_left.data()[0][1], length, 0, 1);
        fields["electrostatic/values/z"].set_external(&host_view_layout_left.data()[0][2], length, 0, 1);

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
