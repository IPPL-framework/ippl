// SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef AsecntAdaptor_h
#define AsecntAdaptor_h

#include "Ippl.h"

#include <ascent.hpp>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "Utility/IpplException.h"


namespace AscentAdaptor {

    ascent::Ascent mAscent;

    using View_vector =
        Kokkos::View<ippl::Vector<double, 3>***, Kokkos::LayoutLeft, Kokkos::HostSpace>;
    inline void setData(conduit::Node& node, const View_vector& view) {
        node["electrostatic/association"].set_string("element");
        node["electrostatic/topology"].set_string("mesh");
        node["electrostatic/volume_dependent"].set_string("false");

        auto length = std::size(view);

        // offset is zero as we start without the ghost cells
        // stride is 1 as we have every index of the array
        node["electrostatic/values/x"].set_external(&view.data()[0][0], length, 0, 1);
        node["electrostatic/values/y"].set_external(&view.data()[0][1], length, 0, 1);
        node["electrostatic/values/z"].set_external(&view.data()[0][2], length, 0, 1);
    }

    using View_scalar = Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::HostSpace>;
    inline void setData(conduit::Node& node, const View_scalar& view) {
        node["density/association"].set_string("element");
        node["density/topology"].set_string("mesh");
        node["density/volume_dependent"].set_string("false");

        node["density/values"].set_external(view.data(), view.size());
    }

    void Initialize(int argc, char* argv[]) {
      MPI_Comm ascent_comm;

      // Split communicator based on the task ID
      MPI_Comm_dup(MPI_COMM_WORLD, &ascent_comm);

      conduit::Node ascent_opts;
      ascent_opts["mpi_comm"] = MPI_Comm_c2f(ascent_comm);
      mAscent.open(ascent_opts);
    }

    template <class Field>
    std::optional<conduit::Node> Execute_Field(int cycle, double time, int rank, Field& field) {
        static_assert(Field::dim == 3, "AscentAdaptor only supports 3D");
        // conduit blueprint definition (v.8.3)
        // https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html
        conduit::Node mesh;

        auto nGhost = field.getNghost();

        typename Field::view_type::host_mirror_type host_view =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field.getView());

        Kokkos::View<typename Field::view_type::data_type, Kokkos::LayoutLeft, Kokkos::HostSpace>
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


        mesh["state/cycle"] = cycle;
        mesh["state/time"] = time;

        mesh["coordsets/coords/type"].set_string("uniform");

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
        mesh["topologies/mesh/type"].set_string("uniform");
        mesh["topologies/mesh/coordset"].set_string("coords");
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
        auto &fields = mesh["fields"];
        setData(fields, host_view_layout_left);

        conduit::Node verify_info;
        if(!conduit::blueprint::mesh::verify(mesh, verify_info))
        {
            std::cerr << "Mesh verification failed!" << std::endl;
            verify_info.print();
            exit(EXIT_FAILURE);
        }
        conduit::Node actions;
        mAscent.publish(mesh);
        mAscent.execute(actions);
        
        return mesh;

    }

    template <class ParticleContainer>
    std::optional<conduit::Node> Execute_Particle(int cycle, double time, int rank, ParticleContainer& particleContainer) {
      assert((particleContainer->ID.getView().data() != nullptr) && "ID view should not be nullptr, might be missing the right execution space");

        //auto layout_view = particleContainer->R.getView();
        typename ippl::ParticleAttrib<ippl::Vector<double, 3>>::HostMirror R_host = particleContainer->R.getHostMirror();
        typename ippl::ParticleAttrib<ippl::Vector<double, 3>>::HostMirror P_host = particleContainer->P.getHostMirror();
        typename ippl::ParticleAttrib<double>::HostMirror q_host = particleContainer->q.getHostMirror();
        typename ippl::ParticleAttrib<std::int64_t>::HostMirror ID_host = particleContainer->ID.getHostMirror();
        Kokkos::deep_copy(R_host, particleContainer->R.getView());
        Kokkos::deep_copy(P_host, particleContainer->P.getView());
        Kokkos::deep_copy(q_host, particleContainer->q.getView());
        Kokkos::deep_copy(ID_host, particleContainer->ID.getView());
        
        conduit::Node mesh;
        
        mesh["state/cycle"] = cycle;
        mesh["state/time"] = time;

        mesh["coordsets/particle_coords/type"].set_string("explicit");

        //mesh["coordsets/coords/values/x"].set_external(&layout_view.data()[0][0], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        //mesh["coordsets/coords/values/y"].set_external(&layout_view.data()[0][1], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        //mesh["coordsets/coords/values/z"].set_external(&layout_view.data()[0][2], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        mesh["coordsets/particle_coords/values/x"].set_external(&R_host.data()[0][0], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        mesh["coordsets/particle_coords/values/y"].set_external(&R_host.data()[0][1], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        mesh["coordsets/particle_coords/values/z"].set_external(&R_host.data()[0][2], particleContainer->getLocalNum(), 0, sizeof(double)*3);

        mesh["topologies/particle_topo/type"].set_string("unstructured");
        mesh["topologies/particle_topo/coordset"].set_string("particle_coords");
        mesh["topologies/particle_topo/elements/shape"].set_string("point");
        //mesh["topologies/mesh/elements/connectivity"].set_external(particleContainer->ID.getView().data(),particleContainer->getLocalNum());
        mesh["topologies/particle_topo/elements/connectivity"].set_external(ID_host.data(),particleContainer->getLocalNum());

        //auto charge_view = particleContainer->getQ().getView();

        // add values for scalar charge field
        auto &fields = mesh["fields"];
        fields["particle_charge/association"].set_string("vertex");
        fields["particle_charge/topology"].set_string("particle_topo");
        fields["particle_charge/volume_dependent"].set_string("false");

        //fields["charge/values"].set_external(particleContainer->q.getView().data(), particleContainer->getLocalNum());
        fields["particle_charge/values"].set_external(q_host.data(), particleContainer->getLocalNum());

        // add values for vector velocity field
        //auto velocity_view = particleContainer->P.getView();
        fields["particle_velocity/association"].set_string("vertex");
        fields["particle_velocity/topology"].set_string("particle_topo");
        fields["particle_velocity/volume_dependent"].set_string("false");

        //fields["velocity/values/x"].set_external(&velocity_view.data()[0][0], particleContainer->getLocalNum(),0 ,sizeof(double)*3);
        //fields["velocity/values/y"].set_external(&velocity_view.data()[0][1], particleContainer->getLocalNum(),0 ,sizeof(double)*3);
        //fields["velocity/values/z"].set_external(&velocity_view.data()[0][2], particleContainer->getLocalNum(),0 ,sizeof(double)*3);
        fields["particle_velocity/values/x"].set_external(&P_host.data()[0][0], particleContainer->getLocalNum(),0 ,sizeof(double)*3);
        fields["particle_velocity/values/y"].set_external(&P_host.data()[0][1], particleContainer->getLocalNum(),0 ,sizeof(double)*3);
        fields["particle_velocity/values/z"].set_external(&P_host.data()[0][2], particleContainer->getLocalNum(),0 ,sizeof(double)*3);

        fields["particle_position/association"].set_string("vertex");
        fields["particle_position/topology"].set_string("particle_topo");
        fields["particle_position/volume_dependent"].set_string("false");

        //fields["position/values/x"].set_external(&layout_view.data()[0][0], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        //fields["position/values/y"].set_external(&layout_view.data()[0][1], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        //fields["position/values/z"].set_external(&layout_view.data()[0][2], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        fields["particle_position/values/x"].set_external(&R_host.data()[0][0], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        fields["particle_position/values/y"].set_external(&R_host.data()[0][1], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        fields["particle_position/values/z"].set_external(&R_host.data()[0][2], particleContainer->getLocalNum(), 0, sizeof(double)*3);

        conduit::Node verify_info;
        if(!conduit::blueprint::mesh::verify(mesh, verify_info))
        {
            std::cerr << "Mesh verification failed!" << std::endl;
            verify_info.print();
            exit(EXIT_FAILURE);
        }
        conduit::Node actions;
        mAscent.publish(mesh);
        mAscent.execute(actions);

        return mesh;
    }


    template <class Field, class ParticleContainer>
    void Execute_Field_Particle(int cycle, double time, int rank, Field& field, ParticleContainer& particle) {

        AscentAdaptor::Execute_Particle(cycle, time, rank, particle);
        AscentAdaptor::Execute_Field(cycle, time, rank, field);
    }

    void Finalize() {
        conduit::Node node;
        mAscent.close();
        
    }
}  // namespace CatalystAdaptor

#endif
