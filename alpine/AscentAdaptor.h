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
    int mFrequency = 1;

    template <typename T, unsigned Dim>
    using FieldVariant = std::variant<Field_t<Dim>*, VField_t<T, Dim>*>;

    template <typename T, unsigned Dim>
    using FieldPair = std::pair<std::string, FieldVariant<T, Dim>>;

    template <typename T, unsigned Dim>
    using ParticlePair = std::pair<std::string, std::shared_ptr<ParticleContainer<T, Dim> > >;

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

    void Initialize(int frequency) {
      MPI_Comm ascent_comm;
      mFrequency = frequency;

      // Split communicator based on the task ID
      MPI_Comm_dup(MPI_COMM_WORLD, &ascent_comm);

      conduit::Node ascent_opts;
      ascent_opts["mpi_comm"] = MPI_Comm_c2f(ascent_comm);
      mAscent.open(ascent_opts);
    }

    void Finalize() {
        conduit::Node node;
        mAscent.close();
        
    }


    void Execute_Particle(
         const auto& particleContainer,
         const auto& R_host, const auto& P_host, const auto& q_host, const auto& ID_host,
         conduit::Node& node) {
            

        node["coordsets/particle_coords/type"].set_string("explicit");

        //mesh["coordsets/coords/values/x"].set_external(&layout_view.data()[0][0], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        //mesh["coordsets/coords/values/y"].set_external(&layout_view.data()[0][1], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        //mesh["coordsets/coords/values/z"].set_external(&layout_view.data()[0][2], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        node["coordsets/particle_coords/values/x"].set_external(&R_host.data()[0][0], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        node["coordsets/particle_coords/values/y"].set_external(&R_host.data()[0][1], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        node["coordsets/particle_coords/values/z"].set_external(&R_host.data()[0][2], particleContainer->getLocalNum(), 0, sizeof(double)*3);

        node["topologies/particle_topo/type"].set_string("unstructured");
        node["topologies/particle_topo/coordset"].set_string("particle_coords");
        node["topologies/particle_topo/elements/shape"].set_string("point");
        //mesh["topologies/mesh/elements/connectivity"].set_external(particleContainer->ID.getView().data(),particleContainer->getLocalNum());
        node["topologies/particle_topo/elements/connectivity"].set_external(ID_host.data(),particleContainer->getLocalNum());

        //auto charge_view = particleContainer->getQ().getView();

        // add values for scalar charge field
        auto &fields = node["fields"];
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
        if(!conduit::blueprint::mesh::verify(node, verify_info))
        {
            std::cerr << "Mesh verification failed!" << std::endl;
            verify_info.print();
            exit(EXIT_FAILURE);
        }
    }


    template <class Field>
    void Execute_Field(Field* field, const std::string& fieldName,
         Kokkos::View<typename Field::view_type::data_type, Kokkos::LayoutLeft, Kokkos::HostSpace>& host_view_layout_left,
         conduit::Node& node) {
        static_assert(Field::dim == 3, "AscentAdaptor only supports 3D");
        

        node["coordsets/coords/type"].set_string("uniform");

        // number of points in specific dimension
        std::string field_node_dim{"coordsets/coords/dims/i"};
        std::string field_node_origin{"coordsets/coords/origin/x"};
        std::string field_node_spacing{"coordsets/coords/spacing/dx"};

        for (unsigned int iDim = 0; iDim < field->get_mesh().getGridsize().dim; ++iDim) {
            // add dimension
            node[field_node_dim].set(field->getLayout().getLocalNDIndex()[iDim].length() + 1);

            // add origin
            node[field_node_origin].set(
                field->get_mesh().getOrigin()[iDim] + field->getLayout().getLocalNDIndex()[iDim].first()
                      * field->get_mesh().getMeshSpacing(iDim));

            // add spacing
            node[field_node_spacing].set(field->get_mesh().getMeshSpacing(iDim));

            // increment last char in string
            ++field_node_dim.back();
            ++field_node_origin.back();
            ++field_node_spacing.back();
        }

        // add topology
        node["topologies/mesh/type"].set_string("uniform");
        node["topologies/mesh/coordset"].set_string("coords");
        std::string field_node_origin_topo{"topologies/mesh/origin/x"};
        for (unsigned int iDim = 0; iDim < field->get_mesh().getGridsize().dim; ++iDim) {
            // shift origin
            node[field_node_origin_topo].set(field->get_mesh().getOrigin()[iDim]
                                             + field->getLayout().getLocalNDIndex()[iDim].first()
                                                   * field->get_mesh().getMeshSpacing(iDim));

            // increment last char in string
            ++field_node_origin_topo.back();
        }

        // B) Set the field values

        // Initialize the existing Kokkos::View
        host_view_layout_left = Kokkos::View<typename Field::view_type::data_type, Kokkos::LayoutLeft, Kokkos::HostSpace>(
           "host_view_layout_left",
           field->getLayout().getLocalNDIndex()[0].length(),
           field->getLayout().getLocalNDIndex()[1].length(),
           field->getLayout().getLocalNDIndex()[2].length());

        // Creates a host-accessible mirror view and copies the data from the device view to the host.
        auto host_view =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field->getView());

        // Copy data from field to the memory+style which will be passed to Catalyst
        auto nGhost = field->getNghost();
        for (size_t i = 0; i < field->getLayout().getLocalNDIndex()[0].length(); ++i) {
            for (size_t j = 0; j < field->getLayout().getLocalNDIndex()[1].length(); ++j) {
                for (size_t k = 0; k < field->getLayout().getLocalNDIndex()[2].length(); ++k) {
                    host_view_layout_left(i, j, k) = host_view(i + nGhost, j + nGhost, k + nGhost);
                }
            }
        }

        auto &fields = node["fields"];
        setData(fields, host_view_layout_left);

        conduit::Node verify_info;
        if(!conduit::blueprint::mesh::verify(node, verify_info))
        {
            std::cerr << "Mesh verification failed!" << std::endl;
            verify_info.print();
            exit(EXIT_FAILURE);
        }
    }

    

    template<typename T, unsigned Dim>
    void Execute(int cycle, double time, int rank,
    // const auto& /* std::shared_ptr<ParticleContainer<double, 3> >& */ particle,
    const std::vector<AscentAdaptor::ParticlePair<T, Dim>>& particles,
    const std::vector<FieldPair<T, Dim>>& fields, double& scaleFactor) {
        conduit::Node node;

        // add time/cycle information
        auto state = node["state"];
        state["cycle"].set(cycle);
        state["time"].set(time);

        // Handle particles

        std::map<std::string, typename ippl::ParticleAttrib<ippl::Vector<double, 3>>::HostMirror> R_host_map;
        std::map<std::string, typename ippl::ParticleAttrib<ippl::Vector<double, 3>>::HostMirror> P_host_map;
        std::map<std::string, typename ippl::ParticleAttrib<double>::HostMirror> q_host_map;
        std::map<std::string, typename ippl::ParticleAttrib<std::int64_t>::HostMirror> ID_host_map;

        // Loop over all particle container
        for (const auto& particlesPair : particles)
        {
            const std::string& particlesName = particlesPair.first;
            const auto particleContainer = particlesPair.second;

            assert((particleContainer->ID.getView().data() != nullptr) && "ID view should not be nullptr, might be missing the right execution space");

            R_host_map[particlesName]  = particleContainer->R.getHostMirror();
            P_host_map[particlesName]  = particleContainer->P.getHostMirror();
            q_host_map[particlesName]  = particleContainer->q.getHostMirror();
            ID_host_map[particlesName] = particleContainer->ID.getHostMirror();

            Kokkos::deep_copy(R_host_map[particlesName],  particleContainer->R.getView());
            Kokkos::deep_copy(P_host_map[particlesName],  particleContainer->P.getView());
            Kokkos::deep_copy(q_host_map[particlesName],  particleContainer->q.getView());
            Kokkos::deep_copy(ID_host_map[particlesName], particleContainer->ID.getView());

            Execute_Particle(
              particleContainer,
              R_host_map[particlesName], P_host_map[particlesName], q_host_map[particlesName], ID_host_map[particlesName],
              particlesName,
              node);
        }


        // Handle fields

        // Map of all Kokkos::Views. This keeps a reference on all Kokkos::Views
        // which ensures that Kokkos does not free the memory before the end of this function.
        std::map<std::string, Kokkos::View<typename Field_t<Dim>::view_type::data_type, Kokkos::LayoutLeft, Kokkos::HostSpace> > scalar_host_views;
        std::map<std::string, Kokkos::View<typename VField_t<T, Dim>::view_type::data_type, Kokkos::LayoutLeft, Kokkos::HostSpace> > vector_host_views;

        // Loop over all fields
        for (const auto& fieldPair : fields)
        {
            const std::string& fieldName = fieldPair.first;
            const auto& fieldVariant = fieldPair.second;

            // If field is a _scalar_ field
            if (std::holds_alternative<Field_t<Dim>*>(fieldVariant)) {
                Field_t<Dim>* field = std::get<Field_t<Dim>*>(fieldVariant);
                // == ippl::Field<double, 3, ippl::UniformCartesian<double, 3>, Cell>*

                Execute_Field(field, fieldName, scalar_host_views[fieldName], node);
            }
            // If field is a _vector_ field
            else if (std::holds_alternative<VField_t<T, Dim>*>(fieldVariant)) {
                VField_t<T, Dim>* field = std::get<VField_t<T, Dim>*>(fieldVariant);
                // == ippl::Field<ippl::Vector<double, 3>, 3, ippl::UniformCartesian<double, 3>, Cell>*

                Execute_Field(field, fieldName, vector_host_views[fieldName], node);     
            }
        }

        conduit::Node actions;
        mAscent.publish(node);
        mAscent.execute(actions);


    }
}  // namespace CatalystAdaptor

#endif
