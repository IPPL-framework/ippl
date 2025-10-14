#ifndef AsecntAdaptor_h
#define AsecntAdaptor_h

#include "Ippl.h"

#include <ascent.hpp>


#include <iostream>
#include <optional>
#include <string>
#include <vector>


#include <cstdlib>
#include <string>
#include <optional>

#include "Utility/IpplException.h"

#include<filesystem>



namespace ippl{
namespace AscentAdaptor {

    ascent::Ascent mAscent;
    int mFrequency = 1;
    conduit::Node mActions;

    template <typename T, unsigned Dim>
    using FieldVariant = std::variant<Field_t<Dim>*, VField_t<T, Dim>*>;

    template <typename T, unsigned Dim>
    using FieldPair = std::pair<std::string, FieldVariant<T, Dim>>;

    template <typename T, unsigned Dim>
    using ParticlePair = std::pair<std::string, std::shared_ptr<ParticleContainer<T, Dim> > >;

    using View_vector =
        Kokkos::View<ippl::Vector<double, 3>***, Kokkos::LayoutLeft, Kokkos::HostSpace>;
    inline void setData(conduit::Node& node, const View_vector& view, const std::string& fieldName) {
        node["electrostatic/association"].set_string("element");
        node["electrostatic/topology"].set_string(fieldName + "_mesh");
        node["electrostatic/volume_dependent"].set_string("false");

        auto length = std::size(view);

        // offset is zero as we start without the ghost cells
        // stride is 1 as we have every index of the array
        node["electrostatic/values/x"].set_external(&view.data()[0][0], length, 0, 1);
        node["electrostatic/values/y"].set_external(&view.data()[0][1], length, 0, 1);
        node["electrostatic/values/z"].set_external(&view.data()[0][2], length, 0, 1);
    }

    using View_scalar = Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::HostSpace>;
    inline void setData(conduit::Node& node, const View_scalar& view, const std::string& fieldName) {
        node["density/association"].set_string("element");
        node["density/topology"].set_string(fieldName + "_mesh");
        node["density/volume_dependent"].set_string("false");

        node["density/values"].set_external(view.data(), view.size());
    }

    void Initialize() {
        
        conduit::Node ascent_options;

        // Split communicator based on the task ID
        MPI_Comm ascent_comm;
        MPI_Comm_dup(MPI_COMM_WORLD, &ascent_comm);
        ascent_options["mpi_comm"] = MPI_Comm_c2f(ascent_comm);

               

        std::filesystem::path output_dir = "./ippl_ascent_output";
        std::filesystem::create_directories(output_dir);
        ascent_options["default_dir"] = output_dir.string();


        const char* ascent_freq_env = std::getenv("ASCENT_EXTRACTION_FREQUENCY");
        int extraction_frequency = 10;

        if (ascent_freq_env) {
            try {
                int parsed = std::stoi(ascent_freq_env);
                if (parsed > 0) {
                    extraction_frequency = parsed;
                    std::cout << "Using ASCENT_EXTRACTION_FREQUENCY from environment: " << extraction_frequency << std::endl;
                } else {
                    std::cerr << "ASCENT_EXTRACTION_FREQUENCY must be > 0. Using default: " << extraction_frequency << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Invalid ASCENT_EXTRACTION_FREQUENCY ('" << ascent_freq_env
                          << "'): " << e.what() << ". Using default: " << extraction_frequency << std::endl;
            }
        } else {
            std::cout << "ASCENT_EXTRACTION_FREQUENCY not set. Using default: " << extraction_frequency << std::endl;
        }
        mFrequency = extraction_frequency;

        


        
        
        // Load actions configuration
        const char* ascent_actions_path = std::getenv("ASCENT_ACTIONS_PATH");
        std::filesystem::path actions_file_path;

        if (ascent_actions_path && std::filesystem::exists(ascent_actions_path)) {
            // Use environment variable path
            actions_file_path = ascent_actions_path;
            std::cout << "Using ASCENT_ACTIONS_PATH defined in environment: " << actions_file_path << std::endl;
        } else {
            // Fallback to relative path from source directory
            std::filesystem::path source_dir = std::filesystem::path(__FILE__).parent_path();
            actions_file_path = source_dir / "ascent_helper_scripts" / "ascent_actions_default.yaml";
            std::cout << "No ASCENT_ACTIONS_PATH defined. Using default ascent_actions: " << actions_file_path << std::endl;
        }

        // Load the actions file
        if (std::filesystem::exists(actions_file_path)) {
            try {
                mActions.load(actions_file_path.string(), "yaml");
                std::cout << "Successfully loaded actions from: " << actions_file_path << std::endl;
            
                // Debug: Print the loaded actions
                std::cout << "Loaded actions:" << std::endl;
                mActions.print();
            
            } catch (const std::exception& e) {
                std::cerr << "Error loading YAML actions file: " << e.what() << std::endl;
                // Initialize empty actions as fallback
                mActions.reset();
            }
        } else {
            std::cerr << "Warning: YAML actions file not found at: " << actions_file_path << std::endl;
            // Initialize empty actions as fallback
            std::cout << "If defined Using ascent_actions.yaml/ascent_actions.json file defined in executable directory, elso no actions." << std::endl;
            mActions.reset();
        }


    
    
        // conduit::Node yaml_config;
        // yaml_config.load(yaml_path.string(), "yaml");
        // ascent_options.update(yaml_config);
        mAscent.open(ascent_options);
    }

    void Finalize() {
        conduit::Node node;
        mAscent.close();
        
    }


    void Execute_Particle(
         const auto& particleContainer,
         const auto& R_host, const auto& P_host, const auto& q_host,
         const auto& magnitude_host,
         const std::string& particlesName,
         conduit::Node& node,
         std::array<double, 3>& center ) {
            
        node["coordsets/" + particlesName + "_coords/type"].set_string("explicit");
        node["coordsets/" + particlesName + "_coords/values/x"].set_external(&R_host.data()[0][0], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        node["coordsets/" + particlesName + "_coords/values/y"].set_external(&R_host.data()[0][1], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        node["coordsets/" + particlesName + "_coords/values/z"].set_external(&R_host.data()[0][2], particleContainer->getLocalNum(), 0, sizeof(double)*3);

        node["topologies/" + particlesName + "_topo/type"].set_string("points");
        node["topologies/" + particlesName + "_topo/coordset"].set_string(particlesName + "_coords");

        node["topologies/" + particlesName + "_topo/type"].set_string("points");
        node["topologies/" + particlesName + "_topo/coordset"].set_string(particlesName + "_coords");

        /* Particle_center */
        node["coordsets/" + particlesName + "_center_coords/type"].set_string("explicit");

        node["coordsets/" + particlesName + "_center_coords/values/x"].set(&center[0], 1);
        node["coordsets/" + particlesName + "_center_coords/values/y"].set(&center[1], 1);
        node["coordsets/" + particlesName + "_center_coords/values/z"].set(&center[2], 1);
        node["topologies/" + particlesName + "_center_topo/type"].set_string("points");
        node["topologies/" + particlesName + "_center_topo/coordset"].set_string(particlesName + "_center_coords");

        std::vector<double> dummy_field = { 1.0f };
        // add a dummy field
        auto &fields = node["fields"];
        fields[particlesName + "_center/association"].set_string("vertex");
        fields[particlesName + "_center/topology"].set_string(particlesName + "_center_topo");
        fields[particlesName + "_center/volume_dependent"].set_string("false");
        fields[particlesName + "_center/values"].set(dummy_field);
        /* end particle center */

        // add values for scalar charge field
        fields[particlesName + "_charge/association"].set_string("vertex");
        fields[particlesName + "_charge/topology"].set_string(particlesName + "_topo");
        fields[particlesName + "_charge/volume_dependent"].set_string("false");

        fields[particlesName + "_charge/values"].set_external(q_host.data(), particleContainer->getLocalNum());

        // add values for vector velocity field
        //auto velocity_view = particleContainer->P.getView();
        fields[particlesName + "_velocity/association"].set_string("vertex");
        fields[particlesName + "_velocity/topology"].set_string(particlesName + "_topo");
        fields[particlesName + "_velocity/volume_dependent"].set_string("false");

        fields[particlesName + "_velocity/values/x"].set_external(&P_host.data()[0][0], particleContainer->getLocalNum(),0 ,sizeof(double)*3);
        fields[particlesName + "_velocity/values/y"].set_external(&P_host.data()[0][1], particleContainer->getLocalNum(),0 ,sizeof(double)*3);
        fields[particlesName + "_velocity/values/z"].set_external(&P_host.data()[0][2], particleContainer->getLocalNum(),0 ,sizeof(double)*3);

        fields[particlesName + "_position/association"].set_string("vertex");
        fields[particlesName + "_position/topology"].set_string(particlesName + "_topo");
        fields[particlesName + "_position/volume_dependent"].set_string("false");

        fields[particlesName + "_position/values/x"].set_external(&R_host.data()[0][0], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        fields[particlesName + "_position/values/y"].set_external(&R_host.data()[0][1], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        fields[particlesName + "_position/values/z"].set_external(&R_host.data()[0][2], particleContainer->getLocalNum(), 0, sizeof(double)*3);


        fields[particlesName + "_magnitude/association"].set_string("vertex");
        fields[particlesName + "_magnitude/topology"].set_string(particlesName + "_topo");
        fields[particlesName + "_magnitude/volume_dependent"].set_string("false");


        fields[particlesName + "_magnitude/values"].set(magnitude_host.data(), magnitude_host.extent(0));

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
        static_assert(Field::dim == 3, "current ippl AscentAdaptor only supports 3D");

        node["coordsets/" + fieldName + "_coords/type"].set_string("uniform");

        // number of points in specific dimension
        std::string field_node_dim{"coordsets/" + fieldName + "_coords/dims/i"};
        std::string field_node_origin{"coordsets/" + fieldName + "_coords/origin/x"};
        std::string field_node_spacing{"coordsets/" + fieldName + "_coords/spacing/dx"};

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
        node["topologies/" + fieldName + "_mesh/type"].set_string("uniform");
        node["topologies/" + fieldName + "_mesh/coordset"].set_string(fieldName + "_coords");
        std::string field_node_origin_topo{"topologies/" + fieldName + "_mesh/origin/x"};
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
        setData(fields, host_view_layout_left, fieldName);

        conduit::Node verify_info;
        if(!conduit::blueprint::mesh::verify(node, verify_info))
        {
            std::cerr << "Mesh verification failed!" << std::endl;
            verify_info.print();
            exit(EXIT_FAILURE);
        }
    }

    std::array<double, 3>  compute_center(const auto& particleContainer, MPI_Comm comm) {
        using memory_space = typename decltype(particleContainer->R)::memory_space;

        auto R_view = particleContainer->R.getView(); // R is a Kokkos::View<double**, MemorySpace>
        auto num_particles = R_view.extent(0);

        Kokkos::View<double[3], memory_space> local_sum("local_sum");
        Kokkos::View<double[3], Kokkos::HostSpace> global_sum("global_sum");

        // Compute local sum using parallel reduction
        Kokkos::parallel_reduce(
            "ComputeLocalSum", num_particles,
            KOKKOS_LAMBDA(const int i, double& sum_x, double& sum_y, double& sum_z) {
                sum_x += R_view(i)[0];
                sum_y += R_view(i)[1];
                sum_z += R_view(i)[2];
            },
            local_sum(0), local_sum(1), local_sum(2));

        // Copy local sum to host
        Kokkos::deep_copy(global_sum, local_sum);


        // Get local particle count
        int local_count = num_particles;
        int global_count;

        // Perform MPI Allreduce to sum positions across all ranks
        MPI_Allreduce(MPI_IN_PLACE, global_sum.data(), 3, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, comm);

        // Compute global center
        if (global_count > 0) {
            global_sum(0) /= global_count;
            global_sum(1) /= global_count;
            global_sum(2) /= global_count;
        }

        // Compute global center
        std::array<double, 3> center = {0.0, 0.0, 0.0};
        if (global_count > 0) {
            center[0] = global_sum(0);
            center[1] = global_sum(1);
            center[2] = global_sum(2);
        }


        return {global_sum(0), global_sum(1), global_sum(2)};
    }

    // Function to compute the magnitude of each point from the center
    Kokkos::View<double*, Kokkos::HostSpace> compute_magnitude_from_center(
        const auto& particleContainer, const std::array<double, 3>& center) {
        using memory_space = typename decltype(particleContainer->R)::memory_space;

        auto R_view = particleContainer->R.getView();
        auto num_particles = R_view.extent(0);

        // Create a view to hold the magnitudes
        Kokkos::View<double*, memory_space> magnitude("magnitude", num_particles);

        // Compute the magnitude (distance) from the center for each particle
        Kokkos::parallel_for(
            "ComputeMagnitude", num_particles, KOKKOS_LAMBDA(const int i) {
                double dx = R_view(i)[0] - center[0];
                double dy = R_view(i)[1] - center[1];
                double dz = R_view(i)[2] - center[2];
                magnitude(i) = sqrt(dx * dx + dy * dy + dz * dz);
            });
        
        // Copy magnitudes to host
        Kokkos::View<double*, Kokkos::HostSpace> host_magnitude("host_magnitude", num_particles);
        Kokkos::deep_copy(host_magnitude, magnitude);

        return host_magnitude;
    }

    template<typename T, unsigned Dim>
    void Execute(int cycle, double time,
    // const auto& /* std::shared_ptr<ParticleContainer<double, 3> >& */ particle,
    const std::vector<AscentAdaptor::ParticlePair<T, Dim>>& particles,
    const std::vector<FieldPair<T, Dim>>& fields) {
        conduit::Node node;

        if((cycle+1) % mFrequency != 0) return;

        // add time/cycle information
        auto state = node["state"];
        state["cycle"].set(cycle);
        state["time"].set(time);

        // Handle particles

        std::map<std::string, typename ippl::ParticleAttrib<ippl::Vector<double, 3>>::HostMirror> R_host_map;
        std::map<std::string, typename ippl::ParticleAttrib<ippl::Vector<double, 3>>::HostMirror> P_host_map;
        std::map<std::string, typename ippl::ParticleAttrib<ippl::Vector<double, 3>>::HostMirror> center_host_map;
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
            

            Kokkos::deep_copy(R_host_map[particlesName],  particleContainer->R.getView());
            Kokkos::deep_copy(P_host_map[particlesName],  particleContainer->P.getView());
            Kokkos::deep_copy(q_host_map[particlesName],  particleContainer->q.getView());
            
            std::array<double, 3> center = compute_center(particleContainer, MPI_COMM_WORLD);
            auto magnitude_host = compute_magnitude_from_center(particleContainer, center);

            Execute_Particle(
              particleContainer,
              R_host_map[particlesName], P_host_map[particlesName], q_host_map[particlesName],
              magnitude_host,
              particlesName,
              node,
              center);
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

                Execute_Field(field, fieldName, vector_host_views[fieldName], node);     
            }
        }






        // conduit::Node actions;
        /* will be overwritten if proper action file exist but needed to call execute ... */
        
        mAscent.publish(node);
        mAscent.execute(mActions);


    }
}  // namespace AscentAdaptor
} // namespace ippl

#endif



// # Copy YAML files to build directory
// configure_file(
//     ${CMAKE_CURRENT_SOURCE_DIR}/ascent_helper_scripts/ascent_actions.yaml
//     ${CMAKE_CURRENT_BINARY_DIR}/ascent_helper_scripts/ascent_actions.yaml
//     COPYONLY
// )