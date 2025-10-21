#pragma once
#include "Stream/InSitu/CatalystAdaptor.h"

#include "Ippl.h"

/* instead of maps storing kokkos view in scope we use the ViewRegistry to keep everything in frame .... and be totally type indepedent
we can set with name (but since we likely will not have the need to ever retrieve we can just stire nameless
to redzcede unncessary computin type ...) */

namespace ippl{

// META PRPOGRAMMING HELPERS

template <typename T, typename = void>
struct has_getRegionLayout : std::false_type {};

template <typename T>
struct has_getRegionLayout<T, std::void_t<decltype(std::declval<T>().getRegionLayout())>> 
    : std::true_type {};

template <typename T>
constexpr bool has_getRegionLayout_v = has_getRegionLayout<T>::value;





#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

/**
 * @brief Replaces all occurrences of a string in a file and writes the content to a new file.
 *
 * @param input_filename The path to the file to read from.
 * @param output_filename The path to the file to write the modified content to.
 * @param search_string The string to find.
 * @param replace_string The string to replace with.
 * @return true if the process was successful, false otherwise.
 */
bool CatalystAdaptor::replace_in_file(const std::string& input_filename,
                     const std::string& output_filename,
                     const std::string& search_string,
                     const std::string& replace_string) {

    // 1. Read the entire content of the input file
    std::ifstream input_file(input_filename);
    if (!input_file.is_open()) {
        std::cerr << "Error: Could not open input file: " << input_filename << std::endl;
        return false;
    }

    std::stringstream buffer;
    buffer << input_file.rdbuf(); // Read the entire file content into the stringstream
    input_file.close(); // Close the input file as soon as we're done reading

    std::string file_content = buffer.str();

    // Handle case where search_string is empty to prevent infinite loop
    if (search_string.empty()) {
        // If the search string is empty, just copy the file content
        // An empty search string usually implies no replacement should happen
    } else {
        // 2. Perform the string replacement
        size_t pos = file_content.find(search_string, 0);
        while (pos != std::string::npos) {
            file_content.replace(pos, search_string.length(), replace_string);
            // Move position past the replaced string to continue search
            pos = file_content.find(search_string, pos + replace_string.length());
        }
    }


    // 3. Write the modified content to the new output file
    std::ofstream output_file(output_filename);
    if (!output_file.is_open()) {
        std::cerr << "Error: Could not open output file: " << output_filename << std::endl;
        return false;
    }

    output_file << file_content;
    output_file.close();

    return true;
}

bool CatalystAdaptor::create_new_proxy_file(const std::string & label){

    const std::string input_filesource_dir  = (source_dir /"catalyst_scripts" / "proxy_default_EXAMPLE.xml").string();
    const std::string output_filesource_dir = (source_dir /"catalyst_scripts" / "catalyst_proxies" / ("proxy_"+ label +".xml")  ).string();
    const std::string search_string  = "EXAMPLE";
    const std::string replace_string =label;
    return replace_in_file(input_filesource_dir, output_filesource_dir, search_string, replace_string  );
}







    
/*  sets a file path to a certain node, first tries to fetch from environment, 
afterwards uses the dafault path passed  */
void CatalystAdaptor::set_node_script(
    conduit_cpp::Node node_path,
    // const char* env_var,
    const std::string env_var,
    const std::filesystem::path default_file_path
){           
        const char* file_path_env = std::getenv(env_var.c_str());
        std::filesystem::path file_path;
        if (file_path_env && std::filesystem::exists(file_path_env)) {
           ca_m << "::Initialize()::set_node_scripts(...):" << endl
                << "                Using " << env_var << " from environment:" << endl
                << "                "<< file_path_env << endl;
           file_path = file_path_env;
        } else {
           ca_m << "::Initialize()::set_node_scripts(...): No valid " << env_var <<" set." << endl 
                << "                Using default:" << endl
                << "                " << default_file_path << endl;
           file_path = default_file_path;
        }
        node_path.set(file_path.string());
}


// init visualisation for SCALAR FIELDS 
// == ippl::Field<double, 3, ippl::UniformCartesian<double, 3>, Cell>*
template<typename T, unsigned Dim, class... ViewArgs>
void CatalystAdaptor::init_entry( 
                    [[maybe_unused]]  
                      const Field<T, Dim, ViewArgs...>& entry
                    , const std::string label
                )
{
        ca_m << "::Initialize()::init_entry(ippl::Field<" << typeid(T).name() << "," << Dim << ">) called" << endl;
            
        const std::string channelName = "ippl_sField_" + label; 
        if(png_extracts){
            const std::string script = "catalyst/scripts/" + label;
            set_node_script( node[script + "/filename"],
                            // "CATALYST_EXTRACTOR_SCRIPT_S",
                            "CATALYST_EXTRACTOR_SCRIPT_" +label,
                            source_dir /"catalyst_scripts" / "catalyst_extractors" /"png_ext_sfield.py"
                        );
            conduit_cpp::Node args = node[script + "/args"];
            args.append().set_string("--channel_name");
            args.append().set_string(channelName);
            if(TestName){
                args.append().set_string("--experiment_name");
                args.append().set_string(std::string(TestName));
            }
        }

        conduit_cpp::Node script_args = node["catalyst/scripts/script/args"];
        script_args.append().set_string(channelName);

    }


// init visualisation for VECTOR FIELDS  
// == ippl::Field<ippl::Vector<double, 3>, 3, ippl::UniformCartesian<double, 3>, Cell>
template<typename T, unsigned Dim, unsigned Dim_v, class... ViewArgs>
void CatalystAdaptor::init_entry( 
                                [[maybe_unused]]  
                                  const Field<Vector<T, Dim_v>, Dim, ViewArgs...>& entry
                                , const std::string label
                            )
{
        ca_m    << "::Initialize()::init_entry(ippl::Field<ippl::Vector<"
                << typeid(T).name() << "," << Dim_v << ">," << Dim 
                << ">) called" << endl;

                
        const std::string channelName = "ippl_vField_" + label;
        if(png_extracts){
            const std::string script = "catalyst/scripts/" + label;

            set_node_script( node[script + "/filename"],
                            // "CATALYST_EXTRACTOR_SCRIPT_S",
                            "CATALYST_EXTRACTOR_SCRIPT_" + label,
                            source_dir /"catalyst_scripts" / "catalyst_extractors" /"png_ext_vfield.py"
                            
                        );
            conduit_cpp::Node args = node[script + "/args"];
            args.append().set_string("--channel_name");
            args.append().set_string(channelName);
            if(TestName){
                args.append().set_string("--experiment_name");
                args.append().set_string(std::string(TestName));
            }
        }

        conduit_cpp::Node script_args = node["catalyst/scripts/script/args"];
        script_args.append().set_string(channelName);

}

// init visualisation for PARTICLECONTAINERS derived from ParticleBaseBase:
// == ippl::ParticleBase<PLayout<T, dim>,...>,...>
template<typename T>
requires (std::derived_from<std::decay_t<T>, ParticleBaseBase>)
void CatalystAdaptor::init_entry( 
                                [[maybe_unused]]  
                                  const T& entry
                                , const std::string label
                            )
{
        ca_m    << "::Initialize()::init_entry(ParticleBase<PLayout<" 
                << typeid(particle_value_t<T>).name() << ","<< particle_dim_v<T> 
                << ",...>...> [or subclass]) called" << endl;
                    
            const std::string channelName = "ippl_particles_" + label;
            if(png_extracts){
                const std::string script = "catalyst/scripts/"+ label;

                set_node_script( 
                            node[script + "/filename"],
                            // "CATALYST_EXTRACTOR_SCRIPT_S",
                            "CATALYST_EXTRACTOR_SCRIPT_" +label,
                            source_dir /"catalyst_scripts" / "catalyst_extractors" /"png_ext_particle.py"
                );

                conduit_cpp::Node args = node[script + "/args"];
                    args.append().set_string("--channel_name");
                    args.append().set_string(channelName);
                if(TestName){
                    args.append().set_string("--experiment_name");
                    args.append().set_string(std::string(TestName));
                }
            }

            conduit_cpp::Node script_args = node["catalyst/scripts/script/args"];
            script_args.append().set_string(channelName);
}
  
/* SHARED_PTR DISPATCHER - automatically unwraps and dispatches to appropriate overload */
template<typename T>
void CatalystAdaptor::init_entry( const std::shared_ptr<T>&   entry, const std::string label)
{
    if (entry) {
        // std::cout << "  dereferencing shared pointer and reattempting execute..." << std::endl;
        init_entry(  *entry
                    , label
        );
    }
    else {
        ca_warn << "::Initialize()init_entry(nullptr):  nullptr passed as entry."         << endl
                << "       ID: "<< label                                    << endl
                << "   ==> Channel will not be registered in Conduit Node." << endl;
    }
}


// BASE CASE: only enabled if EntryT is NOT derived from ippl::ParticleBaseBase
template<typename T>
requires (!std::derived_from<std::decay_t<T>, ParticleBaseBase>)
void CatalystAdaptor::init_entry([[maybe_unused]] const T& entry, const std::string label)
{
        ca_warn <<  "::Initialize()init_entry(nullptr): Entry type can't be processed."   << endl 
                <<  "       ID: "<< label                                   << endl 
                <<  "       Type: "<< typeid(std::decay_t<T>).name()        << endl 
                <<  "   ==>Channel will not be registered in Conduit Node." << endl;
}
  


// ==========================================================
// CHANNEL EXECUTIONERS =====================================
// ==========================================================

// ▶ create_mirror_view:           allocates data only if the host process cannot access view’s data, 
//                                  otherwise hostView references the same data.
// ▶ create_mirror:                always allocates data.
// ▶ create_mirror_view_and_copy:  allocates data if necessary and also copies data.
// Reminder: Kokkos never performs a hidden deep copy

// If needed, deep copy the view’s updated array back to the
// hostView’s array to write file, etc.
// Kokkos :: d e ep c op y ( hostView , view );
               


// execute visualisation for FIELDS: check dimension independence
// == ippl::Field<double, 3, ippl::UniformCartesian<double, 3>, Cell>*
// == ippl::Field<ippl::Vector<double, 3>, 3, ippl::UniformCartesian<double, 3>, Cell>*
template<typename T, unsigned Dim, class... ViewArgs>
void CatalystAdaptor::execute_entry(const Field<T, Dim, ViewArgs...>& entry, const std::string label)
{

        const Field<T, Dim, ViewArgs...>* field = &entry;
        std::string channelName;
        if constexpr (std::is_scalar_v<T>) {
            channelName = "ippl_sField_" + label;
            ca_m << "::Execute()::execute_entry(" << label << ") |Type:ippl::Field<" << typeid(T).name() << "," << Dim << ">) called" << endl;
                
        } else if constexpr (is_vector_v<T>) {
            channelName = "ippl_vField_" + label;
            ca_m << "::Execute()::execute_entry(" << label << ") |Type: ippl::Field<ippl::Vector<" << typeid(T).name() << "," << Field<T, Dim, ViewArgs...>::dim << ">," << Dim << ">)" << endl;
        }else{
            channelName = "ippl_errorField_" + label;

            ca_m    << "::Execute()::execute_entry(Field<"<<typeid(T).name()<< ">)" << endl
                    << "    For this type of Field the Conduit Blueprint description wasnt \n" 
                    << "    implemented in ippl. Therefore this type of field is not \n"
                    << "    supported for visualisation." << endl;
        }
            /* options can diverge in the following levels:
            /type =="mesh" or multimesh
            channels/ *** /data/fields/     *** -> changes
                               /coordsets/  *** -> stays the same
                               /topologies/ *** -> stays the same for al
        
            Currently we are using the same topology and mesh we can
            technically reuse the same channel ->
            but for advanced uses might not be the case
            the labeling og coordsets and topologies in accordances with 
            the layouts ids of ippl would be interesting    */

        // channel for this field
        // channel of type mesh adheres to conduits mesh blueprint
        auto channel = node["catalyst/channels/"+ channelName];
        channel["type"].set_string("mesh");
        auto data   = channel["data"];
        auto fields = channel["data/fields"];
        auto field_node = fields[label];
        // add topology anmed fmesh_topo of type uniform
        data["topologies/fmesh_topo/type"].set_string("uniform");
        // define which coordinates to use 
        data["topologies/fmesh_topo/coordset"].set_string("cart_uniform_coords");
        // add a coordinate set named cart_uniform_coords  of type uniform
        data["coordsets/cart_uniform_coords/type"].set_string("uniform");

        const auto Layout_        = field->getLayout();
        const auto Mesh_          = field->get_mesh();
        const auto nGhost         = field->getNghost();


        const auto LocalNDIndex_  = Layout_.getLocalNDIndex();
        const auto Origin_        = Mesh_.getOrigin();
        const auto Spacing_       = Mesh_.getMeshSpacing();

        {
            data["coordsets/cart_uniform_coords/dims/i"].set(LocalNDIndex_[0].length() + 1);
            data["coordsets/cart_uniform_coords/spacing/dx"].set(Spacing_[0]);
            data["coordsets/cart_uniform_coords/origin/x"].set( Origin_[0] + LocalNDIndex_[0].first() * Spacing_[0] );
            data["topologies/fmesh_topo/origin/x"].set(  Origin_[0] + LocalNDIndex_[0].first() * Spacing_[0] );
        }
        if constexpr(Dim >= 2){
            data["coordsets/cart_uniform_coords/dims/j"].set(LocalNDIndex_[1].length() + 1);
            data["coordsets/cart_uniform_coords/spacing/dy"].set(Spacing_[1]);
            data["coordsets/cart_uniform_coords/origin/y"].set( Origin_[1] + LocalNDIndex_[1].first() * Spacing_[1] );
            data["topologies/fmesh_topo/origin/y"].set(  Origin_[1] + LocalNDIndex_[1].first() * Spacing_[1] );
        }
        if constexpr(Dim >= 3){
            data["coordsets/cart_uniform_coords/dims/k"].set(LocalNDIndex_[2].length() + 1);
            data["coordsets/cart_uniform_coords/spacing/dz"].set(Spacing_[2]);
            data["coordsets/cart_uniform_coords/origin/z"].set( Origin_[2] + LocalNDIndex_[2].first() * Spacing_[2] );
            data["topologies/fmesh_topo/origin/z"].set(  Origin_[2] + LocalNDIndex_[2].first() * Spacing_[2] );
        }

        // host_view_layout_left 
        // Kokkos::View<typename Field<T, Dim, ViewArgs...>::view_type::data_type, Kokkos::LayoutLeft, Kokkos::HostSpace>
        // hostMirrorNoGhosts;
        const size_t nx = LocalNDIndex_[0].length();
        const size_t ny = (Dim >= 2) ? LocalNDIndex_[1].length() : 1;
        const size_t nz = (Dim >= 3) ? LocalNDIndex_[2].length() : 1;
        
        /* upscale to 3D host in all cases ... view not ideal but should work */
        Kokkos::View<typename Field<T, Dim, ViewArgs...>::view_type::data_type, Kokkos::LayoutLeft, Kokkos::HostSpace> 
        hostMirrorNoGhosts
        (
        //    "host_view_layout_left",
            "hostMirrorNoGhosts",
            nx,
            ny,
            nz
        );

        // Creates a host-accessible mirror view and copies the data from the device view to the host.
        auto hostMirror =   Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field->getView());
        
        viewRegistry.set(hostMirrorNoGhosts);


        // // Copy data from field to the memory+style which will be passed to Catalyst
        // for (size_t i = 0; i < nx; ++i) {
        //     if constexpr(Dim >= 2){
        //         for (size_t j = 0; j <ny; ++j) {
        //             if constexpr(Dim >= 2) {
        //                 for (size_t k = 0; k < ny; ++k) {
        //                     hostMirrorNoGhosts(i, j, k) = hostMirror(i + nGhost, j + nGhost, k + nGhost);
        //                 }
        //             }
        //             else{
        //                     hostMirrorNoGhosts(i, j) = hostMirror(i + nGhost, j + nGhost);
        //             }
        //         }
        //     }
        //     else {
        //         hostMirrorNoGhosts(i) = hostMirror(i + nGhost);
        //     }
        // }

        
        // auto at = [&](auto& v, size_t i, size_t j, size_t k) -> auto& {
        //     if constexpr (Dim == 1) return v(i);
        //     else if constexpr (Dim == 2) return v(i, j);
        //     else return v(i, j, k);
        // };
        
        // The compiler will see that ny or nz is 1 for lower dimensions, so the loops 
        // will execute only once, but the loop structure itself will remain. The code 
        // will still perform the extra loop(s), just with a single iteration.
        // But this code will be more maintainable for higher dimension (if there ever are any ...)
        for (size_t i = 0; i < nx; ++i) {
            for (size_t j = 0; j < ny; ++j) {
                for (size_t k = 0; k < nz; ++k) {
                    // at(hostMirrorNoGhosts, i, j, k) =
                    //     at(hostMirror,
                    //        i + nGhost,
                    //        (Dim >= 2) ? j + nGhost : 0,
                    //        (Dim >= 3) ? k + nGhost : 0);
                    hostMirrorNoGhosts(i, j, k) = hostMirror(i + nGhost, j + nGhost, k + nGhost);
                }
            }
        }




        if constexpr (std::is_scalar_v<T>) {
            // --- SCALAR FIELD CASE ---
            field_node["association"].set_string("element");
            field_node["topology"].set_string("fmesh_topo");
            field_node["volume_dependent"].set_string("false");
            field_node["values"].set_external(hostMirrorNoGhosts.data(), hostMirrorNoGhosts.size());

        } else if constexpr (is_vector_v<T>) {
            // --- VECTOR FIELD CASE ---
            field_node["association"].set_string("element");
            field_node["topology"].set_string("fmesh_topo");
            field_node["volume_dependent"].set_string("false");
            
            
            // Use true AoS element size as stride (handles padding)
            // const size_t stride_bytes = sizeof(typename T::value_type)*T::dim;
            // will this surely not be double but vector<double 
            // returning size of vector and amounts of vectorss


            auto n_elems = hostMirrorNoGhosts.size();
            // constexpr auto stride_bytes = T::dim*sizeof(typename T::value_type);
            using elem_t = std::remove_pointer_t<decltype(hostMirrorNoGhosts.data())>;
            static constexpr size_t stride_bytes = sizeof(elem_t);

    
            // offset is zero?? guaranteed?
            // stride was 1 in predecessor code? how did this work?...
                                     field_node["values/x"].set_external(&hostMirrorNoGhosts.data()[0][0], n_elems, 0, stride_bytes);
            if constexpr (T::dim>=2) field_node["values/y"].set_external(&hostMirrorNoGhosts.data()[0][1], n_elems, 0, stride_bytes);
            if constexpr (T::dim>=3) field_node["values/z"].set_external(&hostMirrorNoGhosts.data()[0][2], n_elems, 0, stride_bytes);        
                
        } 
        // else {
            // --- INVALID CASE ---
        // }
}




// execute visualisation for PARTICLECONTAINERS derived from ParticleBaseBase:
// == ippl::ParticleBase<PLayout<T, dim>,...>,...>
template<typename T>
requires (std::derived_from<std::decay_t<T>, ParticleBaseBase>)
void CatalystAdaptor::execute_entry(const T& entry, const std::string label) 
{
        ca_m        << "::Execute()::execute_entry(" << label << ") | Type : ParticleBase<PLayout<" 
                    << typeid(particle_value_t<T>).name() 
                    << ","
                    << particle_dim_v<T> 
                    << ",...>...> [or subclass])" << endl;
        const std::string channelName = "ippl_particles_" + label;

        auto particleContainer = &entry;
        assert((particleContainer->ID.getView().data() != nullptr) && "ID view should not be nullptr, might be missing the right execution space");

        // channel for this particleContainer
        // channel of type mesh adheres to conduits mesh blueprint
        auto channel = node["catalyst/channels/"+ channelName];
        channel["type"].set_string("mesh");
        

        // conduit_cpp::Node data =
        auto data = 
        channel["data"];
        auto fields = channel["data/fields"];
        
        
        
        /* avoid hardcoded and shortenn ,... */
        // ParticleAttrib<std::int64_t>::HostMirror      ID_hostMirror;
        // ParticleAttrib<Vector<double, 3>>::HostMirror R_hostMirror;
        // ID_hostMirror = particleContainer->ID.getHostMirror();
        // R_hostMirror  = particleContainer->R.getHostMirror();
        // Kokkos::deep_copy(ID_hostMirror,  particleContainer->ID.getView());
        // Kokkos::deep_copy(R_hostMirror ,  particleContainer->R.getView());
        
        
        // Creates a host-accessible mirror view and copies the data from the device view to the host.
        // compared to get_mirror and get_mirror_view host space is not guaranteed default behaviour so we specify...
        // comType HostMirror would let the function auto deduct the wanted space ...


        // ID_view = particleContainer->ID.getView();
        // R_view  = particleContainer->R.getView()
        // decltype(ID_view)::host_mirror_type::
        // ID_view::HostMirror 
        /* if original is on host space no copy will b created and any changs will be taken over ... */
        auto ID_hostMirror =   Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), particleContainer->ID.getView());
        auto  R_hostMirror  =   Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), particleContainer->R.getView());
        viewRegistry.set(ID_hostMirror);
        viewRegistry.set(R_hostMirror);



        using PLayout_t = T::Layout_t;
        // using vector_t = T::Layout_t::vector_type;
        // using value_t  = T::Layout_t::value_type;
        //avoids padding etc
        using R_elem_t = std::remove_pointer_t<decltype(R_hostMirror.data())>;
        static constexpr size_t R_stride_bytes = sizeof(R_elem_t);


        /* CHECK IF PLAYOUT IS SPATIAL LAYOUT OR PURE LAYOUT */
        if constexpr (has_getRegionLayout_v<PLayout_t>){

            using RLayout_t  = PLayout_t::RegionLayout_t;
            using NDRegion_t = RLayout_t::NDRegion_t;
            constexpr unsigned dim_ = PLayout_t::dim;
            // using value_type = PLayout_t::value_type;
            const NDRegion_t ndr = particleContainer->getLayout().getRegionLayout().getDomain();
            
            /* HELPER COORDINATES TO PASS THE BOUNDING BOX in vtk format*/
            data["coordsets/bound_helper_coords/type"].set_string("uniform");
            /* HELPER TOPOLOGY    TO PASS THE BOUNDING BOX (??even needed??)  in vtk format */
            data["topologies/bound_helper_topo/coordset"].set_string("bound_helper_coords");
            data["topologies/bound_helper_topo/type"].set_string("uniform");
            /* create unfirom coordinate mesh only consisting of the corner points of the domain */ 
            {
                data["coordsets/bound_helper_coords/dims/i"].set(2);
                data["coordsets/bound_helper_coords/spacing/dx"].set( ndr[0].max()  - ndr[0].min() );
                data["coordsets/bound_helper_coords/origin/x"].set(   ndr[0].min() );
                data["topologies/bound_helper_topo/origin/x"].set(    ndr[0].min() );
            }
            if constexpr(dim_ >= 2){
                data["coordsets/bound_helper_coords/dims/j"].set(2);
                data["coordsets/bound_helper_coords/spacing/dy"].set( ndr[1].max()- ndr[1].min() );
                data["coordsets/bound_helper_coords/origin/y"].set(   ndr[1].min()               );
                data["topologies/bound_helper_topo/origin/y"].set(    ndr[1].min()               );
            }
            if constexpr(dim_ >= 3){
                data["coordsets/bound_helper_coords/dims/k"].set(2);
                data["coordsets/bound_helper_coords/spacing/dz"].set( ndr[2].max()- ndr[1].min() );
                data["coordsets/bound_helper_coords/origin/z"].set(   ndr[2].min()               );
                data["topologies/bound_helper_topo/origin/z"].set(    ndr[2].min()               );
            }
    } 
    // else {
            /* will use raw particle data instead .. */
            /* or do we now anything like unit square ??? */
            // data["coordsets/bound_helper_coords/type"].set_string("uniform");
            // data["topologies/bound_helper_topo/coordset"].set_string("bound_helper_coords");
            // data["topologies/bound_helper_topo/type"].set_string("uniform");
            // {
            //     data["coordsets/bound_helper_coords/dims/i"].set(2);
            //     data["coordsets/bound_helper_coords/spacing/dx"].set(1);
            //     data["coordsets/bound_helper_coords/origin/x"].set(  0);
            //     data["topologies/bound_helper_topo/origin/x"].set(   1);
            // }....
    // }
        
        
        /* ATTRIBUTES HARDCODED IN PARTICELBASE are identity ID and position R */
        /* EXPLICIT COORDINATES -> EACH PARTICLE POSITION */
        data["coordsets/p_explicit_coords/type"].set_string("explicit");
        data["coordsets/p_explicit_coords/values/x"].set_external(&R_hostMirror.data()[0][0], particleContainer->getLocalNum(), 0, R_stride_bytes);
        data["coordsets/p_explicit_coords/values/y"].set_external(&R_hostMirror.data()[0][1], particleContainer->getLocalNum(), 0, R_stride_bytes);
        data["coordsets/p_explicit_coords/values/z"].set_external(&R_hostMirror.data()[0][2], particleContainer->getLocalNum(), 0, R_stride_bytes);
        

        /* UNSTRUCTURED TOPOLOGY RELYING ON UNIQUE PARTICLE ID'S */
        data["topologies/p_unstructured_topo/type"].set_string("unstructured");
        data["topologies/p_unstructured_topo/coordset"].set_string("p_explicit_coords");
        data["topologies/p_unstructured_topo/elements/shape"].set_string("point");
        data["topologies/p_unstructured_topo/elements/connectivity"].set_external(ID_hostMirror.data(),particleContainer->getLocalNum());
        /* concept for no copy in situ vis would be */
        //mesh["topologies/p_unstructured_topo/elements/connectivity"].set_external(particleContainer->ID.getView().data(),particleContainer->getLocalNum());
        
        /* POSITION ATTRIBUTE */
        // this can be left hardcodeed or made part of the for loop, but more efficient to do it right here since we already have the hostView
        fields["position/association"].set_string("vertex");
        fields["position/topology"].set_string("p_unstructured_topo");
        fields["position/volume_dependent"].set_string("false");
        fields["position/values/x"].set_external(&R_hostMirror.data()[0][0], particleContainer->getLocalNum(), 0, R_stride_bytes);
        fields["position/values/y"].set_external(&R_hostMirror.data()[0][1], particleContainer->getLocalNum(), 0, R_stride_bytes);
        fields["position/values/z"].set_external(&R_hostMirror.data()[0][2], particleContainer->getLocalNum(), 0, R_stride_bytes);
            
        /* "no way" to know what dimensions and types particle attributes will have
        from pointer instance since is base pointer make execute member method instead to 
        use virtual functions. Is most pragmatic approach. */

        // entry.template forAllAttributes<void>(
        //     [&]<typename Attributes>(const Attributes& atts) {
        //         for (auto* attribute : atts) {
        const size_t n_attributes =  entry.getAttributeNum();
        for(size_t i = 2; i < n_attributes; ++i){
            const auto attribute = entry.getAttribute(i);
            const std::string  attribute_name = attribute->get_name();
            // ca_m << "Execute attribute: " << attribute_name << endl;
            attribute->signConduitBlueprintNode_rememberHostCopy(particleContainer->getLocalNum(), fields, viewRegistry, ca_m, ca_warn);
        }
        
}

// BASE CASE: only enabled if EntryT is NOT derived from ippl::ParticleBaseBase
template<typename T>
requires (!std::derived_from<std::decay_t<T>, ParticleBaseBase>)
 void CatalystAdaptor::execute_entry(   [[maybe_unused]] T&& entry,  const std::string label)
{
        ca_m << "  Entry type can't be processed: ID "<< label <<" "<< typeid(std::decay_t<T>).name() << endl;
    }

    /* SHARED_PTR DISPATCHER - automatically unwraps and dispatches to appropriate overload */
template<typename T>
 void CatalystAdaptor::execute_entry(   const std::shared_ptr<T>& entry,const std::string  label )
{
        if (entry) {
            ca_m << "  dereferencing shared pointer and reattempting execute..." << endl;
            execute_entry(*entry, label
                // ,  node, vr
            );  // Dereference and dispatch to reference version
        } else {
            ca_m << "  Null shared_ptr encountered" << endl;
        }
}



template<typename T>
void CatalystAdaptor::InitSteerableChannel( [[maybe_unused]] const T& steerable_scalar_forwardpass,  const std::string& label ){
    ca_m << "::Initialize()::InitSteerableChannel(" << label << "):  | Type: " << typeid(T).name() << endl;
    
    
    if(create_new_proxy_file(label)){
        ca_m << "::Initialize()::InitSteerableChannel(" << label << "):  | Creating proxy file 'proxy_" << label << ".xml': SUCCESS "<< endl;
        
        conduit_cpp::Node script_args = node["catalyst/scripts/script/args"];
        script_args.append().set_string(label);
        
        set_node_script(    node["catalyst/proxies/proxy_" + label +"/filename"],
            "CATALYST_PROXYS_PATH_"+label,
            source_dir / "catalyst_scripts" / "catalyst_proxies" / ("proxy_"+label+".xml") );   
    }
    else{
        ca_m << "::Initialize()::InitSteerableChannel(" << label << "):  | Creating proxy file 'proxy_" << label << ".xml': FAILED "<< endl;
    }
                        
}






template<typename T>
void CatalystAdaptor::AddSteerableChannel( const T& steerable_scalar_forwardpass,  const std::string& steerable_suffix ) 
{
        ca_m << "::Execute()::AddSteerableChannel(" << steerable_suffix << ");  | Type: " << typeid(T).name() << endl;

        
        
        
        auto steerable_channel = node["catalyst/channels/steerable_channel_forward_" + steerable_suffix];

        steerable_channel["type"].set("mesh");
        auto steerable_data = steerable_channel["data"];
        steerable_data["coordsets/coords/type"].set_string("explicit");
        steerable_data["coordsets/coords/values/x"].set( 0 );

        steerable_data["topologies/sMesh_topo/type"].set("unstructured");
        steerable_data["topologies/sMesh_topo/coordset"].set("coords");
        steerable_data["topologies/sMesh_topo/elements/shape"].set("point");
        steerable_data["topologies/sMesh_topo/elements/connectivity"].set( 0 );
        
        /* 3D double is not mandatory ?? and irrelevant we want to pass minimal vtk object with 1 data point */
        // steerable_data["coordsets/coords/values/x"].set_float64_vector({ 0 });
        // steerable_data["coordsets/coords/values/y"].set_float64_vector({ 0 });
        // steerable_data["coordsets/coords/values/z"].set_float64_vector({ 0 });

        // steerable_data["topologies/sMesh_topo/elements/connectivity"].set_int32_vector({ 0 });


        conduit_cpp::Node steerable_field = steerable_data["fields/steerable_field_f_" + steerable_suffix];
        steerable_field["association"].set("vertex");
        steerable_field["topology"].set("sMesh_topo");
        steerable_field["volume_dependent"].set("false");

        conduit_cpp::Node values = steerable_field["values"];

        // std::is_scalar_v<std::decay_t<T>>
        if constexpr(std::is_scalar_v<T>){
            values.set(steerable_scalar_forwardpass);
        }        
        else {
            throw IpplException("CatalystAdaptor::AddSteerableChannel", "Unsupported steerable type for channel: " + steerable_suffix);
        }
}

    /* maybe use function overloading instead ... */
        // const std::string value_path = ... 
        // steerable_scalar_backwardpass = static_cast<T>(results[value_path].value());
        // steerable_scalar_backwardpass = results[value_path].value()[0];
        /* ????? this should work?? */
        // if constexpr (std::is_same_v<std::remove_cvref_t<T>, double>) {

template<typename T>
void CatalystAdaptor::FetchSteerableChannelValue( T& steerable_scalar_backwardpass, 
                                                    const std::string& label
                                                ) {
        ca_m << "::Execute()::FetchSteerableChannel(" << label  << ") | Type: " << typeid(T).name() << endl;

            
            conduit_cpp::Node steerable_channel     = results["catalyst/steerable_channel_backward_" + label];
            conduit_cpp::Node steerable_field       = steerable_channel[ "fields/steerable_field_b_" + label];
            conduit_cpp::Node values = steerable_field["values"];

    if (!steerable_field["values"].dtype().is_number()) {
          throw IpplException("CatalystAdaptor::FetchSteerableChannelValue", "Unsupported steerable value type for channel: " + label);
    }
    else{
        if constexpr (std::is_same_v<T, double>) {
                steerable_scalar_backwardpass = steerable_field["values"].to_double();
            // }
            // else if (steerable_field["values"].dtype().is_float64()) {
            //     auto ptr = steerable_field["values"].as_float64_ptr();
            //     if (ptr){
            //         steerable_scalar_backwardpass = ptr[0];
            //         std::cout << "value vector fetched ..." << std::endl;
            //     }
            //     else throw IpplException("CatalystAdaptor::FetchSteerableChannelValue", "Null pointer for steerable value: " + label);
            // }
        }
        else if constexpr (std::is_same_v<T, float>) {
            
                steerable_scalar_backwardpass = steerable_field["values"].to_float();
        
            // else if (steerable_field["values"].dtype().is_float32()) {
            //     auto ptr = steerable_field["values"].as_float64_ptr();
            //     if (ptr) steerable_scalar_backwardpass = ptr[0];
            //     else throw IpplException("CatalystAdaptor::FetchSteerableChannelValue", "Null pointer for steerable value: " + label);
            // }
            
        }
        else if constexpr (std::is_same_v<T, int>) {
                steerable_scalar_backwardpass = steerable_field["values"].to_int32();
            
            // else if (steerable_field["values"].dtype().is_float64()) {
            //     auto ptr = steerable_field["values"].as_float64_ptr();
            //     if (ptr) steerable_scalar_backwardpass = ptr[0];
            //     else throw IpplException("CatalystAdaptor::FetchSteerableChannelValue", "Null pointer for steerable value: " + label);
            // }
            
        }
        else if constexpr (std::is_same_v<T, unsigned int>) {
                steerable_scalar_backwardpass = steerable_field["values"].to_uint32();
            
            // else if (steerable_field["values"].dtype().is_float64()) {
            //     auto ptr = steerable_field["values"].as_float64_ptr();
            //     if (ptr) steerable_scalar_backwardpass = ptr[0];
            //     else throw IpplException("CatalystAdaptor::FetchSteerableChannelValue", "Null pointer for steerable value: " + label);
            // }
        }
        else {
            throw IpplException("CatalystAdaptor::FetchSteerableChannelValue", "failed to fetch scalar value, Unsupported steerable type for channel: " + label);
        } 
    }
    ca_m << "::Execute()::FetchSteerableChannel(" << label  << ") | received:" << steerable_scalar_backwardpass << endl; 
}




void CatalystAdaptor::fetchResults() {
        
        catalyst_status err = catalyst_results(conduit_cpp::c_node(&results));
        if (err != catalyst_status_ok)
        {
            std::cerr << "Failed to execute Catalyst-results: " << err << std::endl;
        }
        // else
        // {
        //     std::cout << "Result Node dump:" << std::endl;
        //     results.print();
        // }   
    }


// =====================================================================================
// Runtime registry based Initialize / Execute (non-templated registry)
// =====================================================================================

void CatalystAdaptor::InitializeRuntime(
                           const std::shared_ptr<VisRegistryRuntime>& visReg,
                           const std::shared_ptr<VisRegistryRuntime>& steerReg
                        ) {


    ca_m << "::Initialize() START============================================================= 0" << endl;
    
        // if ( !(catalyst_steer && std::string(catalyst_steer) == "OFF") ){
        // m << "Catalyst Visualisation was deactivated via setting env variable IPPL_CATALYST_VIS=OFF"
                    
    visRegistry   = visReg;
    steerRegistry = steerReg;


    // Pipeline script (allow override by environment)
    set_node_script(node["catalyst/scripts/script/filename"],
                    "CATALYST_PIPELINE_PATH",
                    source_dir / "catalyst_scripts" / "pipeline_default.py");
    conduit_cpp::Node args = node["catalyst/scripts/script/args"];

    args.append().set_string("--channel_names");
    
    // If PNG extraction requested, run init visitor over visualization registry
    // init_entry will also add channel names here into the node.
    InitVisitor initV{*this};
    visRegistry->for_each(initV);

    

    args.append().set_string("--VTKextract");
    args.append().set_string(std::string(catalyst_vtk));

    args.append().set_string("--steer");
    args.append().set_string(std::string(catalyst_steer));


    args.append().set_string("--steer_channel_names");
        
    if (steer_enabled ) {
        // set_node_script(node["catalyst/proxies/proxy_e/filename"],
        //                 "CATALYST_PROXYS_PATH_E",
        //                 source_dir / "catalyst_scripts" / "proxy_default_electric.xml");
        // set_node_script(node["catalyst/proxies/proxy_m/filename"],
        //                 "CATALYST_PROXYS_PATH_M",
        //                 source_dir / "catalyst_scripts" / "proxy_default_magnetic.xml");


        SteerInitVisitor steerInitV{*this};
        steerRegistry->for_each(steerInitV);
    } 
    // else {
    // }










    ca_m << "::Initialize()   Printing Conduit node passed to catalyst_initialize() =>" << endl;
    // ca_m << node.to_json() << endl;
    ca_m << node.to_yaml() << endl;
        
    catalyst_status err = catalyst_initialize(conduit_cpp::c_node(&node));
    if (err != catalyst_status_ok) {
        ca_m << "::Initialize()   Catalyst initialization failed." << endl;
        throw IpplException("Stream::InSitu::CatalystAdaptor", "Failed to initialize Catalyst!!!");
    } else {
        ca_m << "::Initialize()   Catalyst initialized successfully." << endl;
    }
    node.reset();
    ca_m << "::Initialize()  DONE============================================================= 1" << endl;
    
}

void CatalystAdaptor::ExecuteRuntime( int cycle, double time, int rank /* default = ippl::Comm->rank() */) {
    ca_m << "::Execute() START =============================================================== 0" << endl;
    
    const char* catalyst_steer = std::getenv("IPPL_CATALYST_STEER");
    const char* catalyst_vis = std::getenv("IPPL_CATALYST_VIS");

    auto state = node["catalyst/state"];
    state["cycle"].set(cycle);
    state["time"].set(time);
    state["domain_id"].set(rank);

    // m << "Catalyst Visualisation was deactivated via setting env variable IPPL_CATALYST_VIS=OFF" << endl;
    if ( !(catalyst_vis && std::string(catalyst_vis) == "OFF") ){
        // forward Node: add visualisation channels
        ExecuteVisitor execV{*this};
        visRegistry->for_each(execV); 

    }

    if (catalyst_steer && std::string(catalyst_steer) == "ON") {
        // forward Node: add steering channels
        SteerForwardVisitor steerV{*this};
        steerRegistry->for_each(steerV); 
    }



    if( cycle == 0){
        ca_m << "::Execute()   Printing first Conduit Node passed to catalyst_execute() ==>" << endl;
        if(level >= ca_m.getOutputLevel() && ippl::Comm->rank()==0) node.print();
        // if(level >= 5 && ippl::Comm->rank()==0)  node.print();

        ca_m    << "::Execute() During first catalyst_execute() catalyst will "     << endl
                << "            for each passed script - in order how they were "   << endl 
                << "            passed to the conduit node - run the globa scope,"  << endl 
                << "             the initialize() and the execute()."               << endl;
    }

    
    ca_m << "::Execute()::catalyst_execute() ==>" << endl;
    // Conduit Node Forward pass to Catalyst
    catalyst_status err = catalyst_execute(conduit_cpp::c_node(&node));
    if (err != catalyst_status_ok) {
        std::cerr << "::Execute()   Failed to execute Catalyst (runtime path): " << err << std::endl;
    }

    if (catalyst_steer && std::string(catalyst_steer) == "ON") {
        // Conduit Node Backward pass from Catalyst in results
        // conduit_cpp::Node results;
        fetchResults();
        if(cycle == 0){
            ca_m << "::Execute()   Printing first Conduit Node received from catalyst_execute() ==>" << endl;
            // ca_m << node.to_json() << endl;
            ca_m << results.to_yaml() << endl;

        }
        // backward Node: fetch updated steering values
        SteerFetchVisitor fetchV{*this};
        steerRegistry->for_each(fetchV);
    }
        Kokkos::fence();
        viewRegistry.clear();
        node.reset();
        results.reset();
        
    ca_m << "::Execute()  DONE =============================================================== 1" << endl;
 
}


void CatalystAdaptor::Finalize() {
    conduit_cpp::Node node;
    catalyst_status err = catalyst_finalize(conduit_cpp::c_node(&node));
    if (err != catalyst_status_ok) {
        std::cerr << "Failed to finalize Catalyst: " << err << std::endl;
    }
}



}//ippl