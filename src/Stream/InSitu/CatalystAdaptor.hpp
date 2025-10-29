#pragma once
#include "Stream/InSitu/CatalystAdaptor.h"

#include "Ippl.h"

/* instead of maps storing kokkos view in scope we use the ViewRegistry to keep everything in frame .... and be totally type indepedent
we can set with name (but since we likely will not have the need to ever retrieve we can just stire nameless
to redzcede unncessary computin type ...) */

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
namespace ippl{

// ==============================================================================================
// META PRPOGRAMMING HELPERS ====================================================================
// ==============================================================================================

template <typename T, typename = void>
struct has_getRegionLayout : std::false_type {};

template <typename T>
struct has_getRegionLayout<T, std::void_t<decltype(std::declval<T>().getRegionLayout())>> 
    : std::true_type {};

template <typename T>
constexpr bool has_getRegionLayout_v = has_getRegionLayout<T>::value;




// ==============================================================================================
//  HELPERS =====================================================================================
// ==============================================================================================
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


// ==============================================================================================
// VIS CHANNEL INITIALIZER:======================================================================
// ==============================================================================================

// init visualisation for SCALAR FIELDS 
// == ippl::Field<double, 3, ippl::UniformCartesian<double, 3>, Cell>*
template<typename T, unsigned Dim, class... ViewArgs>
void CatalystAdaptor::init_entry( [[maybe_unused]]  const Field<T, Dim, ViewArgs...>& entry , const std::string label)
{
        ca_m << "::Initialize()::init_entry(ippl::Field<" << typeid(T).name() << "," << Dim << ">) called" << endl;
        forceHostCopy[label] = false;

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
            args.append().set_string("--label");
            args.append().set_string(label);
            if(TestName){
                args.append().set_string("--experiment_name");
                args.append().set_string(std::string(TestName));
            }

            args.append().set_string("--verbosity");
            args.append().set_string(std::to_string(ca_m.getOutputLevel()));
        }

        conduit_cpp::Node script_args = node["catalyst/scripts/script/args"];
        script_args.append().set_string(channelName);
}


// init visualisation for VECTOR FIELDS  
// == ippl::Field<ippl::Vector<double, 3>, 3, ippl::UniformCartesian<double, 3>, Cell>
template<typename T, unsigned Dim, unsigned Dim_v, class... ViewArgs>
void CatalystAdaptor::init_entry( [[maybe_unused]]  const Field<Vector<T, Dim_v>, Dim, ViewArgs...>& entry , const std::string label)
{
        ca_m    << "::Initialize()::init_entry(ippl::Field<ippl::Vector<"
                << typeid(T).name() << "," << Dim_v << ">," << Dim 
                << ">) called" << endl;

        forceHostCopy[label] = false;

                
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
            args.append().set_string("--label");
            args.append().set_string(label);
            if(TestName){
                args.append().set_string("--experiment_name");
                args.append().set_string(std::string(TestName));
            }
            args.append().set_string("--verbosity");
            args.append().set_string(std::to_string(ca_m.getOutputLevel()));
        }


        conduit_cpp::Node script_args = node["catalyst/scripts/script/args"];
        script_args.append().set_string(channelName);

}

// init visualisation for PARTICLECONTAINERS derived from ParticleBaseBase:
// == ippl::ParticleBase<PLayout<T, dim>,...>,...>
template<typename T>
requires (std::derived_from<std::decay_t<T>, ParticleBaseBase>)
void CatalystAdaptor::init_entry( [[maybe_unused]]  const T& entry, const std::string label)
{
        ca_m    << "::Initialize()::init_entry(ParticleBase<PLayout<" 
                << typeid(particle_value_t<T>).name() << ","<< particle_dim_v<T> 
                << ",...>...> [or subclass]) called" << endl;
                    
            
        forceHostCopy[label] = false;

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
                    args.append().set_string("--label");
                    args.append().set_string(label);

                if(TestName){
                    args.append().set_string("--experiment_name");
                    args.append().set_string(std::string(TestName));
                }

                args.append().set_string("--verbosity");
                args.append().set_string(std::to_string(ca_m.getOutputLevel()));
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
  


// ==============================================================================================
// CHANNEL EXECUTIONERS =========================================================================
// ==============================================================================================

// ▶ create_mirror_view:           allocates data only if the host process cannot access view’s data, 
//                                  otherwise hostView references the same data.
// ▶ create_mirror:                always allocates data.
// ▶ create_mirror_view_and_copy:  allocates data if necessary and also copies data.
// Reminder: Kokkos never performs a hidden deep copy

// If needed, deep copy the view’s updated array back to the
// hostView’s array to write file, etc.
// Kokkos :: d e ep c op y ( hostView , view );
               

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




// execute visualisation for FIELDS: check dimension independence
// == ippl::Field<double, 3, ippl::UniformCartesian<double, 3>, Cell>*
// == ippl::Field<ippl::Vector<double, 3>, 3, ippl::UniformCartesian<double, 3>, Cell>*
template<typename T, unsigned Dim, class... ViewArgs>
void CatalystAdaptor::execute_entry(const Field<T, Dim, ViewArgs...>& entry, const std::string label)
{   /* HANDLES CASE IF ENTRY HAS BEEN EXECUTED WITH REMEMBER_NOW */
    if(viewRegistry.contains(label)) return;

    using Field_type = Field<T, Dim, ViewArgs...>;
    const Field_type* field = &entry;

        std::string channelName;
        if constexpr (std::is_scalar_v<T>) {
            channelName = "ippl_sField_" + label;
            ca_m << "::Execute()::execute_entry(" << label << ") | Type:ippl::Field<" << typeid(T).name() << "," << Dim << ">) called" << endl;
                
        } else if constexpr (is_vector_v<T>) {
            channelName = "ippl_vField_" + label;
            ca_m << "::Execute()::execute_entry(" << label << ") | Type: ippl::Field<ippl::Vector<" << typeid(typename T::value_type).name() << "," << Field<T, Dim, ViewArgs...>::dim << ">," << Dim << ">)" << endl;
        }else{
            channelName = "ippl_errorField_" + label;

            ca_m    << "::Execute()::execute_entry(Field<"<<typeid(T).name()<< ">)" << endl
                    << "    For this type of Field the Conduit Blueprint description wasnt \n" 
                    << "    implemented in ippl. Therefore this type of field is not \n"
                    << "    supported for visualisation." << endl;
        }
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

        typename Field_type::Layout_t& Layout_ = field->getLayout();
        typename Field_type::Mesh_t&   Mesh_   = field->get_mesh();

        
        const auto LocalNDIndex_  = Layout_.getLocalNDIndex();
        const auto Origin_        = Mesh_.getOrigin();
        const auto Spacing_       = Mesh_.getMeshSpacing();

        const size_t nGhost       = field->getNghost(); // returns int

        
        const size_t extra        = (use_ghost_masks) ?  size_t(2*nGhost)   :  0 ;
        const size_t index_offset = (use_ghost_masks) ?    size_t(nGhost)   :  0 ; 
        // const size_t index_offset = (use_ghost_masks) ?    nGhost   :  nGhost ;

        const auto Ox = Origin_[0] + double(int(LocalNDIndex_[0].first()) - int(index_offset)) * Spacing_[0];
        const auto Oy = Origin_[1] + double(int(LocalNDIndex_[1].first()) - int(index_offset)) * Spacing_[1];
        const auto Oz = Origin_[2] + double(int(LocalNDIndex_[2].first()) - int(index_offset)) * Spacing_[2];

        const size_t nx =               LocalNDIndex_[0].length() + extra                 ;
        const size_t ny = (Dim >= 2) ?  LocalNDIndex_[1].length() + extra      :         1;
        const size_t nz = (Dim >= 3) ?  LocalNDIndex_[2].length() + extra      :         1;

        const void* meshKey   = static_cast<const void*>(&Mesh_);
        const void* layoutKey = static_cast<const void*>(&Layout_);
        auto ghostKey = GhostKey_t{meshKey, layoutKey, nGhost};


        // std::cout << nGhost << std::endl;
        // std::cout << extra << std::endl;
        // std::cout << index_offset << std::endl;
        // std::cout << 
        // Origin_[0] << " " << 
        // Origin_[1] << " " << 
        // Origin_[2] << std::endl;
        // std::cout << 
        // LocalNDIndex_[0].first() << " " << 
        // LocalNDIndex_[1].first() << " " << 
        // LocalNDIndex_[2].first() << std::endl;
        // std::cout << Ox << " " << Oy << " " << Oz << std::endl;



        // int aa = 0;
        int aa = 1;

        {
            data["coordsets/cart_uniform_coords/dims/i"].set(nx + aa);
            data["coordsets/cart_uniform_coords/spacing/dx"].set(Spacing_[0]);
            data["coordsets/cart_uniform_coords/origin/x"].set( Ox );
            data["topologies/fmesh_topo/origin/x"].set(         Ox );
        }
        if constexpr(Dim >= 2){
            data["coordsets/cart_uniform_coords/dims/j"].set(ny + aa);
            data["coordsets/cart_uniform_coords/spacing/dy"].set(Spacing_[1]);
            data["coordsets/cart_uniform_coords/origin/y"].set( Oy );
            data["topologies/fmesh_topo/origin/y"].set(         Oy );
        }
        if constexpr(Dim >= 3){
            data["coordsets/cart_uniform_coords/dims/k"].set(nz + aa);
            data["coordsets/cart_uniform_coords/spacing/dz"].set(Spacing_[2]);
            data["coordsets/cart_uniform_coords/origin/z"].set( Oz );
            data["topologies/fmesh_topo/origin/z"].set(         Oz );
        }



        const auto& fullDeviceView = field->getView(); // Assuming this is your original view
        using DeviceView_t = typename Field<T, Dim, ViewArgs...>::view_type;
        // using device_memory_space =
                // typename Kokkos::View<typename DeviceView_t::data_type>::memory_space;
        // --- NEEDED OF FIX : LAYOUT ---
        // in genral if Layout of IPPL and Catalyst would align we could use 
        // create_mirror_view_and_copy over enforced deep copy.
        // using HostView_t =
        //     Kokkos::View<typename DeviceView_t::data_type,
        //                  typename DeviceView_t::array_layout,
        //                  Kokkos::HostSpace>;
        // --- START OF FIX ---
        // Change HostView_t to use LayoutLeft, the layout ParaView wants.
        using HostView_t = Kokkos::View<
            typename DeviceView_t::data_type,
            Kokkos::LayoutLeft,  // <-- THE IMPORTANT CHANGE
            Kokkos::HostSpace
        >;
        // --- END OF FIX ---

        auto getHostMirrorView_noGhosts = [&]() -> HostView_t {
            /* CREATE VIEW OF DATA WITHOUT GHOSTS */

            auto r0 = 
                        Kokkos::make_pair(nGhost, nGhost + nx);
            auto r1 = (Dim >= 2) 
                      ? Kokkos::make_pair(nGhost, nGhost + ny)
                      : Kokkos::make_pair(size_t(0), fullDeviceView.extent(1));
            auto r2 = (Dim >= 3) 
                      ? Kokkos::make_pair(nGhost, nGhost + nz)
                      : Kokkos::make_pair(size_t(0), fullDeviceView.extent(2));

            auto deviceSubView = Kokkos::subview(fullDeviceView, r0, r1, r2);
                
            // if (!forceHostCopy[label] && std::is_same<device_memory_space, Kokkos::HostSpace>::value) 
            //      can't use subview direct since data of subview isn't meaningfull accessible in raw format with data() 
            //      further this function can't convert from LayoutRight to LayoutLeft  // so we can't use SHALLOW_COPY
            // B. DEEP COPY (Different memory spaces OR deep copy explicitly forced)
            // --- NEEDED OF FIX: LAYOUT ---
            // HostView_t hostMirrorFinal = HostView_t("hostMirrorNoGhosts",nx,ny,nz); 
            HostView_t hostMirrorFinal = HostView_t("hostMirrorNoGhosts_LayoutLeft", nx, ny, nz);
            // This single deep_copy now performs:
            //    - Device-to-Host transfer
            //    - LayoutRight-to-LayoutLeft transpose
            Kokkos::deep_copy(hostMirrorFinal, deviceSubView);
            // --- END OF FIX ---


            return hostMirrorFinal  ;
        };
        
        
        auto getHostMirrorView_withGhosts = [&]() -> HostView_t 
        {
            
            // --- START OF CACHING LOGIC ---
            // HostMaskView_t hostMaskView; // Declare view
            /* CREATE VIEW OF DATA WITH GHOSTS + ADD GHOST FIELD TO CONDUIT*/
            

            using m_t = unsigned char;

            // 3. Define the N-D Host View type we need for the copy
            using DeviceMaskView_t = typename Field<m_t, Dim, ViewArgs...>::view_type;
            /* STORAGE */
            using HostMaskView1D_t = Kokkos::View<m_t*, Kokkos::LayoutLeft, Kokkos::HostSpace>;
            /* WRAPPER */
            using HostMaskView_t = Kokkos::View<
                typename DeviceMaskView_t::data_type, // e.g., unsigned char***
                Kokkos::LayoutLeft,
                Kokkos::HostSpace
            >;
            
            HostMaskView1D_t hostMaskView1D; // Declare view
            
            
            auto it = ghostMaskCache.find(ghostKey);
            if (it != ghostMaskCache.end()) 
            {
                // --- CACHE HIT ---// Re-use the existing ghost view from the cache
                ca_m << "::Execute()::execute_entry(" << label << ") | GhostCache HIT" << endl;
                hostMaskView1D = it->second;
            } 
            else 
            {   

                // --- CACHE MISS ---
                ca_m << "::Execute()::execute_entry(" << label << ") | GhostCache MISS" << endl;
                // Allocate a mask field matching the source field (same mesh/layout/nghost)
                // auto & meshRef   = field->get_mesh();
                // auto & layoutRef = field->getLayout();
                Field<m_t, Dim, ViewArgs...> ghostMaskField(Mesh_, Layout_, static_cast<int>(nGhost));
                //Fill entire allocation (owned + ghosts) with 1
                ghostMaskField = static_cast<m_t>(1);
                DeviceMaskView_t deviceMaskView   = ghostMaskField.getView();
                auto interior   = ghostMaskField.template getFieldRangePolicy<>(); 
                //  TODO(?): use ippl dimension independent iterators
                if constexpr (Dim == 1) {
                    Kokkos::parallel_for("ZeroOwnedMask1D", interior, KOKKOS_LAMBDA(const int i) {
                        deviceMaskView(i) = static_cast<m_t >(0);
                    });
                } else if constexpr (Dim == 2) {
                    Kokkos::parallel_for("ZeroOwnedMask2D", interior, KOKKOS_LAMBDA(const int i, const int j) {
                        deviceMaskView(i,j) = static_cast<m_t>(0);
                    });
                } else if ( Dim == 3){ // Dim == 3
                    Kokkos::parallel_for("ZeroOwnedMask3D", interior, KOKKOS_LAMBDA(const int i, const int j, const int k) {
                        deviceMaskView(i,j,k) = static_cast<m_t>(0);
                    });
                } else{
                    throw IpplException("CatalystAdaptor::AddSteerableChannel", "Unsupported Field Dimnesion > 3 for Field:" + label);
                }
                Kokkos::fence();
                
                
                //    Allocate the 1D host view that will own the memory
                //    This view has the correct LayoutLeft for ParaView.
                hostMaskView1D = HostMaskView1D_t("hostGhostMask_1D", deviceMaskView.size());

                //    Create a temporary, UNMANAGED N-D view
                //    that wraps the 1D view's data. This is our copy target.
                //    This constructor handles any rank automatically.
                HostMaskView_t hostMaskView_N_Rank(
                    hostMaskView1D.data(),
                    deviceMaskView.extent(0),
                    deviceMaskView.extent(1),
                    deviceMaskView.extent(2),
                    deviceMaskView.extent(3),
                    deviceMaskView.extent(4),
                    deviceMaskView.extent(5),
                    deviceMaskView.extent(6),
                    deviceMaskView.extent(7) // Good for up to 8D
                );
                // The 8-dim extent constructor is flexible for any dimension
                // but eg 3 analog is not .... seems so

                //    Perform the transforming D->H + R->L copy
                //    (Device, LayoutRight) -> (Host, LayoutLeft)
                Kokkos::deep_copy(hostMaskView_N_Rank, deviceMaskView);

                //    Store the 1D view (which owns the memory) in the cache, enough to keep in memory
                ghostMaskCache[ghostKey] = hostMaskView1D;                
                
                // --- NEEDED OF FIX: LAYOUT ---
                // using HostMaskView_t = Kokkos::View<typename decltype(deviceMaskView)::data_type,
                //                                     typename decltype(deviceMaskView)::array_layout,
                //                                     Kokkos::HostSpace>;
                // ---                  ---
                // using HostMaskView_t = Kokkos::View<
                //     typename decltype(deviceMaskView)::data_type,
                //     Kokkos::LayoutLeft, // <-- Use LayoutLeft
                //     Kokkos::HostSpace
                // >;    
                // HostMaskView_t hostMaskView("hostGhostMask_LayoutLeft",
                //                             deviceMaskView.extent(0),
                //                             deviceMaskView.extent(1),
                //                             deviceMaskView.extent(2));
                // Kokkos::deep_copy(hostMaskView, deviceMaskView);
                // //Store the newly created view in the cache
                // ghostMaskCache[ghostKey] = hostMaskView;
                // /* Cache Void pointer is not enough to not discard data->
                // REMEMEBER MASK AND SET CONDUIT NODE */
                // viewRegistry.set(hostMaskView);
            }
            // --- END OF CACHING LOGIC ---
                



            // FIELD NAME IN DATA/FIELDS AND META_DATA/VTK_FIELDS SHOULD COINCIDE
            // auto ghostMask_field_meta  =  data["metadata/vtk_fields/GhostMask_field"]; // can't chooses arbitrary name!!!!
            auto ghostMask_field_meta  =  data["metadata/vtk_fields/vtkGhostType"]; 
            ghostMask_field_meta["attribute_type"] = "Ghosts";  // same as set string??...
            // auto ghostMask_field_node  =  fields["ghostMask_field"]; // can't chooses arbitrary name!!!! must be vtkGhostType
            auto ghostMask_field_node  =  fields["vtkGhostType"];
            ghostMask_field_node["association"].set_string("element");
            ghostMask_field_node["topology"].set_string("fmesh_topo");
            ghostMask_field_node["volume_dependent"].set_string("false");
            // ghostMask_field_node["values"].set_external(hostMaskView.data(), hostMaskView.size());
            ghostMask_field_node["values"].set_external(hostMaskView1D.data(), hostMaskView1D.size());
            // --- NEEDED OF FIX: LAYOUT --- CANTT CHANGE LAYOUT SO WE ARE FORCED TO USE DEEP COPY OVER SHALLOW COPY
            // if (!forceHostCopy[label] && std::is_same<device_memory_space, Kokkos::HostSpace>::value) 
            // {   /* can't use this way since subview isnt accessible this way */
            //     // A. SHALLOW COPY (No deep copy needed, device is on host memory)
            //     // NOTE: This assumes LayoutLeft for the subview, or a layout conversion could be needed.
            //     HostView_t hostMirrorFinal = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fullDeviceView);
            //     return hostMirrorFinal;
            // } 
            // else
            // {   
            //     // B. DEEP COPY (Different memory spaces OR deep copy explicitly forced)
            //     // Kokkos::View<typename Field<T, Dim, ViewArgs...>::view_type::data_type, Kokkos::LayoutLeft, Kokkos::HostSpace>
            //     HostView_t hostMirrorFinal = HostView_t("hostMirrorNoGhosts",nx,ny,nz);
            //     //     fullDeviceView.extent(0),
            //     Kokkos::deep_copy(hostMirrorFinal, fullDeviceView);
            //     return hostMirrorFinal;
            // }
            // --- FIX FOR MAIN FIELD ---
            HostView_t hostMirrorFinal = HostView_t("hostMirrorWithGhosts_LayoutLeft", nx, ny, nz);
            Kokkos::deep_copy(hostMirrorFinal, fullDeviceView);
            return hostMirrorFinal;
            // --- END FIX FOR MAIN FIELD ---
        };

         
        HostView_t hostMirrorFinal = (use_ghost_masks) ? getHostMirrorView_withGhosts() : getHostMirrorView_noGhosts();
        /* FOR BOTH CASES FINAL NODE SETTINS ARE DONE AND WE HAVE THE DATA INSIDE hostMirrorFinal */
        using elem_t = std::remove_pointer_t<decltype(hostMirrorFinal.data())>;
        // will return size of vector and amounts of vectors (not double)
        const auto n_elems = hostMirrorFinal.size();
        // Use true element size as stride (handles padding)
        static constexpr size_t stride_bytes = sizeof(elem_t);
        // offset is zero?? guaranteed?
        const size_t offset = 0;

        if constexpr (std::is_scalar_v<T>) {
            // --- SCALAR FIELD CASE ---
            field_node["association"].set_string("element");
            field_node["topology"].set_string("fmesh_topo");
            field_node["volume_dependent"].set_string("false");
            field_node["values"].set_external(hostMirrorFinal.data(), n_elems);
        } else if constexpr (is_vector_v<T>) {
            // --- VECTOR FIELD CASE ---
            field_node["association"].set_string("element");
            field_node["topology"].set_string("fmesh_topo");
            field_node["volume_dependent"].set_string("false");
                                    // stride was 1 in predecessor code? how did this work?...
                                     field_node["values/x"].set_external(&hostMirrorFinal.data()[0][0], n_elems, offset, stride_bytes);
            if constexpr (T::dim>=2) field_node["values/y"].set_external(&hostMirrorFinal.data()[0][1], n_elems, offset, stride_bytes);
            if constexpr (T::dim>=3) field_node["values/z"].set_external(&hostMirrorFinal.data()[0][2], n_elems, offset, stride_bytes);        
        } 
        // else {
            // --- INVALID CASE ---
        // }
        /* SAVE VIEW SO DATA ISNT DISCARDED*/
        viewRegistry.set(label, hostMirrorFinal);
}




// execute visualisation for PARTICLECONTAINERS derived from ParticleBaseBase:
// == ippl::ParticleBase<PLayout<T, dim>,...>,...>
template<typename T>
requires (std::derived_from<std::decay_t<T>, ParticleBaseBase>)
void CatalystAdaptor::execute_entry(const T& entry, const std::string label) 
{   
    if(viewRegistry.contains(label)) return;

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
        
        
        // Creates a host-accessible mirror view and copies the data from the device view to the host.
        // compared to get_mirror and get_mirror_view host space is not guaranteed default behaviour so we specify...
        // comType HostMirror would let the function auto deduct the wanted space ...
        
        // Get attribute types for ID and R, then their HostMirror types
        using IDAttrib_t = std::remove_reference_t<decltype(particleContainer->ID)>;
        using RAttrib_t  = std::remove_reference_t<decltype(particleContainer->R)>;

        using hostMirror_ID_t = typename IDAttrib_t::HostMirror;
        using hostMirror_R_t  = typename RAttrib_t::HostMirror;

        hostMirror_ID_t ID_hostMirror ;
        hostMirror_R_t  R_hostMirror  ;

        if(forceHostCopy[label]){
            // ParticleAttrib<std::int64_t>::HostMirror      ID_hostMirror;
            // ParticleAttrib<Vector<double, 3>>::HostMirror R_hostMirror;
            // auto ID_hostMirror = particleContainer->ID.getHostMirror();
            // auto R_hostMirror  = particleContainer->R.getHostMirror();
            ID_hostMirror = particleContainer->ID.getHostMirror();
            R_hostMirror  = particleContainer->R.getHostMirror();
            Kokkos::deep_copy(ID_hostMirror,  particleContainer->ID.getView());
            Kokkos::deep_copy(R_hostMirror ,  particleContainer->R.getView());
            viewRegistry.set(label, ID_hostMirror);
            viewRegistry.set(R_hostMirror);
        }else{
            /* if original is on host space no copy will b created and any changs will be taken over ... */
            // auto ID_hostMirror =   Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), particleContainer->ID.getView());
            // auto  R_hostMirror  =   Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), particleContainer->R.getView());
            ID_hostMirror =   Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), particleContainer->ID.getView());
            R_hostMirror  =   Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), particleContainer->R.getView());
            viewRegistry.set(label, ID_hostMirror);
            viewRegistry.set(R_hostMirror);
        }



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
            attribute->signConduitBlueprintNode_rememberHostCopy(particleContainer->getLocalNum(), fields, viewRegistry, ca_m, ca_warn, forceHostCopy[label]);
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



// =====================================================================================
// STEERING:
// =====================================================================================


template<typename T>
requires (!std::is_enum_v<std::decay_t<T>>)
void CatalystAdaptor::InitSteerableChannel( [[maybe_unused]] const T& steerable_scalar_forwardpass,  const std::string& label ){
    ca_m << "::Initialize()::InitSteerableChannel(" << label << "):  | Type: " << typeid(T).name() << endl;
    proxyWriter.include(steerable_scalar_forwardpass, label);
    conduit_cpp::Node script_args = node["catalyst/scripts/script/args"];
    script_args.append().set_string(label);
}

// Enum overload (explicit) to ensure proper dropdown setup even if template above is shadowed
template<typename E>
requires (std::is_enum_v<std::decay_t<E>>)
void CatalystAdaptor::InitSteerableChannel( [[maybe_unused]] const E& e, const std::string& label ){
    ca_m << "::Initialize()::InitSteerableChannel(" << label << "):  | Type: Enum" << endl;
    auto it = enumChoices_.find(label);
    if (it != enumChoices_.end()) {
        proxyWriter.includeEnum(label, it->second, static_cast<int>(e));
    } else {
        // Fallback to an int-like property without dropdown (simplest path is to use Int checkbox placeholder replaced above)
        proxyWriter.includeBool(label, false);
    }
    conduit_cpp::Node script_args = node["catalyst/scripts/script/args"];
    script_args.append().set_string(label);
}

// Bool-like Switch: init (checkbox in GUI)
void CatalystAdaptor::InitSteerableChannel( [[maybe_unused]] const bool& sw, const std::string& label ){
    ca_m << "::Initialize()::InitSteerableChannel(" << label << "):  | Type: Switch" << endl;
    proxyWriter.includeBool(label, static_cast<bool>(sw));

    conduit_cpp::Node script_args = node["catalyst/scripts/script/args"];
    script_args.append().set_string(label);
}

// Button-like: init (push button in GUI)
void CatalystAdaptor::InitSteerableChannel( [[maybe_unused]] const ippl::Button& btn, const std::string& label ){
    ca_m << "::Initialize()::InitSteerableChannel(" << label << "):  | Type: Button" << endl;
    proxyWriter.includeButton(label);
    conduit_cpp::Node script_args = node["catalyst/scripts/script/args"];
    script_args.append().set_string(label);
}

// Vector steerable overloads
template<typename T, unsigned Dim_v>
void CatalystAdaptor::InitSteerableChannel( [[maybe_unused]] const ippl::Vector<T, Dim_v>& steerable_vec_forwardpass, const std::string& label )
{
    ca_m << "::Initialize()::InitSteerableChannel(" << label << "):  | Vector<" << typeid(T).name() << "," << Dim_v << ">" << endl;
    // Register this label as a vector channel in the proxy writer (limit to 3 comps in GUI)
    proxyWriter.includeVector<T, Dim_v>(label);
    (void)steerable_vec_forwardpass;

    // Ensure the Python pipeline receives this label after `--steer_channel_names`
    // so it creates the corresponding forward reader and unified sender wiring.
    conduit_cpp::Node script_args = node["catalyst/scripts/script/args"];
    script_args.append().set_string(label);
}



template<typename T>
requires (!std::is_enum_v<std::decay_t<T>>)
void CatalystAdaptor::AddSteerableChannel( const T& steerable_scalar_forwardpass,  const std::string& steerable_suffix ) 
{
        ca_m << "::Execute()::AddSteerableChannel(" << steerable_suffix << ");  | Type: " << typeid(T).name() << endl;
        
    auto steerable_channel = node["catalyst/channels/steerable_channel_0D_mesh"];
        // auto steerable_channel = node["catalyst/channels/steerable_channels_forward_all"];

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
        if constexpr(std::is_enum_v<std::decay_t<T>>){
            values.set(static_cast<int>(steerable_scalar_forwardpass));
        }
        else if constexpr(std::is_scalar_v<T>){
            values.set(steerable_scalar_forwardpass);
            // lastForwardSteerVal[steerable_suffix] = static_cast<double>(steerable_scalar_forwardpass);
        }        
        else {
            throw IpplException("CatalystAdaptor::AddSteerableChannel", "Unsupported steerable type for channel: " + steerable_suffix);
        }
}

// Enum overload explicit (clarity)
template<typename E>
requires (std::is_enum_v<std::decay_t<E>>)
void CatalystAdaptor::AddSteerableChannel( const E& e, const std::string& steerable_suffix )
{
    ca_m << "::Execute()::AddSteerableChannel(" << steerable_suffix << ");  | Type: Enum" << endl;
    auto steerable_channel = node["catalyst/channels/steerable_channel_0D_mesh"];
    steerable_channel["type"].set("mesh");
    auto steerable_data = steerable_channel["data"];    
    steerable_data["coordsets/coords/type"].set_string("explicit");
    steerable_data["coordsets/coords/values/x"].set( 0 );
    steerable_data["topologies/sMesh_topo/type"].set("unstructured");
    steerable_data["topologies/sMesh_topo/coordset"].set("coords");
    steerable_data["topologies/sMesh_topo/elements/shape"].set("point");
    steerable_data["topologies/sMesh_topo/elements/connectivity"].set( 0 );
    auto steerable_field = steerable_data["fields/steerable_field_f_" + steerable_suffix];
    steerable_field["association"].set("vertex");
    steerable_field["topology"].set("sMesh_topo");
    steerable_field["volume_dependent"].set("false");
    steerable_field["values"].set(static_cast<int>(e));
}

// Bool-like Switch: forward as single-sample scalar (0/1)
void CatalystAdaptor::AddSteerableChannel( const bool& sw, const std::string& steerable_suffix )
{
    ca_m << "::Execute()::AddSteerableChannel(" << steerable_suffix << ");  | Type: bool/Switch" << endl;
    
    auto steerable_channel = node["catalyst/channels/steerable_channel_0D_mesh"];
    steerable_channel["type"].set("mesh");
    auto steerable_data = steerable_channel["data"];
    steerable_data["coordsets/coords/type"].set_string("explicit");
    steerable_data["coordsets/coords/values/x"].set( 0 );

    steerable_data["topologies/sMesh_topo/type"].set("unstructured");
    steerable_data["topologies/sMesh_topo/coordset"].set("coords");
    steerable_data["topologies/sMesh_topo/elements/shape"].set("point");
    steerable_data["topologies/sMesh_topo/elements/connectivity"].set( 0 );

    conduit_cpp::Node steerable_field = steerable_data["fields/steerable_field_f_" + steerable_suffix];
    steerable_field["association"].set("vertex");
    steerable_field["topology"].set("sMesh_topo");
    steerable_field["volume_dependent"].set("false");
    
    int bool_as_int = sw ? 1 : 0;
    steerable_field["values"].set(bool_as_int);
}

// Button-like: forward as single-sample scalar (0/1)
void CatalystAdaptor::AddSteerableChannel( const ippl::Button& btn, const std::string& steerable_suffix )
{
    ca_m << "::Execute()::AddSteerableChannel(" << steerable_suffix << ");  | Type: Button" << endl;
    auto steerable_channel = node["catalyst/channels/steerable_channel_0D_mesh"];
    steerable_channel["type"].set("mesh");
    auto steerable_data = steerable_channel["data"];
    steerable_data["coordsets/coords/type"].set_string("explicit");
    steerable_data["coordsets/coords/values/x"].set( 0 );

    steerable_data["topologies/sMesh_topo/type"].set("unstructured");
    steerable_data["topologies/sMesh_topo/coordset"].set("coords");
    steerable_data["topologies/sMesh_topo/elements/shape"].set("point");
    steerable_data["topologies/sMesh_topo/elements/connectivity"].set( 0 );

    conduit_cpp::Node steerable_field = steerable_data["fields/steerable_field_f_" + steerable_suffix];
    steerable_field["association"].set("vertex");
    steerable_field["topology"].set("sMesh_topo");
    steerable_field["volume_dependent"].set("false");
    steerable_field["values"].set(static_cast<int>(btn ? 1 : 0));
}

// ippl::Vector overload
template<typename T, unsigned Dim_v>
void CatalystAdaptor::AddSteerableChannel( const ippl::Vector<T, Dim_v>& steerable_vec_forwardpass, const std::string& steerable_suffix )
{
    ca_m << "::Execute()::AddSteerableChannel(" << steerable_suffix << ");  | Vector<" << typeid(T).name() << "," << Dim_v << ">" << endl;

    auto steerable_channel = node["catalyst/channels/steerable_channel_0D_mesh"];
    steerable_channel["type"].set("mesh");
    auto steerable_data = steerable_channel["data"];
    steerable_data["coordsets/coords/type"].set_string("explicit");
    steerable_data["coordsets/coords/values/x"].set(0);

    steerable_data["topologies/sMesh_topo/type"].set("unstructured");
    steerable_data["topologies/sMesh_topo/coordset"].set("coords");
    steerable_data["topologies/sMesh_topo/elements/shape"].set("point");
    steerable_data["topologies/sMesh_topo/elements/connectivity"].set(0);

    // Single vector field with 3 components (x,y,z) under one name
    const std::string base = std::string("fields/steerable_field_f_") + steerable_suffix;
    auto fnode = steerable_data[base];
    fnode["association"].set_string("vertex");
    fnode["topology"].set_string("sMesh_topo");
    fnode["volume_dependent"].set_string("false");

    const double vx = (Dim_v >= 1) ? static_cast<double>(steerable_vec_forwardpass[0]) : 0.0;
    const double vy = (Dim_v >= 2) ? static_cast<double>(steerable_vec_forwardpass[1]) : 0.0;
    const double vz = (Dim_v >= 3) ? static_cast<double>(steerable_vec_forwardpass[2]) : 0.0;

    fnode["values/x"].set(vx);
    fnode["values/y"].set(vy);
    fnode["values/z"].set(vz);
}


/* overload for: Scalar, Bool(switch), Button */
template<typename T>
requires (!std::is_enum_v<std::decay_t<T>>)
void CatalystAdaptor::FetchSteerableChannelValue( T& steerable_scalar_backwardpass, const std::string& label) {
    ca_m << "::Execute()::FetchSteerableChannel(" << label  << ") | Type: " << typeid(T).name() << endl;

    std::string unified_path = std::string("catalyst/steerable_channel_backward_all/fields/") +
                               "steerable_field_b_" + label + "/values";

    const std::string* chosen = nullptr;
    if (results.has_path(unified_path)) {
        chosen = &unified_path;
    } else {
        ca_m << "::Execute()::FetchSteerableChannel(" << label << ") | no backward result present; skipping." << endl;
        return;
    }

    conduit_cpp::Node values_node = results[*chosen];
    if (!values_node.dtype().is_number()) {
        ca_m << "::Execute()::FetchSteerableChannel(" << label << ") | backward value not numeric; skipping." << endl;
        return;
    }

    if constexpr (std::is_enum_v<std::decay_t<T>>)      steerable_scalar_backwardpass = static_cast<T>(values_node.to_int32());
    else if constexpr (std::is_same_v<T,double>)        steerable_scalar_backwardpass = values_node.to_double();
    else if constexpr (std::is_same_v<T,float>)         steerable_scalar_backwardpass = static_cast<float>(values_node.to_float());
    else if constexpr (std::is_same_v<T,int>)           steerable_scalar_backwardpass = static_cast<int>(values_node.to_int32());
    else if constexpr (std::is_same_v<T,unsigned>)      steerable_scalar_backwardpass = static_cast<unsigned>(values_node.to_uint32());
    else if constexpr (std::is_same_v<T,bool>)          steerable_scalar_backwardpass = static_cast<bool>(values_node.to_int32());
    else if constexpr (std::is_same_v<T,ippl::Button>)  steerable_scalar_backwardpass = static_cast<bool>(values_node.to_int32());
    else {
        throw IpplException("CatalystAdaptor::FetchSteerableChannelValue", "Unsupported type for channel: " + label);
    }

    ca_m << "::Execute()::FetchSteerableChannel(" << label << ") | received:" << steerable_scalar_backwardpass << endl;
}

// Enum overload explicit (clarity)
template<typename E>
requires (std::is_enum_v<std::decay_t<E>>)
void CatalystAdaptor::FetchSteerableChannelValue( E& e, const std::string& label)
{
    ca_m << "::Execute()::FetchSteerableChannel(" << label  << ") | Type: Enum| Value sent:" << to_string(e) << endl;
    std::string unified_path = std::string("catalyst/steerable_channel_backward_all/fields/") +
                               "steerable_field_b_" + label + "/values";
    if (!results.has_path(unified_path)) {
        ca_m << "  no backward enum found for label '" << label << "'" << endl;
        return;
    }
    conduit_cpp::Node values_node = results[unified_path];
    if (!values_node.dtype().is_number()) return;
    e = static_cast<E>(values_node.to_int32());

    ca_m << "::Execute()::FetchSteerableChannel(" << label  << ") | Type: Enum|  received:" << to_string(e) << endl;
}


/* overload for: ippl::Vector */
template<typename T, unsigned Dim_v>
void CatalystAdaptor::FetchSteerableChannelValue( ippl::Vector<T, Dim_v>& steerable_vec_backwardpass, const std::string& label)
{
    ca_m << "::Execute()::FetchSteerableChannel(" << label  << ") | Vector<" << typeid(T).name() << "," << Dim_v << ">" << endl;

    // static const char* comp_names[3] = {"x","y","z"};
    const unsigned comps = Dim_v > 3 ? 3u : Dim_v;
    // bool any_set = false;
    // // Preferred: unified vector array with values/x,y,z
    // for (unsigned c = 0; c < comps; ++c) {
    //     std::string unified_vec_comp = std::string("catalyst/steerable_channel_backward_all/fields/") +
    //                                    "steerable_field_b_" + label + "/values/" + comp_names[c];
    //     if (results.has_path(unified_vec_comp)) {
    //         conduit_cpp::Node vnode = results[unified_vec_comp];
    //         if (vnode.dtype().is_number()) {
    //             steerable_vec_backwardpass[c] = static_cast<T>(vnode.to_double());
    //             any_set = true;
    //             ca_m << "  read component '" << comp_names[c] << "' from " << unified_vec_comp
    //                  << ": " << vnode.to_double() << endl;
    //         }
    //     }
    // }

    // Fallback: a flat numeric array at .../values holding at least `comps` elements
    // if (!any_set) {
    if (true) {
        std::string unified_vec_values = std::string("catalyst/steerable_channel_backward_all/fields/") +
                                         "steerable_field_b_" + label + "/values";
        // std::string legacy_vec_values  = std::string("catalyst/steerable_channel_backward_") + label +
                                        //  "/fields/steerable_field_b_" + label + "/values";
        const std::string* chosen = nullptr;
        if (results.has_path(unified_vec_values)) chosen = &unified_vec_values;
        // else if (results.has_path(legacy_vec_values)) chosen = &legacy_vec_values;
        if (chosen) {
            conduit_cpp::Node vnode = results[*chosen];
            
            // auto n_elems = vnode.dtype().number_of_elements();
            // if (vnode.dtype().is_number() && n_elems >= comps) {
            //     const double* ptr = vnode.as_float64_ptr();
            //     if (ptr) {
            //         for (unsigned c = 0; c < comps; ++c) {
            //             steerable_vec_backwardpass[c] = static_cast<T>(ptr[c]);
            //         }
            //         ca_m << "  read flat values from " << *chosen << ": ["
            //              << ptr[0] << ", " << (comps>1?ptr[1]:0) << ", " << (comps>2?ptr[2]:0) << "]" << endl;
            //     }
            // }
            // else {
            
                ca_m << "  vector result node with list-like values found at " << *chosen << endl;//" but not numeric or too few elements (" << n_elems << ")" << endl;
                // Handle list-like layout: values/0, values/1, values/2
                bool idx_read = true;
                for (unsigned c = 0; c < comps; ++c) {
                    std::string idx_path = *chosen + "/" + std::to_string(c);
                    if (results.has_path(idx_path)) {
                        conduit_cpp::Node ic = results[idx_path];
                        if (ic.dtype().is_number()) {
                            steerable_vec_backwardpass[c] = static_cast<T>(ic.to_double());
                        } else {
                            idx_read = false;
                        }
                    } else {
                        idx_read = false;
                    }
                }
                if (idx_read) {
                    ca_m << "  read list-like values from " << *chosen << ":" << steerable_vec_backwardpass << endl;
                    // any_set = true;
                }
            // }
        }
        else {
            ca_warn << "  no backward vector found for label '" << label << "' under expected paths." << endl;
        }
    }
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

  
    

    args.append().set_string("--verbosity");
    args.append().set_string(std::to_string(ca_m.getOutputLevel()));


    args.append().set_string("--VTKextract");
    args.append().set_string(std::string(catalyst_vtk));

    args.append().set_string("--steer");
    args.append().set_string(std::string(catalyst_steer));


    args.append().set_string("--steer_channel_names");
        
    auto proxy_path = (source_dir / "catalyst_scripts" / "catalyst_proxy.xml").string() ;
    proxyWriter.initialize( proxy_path  );
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
    set_node_script(    node["catalyst/proxies/proxy_/filename"],
        "CATALYST_PROXYS_PATH",
        proxy_path
    );

    

    // switch (proxy_option){
    //     case "PRODUCE_ONLY":
    //         proxyWriter.produceUnified("SteerableParameters_ALL", "SteerableParameters");
    //         throw IpplException("Stream::InSitu::CatalystAdaptor", "write_proxy_only_run is ON, proxies have been printed");
    //     case "OFF":
    //         break;
    //     case "PRODUCE": 
    //     default:
    //     proxyWriter.produceUnified("SteerableParameters_ALL", "SteerableParameters");
    // }

    if( std::string(proxy_option) == "PRODUCE_ONLY"){
            proxyWriter.produceUnified("SteerableParameters_ALL", "SteerableParameters");
            throw IpplException("Stream::InSitu::CatalystAdaptor", "write_proxy_only_run: proxies have been printed");
    }else if( std::string(proxy_option) == "OFF"){
    }else{
        proxyWriter.produceUnified("SteerableParameters_ALL", "SteerableParameters");
    }









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




void CatalystAdaptor::Remember_now(const std::string label){
    auto it  = forceHostCopy.find(label);
    if (it == forceHostCopy.end()){
            // return false;
            throw IpplException("CatalystAdaptor::FetchSteerableChannelValue", "Label not present in vis executable objects: " + label);
    }
    bool tmp = it->second;
    forceHostCopy[label] = true;
    
    ExecuteVisitor execV{*this};
    visRegistry->for_one(label, execV); 

    forceHostCopy[label] = tmp;

}

void CatalystAdaptor::ExecuteRuntime( int cycle, double time, int rank /* default = ippl::Comm->rank() */) {
    ca_m << "::Execute() START =============================================================== 0" << endl;
    

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



    if(cycle == 0){
        ca_m << "::Execute()   Printing first Conduit Node passed from RANK=1  to catalyst_execute() ==>" << endl;
        if(ca_m.getOutputLevel() > 0 && ippl::Comm->rank()==0) node.print();
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
        fetchResults();
        if(cycle == 0){
        // if(true){
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
        ghostMaskCache.clear();
        node.reset();
        // results.reset();
        
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