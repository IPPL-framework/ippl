#pragma once
#include "Stream/InSitu/CatalystAdaptor.h"

#include "Ippl.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
namespace ippl{

// ==============================================================================================
//  HELPERS =====================================================================================
// ==============================================================================================
void CatalystAdaptor::setNodeScript(
    conduit_cpp::Node nodePath,
    const std::string envVar,
    const std::filesystem::path defaultFilePath
){           
        const char* filePathEnv = std::getenv(envVar.c_str());
        std::filesystem::path filePath;
        if (filePathEnv && std::filesystem::exists(filePathEnv)) {
           catalystInfo_m << level4 <<"::Initialize()::setNodeScripts(...):" << endl
                << "                Using " << envVar << " from environment:" << endl
                << "                "<< filePathEnv << endl;
           filePath = filePathEnv;
        } else {
           catalystInfo_m << level4 <<"::Initialize()::setNodeScripts(...): No valid " << envVar <<" set." << endl 
                << "                Using default:" << endl
                << "                " << defaultFilePath << endl;
           filePath = defaultFilePath;
        }
        nodePath.set(filePath.string());
}


// ==============================================================================================
// VIS CHANNEL INITIALIZER:======================================================================
// ==============================================================================================

// == ippl::Field<double, 3, ippl::UniformCartesian<double, 3>, Cell>
template<typename T, unsigned Dim, class... ViewArgs>
void CatalystAdaptor::InitVizChannel( [[maybe_unused]]  const Field<T, Dim, ViewArgs...>& entry , const std::string label)
{
        catalystInfo_m << level4 <<"::Initialize()::InitVizChannel(ippl::Field<" << typeid(T).name() << "," << Dim << ">) called" << endl;
        forceHostCopy_m[label] = false;

        const std::string channelName = "ippl_sField_" + label; 
        if(pngExtracts_m){
            const std::string script = "catalyst/scripts/" + label;
            setNodeScript( node_m[script + "/filename"],
                            "CATALYST_EXTRACTOR_SCRIPT_" +label,
                            sourceDir_m /"catalyst_scripts" / "catalyst_extractors" /"png_ext_sfield.py"
                        );
            conduit_cpp::Node args = node_m[script + "/args"];
            args.append().set_string("--channel_name");
            args.append().set_string(channelName);
            args.append().set_string("--label");
            args.append().set_string(label);
            if(TestName){
                args.append().set_string("--experiment_name");
                args.append().set_string(std::string(TestName));
            }

            args.append().set_string("--verbosity");
            args.append().set_string(std::to_string(catalystInfo_m.getOutputLevel()));
        }

        conduit_cpp::Node scriptArgs = node_m["catalyst/scripts/script/args"];
        scriptArgs.append().set_string(channelName);
}


// == ippl::Field<ippl::Vector<double, 3>, 3, ippl::UniformCartesian<double, 3>, Cell>
template<typename T, unsigned Dim, unsigned Dim_v, class... ViewArgs>
void CatalystAdaptor::InitVizChannel( [[maybe_unused]]  const Field<Vector<T, Dim_v>, Dim, ViewArgs...>& entry , const std::string label)
{
        catalystInfo_m    << "::Initialize()::InitVizChannel(ippl::Field<ippl::Vector<"
                << typeid(T).name() << "," << Dim_v << ">," << Dim 
                << ">) called" << endl;

        forceHostCopy_m[label] = false;

                
        const std::string channelName = "ippl_vField_" + label;
        if(pngExtracts_m){
            const std::string script = "catalyst/scripts/" + label;

            setNodeScript( node_m[script + "/filename"],
                            "CATALYST_EXTRACTOR_SCRIPT_" + label,
                            sourceDir_m /"catalyst_scripts" / "catalyst_extractors" /"png_ext_vfield.py"
                            
                        );
            conduit_cpp::Node args = node_m[script + "/args"];
            args.append().set_string("--channel_name");
            args.append().set_string(channelName);
            args.append().set_string("--label");
            args.append().set_string(label);
            if(TestName){
                args.append().set_string("--experiment_name");
                args.append().set_string(std::string(TestName));
            }
            args.append().set_string("--verbosity");
            args.append().set_string(std::to_string(catalystInfo_m.getOutputLevel()));
        }


        conduit_cpp::Node scriptArgs = node_m["catalyst/scripts/script/args"];
        scriptArgs.append().set_string(channelName);

}

// PARTICLECONTAINERS derived from ParticleBaseBase:
// == ippl::ParticleBase<PLayout<T, dim>,...>,...>
template<typename T>
requires (std::derived_from<std::decay_t<T>, ParticleBaseBase>)
void CatalystAdaptor::InitVizChannel( [[maybe_unused]]  const T& entry, const std::string label)
{
        catalystInfo_m    << "::Initialize()::InitVizChannel(ParticleBase<PLayout<" 
                << typeid(particle_value_t<T>).name() << ","<< particle_dim_v<T> 
                << ",...>...> [or subclass]) called" << endl;
                    
        forceHostCopy_m[label] = false;

            const std::string channelName = "ippl_particles_" + label;
            if(pngExtracts_m){
                const std::string script = "catalyst/scripts/"+ label;

                setNodeScript( 
                            node_m[script + "/filename"],
                            "CATALYST_EXTRACTOR_SCRIPT_" +label,
                            sourceDir_m /"catalyst_scripts" / "catalyst_extractors" /"png_ext_particle.py"
                );

                conduit_cpp::Node args = node_m[script + "/args"];
                    args.append().set_string("--channel_name");
                    args.append().set_string(channelName);
                    args.append().set_string("--label");
                    args.append().set_string(label);

                if(TestName){
                    args.append().set_string("--experiment_name");
                    args.append().set_string(std::string(TestName));
                }

                args.append().set_string("--verbosity");
                args.append().set_string(std::to_string(catalystInfo_m.getOutputLevel()));
            }

            conduit_cpp::Node scriptArgs = node_m["catalyst/scripts/script/args"];
            scriptArgs.append().set_string(channelName);
}
  
/* SHARED_PTR DISPATCHER - automatically unwraps and dispatches to appropriate overload */
template<typename T>
void CatalystAdaptor::InitVizChannel( const std::shared_ptr<T>&   entry, const std::string label)
{
    if (entry) {
        InitVizChannel(  *entry
                        , label
        );
    }
    else {
        catalystWarn_m << "::Initialize()InitVizChannel(nullptr):  nullptr passed as entry."         << endl
                << "       ID: "<< label                                    << endl
                << "   ==> Channel will not be registered in Conduit Node." << endl;
    }
}


// BASE CASE: 
template<typename T>
requires (!std::derived_from<std::decay_t<T>, ParticleBaseBase>)
void CatalystAdaptor::InitVizChannel([[maybe_unused]] const T& entry, const std::string label)
{
        catalystWarn_m <<  "::Initialize()InitVizChannel(nullptr): Entry type can't be processed."   << endl 
                <<  "       ID: "<< label                                   << endl 
                <<  "       Type: "<< typeid(std::decay_t<T>).name()        << endl 
                <<  "   ==>Channel will not be registered in Conduit Node." << endl
                <<  "   If you see this something is wrong with: CatalystAdaptor::InitVisitor!!" << endl;
}
  



// == ippl::Field<double, 3, ippl::UniformCartesian<double, 3>, Cell>*
// == ippl::Field<ippl::Vector<double, 3>, 3, ippl::UniformCartesian<double, 3>, Cell>*
template<typename T, unsigned Dim, class... ViewArgs>
void CatalystAdaptor::ExecVizChannel(const Field<T, Dim, ViewArgs...>& entry, const std::string label)
{   
    const bool refreshOnly = viewRegistry_m.contains(label);
    if (refreshOnly && !forceHostCopy_m[label]) {
        return;
    }

    using Field_type = Field<T, Dim, ViewArgs...>;
    const Field_type* field = &entry;

        std::string channelName;
        if constexpr (std::is_scalar_v<T>) {
            channelName = "ippl_sField_" + label;
            catalystInfo_m << level4 <<"::Execute()::ExecVizChannel(" << label << ") | Type:ippl::Field<" << typeid(T).name() << "," << Dim << ">) called" << endl;
                
        } else if constexpr (is_vector_v<T>) {
            channelName = "ippl_vField_" + label;
            catalystInfo_m << level4 <<"::Execute()::ExecVizChannel(" << label << ") | Type: ippl::Field<ippl::Vector<" << typeid(typename T::value_type).name() << "," << Field<T, Dim, ViewArgs...>::dim << ">," << Dim << ">)" << endl;
        }else{
            channelName = "ippl_errorField_" + label;

            catalystInfo_m    << "::Execute()::ExecVizChannel(Field<"<<typeid(T).name()<< ">)" << endl
                    << "    For this type of Field the Conduit Blueprint description wasnt \n" 
                    << "    implemented in ippl. Therefore this type of field is not \n"
                    << "    supported for visualisation." << endl;
        }

        conduit_cpp::Node field_node;
        typename Field_type::Layout_t& Layout_ = field->getLayout();
        typename Field_type::Mesh_t&   Mesh_   = field->get_mesh();

        const auto LocalNDIndex_  = Layout_.getLocalNDIndex();
        const auto Origin_        = Mesh_.getOrigin();
        const auto Spacing_       = Mesh_.getMeshSpacing();

        const size_t nGhost       = field->getNghost(); // returns int
        
        const size_t extra        = (useGhostMasks_m) ?  size_t(2*nGhost)   :  0 ;
        const size_t index_offset = (useGhostMasks_m) ?    size_t(nGhost)   :  0 ; 

        int dims_n=1;
        double extra_origin=0;

        const auto Ox = Origin_[0] + (double(int(LocalNDIndex_[0].first()) - int(index_offset)) + extra_origin) * Spacing_[0];
        const auto Oy = Origin_[1] + (double(int(LocalNDIndex_[1].first()) - int(index_offset)) + extra_origin) * Spacing_[1];
        const auto Oz = Origin_[2] + (double(int(LocalNDIndex_[2].first()) - int(index_offset)) + extra_origin) * Spacing_[2];

        const size_t nx =               LocalNDIndex_[0].length() + extra                 ;
        const size_t ny = (Dim >= 2) ?  LocalNDIndex_[1].length() + extra      :         1;
        const size_t nz = (Dim >= 3) ?  LocalNDIndex_[2].length() + extra      :         1;

        const auto& fullDeviceView = field->getView(); // original view
        using DeviceView_t = typename Field<T, Dim, ViewArgs...>::view_type;
        using HostView_t = Kokkos::View<
            typename DeviceView_t::data_type,
            Kokkos::LayoutLeft,  
            Kokkos::HostSpace
        >;

        if (refreshOnly) {
            HostView_t* hostMirrorFinal = viewRegistry_m.find<HostView_t>(label);
            if (!hostMirrorFinal) {
                throw IpplException("Stream::InSitu::CatalystAdaptor::ExecVizChannel",
                                    "Missing host mirror for refresh: " + label);
            }
            if (useGhostMasks_m) {
                Kokkos::deep_copy(*hostMirrorFinal, fullDeviceView);
            } else {
                auto r0 = Kokkos::make_pair(nGhost, nGhost + nx);
                auto r1 = (Dim >= 2)
                              ? Kokkos::make_pair(nGhost, nGhost + ny)
                              : Kokkos::make_pair(size_t(0), fullDeviceView.extent(1));
                auto r2 = (Dim >= 3)
                              ? Kokkos::make_pair(nGhost, nGhost + nz)
                              : Kokkos::make_pair(size_t(0), fullDeviceView.extent(2));
                Kokkos::deep_copy(*hostMirrorFinal, Kokkos::subview(fullDeviceView, r0, r1, r2));
            }
            return;
        }

        // channel for this field of type mesh adheres to conduits mesh blueprint
        auto channel = node_m["catalyst/channels/"+ channelName];
        auto channel_state = channel["state"];

        channel["type"].set_string("mesh");
        auto data   = channel["data"];
        
        
        auto fields = data["fields"];
        field_node = fields[label];
        data["topologies/fmesh_topo/type"].set_string("uniform");
        data["topologies/fmesh_topo/coordset"].set_string("cart_uniform_coords");
        data["coordsets/cart_uniform_coords/type"].set_string("uniform");

        const void* meshKey   = static_cast<const void*>(&Mesh_);
        const void* layoutKey = static_cast<const void*>(&Layout_);
        auto ghostKey = GhostKey_t{meshKey, layoutKey, nGhost};

        const size_t localNumCells = nx * ny * nz;
        const int rank = ippl::Comm->rank();

        // using RankViewCells_t = Kokkos::View<int*, Kokkos::HostSpace>;
        // RankViewCells_t rank_id_view_cells("rank_id_view_cells", localNumCells);
        // auto host_policy = getRangePolicy(rank_id_view_cells);

        using RankViewCells_t = Kokkos::View<int***, Kokkos::HostSpace>;
        using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
        RankViewCells_t rank_id_view_cells("rank_id_view_cells_3D", nx, ny, nz);

        if (localNumCells > 0) {
            Kokkos::MDRangePolicy<HostExecSpace, Kokkos::Rank<3>> host_policy(
                {0, 0, 0},      // Start indices {i, j, k}
                {nx, ny, nz}    // End indices {i, j, k}
            );
            Kokkos::parallel_for("fill_rank_ids_3D", host_policy, 
                KOKKOS_LAMBDA(const int i, const int j, const int k) {
                rank_id_view_cells(i, j, k) = rank;
            });
        }

        auto rank_field = fields["RankID"];
        rank_field["association"].set_string("element"); // associate_m);
        rank_field["topology"].set_string("fmesh_topo");
        rank_field["volume_dependent"].set_string("false");
        if (localNumCells > 0) {
            rank_field["values"].set_external(rank_id_view_cells.data(), localNumCells);
        } else {
            rank_field["values"].set_external(static_cast<int*>(nullptr), 0);
        }
        data["metadata/vtk_fields/RankID/attribute_type"].set_string("ProcessIds");
        viewRegistry_m.set(label + "_rank_id_cells", rank_id_view_cells);


        // auto print_ranked_mesh_info = [&](){
        //     catalystWarn_m << "[  rank="          << ippl::Comm->rank() << "]"
        //             << " | dims(points)="  << nx << "x" << ny << "x" << nz
        //             << " | ghost: "        << nGhost 
        //             << " | origin=("       << Ox << "," << Oy << "," << Oz << ")"
        //             << " | spacing=("      << Spacing_[0] << "," << (Dim>=2?Spacing_[1]:0)<< "," << (Dim>=3?Spacing_[2]:0) << ")" << endl;
        // };

        // #if defined(MPI_VERSION)
        // MPI_Barrier(MPI_COMM_WORLD);
        // if(ippl::Comm->rank()==0)  print_ranked_mesh_info();
        // MPI_Barrier(MPI_COMM_WORLD);
        // if(ippl::Comm->rank()==1)  print_ranked_mesh_info();
        // MPI_Barrier(MPI_COMM_WORLD);
        // #endif
     
        {
            data["coordsets/cart_uniform_coords/dims/i"].set(nx+ dims_n );
            data["coordsets/cart_uniform_coords/spacing/dx"].set(Spacing_[0]);
            data["coordsets/cart_uniform_coords/origin/x"].set( Ox );
            data["topologies/fmesh_topo/origin/x"].set(         Ox );
        }
        if constexpr(Dim >= 2){
            data["coordsets/cart_uniform_coords/dims/j"].set(ny+ dims_n);
            data["coordsets/cart_uniform_coords/spacing/dy"].set(Spacing_[1]);
            data["coordsets/cart_uniform_coords/origin/y"].set( Oy );
            data["topologies/fmesh_topo/origin/y"].set(         Oy );
        }
        if constexpr(Dim >= 3){
            data["coordsets/cart_uniform_coords/dims/k"].set(nz+ dims_n);
            data["coordsets/cart_uniform_coords/spacing/dz"].set(Spacing_[2]);
            data["coordsets/cart_uniform_coords/origin/z"].set( Oz );
            data["topologies/fmesh_topo/origin/z"].set(         Oz );
        }

        // ==================================
        // Prepare Field Data in a HostMirror
        // ==================================

        // Version 1: Cut out Ghost Cells from data during a deep copy into a new Kokkos View.
        auto getHostMirrorView_noGhosts = [&]() -> HostView_t {

            auto r0 = 
                        Kokkos::make_pair(nGhost, nGhost + nx);
            auto r1 = (Dim >= 2) 
                      ? Kokkos::make_pair(nGhost, nGhost + ny)
                      : Kokkos::make_pair(size_t(0), fullDeviceView.extent(1));
            auto r2 = (Dim >= 3) 
                      ? Kokkos::make_pair(nGhost, nGhost + nz)
                      : Kokkos::make_pair(size_t(0), fullDeviceView.extent(2));

            
            auto deviceSubView = Kokkos::subview(fullDeviceView, r0, r1, r2);
                
            HostView_t hostMirrorFinal = HostView_t("hostMirrorNoGhosts_LayoutLeft", nx, ny, nz);
            // This single deep_copy now performs:
            //    - Device-to-Host transfer
            //    - LayoutRight-to-LayoutLeft transpose
            //    - Cutting Ghost Cells from data.
            Kokkos::deep_copy(hostMirrorFinal, deviceSubView);
            return hostMirrorFinal  ;

            /////////////////////////////////////////////////////////////////////////////////////////////////////////
            // NOTE: 
            // for both lambdas:
            // Technically, we need to use deep copy: 1. when explicitly forced 2. In case of Different memory spaces.
            // 3.(?) when layouts don't match up
            // Even if we can't use a subview directly since data of subview isn't meaningfull accessible in raw format 
            // with data() (so the subview can't be used by Conduit), but Conduit has it's own methods to access a 
            // substructures of arrays. See: Conduit Strided Structured Field descriptions.
            // more efficient versions without default deep copies should be possible partially relying on shallow copies.
            //
            // if (!forceHostCopy_m[label] && std::is_same<device_memory_space, Kokkos::HostSpace>::value) 
            //      HostView_t hostMirrorFinal = HostView_t("hostMirrorNoGhosts",nx,ny,nz); 
            //      -> use data directy if possible eg
            //     HostView_t hostMirrorFinal = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fullDeviceView);
            //     return hostMirrorFinal;
            //
            //
            // (?) create_mirror_and_copy, adapts to spaces! but can convert from LayoutRight to LayoutLeft?  
            /////////////////////////////////////////////////////////////////////////////////////////////////////////
        };
        
        
        // Version 2: Mark Ghost Cells in data with VTK meta data.
        auto getHostMirrorView_withGhosts = [&]() -> HostView_t 
        {
            using m_t = unsigned char;  // Element Type for Field Mask 
            
            /* Define the N-D Host View types we needed for the upcoming section/copying. */
            using DeviceMaskView_t = typename Field<m_t, Dim, ViewArgs...>::view_type; //Fetch original Field View Type 
            using HostMaskView1D_t = Kokkos::View<m_t*, Kokkos::LayoutLeft, Kokkos::HostSpace>; //  Host STORAGE
            using HostMaskView_t = Kokkos::View<
                    typename DeviceMaskView_t::data_type, // e.g., unsigned char***
                    Kokkos::LayoutLeft,
                    Kokkos::HostSpace
                >;    // Host WRAPPER
            HostMaskView1D_t hostMaskView1D; // Declare storage view
            
            
            // --- START OF CACHING LOGIC ---
            auto it = ghostMaskCache_m.find(ghostKey);
            if (it != ghostMaskCache_m.end()) 
            {
                // --- CACHE HIT ---// Re-use the existing ghost view from the cache
                catalystInfo_m << level4 <<"::Execute()::ExecVizChannel(" << label << ") | GhostCache HIT" << endl;
                hostMaskView1D = it->second;
            } 
            else 
            {   
                // --- CACHE MISS ---
                catalystInfo_m << level4 <<"::Execute()::ExecVizChannel(" << label << ") | GhostCache MISS" << endl;

                /* Allocate a mask field matching the source field (same mesh/layout/nghost) */
                Field<m_t, Dim, ViewArgs...> ghostMaskField(Mesh_, Layout_, static_cast<int>(nGhost));
                
                /* Fill entire allocation (owned + ghosts) with 1 */
                ghostMaskField = static_cast<m_t>(1);
                DeviceMaskView_t deviceMaskView   = ghostMaskField.getView();
                auto interior   = ghostMaskField.template getFieldRangePolicy<>(); 

                /* Fill inner cells with 0 */
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
                    throw IpplException("Stream::InSitu::CatalystAdaptor::Execute()::ExecVizChannel(" + label + ") | Type:ippl::Field<" + typeid(T).name() + "," + std::to_string(Dim) + ">)", 
                                        "Unsupported Field Dimnesion (Dim > 3) for Visualisation with Catalyst Paraview");
                }
                Kokkos::fence();
                
                /* Allocate the 1D host view that will own the memory */
                hostMaskView1D = HostMaskView1D_t("hostGhostMask_1D", deviceMaskView.size());

                /*  Create a temporary, UNMANAGED (8)N-D view that wraps the 1D view's data. 
                    This is our copy target. The constructor handles any rank automatically.  
                    (?) The 8-dim extent constructor is flexible for any dimension up to 8
                    but eg 3 analog is not (or it seems so at the moment).
                */
                HostMaskView_t hostMaskView_N_Rank(
                    hostMaskView1D.data(),
                    deviceMaskView.extent(0),
                    deviceMaskView.extent(1),
                    deviceMaskView.extent(2),
                    deviceMaskView.extent(3),
                    deviceMaskView.extent(4),
                    deviceMaskView.extent(5),
                    deviceMaskView.extent(6),
                    deviceMaskView.extent(7) 
                );

                
                // The deep_copy now performs:
                    //  - Device-to-Host transfer
                    //  - LayoutRight-to-LayoutLeft
                    //  - Since ND wraps 1D; Copied data is in the hostMaskView1D owned data
                Kokkos::deep_copy(hostMaskView_N_Rank, deviceMaskView);

                //  Store the 1D view (which owns the memory) in the cache, enough to keep in memory
                ghostMaskCache_m[ghostKey] = hostMaskView1D;                
                // --- END OF CACHING LOGIC ---
                
                ////////////////////////////////////////////////////////////////////////////////////////
                // NOTE: 
                // I am unsure why a 3D wrapper would not succeed also, but currently a
                // code along the following line will run into compilation issues (?).
                //
                // using HostMaskView_t = Kokkos::View<
                //     typename decltype(deviceMaskView)::data_type,
                //     typename decltype(deviceMaskView)::array_layout,
                //     Kokkos::HostSpace
                // >;
                // ---      or              ---
                // using HostMaskView_t = Kokkos::View<
                //     typename decltype(deviceMaskView)::data_type,
                //     Kokkos::LayoutLeft, // <-- Use LayoutLeft
                //     Kokkos::HostSpace
                // >;    
                //
                // HostMaskView_t hostMaskView(hostMaskView1D.data(),
                //                             deviceMaskView.extent(0),
                //                             deviceMaskView.extent(1),
                //                             deviceMaskView.extent(2));
                //
                // Kokkos::deep_copy(hostMaskView, deviceMaskView);
                //
                // ghostMaskCache_m[ghostKey] = hostMaskView1D;
                ////////////////////////////////////////////////////////////////////////////////////////
            }
            



            // auto ghostMask_field_meta  =  data["metadata/vtk_fields/GhostMask_field"]; // can't chooses arbitrary name!!!!
            auto ghostMask_field_meta  =  data["metadata/vtk_fields/vtkGhostType"]; 
            ghostMask_field_meta["attribute_type"] = "Ghosts";  // same as set string??...
            // auto ghostMask_field_node  =  fields["ghostMask_field"]; // can't chooses arbitrary name!!!! must be vtkGhostType
            auto ghostMask_field_node  =  fields["vtkGhostType"];
            ghostMask_field_node["association"].set_string("element");  //associate_m); // vs vertex ...
            ghostMask_field_node["topology"].set_string("fmesh_topo");
            ghostMask_field_node["volume_dependent"].set_string("false");
            // ghostMask_field_node["values"].set_external(hostMaskView.data(), hostMaskView.size());
            ghostMask_field_node["values"].set_external(hostMaskView1D.data(), hostMaskView1D.size());
            

            ////////////////////////////////////////////////////////////////////////////////////////////
            // Note:
            // Field name in the conduit nodes for data/fields and metadata/vtk_fields has to coincide
            // so the meta data can be properly associated with the data.
            ////////////////////////////////////////////////////////////////////////////////////////////

            HostView_t hostMirrorFinal = HostView_t("hostMirrorWithGhosts_LayoutLeft", nx, ny, nz);
            Kokkos::deep_copy(hostMirrorFinal, fullDeviceView);

            return hostMirrorFinal;
            // --- END FIX FOR MAIN FIELD ---
        };

         
        HostView_t hostMirrorFinal = (useGhostMasks_m) ? getHostMirrorView_withGhosts() : getHostMirrorView_noGhosts();
            /* FOR BOTH CASES FINAL NODE SETTINGS ARE DONE AND WE HAVE THE DATA INSIDE hostMirrorFinal */
        using elem_t = std::remove_pointer_t<decltype(hostMirrorFinal.data())>;
            // will return size of vector and amounts of vectors (not size of multiple doubles)
        const auto n_elems = hostMirrorFinal.size();
            // Use true element size as stride (handles padding)
        static constexpr size_t stride_bytes = sizeof(elem_t);
            // offset is zero?? guaranteed?
        const size_t offset = 0;


        field_node["association"].set_string("element"); //associate_m);
        field_node["topology"].set_string("fmesh_topo");
        field_node["volume_dependent"].set_string("false");
        if constexpr (std::is_scalar_v<T>) {
            // --- SCALAR FIELD CASE ---
            field_node["values"].set_external(hostMirrorFinal.data(), n_elems);
        } else if constexpr (is_vector_v<T>) {
            // --- VECTOR FIELD CASE ---
                                    // stride was 1 in predecessor code? how did this work?...
                                     field_node["values/x"].set_external(&hostMirrorFinal.data()[0][0], n_elems, offset, stride_bytes);
            if constexpr (T::dim>=2) field_node["values/y"].set_external(&hostMirrorFinal.data()[0][1], n_elems, offset, stride_bytes);
            if constexpr (T::dim>=3) field_node["values/z"].set_external(&hostMirrorFinal.data()[0][2], n_elems, offset, stride_bytes);        
        } 
        // else {
            // --- INVALID CASE ---
        // }

        /* save view so data isn't discarded */
        viewRegistry_m.set(label, hostMirrorFinal);
}



// ==  PARTICLECONTAINERS derived from ippl::ParticleBase<PLayout<T, dim>,...>,...>
template<typename T>
requires (std::derived_from<std::decay_t<T>, ParticleBaseBase>)
void CatalystAdaptor::ExecVizChannel(const T& entry, const std::string label) 
{   
    const bool refreshOnly = viewRegistry_m.contains(label);
    if (refreshOnly && !forceHostCopy_m[label]) {
        return;
    }

        catalystInfo_m        << "::Execute()::ExecVizChannel(" << label << ") | Type : ParticleBase<PLayout<" 
                    << typeid(particle_value_t<T>).name() 
                    << ","
                    << particle_dim_v<T> 
                    << ",...>...> [or subclass])" << endl;
        const std::string channelName = "ippl_particles_" + label;

        auto particleContainer = &entry;
        assert((particleContainer->R.getView().data() != nullptr) && "R view should not be nullptr, might be missing the right execution space");

        const std::string blockName = "block_allRanks";
        // const std::string blockName = "block_rank" + std::to_string(ippl::Comm->rank());

        // channel for this particleContainer
        // channel of type mesh adheres to conduits mesh blueprint


        auto channel = node_m["catalyst/channels/"+ channelName];



        channel["type"].set_string("multimesh");
        
        
        auto data =     channel["data/block_main"];
        auto data_help =     channel["data/block_help"];
        channel["assembly/main"] = "block_main";
        channel["assembly/help"] = "block_help";
        
        ////////////////////////////////////////////////////////
        // Note:
        // Multimesh currently seems to have caused more headaches compared to what
        // we have seemed to have gained from using it. So if the bugs or inconveniences stay
        // in upcoming updates for ParaView catalyst we might want to use two normal meshes
        // instead.
        // channel["type"].set_string("mesh");
        ////////////////////////////////////////////////////////
        
        auto fields = data["fields"];
        data["type"].set_string("mesh");

        const size_t localNum = particleContainer->getLocalNum();
        const int rank = ippl::Comm->rank();

        using IotaView_t = Kokkos::View<int64_t*, Kokkos::HostSpace>;
        using RankView_t = Kokkos::View<int*, Kokkos::HostSpace>;
        IotaView_t iota_view("iota", localNum);
        RankView_t rank_id_view("rank_id_view", localNum);
        if (localNum > 0) {
            using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
            Kokkos::RangePolicy<HostExecSpace> host_policy(0, localNum);
            Kokkos::parallel_for("fill_iota_host", host_policy, KOKKOS_LAMBDA(const int64_t i) {
                iota_view(i) = i;
            });
            Kokkos::parallel_for("fill_rank_ids", host_policy, KOKKOS_LAMBDA(const int64_t i) {
                rank_id_view(i) = rank;
            });
        }

        viewRegistry_m.set(label + "_iota", iota_view);
        viewRegistry_m.set(label + "_rank_id", rank_id_view);

        // Creates a host-accessible mirror view and copies the data from the device view to the host.
        using RAttrib_t = std::remove_reference_t<decltype(particleContainer->R)>;
        using hostMirror_R_t = typename RAttrib_t::host_mirror_type;
        hostMirror_R_t R_hostMirror;

        if (forceHostCopy_m[label]) {
            R_hostMirror = particleContainer->R.getHostMirror();
            Kokkos::deep_copy(R_hostMirror, particleContainer->R.getView());
            viewRegistry_m.set(R_hostMirror);
        } else {
            R_hostMirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), particleContainer->R.getView());
            viewRegistry_m.set(R_hostMirror);
        }

        using hostMirror_ID_t = typename std::remove_reference_t<decltype(particleContainer->ID)>::host_mirror_type;
        hostMirror_ID_t ID_hostMirror;
        if constexpr (T::EnableIDs) {
            if (forceHostCopy_m[label]) {
                ID_hostMirror = particleContainer->ID.getHostMirror();
                Kokkos::deep_copy(ID_hostMirror, particleContainer->ID.getView());
                viewRegistry_m.set(label, ID_hostMirror);
            } else {
                ID_hostMirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), particleContainer->ID.getView());
                viewRegistry_m.set(label, ID_hostMirror);
            }
        }

        using PLayout_t = T::Layout_t;
        // using vector_t = T::Layout_t::vector_type;
        // using value_t  = T::Layout_t::value_type;
        using R_elem_t = std::remove_pointer_t<decltype(R_hostMirror.data())>; //avoids padding etc  (?) Rattrib_t
        static constexpr size_t R_stride_bytes = sizeof(R_elem_t);

        /* checks if playput it spatial or pure layout */
        if constexpr (has_getRegionLayout_v<PLayout_t>){

            using RLayout_t  = PLayout_t::RegionLayout_t;
            using NDRegion_t = RLayout_t::NDRegion_t;
            constexpr unsigned dim_ = PLayout_t::dim;
            const NDRegion_t ndr = particleContainer->getLayout().getRegionLayout().getDomain();
            
            /* HELPER COORDINATES TO PASS THE BOUNDING BOX in vtk format*/
            /* HELPER TOPOLOGY    TO PASS THE BOUNDING BOX (??even needed??)  in vtk format */
            data_help["coordsets/bound_helper_coords/type"].set_string("uniform");
            data_help["topologies/bound_helper_topo/coordset"].set_string("bound_helper_coords");
            data_help["topologies/bound_helper_topo/type"].set_string("uniform");
            /* create unfirom coordinate mesh only consisting of the corner points of the domain */ 
            {
                data_help["coordsets/bound_helper_coords/dims/i"].set(2);
                data_help["coordsets/bound_helper_coords/spacing/dx"].set( ndr[0].max()  - ndr[0].min() );
                data_help["coordsets/bound_helper_coords/origin/x"].set(   ndr[0].min() );
                // data_help["topologies/bound_helper_topo/origin/x"].set(    ndr[0].min() );
            }
            if constexpr(dim_ >= 2){
                data_help["coordsets/bound_helper_coords/dims/j"].set(2);
                data_help["coordsets/bound_helper_coords/spacing/dy"].set( ndr[1].max()- ndr[1].min() );
                data_help["coordsets/bound_helper_coords/origin/y"].set(   ndr[1].min()               );
                // data_help["topologies/bound_helper_topo/origin/y"].set(    ndr[1].min()               );
            }
            if constexpr(dim_ >= 3){
                data_help["coordsets/bound_helper_coords/dims/k"].set(2);
                data_help["coordsets/bound_helper_coords/spacing/dz"].set( ndr[2].max()- ndr[1].min() );
                data_help["coordsets/bound_helper_coords/origin/z"].set(   ndr[2].min()               );
                // data_help["topologies/bound_helper_topo/origin/z"].set(    ndr[2].min()               );
            }
    } 
    // else {
            /* will use raw particle data instead .. */
        // }
        
        
        /* ATTRIBUTES HARDCODED IN PARTICELBASE are identity ID and position R */
        /* EXPLICIT COORDINATES -> EACH PARTICLE'S POSITION */

        /////////////////////////////////////////////////////////////
        // Note:
        // For debuggng purposes checking distribution of particles on ranks...
        // #if defined(MPI_VERSION)
        // // MPI_Barrier(MPI_COMM_WORLD);
        // // if(ippl::Comm->rank()==0)  catalystInfo_m << level4 <<"[Rank 0] Local Particles: " << localNum << endl;
        // MPI_Barrier(MPI_COMM_WORLD);
        // catalystWarn_m << "Local Particles: " << localNum << endl;
        // MPI_Barrier(MPI_COMM_WORLD);
        // #endif
        /////////////////////////////////////////////////////////////

            data["coordsets/p_explicit_coords/type"].set_string("explicit");
            /* unstructured topology relying on per rank unique particle ID */
            data["topologies/p_unstructured_topo/coordset"].set_string("p_explicit_coords");
            data["topologies/p_unstructured_topo/type"].set_string("unstructured");
            data["topologies/p_unstructured_topo/elements/shape"].set_string("point");
            data["topologies/p_unstructured_topo/elements/connectivity"].set_external(iota_view.data(),particleContainer->getLocalNum());
            
            // left hardcodeed we already have the hostViews (instead of integrating the into the loop)

            /* Process ID ATTRIBUTE */
            auto rank_field = fields["RankID"];
            rank_field["association"].set_string("vertex");
            rank_field["topology"].set_string("p_unstructured_topo");
            rank_field["volume_dependent"].set_string("false");
            rank_field["values"].set_external(rank_id_view.data(), localNum);
            data["metadata/vtk_fields/RankID/attribute_type"].set_string("ProcessIds");

            /* Global ID ATTRIBUTE (only when particle IDs are enabled in ParticleBase) */
            if constexpr (T::EnableIDs) {
                auto id_field = fields["ParticleIDs"];
                id_field["association"].set_string("vertex");
                id_field["topology"].set_string("p_unstructured_topo");
                id_field["volume_dependent"].set_string("false");
                if (localNum > 0) {
                    id_field["values"].set_external(ID_hostMirror.data(), localNum);
                } else {
                    using id_value_t = typename hostMirror_ID_t::value_type;
                    id_field["values"].set_external(static_cast<id_value_t*>(nullptr), 0);
                }
                data["metadata/vtk_fields/ParticleIDs/attribute_type"].set_string("GlobalIds");
            }

            /* POSITION ATTRIBUTE */
            auto R_field = fields["position"];
            R_field["association"].set_string("vertex");
            R_field["topology"].set_string("p_unstructured_topo");
            R_field["volume_dependent"].set_string("false");


        if (localNum > 0)
        {   
            /* COORDINATE DEFINITION... */
            data["coordsets/p_explicit_coords/values/x"].set_external(&R_hostMirror.data()[0][0], particleContainer->getLocalNum(), 0, R_stride_bytes);
            data["coordsets/p_explicit_coords/values/y"].set_external(&R_hostMirror.data()[0][1], particleContainer->getLocalNum(), 0, R_stride_bytes);
            data["coordsets/p_explicit_coords/values/z"].set_external(&R_hostMirror.data()[0][2], particleContainer->getLocalNum(), 0, R_stride_bytes);
            
            /* POSITION ATTRIBUTE */
            R_field["values/x"].set_external(&R_hostMirror.data()[0][0], particleContainer->getLocalNum(), 0, R_stride_bytes);
            R_field["values/y"].set_external(&R_hostMirror.data()[0][1], particleContainer->getLocalNum(), 0, R_stride_bytes);
            R_field["values/z"].set_external(&R_hostMirror.data()[0][2], particleContainer->getLocalNum(), 0, R_stride_bytes);
            
            /* concept for no copy in situ vis would be */
            //mesh["topologies/p_unstructured_topo/elements/connectivity"].set_external(particleContainer->ID.getView().data(),particleContainer->getLocalNum());
        }else 
        {
            // In case a rank has no particles-> data()[0] is nulllptr dereferencing !!!!!
            using component_type = typename R_elem_t::value_type;
            data["coordsets/p_explicit_coords/values/x"].set_external(static_cast<component_type*>(nullptr), 0);
            data["coordsets/p_explicit_coords/values/y"].set_external(static_cast<component_type*>(nullptr), 0);
            data["coordsets/p_explicit_coords/values/z"].set_external(static_cast<component_type*>(nullptr), 0);       
            R_field["values/x"].set_external(static_cast<component_type*>(nullptr), 0);
            R_field["values/y"].set_external(static_cast<component_type*>(nullptr), 0);
            R_field["values/z"].set_external(static_cast<component_type*>(nullptr), 0);
        }

        // Skip built-in attributes (ID and/or R) and call signConduitBlueprintNode
        // on every remaining user-defined attribute.
        //
        // ?NOTE?: getAttributeNum() counts across ALL memory spaces, while getAttribute(i)
        // only indexes into the DefaultExecutionSpace::memory_space vector — a mismatch
        // that causes an out-of-bounds segfault when attributes exist in multiple spaces.
        // forAllAttributes with void MemorySpace passes each per-space *vector* to the
        // functor, so we iterate the vectors ourselves with a global skip counter.
        constexpr size_t builtinAttribs = T::EnableIDs ? 2 : 1;

        {
            size_t attrib_idx = 0;
            entry.forAllAttributes([&]<typename Attributes>(const Attributes& atts) {
                for (auto* attribute : atts) {
                    if (attrib_idx >= builtinAttribs) {
                        attribute->signConduitBlueprintNode(localNum, fields, viewRegistry_m, catalystInfo_m, catalystWarn_m, forceHostCopy_m[label]);
                    }
                    ++attrib_idx;
                }
            });
        }
         ////////////////////////////////////////////////////////////////////////////////////////////
         // Note:
         // All ways on how to iterate over particle attributes rely on base class pointers.
         // In whih case dimensions and types particle attributes is not retrievable from the 
         // a pointer instance. Therefore conduit maniupulation are done as a membermethod
         // for paricleAttrib (overriding a virtual method in the base class).
         //
         // entry.template forAllAttributes<void>(
         //     [&]<typename Attributes>(const Attributes& atts) {
         //         for (auto* attribute : atts) {
         ////////////////////////////////////////////////////////////////////////////////////////////
        
}


// BASE CASE: only enabled if EntryT is NOT derived from ippl::ParticleBaseBase
template<typename T>
requires (!std::derived_from<std::decay_t<T>, ParticleBaseBase>)
 void CatalystAdaptor::ExecVizChannel(   [[maybe_unused]] T&& entry,  const std::string label)
{
        catalystInfo_m << level4 <<"  Entry type can't be processed: ID "<< label <<" "<< typeid(std::decay_t<T>).name() << endl;
}


/* SHARED_PTR DISPATCHER - automatically unwraps and dispatches to appropriate overload */
template<typename T>
 void CatalystAdaptor::ExecVizChannel(   const std::shared_ptr<T>& entry,const std::string  label )
{
        if (entry) {
            catalystInfo_m << level4 <<"  dereferencing shared pointer and reattempting execute..." << endl;
            ExecVizChannel(*entry, label ); 
        } else {
            catalystInfo_m << level4 <<"  Null shared_ptr encountered" << endl;
        }
}


}


// =====================================================================================
// STEERING:
// =====================================================================================
#include "Stream/InSitu/CatalystAdaptorSteering.hpp"
// =====================================================================================

namespace ippl{
///////////////////////////////////////////////////////////////
// Note:
// this does not really need to be a separate function call,
// May we should just inline this into the execute functions ...
///////////////////////////////////////////////////////////////
void CatalystAdaptor::fetchResults() {
        
        catalyst_status err = catalyst_results(conduit_cpp::c_node(&results_m));
        if (err != catalyst_status_ok)
        {
            std::cerr << "Failed to execute Catalyst-results: " << err << std::endl;
        }
    }


// =====================================================================================
// Runtime registry based Initialize / Execute (non-templated registry)
// =====================================================================================

void CatalystAdaptor::Initialize(
                           const std::shared_ptr<VisRegistryRuntime>& visReg,
                           const std::shared_ptr<VisRegistryRuntime>& steerReg
                        ) {
if ( !visEnabled_m) return;


    catalystInfo_m << level4 <<"::Initialize() START============================================================= 0" << endl;

    int all_ready = 1;
    #if defined(MPI_VERSION)
        MPI_Allreduce(MPI_IN_PLACE, &all_ready, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    #endif
    catalystInfo_m << level4 <<"::InitializeRuntime() Ranks ready for catalyst int: " << all_ready << " ranks" << endl;  
                    
    visRegistry_m   = visReg;
    steerRegistry_m = steerReg;


    const int fcomm = MPI_Comm_c2f(MPI_COMM_WORLD);
    const int64_t fcomm64 = static_cast<int64_t>(fcomm);
    node_m["catalyst/mpi_comm"].set(fcomm64);   


    setNodeScript(  node_m["catalyst/scripts/script/filename"], //where in node_m
                    "CATALYST_PIPELINE_PATH",                   // environment override
                    sourceDir_m / "catalyst_scripts" / "pipeline_default.py") //default
                ;
    conduit_cpp::Node args = node_m["catalyst/scripts/script/args"];

    
    args.append().set_string("--channel_names");
    InitVisitor initV{*this};
    visRegistry_m->forEach(initV); 
    // Visitor will (also) append channel names here into the node_m (sequence of overall arguments is important!!)

    args.append().set_string("--verbosity");
    args.append().set_string(std::to_string(catalystInfo_m.getOutputLevel()));


    args.append().set_string("--VTKextract");
    args.append().set_string(std::string(catalystVtk_m));

    args.append().set_string("--live");
    args.append().set_string(std::string(catalystLive_m));

    args.append().set_string("--steer");
    args.append().set_string(std::string(catalystSteer_m));


    args.append().set_string("--steer_channel_names");
        
    auto proxyPath = (sourceDir_m / "catalyst_scripts" / "catalyst_proxy.xml").string() ;
    std::string cfgYaml;
    if (const char* cfg_env = std::getenv("IPPL_PROXY_CONFIG_YAML")) {
        if (std::filesystem::exists(cfg_env)) {
            cfgYaml = std::string(cfg_env);
        } else {
            catalystInfo_m << level4 <<"::Initialize() IPPL_PROXY_CONFIG_YAML set but file not found: '" << cfg_env << "', using default." << endl;
        }
    }
    if (cfgYaml.empty()) {
        auto default_cfgYaml = (sourceDir_m / "catalyst_scripts" / "proxy_default_config.yaml").string();
        if (std::filesystem::exists(default_cfgYaml)) {
            cfgYaml = std::move(default_cfgYaml);
        } // else leave empty -> ProxyWriter can proceed without config
    }


    proxyWriter_m.initialize(proxyPath, cfgYaml);
    if (steerEnabled_m ) {
        SteerInitVisitor steerInitV{*this};
        steerRegistry_m->forEach(steerInitV);
    } 
    setNodeScript(    node_m["catalyst/proxies/proxy_/filename"],
        "CATALYST_PROXYS_PATH",
        proxyPath
    );

    

    if( std::string(proxyOption_m) == "PRODUCE_ONLY"){
            proxyWriter_m.produceUnified("SteerableParameters_SCALARS", "SteerableParameters");
            throw IpplException("Stream::InSitu::CatalystAdaptor", "write_proxy_only_run: proxies have been printed");
    }else if( std::string(proxyOption_m) == "OFF"){
    }else{
        proxyWriter_m.produceUnified("SteerableParameters_SCALARS", "SteerableParameters");
    }


    catalystInfo_m << level4 <<"::Initialize()   Printing Conduit `node_m` instance passed to catalyst_initialize() =>" << endl;
    catalystInfo_m << level4 <<node_m.to_yaml() << endl; // or node.to_json() 
        
    catalyst_status err = catalyst_initialize(conduit_cpp::c_node(&node_m));
    if (err != catalyst_status_ok) {
        catalystInfo_m << level4 <<"::Initialize()   Catalyst initialization failed." << endl;
        throw IpplException("Stream::InSitu::CatalystAdaptor::Initialize()", "Failed to initialize Catalyst!!!");
    } else {
        catalystInfo_m << level4 <<"::Initialize()   Catalyst initialized successfully." << endl;
    }
    node_m.reset();
    catalystInfo_m << level4 <<"::Initialize()  DONE============================================================= 1" << endl;
    
}




void CatalystAdaptor::rememberNow(const std::string label){
if ( !visEnabled_m) return;

    // Validate inputs and state
    auto it  = forceHostCopy_m.find(label);
    if (it == forceHostCopy_m.end()){
        throw IpplException("Stream::InSitu::CatalystAdaptor::rememberNow", "Label not present in Visualisation Registry: " + label);
    }
    if (!visRegistry_m) {
        throw IpplException("Stream::InSitu::CatalystAdaptor::rememberNow", "Visualization registry is not initialized (nullptr)");
    }

    // Temporarily force a host copy for this label during an execute
    bool tmp = it->second;
    forceHostCopy_m[label] = true;
    ExecVisitor execV{*this};
    const bool ok = visRegistry_m->forOne(label, execV);

    // Restore prior state
    forceHostCopy_m[label] = tmp;
    if (!ok) {
        throw IpplException("Stream::InSitu::CatalystAdaptor::rememberNow", "Label not found in executable entries or has no execute callback: " + label);
    }

}

void CatalystAdaptor::Execute( int cycle, double time, int rank /* default = ippl::Comm->rank() */) {
    if ( !visEnabled_m) return;

    catalystInfo_m << level4 <<"::Execute() START =============================================================== 0" << endl;
    
    static IpplTimings::TimerRef TMRcatalyst_execute = IpplTimings::getTimer("catalyst_execute");
    static IpplTimings::TimerRef TMRexecVizVisitor   = IpplTimings::getTimer("execVizVisitor");
    static IpplTimings::TimerRef TMRexecSteerVisitor = IpplTimings::getTimer("execSteerVisitor");


    auto state = node_m["catalyst/state"];
    state["cycle"].set(cycle);
    state["time"].set(time);
    state["domain_id"].set(rank);    

    IpplTimings::startTimer(TMRexecVizVisitor);
    if ( !!visEnabled_m){
        // edit forward Node: add visualisation channels
        ExecVisitor execV{*this};
        visRegistry_m->forEach(execV); 
    }
    IpplTimings::stopTimer(TMRexecVizVisitor);


    IpplTimings::startTimer(TMRexecSteerVisitor);
    if (catalystSteer_m && std::string(catalystSteer_m) == "ON") {
        // edit forward Node: add steering channels
        SteerForwardVisitor steerV{*this};
        steerRegistry_m->forEach(steerV); 
    }
    IpplTimings::stopTimer(TMRexecSteerVisitor);


    if(cycle == 0){

        #if defined(MPI_VERSION)
        MPI_Barrier(MPI_COMM_WORLD);
            catalystInfo_m << level4 <<"::Execute() [rank = 0]  Printing first Conduit Node passed from  to catalyst_execute() ==>" << endl;
            if(catalystInfo_m.getOutputLevel() > 0 && ippl::Comm->rank()==0) node_m.print();
            catalystInfo_m << level4 <<"::Execute() [rank = 1]  Printing first Conduit Node passed from  to catalyst_execute() ==>" << endl;
        MPI_Barrier(MPI_COMM_WORLD);
            if(catalystInfo_m.getOutputLevel() > 0 && ippl::Comm->rank()==1) node_m.print();
        MPI_Barrier(MPI_COMM_WORLD);
        #endif
        // if(level >= 5 && ippl::Comm->rank()==0)  node_m.print();

        catalystInfo_m    << "::Execute() During first catalyst_execute() catalyst will "     << endl
                << "            for each passed script - in order how they were "   << endl 
                << "            passed to the conduit node - run the globa scope,"  << endl 
                << "             the initialize() and the execute()."               << endl;
    }


    ////////////////////////////////////////////////////////////////
    // Note:
    // Possibly helpful for further debugging.
    //
    // Kokkos::fence();
    // #if defined(MPI_VERSION)
    // MPI_Barrier(MPI_COMM_WORLD);
    // #endif
    //
    // int all_ready = 1;
    // #if defined(MPI_VERSION)
    //     MPI_Allreduce(MPI_IN_PLACE, &all_ready, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    // #endif
    // catalystInfo_m << level4 <<"::Execute() All ranks ready for catalyst_execute:    " << all_ready << " ranks" << endl;  
    ////////////////////////////////////////////////////////////////        
        
    catalystInfo_m << level4 <<"::Execute()::catalyst_execute() ==>" << endl;
    IpplTimings::startTimer(TMRcatalyst_execute);
        catalyst_status err = catalyst_execute(conduit_cpp::c_node(&node_m));
    IpplTimings::stopTimer(TMRcatalyst_execute);

    ////////////////////////////////////////////////////////////////////////////////
    // Note: 
    // catalyst execute seems to be the current bottleneck of a medium sized simulation... 
    ////////////////////////////////////////////////////////////////////////////////


    if (err != catalyst_status_ok) {
        std::cerr << "::Execute()   Failed to execute Catalyst (runtime path): " << err << std::endl;
    }

    if (catalystSteer_m && std::string(catalystSteer_m) == "ON") {
        
        static IpplTimings::TimerRef TMRfetchResult = IpplTimings::getTimer("fetchSteerParameters");
        IpplTimings::startTimer(TMRfetchResult);

            fetchResults();
            // backward Node: fetch updated steering values
            SteerFetchVisitor fetchV{*this};
            steerRegistry_m->forEach(fetchV);

        IpplTimings::stopTimer(TMRfetchResult);


        if(true){
        // if(cycle == 0){
            catalystInfo_m << level4 <<"::Execute()   Printing Conduit Node received from catalyst_execute() ==>" << endl;
            catalystInfo_m << level4 << results_m.to_yaml() << endl;
        }

    }


        viewRegistry_m.clear();
        ghostMaskCache_m.clear();
        node_m.reset();

        ///////////////////////////////////////////////////
        // Note: 
        // We deliberately don't reset results since
        //  1. they will be properly overwritten by Catalyst.
        //  2. If the part of the Catalyst backend crashes and the 
        //      results are not sent back, the old results will be used, possibly 
        //      avoiding a problems during result retrieval.
        //
        // results.reset();
        ///////////////////////////////////////////////////

        
    catalystInfo_m << level4 <<"::Execute()  DONE =============================================================== 1" << endl;
 
}


void CatalystAdaptor::Finalize() {
    if ( !visEnabled_m) return;

    conduit_cpp::Node node;
    catalyst_status err = catalyst_finalize(conduit_cpp::c_node(&node_m));
    if (err != catalyst_status_ok) {
        std::cerr << "Failed to finalize Catalyst: " << err << std::endl;
    }
}



}//ippl