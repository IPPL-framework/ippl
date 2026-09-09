# -----------------------------------------------------------------------------
# src/Stream/InSitu/CMakeLists.txt
# -----------------------------------------------------------------------------
target_include_directories(ippl
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

if(IPPL_ENABLE_CATALYST)
    # ProxyWriter uses Catalyst's Conduit API and is only needed by the Catalyst adaptor.
    target_sources(ippl
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/ProxyWriter.cpp
    )

    set(_ippl_catalyst_resource_source_dir
        "${CMAKE_CURRENT_SOURCE_DIR}/catalyst_scripts")
    set(_ippl_catalyst_resource_build_dir
        "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATADIR}/ippl/catalyst_scripts")

    # CONFIGURE_DEPENDS notices added or removed resources. Each generated rule
    # also depends on its source file, so editing a script resynchronizes it on
    # the next normal build without copying Python cache files.
    file(
        GLOB_RECURSE _ippl_catalyst_resources
        CONFIGURE_DEPENDS
        LIST_DIRECTORIES FALSE
        RELATIVE "${_ippl_catalyst_resource_source_dir}"
        "${_ippl_catalyst_resource_source_dir}/*.py"
        "${_ippl_catalyst_resource_source_dir}/*.yaml"
        "${_ippl_catalyst_resource_source_dir}/__init__py"
    )

    set(_ippl_catalyst_staged_resources)
    foreach(_ippl_catalyst_resource IN LISTS _ippl_catalyst_resources)
        set(_ippl_catalyst_resource_source
            "${_ippl_catalyst_resource_source_dir}/${_ippl_catalyst_resource}")
        set(_ippl_catalyst_resource_output
            "${_ippl_catalyst_resource_build_dir}/${_ippl_catalyst_resource}")
        get_filename_component(
            _ippl_catalyst_resource_output_dir
            "${_ippl_catalyst_resource_output}"
            DIRECTORY
        )

        add_custom_command(
            OUTPUT "${_ippl_catalyst_resource_output}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory
                    "${_ippl_catalyst_resource_output_dir}"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                    "${_ippl_catalyst_resource_source}"
                    "${_ippl_catalyst_resource_output}"
            DEPENDS "${_ippl_catalyst_resource_source}"
            VERBATIM
        )
        list(APPEND _ippl_catalyst_staged_resources
             "${_ippl_catalyst_resource_output}")
    endforeach()

    add_custom_target(
        ippl_catalyst_resources ALL
        DEPENDS ${_ippl_catalyst_staged_resources}
        COMMENT "Staging Catalyst Python and YAML resources"
    )
    add_dependencies(ippl ippl_catalyst_resources)

    unset(_ippl_catalyst_resource_source_dir)
    unset(_ippl_catalyst_resource_build_dir)
    unset(_ippl_catalyst_resources)
    unset(_ippl_catalyst_staged_resources)
    unset(_ippl_catalyst_resource)
    unset(_ippl_catalyst_resource_source)
    unset(_ippl_catalyst_resource_output)
    unset(_ippl_catalyst_resource_output_dir)
endif()


install(FILES
    ProxyWriter.h
    ProxyWriter.hpp
    CatalystAdaptor.h
    CatalystAdaptorSteering.hpp
    CatalystAdaptor.hpp
    CatalystVisitors.h
    DESTINATION include/Stream/InSitu
)

# Catalyst Python pipelines and their YAML configuration are runtime resources,
# not Python packages. Keep them under the platform-independent data directory.
install(
    DIRECTORY catalyst_scripts/
    DESTINATION ${CMAKE_INSTALL_DATADIR}/ippl/catalyst_scripts
    FILES_MATCHING
        PATTERN "__pycache__" EXCLUDE
        PATTERN "*.py"
        PATTERN "*.yaml"
        PATTERN "__init__py"
)
