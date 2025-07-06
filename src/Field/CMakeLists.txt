# -----------------------------------------------------------------------------
# src/Field/CMakeLists.txt
#
# Adds core Field data structure headers to the IPPL target.
# -----------------------------------------------------------------------------

# Add this directory to public include path
target_include_directories(ippl
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

# Install Field-related headers
install(FILES
    BareField.hpp
    BareField.h
    BareFieldOperations.hpp
    BConds.h
    BConds.hpp
    BcTypes.h
    BcTypes.hpp
    Field.h
    Field.hpp
    FieldOperations.hpp
    HaloCells.h
    HaloCells.hpp
    DESTINATION include/Field
)

