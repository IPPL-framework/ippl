# -----------------------------------------------------------------------------
# src/Manager/CMakeLists.txt
#
# Adds simulation manager interfaces to the IPPL target.
# -----------------------------------------------------------------------------

target_include_directories(ippl
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Install Manager-related headers
install(FILES
    FieldSolverBase.h
    PicManager.h
    BaseManager.h
    DESTINATION include/Manager
)

