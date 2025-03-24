# -----------------------------------------------------------------------------
# src/Expression/CMakeLists.txt
#
# Adds expression-related headers to the IPPL target.
# -----------------------------------------------------------------------------

target_include_directories(ippl
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Install public headers
install(FILES
    IpplExpressions.h
    IpplOperations.h
    DESTINATION include/Expression
)

