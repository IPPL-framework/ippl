file(RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Adding unit tests found in ${_relPath}")

add_ippl_test(ParticleBase)
# Retain a non-executed instantiation before the active caller to exercise NVCC helper coalescing.
add_ippl_test(ParticleAttribAssignment
  SOURCES ParticleAttribAssignmentUnused.cpp ParticleAttribAssignment.cpp)
add_ippl_test(ParticleBC)
add_ippl_test(ParticleSendRecv)
add_ippl_test(GatherScatterTest)
add_ippl_test(ParticleUpdate)
add_ippl_test(ParticleUpdateNonuniform)
