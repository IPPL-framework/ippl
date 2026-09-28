# Distributed IPPL mesh with a cell-centred dielectric prism or brick.
add_executable(mesh-1 mesh-1.cpp)
target_link_libraries(mesh-1 PRIVATE IPPL::ippl catalyst::catalyst)
target_include_directories(mesh-1 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../fel")

foreach(_config IN ITEMS config.yaml config-scale.yaml config-brick.yaml)
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${_config}"
                 "${CMAKE_CURRENT_BINARY_DIR}/${_config}" COPYONLY)
endforeach()

target_compile_definitions(
  mesh-1 PRIVATE CHDR_DEFAULT_CONFIG="${CMAKE_CURRENT_BINARY_DIR}/config.yaml")

# Optional local API/student documentation; never part of the default build.
find_program(CHDR_DOXYGEN_EXECUTABLE NAMES doxygen)
if(CHDR_DOXYGEN_EXECUTABLE)
  add_custom_target(chdr-docs
    COMMAND "${CHDR_DOXYGEN_EXECUTABLE}" Doxyfile
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Generating focused ChDR student documentation in src/docs/html"
    VERBATIM
    SOURCES Doxyfile StudentGuide.md Documentation.dox)
endif()

if(BUILD_TESTING)
  add_executable(test-chdr-geometry test_geometry.cpp)
  target_link_libraries(test-chdr-geometry PRIVATE IPPL::ippl catalyst::catalyst)
  target_include_directories(test-chdr-geometry PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../fel")

  add_test(NAME chdr.mesh.geometry COMMAND test-chdr-geometry)
  add_test(
    NAME chdr.mesh.serial
    COMMAND mesh-1 "${CMAKE_CURRENT_BINARY_DIR}/config.yaml"
            --output "${CMAKE_CURRENT_BINARY_DIR}/test-serial")
  set_tests_properties(chdr.mesh.geometry chdr.mesh.serial
                       PROPERTIES LABELS "chdr;mesh" TIMEOUT 60
                                  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")

  add_test(
    NAME chdr.mesh.brick.serial
    COMMAND mesh-1 "${CMAKE_CURRENT_BINARY_DIR}/config-brick.yaml"
            --output "${CMAKE_CURRENT_BINARY_DIR}/test-brick-serial")
  set_tests_properties(chdr.mesh.brick.serial
                       PROPERTIES LABELS "chdr;mesh" TIMEOUT 60
                                  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")

  if(MPIEXEC_EXECUTABLE)
    add_test(
      NAME chdr.mesh.mpi2
      COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2 ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:mesh-1> ${MPIEXEC_POSTFLAGS}
              "${CMAKE_CURRENT_BINARY_DIR}/config.yaml"
              --output "${CMAKE_CURRENT_BINARY_DIR}/test-mpi2")
    set_tests_properties(chdr.mesh.mpi2
                         PROPERTIES LABELS "chdr;mesh;mpi" TIMEOUT 60 PROCESSORS 4
                                    ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
    add_test(
      NAME chdr.mesh.brick.mpi2
      COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2 ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:mesh-1> ${MPIEXEC_POSTFLAGS}
              "${CMAKE_CURRENT_BINARY_DIR}/config-brick.yaml"
              --output "${CMAKE_CURRENT_BINARY_DIR}/test-brick-mpi2")
    set_tests_properties(chdr.mesh.brick.mpi2
                         PROPERTIES LABELS "chdr;mesh;mpi" TIMEOUT 60 PROCESSORS 4
                                    ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")

    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
      add_test(NAME chdr.mesh.compare
               COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/verify_mesh_output.py"
                       "${CMAKE_CURRENT_BINARY_DIR}/test-serial"
                       "${CMAKE_CURRENT_BINARY_DIR}/test-mpi2")
      set_tests_properties(chdr.mesh.serial PROPERTIES FIXTURES_SETUP ChdrSerial)
      set_tests_properties(chdr.mesh.mpi2 PROPERTIES FIXTURES_SETUP ChdrMpi2)
      set_tests_properties(chdr.mesh.compare PROPERTIES LABELS "chdr;mesh;mpi"
                           FIXTURES_REQUIRED "ChdrSerial;ChdrMpi2" TIMEOUT 60)
      add_test(NAME chdr.mesh.brick.compare
               COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/verify_mesh_output.py"
                       "${CMAKE_CURRENT_BINARY_DIR}/test-brick-serial"
                       "${CMAKE_CURRENT_BINARY_DIR}/test-brick-mpi2")
      set_tests_properties(chdr.mesh.brick.serial PROPERTIES FIXTURES_SETUP ChdrBrickSerial)
      set_tests_properties(chdr.mesh.brick.mpi2 PROPERTIES FIXTURES_SETUP ChdrBrickMpi2)
      set_tests_properties(chdr.mesh.brick.compare PROPERTIES LABELS "chdr;mesh;mpi"
                           FIXTURES_REQUIRED "ChdrBrickSerial;ChdrBrickMpi2" TIMEOUT 60)
    endif()
  endif()
endif()
