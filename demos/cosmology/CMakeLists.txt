# -----------------------------------------------------------------------------
# demos/cosmology/CMakeLists.txt
#
# Cosmology demo for large-scale structure formation simulations.
# -----------------------------------------------------------------------------
message(STATUS "Configuring demos/cosmology/")

add_executable(StructureFormation StructureFormation.cpp)

target_link_libraries(StructureFormation PRIVATE IPPL::ippl)

if(NOT IPPL_ENABLE_FFT)
  message(FATAL_ERROR "Cosmology requires IPPL_ENABLE_FFT=ON")
endif()
add_executable(Cosmology Cosmology.cpp)
target_link_libraries(Cosmology PRIVATE IPPL::ippl)

if(BUILD_TESTING)
  add_executable(TestCosmologyPhysics tests/TestCosmologyPhysics.cpp)
  target_include_directories(TestCosmologyPhysics PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(TestCosmologyPhysics PRIVATE IPPL::ippl)
  add_test(NAME cosmology_physics COMMAND TestCosmologyPhysics)
  set_tests_properties(cosmology_physics PROPERTIES LABELS cosmology TIMEOUT 120)
  foreach(ranks 1 2 3 4)
    add_test(NAME cosmology_spectral_${ranks}r
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:Cosmology> --self-test ${MPIEXEC_POSTFLAGS})
    set_tests_properties(cosmology_spectral_${ranks}r PROPERTIES
                         ENVIRONMENT "OMP_NUM_THREADS=1;OMP_PROC_BIND=false"
                         PROCESSORS ${ranks} TIMEOUT 120 LABELS cosmology)
  endforeach()

  option(IPPL_COSMOLOGY_PYTHON_VALIDATION "Enable local MPI evolution tests (NumPy and pandas required)" OFF)
  if(IPPL_COSMOLOGY_PYTHON_VALIDATION)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy; import pandas"
                    RESULT_VARIABLE cosmology_python_status)
    if(NOT cosmology_python_status EQUAL 0)
      message(FATAL_ERROR "Cosmology validation requires NumPy and pandas in Python3_EXECUTABLE")
    endif()
    set(cosmology_validation_command ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/validate_linear.py
        --exe $<TARGET_FILE:Cosmology> --mpiexec ${MPIEXEC_EXECUTABLE}
        --numproc-flag=${MPIEXEC_NUMPROC_FLAG})
    foreach(flag IN LISTS MPIEXEC_PREFLAGS)
      list(APPEND cosmology_validation_command --mpi-arg=${flag})
    endforeach()
    add_test(NAME cosmology_linear_quick COMMAND ${cosmology_validation_command} --quick)
    set_tests_properties(cosmology_linear_quick PROPERTIES
                         PROCESSORS 4 TIMEOUT 900 LABELS "cosmology;cosmology_evolution")
    add_custom_target(cosmology_validate COMMAND ${cosmology_validation_command}
                      DEPENDS Cosmology USES_TERMINAL
                      COMMENT "Full 1-4 rank cosmology IC, evolution, and convergence validation")
  endif()
endif()
