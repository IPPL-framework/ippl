# -----------------------------------------------------------------------------
# demos/cosmology/CMakeLists.txt
#
# Cosmology demo for large-scale structure formation simulations.
# -----------------------------------------------------------------------------
message(STATUS "Configuring demos/cosmology/")

if(NOT IPPL_ENABLE_FFT)
  message(FATAL_ERROR "Cosmology requires IPPL_ENABLE_FFT=ON")
endif()
add_executable(Cosmology Cosmology.cpp)
target_link_libraries(Cosmology PRIVATE IPPL::ippl)

if(BUILD_TESTING)
  add_executable(CompareCosmologyEvolution tests/CompareCosmologyEvolution.cpp)
  target_include_directories(CompareCosmologyEvolution PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(CompareCosmologyEvolution PRIVATE IPPL::ippl)
  add_executable(CompareCosmologyForce tests/CompareCosmologyForce.cpp)
  target_include_directories(CompareCosmologyForce PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(CompareCosmologyForce PRIVATE IPPL::ippl)
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

  option(IPPL_COSMOLOGY_PYTHON_VALIDATION "Enable local MPI evolution tests (NumPy, pandas and SciPy required)" OFF)
  if(IPPL_COSMOLOGY_PYTHON_VALIDATION)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy; import pandas; import scipy.fft"
                    RESULT_VARIABLE cosmology_python_status)
    if(NOT cosmology_python_status EQUAL 0)
      message(FATAL_ERROR "Cosmology validation requires NumPy, pandas and SciPy in Python3_EXECUTABLE")
    endif()
    set(cosmology_validation_command ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/python/validate_linear.py
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

    add_test(NAME cosmology_zarija_analysis
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/python/tests/test_validate_zarija.py)
    set_tests_properties(cosmology_zarija_analysis PROPERTIES LABELS cosmology TIMEOUT 120)

    foreach(analysis frozen_force pancake evolution)
      add_test(NAME cosmology_${analysis}_analysis
               COMMAND ${Python3_EXECUTABLE} -B
                       ${CMAKE_CURRENT_SOURCE_DIR}/python/tests/test_validate_${analysis}.py)
      set_tests_properties(cosmology_${analysis}_analysis PROPERTIES LABELS cosmology TIMEOUT 120)
    endforeach()
    set(cosmology_frozen_command ${Python3_EXECUTABLE} -B
        ${CMAKE_CURRENT_SOURCE_DIR}/python/validate_frozen_force.py
        --ippl-exe $<TARGET_FILE:CompareCosmologyForce> --mpiexec ${MPIEXEC_EXECUTABLE}
        --numproc-flag=${MPIEXEC_NUMPROC_FLAG})
    set(cosmology_pancake_command ${Python3_EXECUTABLE} -B
        ${CMAKE_CURRENT_SOURCE_DIR}/python/validate_pancake.py
        --exe $<TARGET_FILE:Cosmology> --mpiexec ${MPIEXEC_EXECUTABLE}
        --numproc-flag=${MPIEXEC_NUMPROC_FLAG})
    set(cosmology_adapter_command ${Python3_EXECUTABLE} -B
        ${CMAKE_CURRENT_SOURCE_DIR}/python/tests/test_frozen_adapter.py
        --exe $<TARGET_FILE:CompareCosmologyForce> --mpiexec ${MPIEXEC_EXECUTABLE}
        --numproc-flag=${MPIEXEC_NUMPROC_FLAG})
    set(cosmology_evolution_adapter_command ${Python3_EXECUTABLE} -B
        ${CMAKE_CURRENT_SOURCE_DIR}/python/tests/test_evolution_adapter.py
        --exe $<TARGET_FILE:CompareCosmologyEvolution> --production-exe $<TARGET_FILE:Cosmology>
        --mpiexec ${MPIEXEC_EXECUTABLE} --numproc-flag=${MPIEXEC_NUMPROC_FLAG})
    set(cosmology_evolution_command ${Python3_EXECUTABLE} -B
        ${CMAKE_CURRENT_SOURCE_DIR}/python/validate_evolution.py
        --ippl-exe $<TARGET_FILE:CompareCosmologyEvolution> --mpiexec ${MPIEXEC_EXECUTABLE}
        --numproc-flag=${MPIEXEC_NUMPROC_FLAG})
    foreach(flag IN LISTS MPIEXEC_PREFLAGS)
      foreach(command frozen pancake adapter evolution_adapter evolution)
        list(APPEND cosmology_${command}_command --mpi-arg=${flag})
      endforeach()
    endforeach()
    add_test(NAME cosmology_frozen_quick COMMAND ${cosmology_frozen_command} --quick)
    add_test(NAME cosmology_pancake_quick COMMAND ${cosmology_pancake_command} --quick)
    add_test(NAME cosmology_frozen_adapter COMMAND ${cosmology_adapter_command})
    add_test(NAME cosmology_evolution_adapter COMMAND ${cosmology_evolution_adapter_command})
    set_tests_properties(cosmology_frozen_quick cosmology_pancake_quick cosmology_frozen_adapter cosmology_evolution_adapter
      PROPERTIES PROCESSORS 4 TIMEOUT 900 LABELS "cosmology;cosmology_evolution")
    set_tests_properties(cosmology_evolution_analysis cosmology_evolution_adapter PROPERTIES
      ENVIRONMENT "OMP_NUM_THREADS=1;OMP_PROC_BIND=false;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
    add_test(NAME cosmology_evolution_refinement_analysis COMMAND ${Python3_EXECUTABLE} -B
      ${CMAKE_CURRENT_SOURCE_DIR}/python/tests/test_evolution_refinement.py)
    set_tests_properties(cosmology_evolution_refinement_analysis PROPERTIES LABELS cosmology TIMEOUT 120
      ENVIRONMENT "OMP_NUM_THREADS=1;OMP_PROC_BIND=false;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
    foreach(analysis quijote_io quijote_benchmark)
      add_test(NAME cosmology_${analysis} COMMAND ${Python3_EXECUTABLE} -B
        ${CMAKE_CURRENT_SOURCE_DIR}/python/tests/test_${analysis}.py)
      set_tests_properties(cosmology_${analysis} PROPERTIES LABELS "cosmology;quijote" TIMEOUT 180
        ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
    endforeach()
    set(cosmology_external_command ${Python3_EXECUTABLE} -B
      ${CMAKE_CURRENT_SOURCE_DIR}/python/tests/test_external_cosmology.py
      --exe $<TARGET_FILE:Cosmology> --reference-exe $<TARGET_FILE:CompareCosmologyEvolution>
      --mpiexec ${MPIEXEC_EXECUTABLE} --numproc-flag=${MPIEXEC_NUMPROC_FLAG})
    foreach(flag IN LISTS MPIEXEC_PREFLAGS)
      list(APPEND cosmology_external_command --mpi-arg=${flag})
    endforeach()
    add_test(NAME cosmology_external_ic COMMAND ${cosmology_external_command})
    set_tests_properties(cosmology_external_ic PROPERTIES PROCESSORS 4 TIMEOUT 900
      LABELS "cosmology;quijote;cosmology_evolution"
      ENVIRONMENT "OMP_NUM_THREADS=1;OMP_PROC_BIND=false;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
    foreach(analysis gaussian_fixture study_spectra study_storage resolution_study)
      add_test(NAME cosmology_${analysis}_analysis COMMAND ${Python3_EXECUTABLE} -B
        ${CMAKE_CURRENT_SOURCE_DIR}/python/tests/test_${analysis}.py)
      set_tests_properties(cosmology_${analysis}_analysis PROPERTIES LABELS cosmology TIMEOUT 120
        ENVIRONMENT "OMP_NUM_THREADS=1;OMP_PROC_BIND=false;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
    endforeach()
    foreach(analysis runtime_metadata compare_fixture_files merlin_cpu_launcher
                     merlin_gpu_rank merlin_gpu_mpiexec merlin_a100_launcher a100_completion_audit)
      add_test(NAME cosmology_${analysis}_analysis COMMAND ${Python3_EXECUTABLE} -B
        ${CMAKE_CURRENT_SOURCE_DIR}/python/tests/test_${analysis}.py)
      set_tests_properties(cosmology_${analysis}_analysis PROPERTIES LABELS cosmology TIMEOUT 120
        ENVIRONMENT "OMP_NUM_THREADS=1;OMP_PROC_BIND=false;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
    endforeach()
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import matplotlib"
                    RESULT_VARIABLE cosmology_plot_python_status OUTPUT_QUIET ERROR_QUIET)
    if(cosmology_plot_python_status EQUAL 0)
      add_test(NAME cosmology_three_code_campaign_analysis COMMAND ${Python3_EXECUTABLE} -B
        ${CMAKE_CURRENT_SOURCE_DIR}/python/tests/test_three_code_campaign.py)
      set_tests_properties(cosmology_three_code_campaign_analysis PROPERTIES LABELS cosmology TIMEOUT 120
        ENVIRONMENT "OMP_NUM_THREADS=1;OMP_PROC_BIND=false;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
      add_test(NAME cosmology_resolution_plot_analysis COMMAND ${Python3_EXECUTABLE} -B
        ${CMAKE_CURRENT_SOURCE_DIR}/python/tests/test_plot_resolution_study.py)
      set_tests_properties(cosmology_resolution_plot_analysis PROPERTIES LABELS cosmology TIMEOUT 120
        ENVIRONMENT "OMP_NUM_THREADS=1;OMP_PROC_BIND=false;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
    endif()
    add_custom_target(cosmology_validate_pancake COMMAND ${cosmology_pancake_command}
      DEPENDS Cosmology USES_TERMINAL
      COMMENT "Finite-amplitude analytical pancake, convergence and ranks 1-4")
    set(IPPL_COSMOLOGY_FASTPM_EXECUTABLE "" CACHE FILEPATH "Pinned native FastPM frozen-force harness")
    set(IPPL_COSMOLOGY_FASTPM_MANIFEST "" CACHE FILEPATH "Pinned FastPM build provenance manifest")
    if(IPPL_COSMOLOGY_FASTPM_EXECUTABLE)
      list(APPEND cosmology_frozen_command --fastpm-exe ${IPPL_COSMOLOGY_FASTPM_EXECUTABLE})
      if(IPPL_COSMOLOGY_FASTPM_MANIFEST)
        list(APPEND cosmology_frozen_command --fastpm-manifest ${IPPL_COSMOLOGY_FASTPM_MANIFEST})
      endif()
      add_custom_target(cosmology_validate_frozen_force COMMAND ${cosmology_frozen_command}
        DEPENDS CompareCosmologyForce USES_TERMINAL
        COMMENT "Frozen native FastPM/IPPL forces and independent mesh oracle, ranks 1-4")
    endif()
    set(IPPL_COSMOLOGY_FASTPM_EVOLUTION_EXECUTABLE "" CACHE FILEPATH "Pinned native FastPM evolution harness")
    set(IPPL_COSMOLOGY_FASTPM_EVOLUTION_MANIFEST "" CACHE FILEPATH "Separate FastPM evolution build manifest")
    if(IPPL_COSMOLOGY_FASTPM_EVOLUTION_EXECUTABLE)
      list(APPEND cosmology_evolution_command --fastpm-exe ${IPPL_COSMOLOGY_FASTPM_EVOLUTION_EXECUTABLE})
      if(IPPL_COSMOLOGY_FASTPM_EVOLUTION_MANIFEST)
        list(APPEND cosmology_evolution_command --fastpm-manifest ${IPPL_COSMOLOGY_FASTPM_EVOLUTION_MANIFEST})
      endif()
      add_test(NAME cosmology_matched_evolution_quick COMMAND ${cosmology_evolution_command} --quick)
      set_tests_properties(cosmology_matched_evolution_quick PROPERTIES PROCESSORS 4 TIMEOUT 900
        LABELS "cosmology;cosmology_evolution;cosmology_reference"
        ENVIRONMENT "OMP_NUM_THREADS=1;OMP_PROC_BIND=false;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
      add_custom_target(cosmology_validate_evolution COMMAND ${CMAKE_COMMAND} -E env
        "OMP_NUM_THREADS=1" "OMP_PROC_BIND=false" "OPENBLAS_NUM_THREADS=1" "MKL_NUM_THREADS=1"
        ${cosmology_evolution_command}
        DEPENDS CompareCosmologyEvolution USES_TERMINAL
        COMMENT "Matched native plain-PM/IPPL evolution, fixed-particle mesh/time studies and ranks 1-4")
    endif()

    # Reference qualification is opt-in: never download or build external code
    # during a normal configure, build, or CTest invocation.
    set(IPPL_COSMOLOGY_ZARIJA_SOURCE "" CACHE PATH "Preserved original Zarija source directory")
    set(IPPL_COSMOLOGY_ZARIJA_EXECUTABLE "" CACHE FILEPATH "Separately built Zarija initializer")
    if(IPPL_COSMOLOGY_ZARIJA_SOURCE)
      add_custom_target(cosmology_compare_zarija_physics
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/python/reference/compare_zarija_physics.py
                --reference-source ${IPPL_COSMOLOGY_ZARIJA_SOURCE}
                --ippl-build ${CMAKE_BINARY_DIR} --box-size 168.75
        USES_TERMINAL COMMENT "Compare original Zarija and IPPL cosmology routines")
      if(IPPL_COSMOLOGY_ZARIJA_EXECUTABLE)
        set(cosmology_zarija_command ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/python/validate_zarija.py
            --ippl-exe $<TARGET_FILE:Cosmology>
            --zarija-exe ${IPPL_COSMOLOGY_ZARIJA_EXECUTABLE}
            --zarija-source ${IPPL_COSMOLOGY_ZARIJA_SOURCE}
            --mpiexec ${MPIEXEC_EXECUTABLE} --numproc-flag=${MPIEXEC_NUMPROC_FLAG})
        foreach(flag IN LISTS MPIEXEC_PREFLAGS)
          list(APPEND cosmology_zarija_command --mpi-arg=${flag})
        endforeach()
        add_custom_target(cosmology_validate_zarija COMMAND ${cosmology_zarija_command}
          DEPENDS Cosmology cosmology_compare_zarija_physics USES_TERMINAL
          COMMENT "Matched-cosmology IC ensembles against the original Zarija generator")
      endif()
    endif()
  endif()
endif()

# Documentation is an opt-in target; no generators run during a normal build.
find_package(Doxygen 1.14 QUIET)
find_package(Python3 QUIET COMPONENTS Interpreter)
if(DOXYGEN_FOUND AND Python3_Interpreter_FOUND)
  add_custom_target(cosmology_docs
    COMMAND ${Python3_EXECUTABLE} -B
            ${CMAKE_CURRENT_SOURCE_DIR}/python/build_documentation.py
            --output ${CMAKE_BINARY_DIR}/docs/cosmology
            --doxygen ${DOXYGEN_EXECUTABLE}
    VERBATIM
    COMMENT "Strict cosmology Doxygen manual, API and coverage checks")
  if(BUILD_TESTING)
    add_test(NAME cosmology_documentation_analysis
      COMMAND ${Python3_EXECUTABLE} -B
              ${CMAKE_CURRENT_SOURCE_DIR}/python/tests/test_documentation.py)
    set_tests_properties(cosmology_documentation_analysis PROPERTIES
      LABELS "cosmology;documentation" TIMEOUT 120)
  endif()
endif()
