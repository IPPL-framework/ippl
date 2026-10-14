file(RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Adding unit tests found in ${_relPath}")

# The DLPack round trip needs a framework to consume the export. For now that is
# PyTorch: the test is only added if the Python embedded by pybind11 can import torch.
if(Python_EXECUTABLE)
  set(_ipplPython "${Python_EXECUTABLE}")
else()
  set(_ipplPython "${PYTHON_EXECUTABLE}")
endif()
if(_ipplPython)
  execute_process(COMMAND "${_ipplPython}" -c "import torch" RESULT_VARIABLE _torchMissing
                  OUTPUT_QUIET ERROR_QUIET)
endif()

if(_ipplPython AND NOT _torchMissing)
  # PYTHONDONTWRITEBYTECODE keeps the source tree free of __pycache__
  add_ippl_test(DLPackExport LINK_LIBS IPPL::python_bridge TIMEOUT 120
                PROPERTIES ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
  # the test loads torch_consumer.py from this directory through InferenceBridge
  target_compile_definitions(DLPackExport PRIVATE IPPL_TEST_PY_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  # pybind11 types are hidden; match it to avoid visibility warnings on InferenceBridge
  set_target_properties(DLPackExport PROPERTIES CXX_VISIBILITY_PRESET hidden
                                                VISIBILITY_INLINES_HIDDEN ON)
else()
  message(STATUS "PyTorch not importable from '${_ipplPython}': skipping DLPackExport")
endif()
