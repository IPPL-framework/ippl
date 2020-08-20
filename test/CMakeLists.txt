macro(list_subdirectories retval curdir)
  file(GLOB sub-dir RELATIVE ${curdir} *)
  set(list_of_dirs "")
  foreach(dir ${sub-dir})
    if(IS_DIRECTORY ${curdir}/${dir})
      set(list_of_dirs ${list_of_dirs} ${dir})
    endif()
  endforeach()
  set(${retval} ${list_of_dirs})
endmacro()

list_subdirectories("TESTS" ${CMAKE_CURRENT_SOURCE_DIR})
foreach (test ${TESTS})
  add_subdirectory (${test})
endforeach()

# vi: set et ts=4 sw=4 sts=4:
  
# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:
