# update files with correct build dir vars etc

# ------------------------------------------------------------------------------
# for slurm submission, we will need an account name
# ------------------------------------------------------------------------------
if(IPPL_ENABLE_SCRIPTS)
  set(IPPL_JOB_SUBMISSION_ACCOUNT "c41" CACHE STRING "Account to use for job submission templates")
endif()

# ------------------------------------------------------------------------------
# utility function to get target path/name since we can't use generator expressions to set variables
# directly
# ------------------------------------------------------------------------------
function(get_target_location target result)
  file(TO_NATIVE_PATH "/" _separator)
  #
  get_target_property(_dir ${target} BINARY_DIR)
  get_target_property(_name ${target} NAME)
  #
  set(${result} "${_dir}${_separator}${_name}" PARENT_SCOPE)
endfunction()

# ------------------------------------------------------------------------------
# Landau script generator variables substituted into the landau scripts are LANDAU_SCRIPT_NAME :
# usually jobscript-gh200.slurm for alps settings LANDAU_BINARY : full path/name to wherever cmake
# build the target binary
# ------------------------------------------------------------------------------

if(TARGET LandauDamping)
  get_target_location(LandauDamping LANDAU_BINARY)
  #
  set(LANDAU_SCRIPT_NAME jobscript-gh200.slurm)
  #
  set(_script_path landau/strong-scaling-alps)
  set(_landau_files generate.sh ${LANDAU_SCRIPT_NAME})
  #
  foreach(_file ${_landau_files})
    configure_file(${_script_path}/${_file} ${PROJECT_BINARY_DIR}/scripts/${_script_path}/${_file}
                   @ONLY)
  endforeach()
endif()

# ------------------------------------------------------------------------------
# done, let the user know where we put the files we generated
# ------------------------------------------------------------------------------
colour_message(STATUS ${Green} "Scripts configured in ${PROJECT_BINARY_DIR}/scripts")
