# ------------------------------------------------------------------------------
# If we can detect which cluster/machine we are running on, then set scripts up for that
# ------------------------------------------------------------------------------
include(MachineName)
message(STATUS "IPPL_MACHINENAME for script generation is: ${IPPL_MACHINENAME}")

# ------------------------------------------------------------------------------
# for slurm submission, we will need an account name
# ------------------------------------------------------------------------------
set(IPPL_JOB_SUBMISSION_ACCOUNT "c41" CACHE STRING "Account to use for job submission templates")

# ------------------------------------------------------------------------------
# utility function to get target path/name since we can't use generator expressions to set variables
# directly
# ------------------------------------------------------------------------------
function(get_target_location target result)
  file(TO_NATIVE_PATH "/" _separator)
  # use BiNARY_DIR unless IPPL_USE_STANDARD_FOLDERS is ON, then use RUNTIME_OUTPUT_DIRECTORY
  get_target_property(_dir ${target} BINARY_DIR)
  get_target_property(_name ${target} NAME)
  get_target_property(_rundir ${target} RUNTIME_OUTPUT_DIRECTORY)
  if(_rundir)
    set(_dir ${_rundir})
  endif()
  #
  set(${result} "${_dir}${_separator}${_name}" PARENT_SCOPE)
endfunction()

set(ALPS_GH200 "daint;santis;clariden")
set(ALPS_MI300 "beverin")
set(MI250X "lumi")

# ------------------------------------------------------------------------------
# ~~~
# Landau script generator variables substituted into the landau scripts are
# LANDAU_BINARY : full path/name to wherever cmake puts the executable
# JOB_LAUNCH_SCRIPT_NAME : like ... jobscript-gh200.slurm
# JOB_WRAPPER_SCRIPT : utility to set bindings and/or env vars
# ~~~
# ------------------------------------------------------------------------------

if(TARGET LandauDamping)
  get_target_location(LandauDamping LANDAU_BINARY)

  if(IPPL_MACHINENAME IN_LIST ALPS_GH200)
    set(JOB_LAUNCH_SCRIPT_NAME "jobscript-gh200.slurm")
    set(JOB_WRAPPER_SCRIPT "wrapper-gh200.sh")

  elseif(IPPL_MACHINENAME IN_LIST ALPS_MI300)
    set(JOB_LAUNCH_SCRIPT_NAME "jobscript-mi300.slurm")
    set(JOB_WRAPPER_SCRIPT "wrapper-mi300.sh")

  elseif(IPPL_MACHINENAME IN_LIST MI250X)
    set(JOB_LAUNCH_SCRIPT_NAME "jobscript-lumig.slurm")

    # T.B.D.
  else()
    set(JOB_LAUNCH_SCRIPT_NAME "jobscript-generic.slurm")
  endif()
  #
  set(JOB_SCRIPTS_PATH scripts/landau/strong-scaling-alps)
  set(_configure_files generate.sh ${JOB_LAUNCH_SCRIPT_NAME} ${JOB_WRAPPER_SCRIPT})

  # Copy with cmake variable substitution
  foreach(_file ${_configure_files})
    configure_file(${PROJECT_SOURCE_DIR}/${JOB_SCRIPTS_PATH}/${_file}
                   ${PROJECT_BINARY_DIR}/${JOB_SCRIPTS_PATH}/${_file} @ONLY)
  endforeach()
endif()

colour_message(
  STATUS ${Yellow}
  "Configured scripts for ${IPPL_MACHINENAME} in ${PROJECT_BINARY_DIR}/${JOB_SCRIPTS_PATH}")
