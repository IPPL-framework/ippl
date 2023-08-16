#!/bin/bash
#
# Setup script for the IPPL Toolchain Builder.
#
# Notes:
# - This script must be sourced not executed!
# - Supported shells are BASH and ZSH (new default on macOS)
#
if [[ $_ == $0 ]]; then
	echo "This file must be sourced not executed!"
	exit 1
fi

if [[ -n "${BASH_VERSION}" ]]; then
	#
	# BASH specific code
	#
	declare __my_name="$(basename -- "${BASH_SOURCE[0]}")"
	declare __my_dir="$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
	__otb_path_munge () {
		local _path=$1
    		case ":${!_path}:" in
			*:"$2":*)
            		;;
        		*)
            			if [ -z "${!_path}" ] ; then
                			eval ${_path}=\"$2\"
            			else
                			eval ${_path}=\"$2:${!_path}\"
            			fi
    		esac
	}
elif [[ -n "${ZSH_VERSION}" ]]; then
	#
	# ZSH specific code
	#
	declare __my_name="$(basename -- "${(%):-%N}")"
	declare __my_dir="$(cd "$(dirname -- "${(%):-%N}")" && pwd)"
	__otb_path_munge () {
		local _path=$1
    		case ":${(P)_path}:" in
			*:"$2":*)
            		;;
        		*)
            			if [ -z "${(P)_path}" ] ; then
                			eval ${_path}=\"$2\"
            			else
                			eval ${_path}=\"$2:${(P)_path}\"
				fi
    		esac
	}
else
	echo "Unsupported shell!" 1>&2
	return
fi

export OTB_OS=$(uname -s)
case ${OTB_OS} in
	Darwin )
		OTB_OS='macOS'
		;;
	Linux )
		:
		;;
	* )
		echo "Unsupported OS -- ${OS}" 1>&2
		;;
esac

declare -ix OTB_ERR_ARG=1
declare -ix OTB_ERR_SETUP=2
declare -ix OTB_ERR_SYSTEM=3
declare -ix OTB_ERR_DOWNLOAD=4
declare -ix OTB_ERR_UNTAR=5
declare -ix OTB_ERR_CONFIGURE=6
declare -ix OTB_ERR_MAKE=7
declare -ix OTB_ERR_PRE_INSTALL=8
declare -ix OTB_ERR_INSTALL=9
declare -ix OTB_ERR_POST_INSTALL=10
declare -ix OTB_ERR=255

#
# cleanup settings from previous calls
#
unset OTB_RECIPES
unset OTB_SYMLINKS

__usage(){
	echo "
Usage:
	source ${_my_name} [--prefix DIR] [CONFIG_FILE]

This script setup the environment for the IPPL Toolchain Builder.

OPTIONS and ARGS
--prefix
	Installation prefix. The default is
	\${HOME}/IPPL
	Everything will go into this directory including
	downloaded and temporary files.

CONFIG_FILE
	In the configuration file you can set the compiler
	collection and the MPI flavour you want to use.
	On Linux the compiler collection defaults to GCC
	on Linux and Clang/LLVM on macOS. 
	The default MPI flavour is OpenMPI.

	In the configuration file you can also define 
	which build-script have to be called.

	Predefined configuration files are available in
	the sub-directory 'config/{linux,macOS}'.

" 1>&2
	return ${OTB_ERR_ARG}
}

#
# This function is used in the build recipes to trap EXIT.
#
# Note: we cannot use it in this script due to the exit call!
#
otb_exit() {
        local -i ec=$?
	if [[ -n "${BASH_VERSION}" ]]; then
        	local -i n=${#BASH_SOURCE[@]}
        	local -r recipe_name="${BASH_SOURCE[n]}"
	else
		local -r recipe_name="${ZSH_ARGZERO}"
	fi
        echo -n "${recipe_name}: "
        if (( ec == 0 )); then
                echo "done!"
        elif (( ec == OTB_ERR_ARG )); then
                echo "argument error!"
        elif (( ec == OTB_ERR_SETUP )); then
                echo "error in setting everything up!"
        elif (( ec == OTB_ERR_SYSTEM )); then
                echo "unexpected system error!"
        elif (( ec == OTB_ERR_DOWNLOAD )); then
                echo "error in downloading the source file!"
        elif (( ec == OTB_ERR_UNTAR )); then
                echo "error in un-taring the source file!"
        elif (( ec == OTB_ERR_CONFIGURE )); then
                echo "error in configuring the software!"
        elif (( ec == OTB_ERR_MAKE )); then
                echo "error in compiling the software!"
        elif (( ec == OTB_ERR_PRE_INSTALL )); then
                echo "error in pre-installing the software!"
        elif (( ec == OTB_ERR_INSTALL )); then
                echo "error in installing the software!"
        elif (( ec == OTB_ERR_POST_INSTALL )); then
                echo "error in post-installing the software!"
        else
                echo "oops, unknown error!!!"
        fi
        exit ${ec}
}

export -f otb_exit > /dev/null

declare config_file="${__my_dir}/config.sh"
if [[ ${__my_dir} == */etc/profile.d ]]; then
        if [[ -r ${config_file} ]]; then
                source "${config_file}"
                OTB_PREFIX=$(cd "${__my_dir}/../.." && pwd)
                
        fi
else
        while (($# > 0)); do
	        case $1 in
	                --prefix )
		                OTB_PREFIX="$2"
		                shift 1
		                ;;
	                -h | --help | -\? )
		                __usage
		                return $?
		                ;;
	                -* )
		                echo "Illegal option -- $1" 1>&2
		                __usage
		                return $?
		                ;;
	                * )
		                if [[ ! -r "$1" ]]; then
			                echo "File doesn't exist or is not readable -- $1" 1>&2
			                return ${OTB_ERR_ARG}
		                fi
		                source "$1" || return ${OTB_ERR_SETUP}
				#
				# a list of recipes might be defined if we are building the
				# tool-chain.
				#
				if [[ -n "${BASH_VERSION}" ]]; then
					OTB_RECIPES=( ${OTB_RECIPES[@]/#/${__my_dir}/} )
				elif [[ -n "${ZSH_VERSION}" ]]; then
					OTB_RECIPES=${OTB_RECIPES/#/${__my_dir}/}
				fi
				;;
	        esac
	        shift 1
        done
fi

#
# OTB_TOOLSET is used in the boost build recipe
#
if [[ -z "${OTB_TOOLSET}" ]]; then
	if [[ "${OTB_OS}" == 'Darwin' ]]; then
		OTB_TOOLSET='clang'
	else
		OTB_TOOLSET='gcc'
	fi
	echo "TOOLSET not set, using '${OTB_TOOLSET}'!" 1>&2
fi
export OTB_TOOLSET

export OTB_PREFIX="${OTB_PREFIX:-${HOME}/IPPL}"
export OTB_DOWNLOAD_DIR="${OTB_PREFIX}/tmp/Downloads"
export OTB_SRC_DIR="${OTB_PREFIX}/tmp/src"
export OTB_PROFILE_DIR="${OTB_PREFIX}/etc/profile.d"

__otb_path_munge PATH			"${OTB_PREFIX}/bin"
__otb_path_munge C_INCLUDE_PATH		"${OTB_PREFIX}/include"
__otb_path_munge CPLUS_INCLUDE_PATH	"${OTB_PREFIX}/include"
__otb_path_munge LIBRARY_PATH		"${OTB_PREFIX}/lib"
__otb_path_munge LD_LIBRARY_PATH	"${OTB_PREFIX}/lib"
__otb_path_munge PKG_CONFIG_PATH	"${OTB_PREFIX}/lib/pkgconfig"

export PATH
export C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH
export LIBRARY_PATH
export LD_LIBRARY_PATH
export PKG_CONFIG_PATH

export BOOST_DIR="${OTB_PREFIX}"
export BOOST_ROOT="${OTB_PREFIX}"
export HDF5_ROOT="${OTB_PREFIX}"
export MITHRA_PREFIX="${OTB_PREFIX}"

__ncores=$(getconf _NPROCESSORS_ONLN)
[[ ${__ncores} > 10 ]] && __ncores=10
export NJOBS=${NJOBS:-${__ncores}}

mkdir -p "${OTB_PREFIX}/bin" "${OTB_PREFIX}/lib" "${OTB_DOWNLOAD_DIR}" "${OTB_SRC_DIR}" \
	|| return ${OTB_ERR_SETUP}

if [[ -n "${BASH_VERSION}" ]]; then
	for link_name in "${!OTB_SYMLINKS[@]}"; do
		ln -fs "${OTB_SYMLINKS[${link_name}]}" "${OTB_PREFIX}/${link_name}"
	done
elif [[ -n "${ZSH_VERSION}" ]]; then
	for link_name in ${(k)OTB_SYMLINKS}; do
		ln -fs "${OTB_SYMLINKS[${link_name}]}" "${OTB_PREFIX}/${link_name}"
	done
fi

if [[ "${OTB_OS}" == "Linux" ]]; then
	( cd "${OTB_PREFIX}" && ln -fs lib lib64 || return ${OTB_ERR_SETUP};)
	LIBRARY_PATH="${OTB_PREFIX}/lib64:${LIBRARY_PATH}"
	LD_LIBRARY_PATH="${OTB_PREFIX}/lib64:${LD_LIBRARY_PATH}"
	# this has been added for Ubuntu - but is this really needed?
	if [[ -d /usr/lib/x86_64-linux-gnu ]]; then
		LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH}"
	fi
fi

if [[  ${__my_dir} != */etc/profile.d ]]; then
	# we are *building* the binary package
	mkdir -p "${OTB_PROFILE_DIR}"
	cp "${__my_dir}/setup.sh"      "${OTB_PROFILE_DIR}/ippl.sh"

	{
		echo "OTB_TOOLSET=${OTB_TOOLSET}"
		[[ -n ${OTB_COMPILER_VERSION} ]] && \
			echo "OTB_COMPILER_VERSION=${OTB_COMPILER_VERSION}"
		[[ -n ${OTB_MPI} ]] && \
             	   	echo "OTB_MPI=${OTB_MPI}"
		[[ -n ${OTB_MPI_VERSION} ]] && \
			echo "OTB_MPI_VERSION=${OTB_MPI_VERSION}"
	}  > "${OTB_PROFILE_DIR}/config.sh"
else
	# we are *using* the binary package
	export IPPL_PREFIX="${OTB_PREFIX}"
fi

echo "Using:"
echo "    Prefix:       ${OTB_PREFIX}"
echo "    Compiler:     ${OTB_TOOLSET}"
if [[ -n ${OTB_COMPILER_VERSION} ]]; then
	echo "    Version:      ${OTB_COMPILER_VERSION}"
fi
if [[ -n ${OTB_MPI} ]]; then
	echo "    MPI:          ${OTB_MPI}"
fi
if [[ -n ${OTB_MPI_VERSION} ]]; then
	export OTB_MPI_VERSION
	echo "    Version:      ${OTB_MPI_VERSION}"
fi

unset __my_dir
unset __ncores
unset __usage

# Local Variables:
# mode: shell-script-mode
# sh-basic-offset: 8
# End:

