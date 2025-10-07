#!/bin/bash


cd build
make PenningTrap  -j
# make PenningTrap LandauDamping BumponTailInstability -j
# make LandauDamping -j 
# make BumponTailInstability -j 

# Report linkage
echo "Linked libs (MPI/Catalyst/Python):"
ldd alpine/PenningTrap | grep -E "libmpi|libmpicxx|libcatalyst|libpython|libascent" || true


# check for multiple MPI versions
mpi_libs=$(ldd alpine/PenningTrap | grep -o 'libmpi\.[^ ]*' | sort -u)
mpi_lib_count=$(echo "${mpi_libs}" | wc -l)

if [ "${mpi_lib_count}" -gt 1 ]; then
    echo "------------------------------------------------------------------" >&2
    echo "WARNING: Multiple MPI implementations linked into PenningTrap!" >&2
    echo "Found the following conflicting libraries:" >&2
    echo "${mpi_libs}" | sed 's/^/  - /' >&2
    echo "This is caused by linking components (like Catalyst and IPPL)" >&2
    echo "that were built against different MPI versions." >&2
    echo "Ensure all components are built with the same MPI (e.g., MPICH)." >&2
    echo "This will very LIKELY CAUSE ERRORS down the line" >&2
    echo "------------------------------------------------------------------" >&2

else
    if [ "${mpi_lib_count}" -eq 1 ]; then
        echo "------------------------------------------------------------------"
        echo "✅ SUCCESS: Single MPI implementation linked into PenningTrap"
        echo "Using: ${mpi_libs}"
        echo "------------------------------------------------------------------"
    else
        echo "No MPI libraries detected in the binary."
    fi
fi


cd ..