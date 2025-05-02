export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"
export no_proxy="admin,polaris-adminvm-01,localhost,*.cm.polaris.alcf.anl.gov,polaris-*,*.polaris.alcf.anl.gov,*.alcf.anl.gov"


module reset
module use /soft/modulefiles
module switch PrgEnv-nvhpc PrgEnv-gnu
module load spack-pe-base cmake
#module load kokkos/4.2.01
module load cudatoolkit-standalone
module load visualization/ascent

export CXX=CC
export CC=cc
