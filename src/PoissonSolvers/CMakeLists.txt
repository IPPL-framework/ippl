set (_SRCS
    )

set (_HDRS
    PoissonCG.h
    Poisson.h
    NullSolver.h
    LaplaceHelpers.h
)
if (ENABLE_FFT)
    list (APPEND _HDRS
                 FFTOpenPoissonSolver.h
                 FFTOpenPoissonSolver.hpp
                 FFTPeriodicPoissonSolver.h
                 FFTPeriodicPoissonSolver.hpp
                 P3MSolver.h
                 P3MSolver.hpp
    )
endif ()

include_DIRECTORIES (
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

add_ippl_sources (${_SRCS})
add_ippl_headers (${_HDRS})

install (FILES ${_HDRS} DESTINATION include/PoissonSolvers)
