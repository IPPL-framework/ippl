set (_SRCS
    )

set (_HDRS
    SolverAlgorithm.h
    ElectrostaticsCG.h
    Electrostatics.h
    PCG.h
    Solver.h
)

if (ENABLE_FFT)
    list (APPEND _HDRS
                 FFTPoissonSolver.h
                 FFTPoissonSolver.hpp
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

add_subdirectory( test/double_precision )
add_subdirectory( test/mixed_precision )

install (FILES ${_HDRS} DESTINATION include/Solver)
