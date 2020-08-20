set (_SRCS
    fftpack.cpp
    )

set (_HDRS
    FFTBase.hpp
    FFTBase.h
    FFT.hpp
    FFT.h
    fftpack_FFT.h
    )

include_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

add_ippl_sources (${_SRCS})
add_ippl_headers (${_HDRS})

# vi: set et ts=4 sw=4 sts=4:

# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:
