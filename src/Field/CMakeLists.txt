set (_SRCS
    )

set (_HDRS
#     Assign.hpp
#     AssignDefs.h
#     AssignGeneralBF.hpp
#     AssignGeneralIBF.hpp
#     Assign.h
#     AssignTags.h
    BareField.h
    BareField.hpp
#     BareFieldIterator.h
    BCond.h
    BCond.hpp
#     BrickExpression.hpp
#     BrickExpression.h
#     BrickIterator.hpp
#     BrickIterator.h
#     CompressedBrickIterator.hpp
#     CompressedBrickIterator.h
#     Field.hpp
#     Field.h
    Kokkos_Field.h
    Kokkos_Field.hpp
#     FieldLoc.h
#     FieldSpec.h
    GuardCellSizes.h
    GuardCellSizes.hpp
#     IndexedBareField.hpp
#     IndexedBareField.h
#     IndexedField.h
    LField.h
    LField.hpp
#     ReductionLoc.hpp
#     ReductionLoc.h
    )

include_DIRECTORIES (
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

add_ippl_sources (${_SRCS})
add_ippl_headers (${_HDRS})

install (FILES ${_HDRS} DESTINATION include/Field)

# vi: set et ts=4 sw=4 sts=4:

# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:
