set (_SRCS
  DataSource.cpp
  DataConnect.cpp
  DataConnectCreator.cpp
  DataSource.cpp
  PtclAttribDataSource.cpp
  PtclBaseDataSource.cpp
  )

set (_HDRS
  DataConnectCreator.h
  DataConnect.h
  DataSource.h
  DataSourceObject.h
  FieldDataSource.hpp
  FieldDataSource.h
  FileDataConnect.h
  FileFieldDataSource.hpp
  FileFieldDataSource.h
  FilePtclAttribDataSource.hpp
  FilePtclAttribDataSource.h
  FilePtclBaseDataSource.hpp
  FilePtclBaseDataSource.h
  MakeDataSource.hpp
  MakeDataSource.h
  PtclAttribDataSource.h
  PtclBaseDataSource.h
  )

include_directories (
  ${CMAKE_CURRENT_SOURCE_DIR}
)

ADD_IPPL_SOURCES (${_SRCS})
ADD_IPPL_HEADERS (${_HDRS})

install (FILES ${_HDRS} DESTINATION include/DataSource)

# vi: set et ts=4 sw=4 sts=4:
  
# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:
