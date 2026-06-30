find_package(PkgConfig)

PKG_CHECK_MODULES(PC_GR_DEEPJSCC gnuradio-deepjscc)

FIND_PATH(
    GR_DEEPJSCC_INCLUDE_DIRS
    NAMES gnuradio/deepjscc/api.h
    HINTS $ENV{DEEPJSCC_DIR}/include
        ${PC_DEEPJSCC_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    GR_DEEPJSCC_LIBRARIES
    NAMES gnuradio-deepjscc
    HINTS $ENV{DEEPJSCC_DIR}/lib
        ${PC_DEEPJSCC_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/gnuradio-deepjsccTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GR_DEEPJSCC DEFAULT_MSG GR_DEEPJSCC_LIBRARIES GR_DEEPJSCC_INCLUDE_DIRS)
MARK_AS_ADVANCED(GR_DEEPJSCC_LIBRARIES GR_DEEPJSCC_INCLUDE_DIRS)
