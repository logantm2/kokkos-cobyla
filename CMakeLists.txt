cmake_minimum_required(VERSION 3.12)

project(
    "kokkos-cobyla"
    VERSION 1.0.0
    DESCRIPTION "A header-only C++ port of COBYLA designed for use in Kokkos parallel kernels."
    HOMEPAGE_URL "https://github.com/logantm2/kokkos-cobyla"
)

add_library(${PROJECT_NAME} INTERFACE)
# add alias so the project can be uses with add_subdirectory
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

include(GNUInstallDirs)

# Adding the install interface generator expression makes sure that the include
# files are installed to the proper location (provided by GNUInstallDirs)
target_include_directories(
  ${PROJECT_NAME}
  INTERFACE $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Unnecessary
# target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_14)

if(WITH_TESTS)
    find_package(Kokkos REQUIRED)
    enable_testing()
    find_package(GTest REQUIRED)
    include(GoogleTest)
    add_subdirectory(test)
else()
    find_package(Kokkos)
    if(NOT Kokkos_FOUND)
        message(WARNING "
            Could not find Kokkos!
            We can still install kokkos-cobyla,
            but your project must link to Kokkos for kokkos-cobyla to used."
        )
    endif()
endif()

# locations are provided by GNUInstallDirs
install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}_Targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake.in"
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION
  ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
)

install(
    EXPORT ${PROJECT_NAME}_Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
)

install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
              "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

install(FILES "${PROJECT_SOURCE_DIR}/kokkos_cobyla.hpp" DESTINATION include)
