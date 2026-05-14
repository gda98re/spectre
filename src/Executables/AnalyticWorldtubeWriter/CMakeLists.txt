# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EXECUTABLE AnalyticWorldtubeWriter)

add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  AnalyticWorldtubeWriter.cpp
  )

target_link_libraries(
  ${EXECUTABLE}
  PRIVATE
  Boost::boost
  Boost::program_options
  Cce
  Informer
  Options
  Parallel
  Printf
  Spectral
  SpinWeightedSphericalHarmonics
  )

if(BUILD_TESTING)
  add_dependencies(test-executables ${EXECUTABLE})
endif()
