# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  CauchyFirstOrder.cpp
  CauchySecondOrder.cpp
  ComputeSecondOrderRadialDerivativeJ.cpp
  ConformalFactor.cpp
  InitializeJ.cpp
  InverseCubic.cpp
  NoIncomingRadiation.cpp
  ZeroNonSmooth.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  CauchyFirstOrder.hpp
  CauchySecondOrder.hpp
  ComputeSecondOrderRadialDerivativeJ.hpp
  ConformalFactor.hpp
  InitializeJ.hpp
  InverseCubic.hpp
  NoIncomingRadiation.hpp
  RegisterInitializeJWithCharm.hpp
  ZeroNonSmooth.hpp
  )
