// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Options/Options.hpp"

namespace Cce::InitializeJ {

/// Possible iteration heuristics to use for optimizing the value of the
/// conformal factor \f$\omega\f$ to fix the initial data.
enum class ConformalFactorIterationHeuristic {
  /// Assumes that the spin-weighted Jacobian perturbations obey
  /// \f$c = \hat \eth f\f$,\f$d = \hat{\bar\eth} f\f$, for some spin-weight-1
  /// value\f$f\f$.
  SpinWeight1CoordPerturbation,
  /// Varies only the \f$d\f$ spin-weighted Jacobian when constructing the
  /// itertion heuristic, leaving \f$c\f$ fixed.
  OnlyVaryGaugeD
};

std::ostream& operator<<(
    std::ostream& os,
    const Cce::InitializeJ::ConformalFactorIterationHeuristic& heuristic_type);

}  // namespace Cce::InitializeJ

template <>
struct Options::create_from_yaml<
    Cce::InitializeJ::ConformalFactorIterationHeuristic> {
  template <typename Metavariables>
  static Cce::InitializeJ::ConformalFactorIterationHeuristic create(
      const Options::Option& options) {
    return create<void>(options);
  }
};
template <>
Cce::InitializeJ::ConformalFactorIterationHeuristic
Options::create_from_yaml<Cce::InitializeJ::ConformalFactorIterationHeuristic>::
    create<void>(const Options::Option& options);
