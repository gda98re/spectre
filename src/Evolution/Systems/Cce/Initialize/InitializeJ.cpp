// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"

#include <complex>
#include <cstddef>
#include <memory>
#include <type_traits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshDerivatives.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshInterpolation.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce::InitializeJ {

namespace detail {
void compute_inverse_jacobian_target(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> target_c_inv,
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> target_d_inv,
    const SpinWeighted<ComplexDataVector, 2>& forward_gauge_c,
    const SpinWeighted<ComplexDataVector, 0>& forward_gauge_d,
    const size_t /*l_max*/) {
  // Four times the squared conformal factor of the forward map,
  // (2 omega)^2 = d \bar d - c \bar c.
  const ComplexDataVector two_omega_squared =
      forward_gauge_d.data() * conj(forward_gauge_d.data()) -
      forward_gauge_c.data() * conj(forward_gauge_c.data());
  // TODO(physics): replace with the exact inverse-Jacobian relation. The
  // expressions below are the leading-order (near-identity) inverse Jacobians,
  // provided as a placeholder so the surrounding inversion machinery can be
  // built and tested. The metamorphic inverse-of-forward angular-map test
  // quantifies the residual once the exact relation is supplied here.
  target_d_inv->data() = conj(forward_gauge_d.data()) / two_omega_squared *
                         std::complex<double>(2.0, 0.0);
  target_c_inv->data() = -forward_gauge_c.data() / two_omega_squared *
                         std::complex<double>(2.0, 0.0);
}

void jacobian_match_heuristic(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
        gauge_c_step,
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
        gauge_d_step,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
    const SpinWeighted<ComplexDataVector, 2>& target_c,
    const SpinWeighted<ComplexDataVector, 0>& target_d,
    const size_t /*l_max*/) {
  // Newton-like step driving the current (inverse-solve) Jacobians toward the
  // interpolated targets. Generalizes the omega-only heuristic used by
  // ConformalFactor to both spin-weight-2 (c) and spin-weight-0 (d) factors; it
  // can be refined (damping, filtering) if convergence requires.
  get(*gauge_c_step).data() = target_c.data() - get(gauge_c).data();
  get(*gauge_d_step).data() = target_d.data() - get(gauge_d).data();
}
}  // namespace detail

void GaugeAdjustInitialJ::apply(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> volume_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_omega,
    const tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>&
    /*cauchy_angular_coordinates*/,
    const Spectral::Swsh::SwshInterpolator& interpolator, const size_t l_max) {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  const size_t number_of_radial_points =
      get(*volume_j).size() /
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  Scalar<SpinWeighted<ComplexDataVector, 2>> evolution_coords_j_buffer{
      number_of_angular_points};
  for (size_t i = 0; i < number_of_radial_points; ++i) {
    Scalar<SpinWeighted<ComplexDataVector, 2>> angular_view_j;
    get(angular_view_j)
        .set_data_ref(
            get(*volume_j).data().data() + i * number_of_angular_points,
            number_of_angular_points);
    get(evolution_coords_j_buffer) = get(angular_view_j);
    GaugeAdjustedBoundaryValue<Tags::BondiJ>::apply(
        make_not_null(&angular_view_j), evolution_coords_j_buffer, gauge_c,
        gauge_d, gauge_omega, interpolator);
  }
}
}  // namespace Cce::InitializeJ
