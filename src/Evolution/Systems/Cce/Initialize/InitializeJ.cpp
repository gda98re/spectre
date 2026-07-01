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
  // Inverse Jacobians at the same spacetime point, from the identity
  // \partial_\hat{A} x^A \partial_A \hat{x}^\hat{B} = \delta (Moxon2020 Eq.
  // 4.18, with a -> c, b -> d). The forward solve supplies the partially flat
  // ("hat") Jacobians forward_gauge_c = \hat a, forward_gauge_d = \hat b, with
  // forward conformal factor \hat\omega^2 = (1/4)(\hat b \bar{\hat b}
  //                                                - \hat a \bar{\hat a}).
  // The inverse (Cauchy) Jacobians the inertial solve must reproduce are then
  //   c = - \hat a / \hat\omega^2,   d = conj(\hat b) / \hat\omega^2.
  const ComplexDataVector omega_squared =
      0.25 * (forward_gauge_d.data() * conj(forward_gauge_d.data()) -
              forward_gauge_c.data() * conj(forward_gauge_c.data()));
  target_c_inv->data() = -forward_gauge_c.data() / omega_squared;
  target_d_inv->data() = conj(forward_gauge_d.data()) / omega_squared;
}

void jacobian_match_heuristic(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
        gauge_c_step,
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
        gauge_d_step,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
    const SpinWeighted<ComplexDataVector, 2>& target_c,
    const size_t /*l_max*/) {
  // Newton-like step for the inverse angular-coordinate solve. The
  // spin-weight-2 factor is driven toward its target, `\delta c = c_target -
  // c`, and the spin-weight-0 factor step is *slaved* to it through the same
  // integrability constraint the forward solve uses, `\delta d = \delta c \bar
  // c / \bar d`. Driving `c` and `d` independently (toward separately-computed
  // targets) is not consistent with a genuine coordinate map: the resulting
  // step has a non-integrable component that lies in the null space of the
  // coordinate update, so the iteration stalls at a fixed point well short of
  // the true inverse. Because the Jacobian factors of a coordinate map are not
  // independent, fixing `c` (which is the exact inverse factor, Moxon2020 Eq.
  // 4.18) determines `d`, and the iteration converges to the true inverse.
  get(*gauge_c_step).data() = target_c.data() - get(gauge_c).data();
  get(*gauge_d_step).data() = get(*gauge_c_step).data() *
                              conj(get(gauge_c).data()) /
                              conj(get(gauge_d).data());
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
