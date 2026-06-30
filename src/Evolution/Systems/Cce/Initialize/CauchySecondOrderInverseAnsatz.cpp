// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/Initialize/CauchySecondOrderInverseAnsatz.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/GaugeTransformBoundaryData.hpp"
#include "Evolution/Systems/Cce/Initialize/ComputeSecondOrderRadialDerivativeJ.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Evolution/Systems/Cce/LinearOperators.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshInterpolation.hpp"
#include "Parallel/NodeLock.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace Cce::InitializeJ {
namespace {

// Transform the worldtube second radial derivative of J into the evolution
// gauge, extending `GaugeAdjustedBoundaryValue<Tags::Dr<Tags::BondiJ>>` to
// second order. With \f$\tilde J\f$ the Cauchy-gauge J interpolated to the
// transformed angle, the evolution-gauge J is
// \f[
//   \hat J = \frac{1}{4 \omega^2}
//     \left( \bar d^2 \tilde J + c^2 \bar{\tilde J}
//            + 2 c \bar d \sqrt{1 + \tilde J \bar{\tilde J}} \right).
// \f]
// Since c, d, omega depend only on angle, radial derivatives commute with the
// angular transform, and \f$\partial_{\hat r} = \partial_r / \omega\f$, so
// \f$\partial_{\hat r}^2 = \partial_r^2 / \omega^2\f$. Writing
// \f$f = \sqrt{1 + \tilde J \bar{\tilde J}}\f$ with
// \f$N = \tilde J' \bar{\tilde J} + \tilde J \bar{\tilde J}'\f$,
// \f$f' = N / (2 f)\f$ and
// \f$f'' = N' / (2 f) - (f')^2 / f\f$ where
// \f$N' = \tilde J'' \bar{\tilde J} + \tilde J \bar{\tilde J}''
//         + 2 \tilde J' \bar{\tilde J}'\f$, giving
// \f[
//   \partial_{\hat r}^2 \hat J = \frac{1}{4 \omega^4}
//     \left( \bar d^2 \tilde J'' + c^2 \bar{\tilde J}''
//            + 2 c \bar d \, f'' \right).
// \f]
void evolution_gauge_dr_dr_j(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
        evolution_gauge_dr_dr_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& cauchy_gauge_dr_dr_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& cauchy_gauge_dr_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& cauchy_gauge_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
    const Spectral::Swsh::SwshInterpolator& interpolator, const size_t l_max) {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  SpinWeighted<ComplexDataVector, 2> interpolated_j{number_of_angular_points};
  SpinWeighted<ComplexDataVector, 2> interpolated_dr_j{
      number_of_angular_points};
  SpinWeighted<ComplexDataVector, 2> interpolated_dr_dr_j{
      number_of_angular_points};
  interpolator.interpolate(make_not_null(&interpolated_j), get(cauchy_gauge_j));
  interpolator.interpolate(make_not_null(&interpolated_dr_j),
                           get(cauchy_gauge_dr_j));
  interpolator.interpolate(make_not_null(&interpolated_dr_dr_j),
                           get(cauchy_gauge_dr_dr_j));

  const ComplexDataVector f =
      sqrt(1.0 + interpolated_j.data() * conj(interpolated_j.data()));
  const ComplexDataVector n_term =
      interpolated_dr_j.data() * conj(interpolated_j.data()) +
      interpolated_j.data() * conj(interpolated_dr_j.data());
  const ComplexDataVector f_prime = n_term / (2.0 * f);
  const ComplexDataVector n_prime =
      interpolated_dr_dr_j.data() * conj(interpolated_j.data()) +
      interpolated_j.data() * conj(interpolated_dr_dr_j.data()) +
      2.0 * interpolated_dr_j.data() * conj(interpolated_dr_j.data());
  const ComplexDataVector f_double_prime =
      n_prime / (2.0 * f) - f_prime * f_prime / f;

  get(*evolution_gauge_dr_dr_j).data() =
      (square(conj(get(gauge_d).data())) * interpolated_dr_dr_j.data() +
       square(get(gauge_c).data()) * conj(interpolated_dr_dr_j.data()) +
       2.0 * get(gauge_c).data() * conj(get(gauge_d).data()) * f_double_prime) /
      (4.0 * pow<4>(get(omega).data()));
}
}  // namespace

CauchySecondOrderInverseAnsatz::CauchySecondOrderInverseAnsatz(
    const double angular_coordinate_tolerance, const size_t max_iterations,
    const bool require_convergence, const double max_scri_second_derivative,
    const bool use_quartic_ansatz)
    : require_convergence_{require_convergence},
      angular_coordinate_tolerance_{angular_coordinate_tolerance},
      max_iterations_{max_iterations},
      max_scri_second_derivative_{max_scri_second_derivative},
      use_quartic_ansatz_{use_quartic_ansatz} {}

std::unique_ptr<InitializeJ<false>> CauchySecondOrderInverseAnsatz::get_clone()
    const {
  return std::make_unique<CauchySecondOrderInverseAnsatz>(*this);
}

void CauchySecondOrderInverseAnsatz::operator()(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& boundary_u,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_w,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_beta,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& boundary_q,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_du_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_du_dr_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_du_r,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& r, const size_t l_max,
    const size_t number_of_radial_points,
    const gsl::not_null<Parallel::NodeLock*> /*hdf5_lock*/) const {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  // Solve the H hypersurface equation at y = -1 for dy^2 J using the worldtube
  // boundary data, then build the volume J as a polynomial in (1 - y) that
  // matches J, dr_j, and dy^2 J at the worldtube.
  Scalar<SpinWeighted<ComplexDataVector, 2>> boundary_dy2_j{
      number_of_angular_points};
  CauchySecondOrder_detail::compute_dy_dy_j(
      make_not_null(&boundary_dy2_j), boundary_j, boundary_u, boundary_w,
      boundary_beta, boundary_q, boundary_du_j, boundary_dr_j, boundary_du_dr_j,
      boundary_du_r, r, l_max);

  const DataVector one_minus_y_collocation =
      1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::GaussLobatto>(
                number_of_radial_points);

  for (size_t i = 0; i < number_of_radial_points; ++i) {
    ComplexDataVector angular_view_j{
        get(*j).data().data() + get(boundary_j).size() * i,  // NOLINT
        get(boundary_j).size()};
    const auto constant_term = get(boundary_j).data() +
                               get(r).data() * get(boundary_dr_j).data() +
                               4.0 * get(boundary_dy2_j).data() / 3.0;
    const auto one_minus_y_coefficient =
        -(0.5 * get(r).data() * get(boundary_dr_j).data() +
          get(boundary_dy2_j).data());
    const auto one_minus_y_cubed_coefficient =
        get(boundary_dy2_j).data() / 12.0;

    angular_view_j =
        constant_term + one_minus_y_collocation[i] * one_minus_y_coefficient +
        pow<3>(one_minus_y_collocation[i]) * one_minus_y_cubed_coefficient;
  }

  // Iteratively adjust the angular coordinates so that J vanishes at scri+
  // (identical to the procedure in NoIncomingRadiation).
  const SpinWeighted<ComplexDataVector, 2> j_at_scri_view;
  make_const_view(make_not_null(&j_at_scri_view), get(*j),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  Variables<
      tmpl::list<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                      std::integral_constant<int, 2>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<1, ComplexDataVector>,
                                      std::integral_constant<int, 0>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<2, ComplexDataVector>,
                                      std::integral_constant<int, 0>>>>
      iteration_buffers{number_of_angular_points};

  auto& evolution_gauge_surface_j =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                   std::integral_constant<int, 2>>>(
          iteration_buffers));
  auto& interpolated_k =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<1, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          iteration_buffers));
  auto& gauge_omega =
      get<::Tags::SpinWeighted<::Tags::TempScalar<2, ComplexDataVector>,
                               std::integral_constant<int, 0>>>(
          iteration_buffers);

  // Worldtube second radial derivative of J in the Cauchy gauge, needed only
  // for the quartic ansatz. The numerical-coordinate value computed above is
  // dy^2 J; at the worldtube dy^2 = (R / 2) dr + (R^2 / 4) dr^2, so
  // dr^2 J = (4 / R^2) dy^2 J - (2 / R) dr J.
  Scalar<SpinWeighted<ComplexDataVector, 2>> boundary_dr_dr_j{
      number_of_angular_points};
  get(boundary_dr_dr_j).data() =
      4.0 * get(boundary_dy2_j).data() / square(get(r).data()) -
      2.0 * get(boundary_dr_j).data() / get(r).data();

  // Persistent buffers for the worldtube data transformed into the evolution
  // gauge by `finalize_function`; used to rebuild the volume J after the
  // angular-coordinate iteration converges.
  Variables<
      tmpl::list<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                      std::integral_constant<int, 2>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<1, ComplexDataVector>,
                                      std::integral_constant<int, 2>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<2, ComplexDataVector>,
                                      std::integral_constant<int, 0>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<3, ComplexDataVector>,
                                      std::integral_constant<int, 2>>>>
      surface_buffers{number_of_angular_points};
  auto& surface_j =
      get<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                               std::integral_constant<int, 2>>>(
          surface_buffers);
  auto& surface_dr_j =
      get<::Tags::SpinWeighted<::Tags::TempScalar<1, ComplexDataVector>,
                               std::integral_constant<int, 2>>>(
          surface_buffers);
  auto& surface_r =
      get<::Tags::SpinWeighted<::Tags::TempScalar<2, ComplexDataVector>,
                               std::integral_constant<int, 0>>>(
          surface_buffers);
  auto& surface_dr_dr_j =
      get<::Tags::SpinWeighted<::Tags::TempScalar<3, ComplexDataVector>,
                               std::integral_constant<int, 2>>>(
          surface_buffers);

  auto iteration_function =
      [&interpolated_k, &gauge_omega, &evolution_gauge_surface_j,
       &j_at_scri_view](
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
              gauge_c_step,
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
              gauge_d_step,
          const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
          const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
          const Spectral::Swsh::SwshInterpolator& iteration_interpolator) {
        iteration_interpolator.interpolate(
            make_not_null(&evolution_gauge_surface_j), j_at_scri_view);
        interpolated_k.data() =
            sqrt(1.0 + evolution_gauge_surface_j.data() *
                           conj(evolution_gauge_surface_j.data()));
        get(gauge_omega).data() =
            0.5 * sqrt(get(gauge_d).data() * conj(get(gauge_d).data()) -
                       get(gauge_c).data() * conj(get(gauge_c).data()));
        evolution_gauge_surface_j.data() =
            0.25 *
            (square(conj(get(gauge_d).data())) *
                 evolution_gauge_surface_j.data() +
             square(get(gauge_c).data()) *
                 conj(evolution_gauge_surface_j.data()) +
             2.0 * get(gauge_c).data() * conj(get(gauge_d).data()) *
                 interpolated_k.data()) /
            square(get(gauge_omega).data());

        const double max_error = max(abs(evolution_gauge_surface_j.data()));
        get(*gauge_c_step).data() =
            -0.5 * evolution_gauge_surface_j.data() *
            square(get(gauge_omega).data()) /
            (get(gauge_d).data() * interpolated_k.data());
        get(*gauge_d_step).data() = get(*gauge_c_step).data() *
                                    conj(get(gauge_c).data()) /
                                    conj(get(gauge_d).data());
        return max_error;
      };

  // Instead of gauge-transforming the Cauchy-gauge volume J, transform the
  // worldtube data needed to rebuild the volume J from an inverse ansatz. The
  // Cauchy-gauge volume J above is therefore used only to drive the
  // angular-coordinate iteration and is discarded afterwards.
  auto finalize_function =
      [&gauge_omega, &surface_j, &surface_dr_j, &surface_r, &surface_dr_dr_j,
       &boundary_j, &boundary_dr_j, &boundary_dr_dr_j, &r, &l_max, this](
          const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
          const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
          const tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>&
          /*local_angular_cauchy_coordinates*/,
          const Spectral::Swsh::SwshInterpolator& interpolator) {
        get(gauge_omega).data() =
            0.5 * sqrt(get(gauge_d).data() * conj(get(gauge_d).data()) -
                       get(gauge_c).data() * conj(get(gauge_c).data()));
        GaugeAdjustedBoundaryValue<Tags::Dr<Tags::BondiJ>>::apply(
            make_not_null(&surface_dr_j), boundary_dr_j, boundary_j, gauge_c,
            gauge_d, gauge_omega, interpolator, l_max);
        GaugeAdjustedBoundaryValue<Tags::BondiJ>::apply(
            make_not_null(&surface_j), boundary_j, gauge_c, gauge_d,
            gauge_omega, interpolator);
        GaugeAdjustedBoundaryValue<Tags::BondiR>::apply(
            make_not_null(&surface_r), r, gauge_omega, interpolator);
        if (use_quartic_ansatz_) {
          evolution_gauge_dr_dr_j(
              make_not_null(&surface_dr_dr_j), boundary_dr_dr_j, boundary_dr_j,
              boundary_j, gauge_c, gauge_d, gauge_omega, interpolator, l_max);
        }
      };

  detail::iteratively_adapt_angular_coordinates(
      cartesian_cauchy_coordinates, angular_cauchy_coordinates, l_max,
      angular_coordinate_tolerance_, max_iterations_, 1.0e-2,
      iteration_function, require_convergence_, finalize_function);

  // Rebuild the volume J from an inverse ansatz in (1 - y) using the worldtube
  // data transformed into the evolution gauge, discarding the Cauchy-gauge
  // volume J that drove the iteration. Both ansatze omit the constant term so
  // that J vanishes at scri+ by construction.
  if (not use_quartic_ansatz_) {
    // Inverse-cubic ansatz J = A (1 - y) + B (1 - y)^3, matched to the
    // transformed J and dr_j (identical to ConformalFactor with inverse-cubic
    // data).
    const auto a_coefficient =
        0.25 * (3.0 * get(surface_j).data() +
                get(surface_r).data() * get(surface_dr_j).data());
    const auto b_coefficient =
        -0.0625 * (get(surface_j).data() +
                   get(surface_r).data() * get(surface_dr_j).data());
    for (size_t i = 0; i < number_of_radial_points; ++i) {
      ComplexDataVector angular_view_j{
          get(*j).data().data() + number_of_angular_points * i,  // NOLINT
          number_of_angular_points};
      angular_view_j = one_minus_y_collocation[i] * a_coefficient +
                       pow<3>(one_minus_y_collocation[i]) * b_coefficient;
    }
  } else {
    // Inverse-quartic ansatz J = A (1 - y) + B (1 - y)^3 + C (1 - y)^4 matched
    // at the worldtube (1 - y = 2) to J0 = J, J1 = dy J = (R / 2) dr_j, and
    // J2 = dy^2 J = (R / 2) dr_j + (R^2 / 4) dr^2 J, all in the evolution
    // gauge. Solving the three matching conditions gives
    //   A = J0 + J1 + J2 / 3,
    //   B = -J0 / 4 - J1 / 2 - J2 / 4,
    //   C = J0 / 16 + J1 / 8 + J2 / 12.
    // The absence of a constant and a (1 - y)^2 term forces both J and dy^2 J
    // to vanish at scri+.
    const ComplexDataVector j0 = get(surface_j).data();
    const ComplexDataVector j1 =
        0.5 * get(surface_r).data() * get(surface_dr_j).data();
    const ComplexDataVector j2 =
        j1 + 0.25 * square(get(surface_r).data()) * get(surface_dr_dr_j).data();
    const ComplexDataVector a_coefficient = j0 + j1 + j2 / 3.0;
    const ComplexDataVector b_coefficient = -0.25 * j0 - 0.5 * j1 - 0.25 * j2;
    const ComplexDataVector c_coefficient = j0 / 16.0 + j1 / 8.0 + j2 / 12.0;
    for (size_t i = 0; i < number_of_radial_points; ++i) {
      ComplexDataVector angular_view_j{
          get(*j).data().data() + number_of_angular_points * i,  // NOLINT
          number_of_angular_points};
      angular_view_j = one_minus_y_collocation[i] * a_coefficient +
                       pow<3>(one_minus_y_collocation[i]) * b_coefficient +
                       pow<4>(one_minus_y_collocation[i]) * c_coefficient;
    }
  }

  // Safeguard: both inverse ansatze omit the (1 - y)^2 term, so the second
  // radial derivative of J vanishes at scri+ by construction. A large residual
  // value (beyond spectral round-off) signals a poorly matched solution that
  // should not be evolved.
  const Mesh<3> volume_mesh{
      {{Spectral::Swsh::number_of_swsh_theta_collocation_points(l_max),
        Spectral::Swsh::number_of_swsh_phi_collocation_points(l_max),
        number_of_radial_points}},
      Spectral::Basis::Legendre,
      Spectral::Quadrature::GaussLobatto};
  SpinWeighted<ComplexDataVector, 2> dy_j{number_of_angular_points *
                                          number_of_radial_points};
  SpinWeighted<ComplexDataVector, 2> dy_dy_j{number_of_angular_points *
                                             number_of_radial_points};
  // dimension 2 is the radial (y) direction; see PreSwshDerivatives.
  logical_partial_directional_derivative_of_complex(
      make_not_null(&dy_j.data()), get(*j).data(), volume_mesh, 2);
  logical_partial_directional_derivative_of_complex(
      make_not_null(&dy_dy_j.data()), dy_j.data(), volume_mesh, 2);
  const SpinWeighted<ComplexDataVector, 2> scri_dy_dy_j;
  make_const_view(make_not_null(&scri_dy_dy_j), dy_dy_j,
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);
  const double max_scri_dy_dy_j = max(abs(scri_dy_dy_j.data()));
  if (max_scri_dy_dy_j > max_scri_second_derivative_) {
    ERROR(
        "The second-order CCE initial data has a second radial derivative of J "
        "at scri+ of magnitude "
        << max_scri_dy_dy_j << ", which exceeds the threshold "
        << max_scri_second_derivative_
        << " set by the MaxScriSecondDerivative option. The matched solution "
           "is not asymptotically well-behaved; check the worldtube boundary "
           "data or raise the threshold.");
  }
}

void CauchySecondOrderInverseAnsatz::pup(PUP::er& p) {
  p | require_convergence_;
  p | angular_coordinate_tolerance_;
  p | max_iterations_;
  p | max_scri_second_derivative_;
  p | use_quartic_ansatz_;
}

PUP::able::PUP_ID CauchySecondOrderInverseAnsatz::my_PUP_ID = 0;  // NOLINT
}  // namespace Cce::InitializeJ
