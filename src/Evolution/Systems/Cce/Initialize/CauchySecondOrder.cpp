// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/Initialize/CauchySecondOrder.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/Initialize/ComputeSecondOrderRadialDerivativeJ.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Evolution/Systems/Cce/LinearOperators.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "Parallel/NodeLock.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace Cce::InitializeJ {

namespace {
// Shared second-order Cauchy solve and volume-J construction used by both
// `CauchySecondOrder<false>` and `CauchySecondOrder<true>`. The converged
// forward gauge Jacobians are written to `converged_gauge_c` and
// `converged_gauge_d` so the `evolve_ccm = true` specialization can build the
// inverse-Jacobian target for the inertial (partially flat) coordinate solve.
void cauchy_second_order_apply_impl(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
        converged_gauge_c,
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
        converged_gauge_d,
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
    const double angular_coordinate_tolerance, const size_t max_iterations,
    const bool require_convergence, const double max_scri_second_derivative) {
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
    // Store expressions so that the later equation for `j` is easier to read.
    const auto constant_term = get(boundary_j).data() +
                               get(r).data() * get(boundary_dr_j).data() +
                               (4.0 / 3.0) * get(boundary_dy2_j).data();
    const auto one_minus_y_coefficient =
        -(0.5 * get(r).data() * get(boundary_dr_j).data() +
          get(boundary_dy2_j).data());
    const auto one_minus_y_cubed_coefficient =
        (1.0 / 12.0) * get(boundary_dy2_j).data();

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

  auto finalize_function =
      [&j, &gauge_omega, &l_max, &converged_gauge_c, &converged_gauge_d](
          const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
          const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
          const tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>&
              local_angular_cauchy_coordinates,
          const Spectral::Swsh::SwshInterpolator& interpolator) {
        get(gauge_omega).data() =
            0.5 * sqrt(get(gauge_d).data() * conj(get(gauge_d).data()) -
                       get(gauge_c).data() * conj(get(gauge_c).data()));
        GaugeAdjustInitialJ::apply(j, gauge_c, gauge_d, gauge_omega,
                                   local_angular_cauchy_coordinates,
                                   interpolator, l_max);
        // Store the converged forward gauge Jacobians so an `evolve_ccm = true`
        // caller can build the inverse-Jacobian target.
        get(*converged_gauge_c) = get(gauge_c);
        get(*converged_gauge_d) = get(gauge_d);
      };

  // The angular solve can only eliminate J at scri+ through a well-behaved
  // alteration of the spherical mesh, so it tolerates only a small asymptotic
  // J. Guard against a large asymptotic initial J in Cauchy coordinates before
  // attempting the solve, which would otherwise fail less informatively.
  const double max_angular_solve_error = 1.0e-2;
  const double max_asymptotic_j = max(abs(j_at_scri_view.data()));
  if (max_asymptotic_j > max_angular_solve_error) {
    ERROR(
        "The asymptotic value of the initial J in Cauchy coordinates has "
        "magnitude "
        << max_asymptotic_j
        << ", which is too large for the angular-coordinate solve to "
           "eliminate. The worldtube data may be incorrect, or the worldtube "
           "may be too close to the strong-field region. Consider using the "
           "ConformalFactor initial-data generator instead.");
  }

  detail::iteratively_adapt_angular_coordinates(
      cartesian_cauchy_coordinates, angular_cauchy_coordinates, l_max,
      angular_coordinate_tolerance, max_iterations, max_angular_solve_error,
      iteration_function, require_convergence, finalize_function);

  // Safeguard: the second-order construction forces the second radial
  // derivative of J to vanish at scri+, and the angular gauge transform only
  // adds a strain-suppressed contribution, so the final initial data must
  // retain a tiny second derivative there. A large value signals a poorly
  // matched solution that should not be evolved.
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
  if (max_scri_dy_dy_j > max_scri_second_derivative) {
    ERROR("The initial J has a second radial derivative at scri+ of magnitude "
          << max_scri_dy_dy_j << ", which exceeds the threshold "
          << max_scri_second_derivative
          << " set by the MaxScriSecondDerivative option. The matched solution "
             "is not asymptotically well-behaved; check the worldtube boundary "
             "data or raise the threshold.");
  }
}
}  // namespace

CauchySecondOrder<false>::CauchySecondOrder(
    const double angular_coordinate_tolerance, const size_t max_iterations,
    const bool require_convergence, const double max_scri_second_derivative)
    : require_convergence_{require_convergence},
      angular_coordinate_tolerance_{angular_coordinate_tolerance},
      max_iterations_{max_iterations},
      max_scri_second_derivative_{max_scri_second_derivative} {}

std::unique_ptr<InitializeJ<false>> CauchySecondOrder<false>::get_clone()
    const {
  return std::make_unique<CauchySecondOrder>(*this);
}

void CauchySecondOrder<false>::operator()(
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
  Scalar<SpinWeighted<ComplexDataVector, 2>> converged_gauge_c{};
  Scalar<SpinWeighted<ComplexDataVector, 0>> converged_gauge_d{};
  cauchy_second_order_apply_impl(
      j, cartesian_cauchy_coordinates, angular_cauchy_coordinates,
      make_not_null(&converged_gauge_c), make_not_null(&converged_gauge_d),
      boundary_j, boundary_u, boundary_w, boundary_beta, boundary_q,
      boundary_du_j, boundary_dr_j, boundary_du_dr_j, boundary_du_r, r, l_max,
      number_of_radial_points, angular_coordinate_tolerance_, max_iterations_,
      require_convergence_, max_scri_second_derivative_);
}

void CauchySecondOrder<false>::pup(PUP::er& p) {
  p | require_convergence_;
  p | angular_coordinate_tolerance_;
  p | max_iterations_;
  p | max_scri_second_derivative_;
}

PUP::able::PUP_ID CauchySecondOrder<false>::my_PUP_ID = 0;  // NOLINT

CauchySecondOrder<true>::CauchySecondOrder(
    const double angular_coordinate_tolerance, const size_t max_iterations,
    const bool require_convergence, const double max_scri_second_derivative)
    : require_convergence_{require_convergence},
      angular_coordinate_tolerance_{angular_coordinate_tolerance},
      max_iterations_{max_iterations},
      max_scri_second_derivative_{max_scri_second_derivative} {}

std::unique_ptr<InitializeJ<true>> CauchySecondOrder<true>::get_clone() const {
  return std::make_unique<CauchySecondOrder>(*this);
}

void CauchySecondOrder<true>::operator()(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_inertial_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_inertial_coordinates,
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
  Scalar<SpinWeighted<ComplexDataVector, 2>> converged_gauge_c{
      number_of_angular_points};
  Scalar<SpinWeighted<ComplexDataVector, 0>> converged_gauge_d{
      number_of_angular_points};
  cauchy_second_order_apply_impl(
      j, cartesian_cauchy_coordinates, angular_cauchy_coordinates,
      make_not_null(&converged_gauge_c), make_not_null(&converged_gauge_d),
      boundary_j, boundary_u, boundary_w, boundary_beta, boundary_q,
      boundary_du_j, boundary_dr_j, boundary_du_dr_j, boundary_du_r, r, l_max,
      number_of_radial_points, angular_coordinate_tolerance_, max_iterations_,
      require_convergence_, max_scri_second_derivative_);

  // Solve for the inertial (partially flat) coordinates that invert the Cauchy
  // angular transformation, using the same shared inverse solve as the other
  // `evolve_ccm = true` generators.
  SpinWeighted<ComplexDataVector, 2> target_c_inv{number_of_angular_points};
  SpinWeighted<ComplexDataVector, 0> target_d_inv{number_of_angular_points};
  detail::compute_inverse_jacobian_target(
      make_not_null(&target_c_inv), make_not_null(&target_d_inv),
      get(converged_gauge_c), get(converged_gauge_d), l_max);
  detail::invert_angular_coordinates(
      cartesian_inertial_coordinates, angular_inertial_coordinates,
      target_c_inv, target_d_inv, l_max, angular_coordinate_tolerance_,
      max_iterations_, 1.0e-2, require_convergence_);
}

void CauchySecondOrder<true>::pup(PUP::er& p) {
  p | require_convergence_;
  p | angular_coordinate_tolerance_;
  p | max_iterations_;
  p | max_scri_second_derivative_;
}

PUP::able::PUP_ID CauchySecondOrder<true>::my_PUP_ID = 0;  // NOLINT
}  // namespace Cce::InitializeJ
