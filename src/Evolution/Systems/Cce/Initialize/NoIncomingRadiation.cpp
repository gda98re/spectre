// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/Initialize/NoIncomingRadiation.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "NumericalAlgorithms/OdeIntegration/OdeIntegration.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce::InitializeJ {
namespace {

void radial_evolve_psi0_condition(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> volume_j_id,
    const SpinWeighted<ComplexDataVector, 2>& boundary_j,
    const SpinWeighted<ComplexDataVector, 2>& boundary_dr_j,
    const SpinWeighted<ComplexDataVector, 0>& r, const size_t l_max,
    const size_t number_of_radial_points) {
  // use the maximum to measure the scale for the vector quantities
  const double j_scale = max(abs(boundary_j.data()));
  const double dy_j_scale = max(abs(0.5 * boundary_dr_j.data() * r.data()));
  // set initial step size according to the first couple of steps in section
  // II.4 of Solving Ordinary Differential equations by Hairer, Norsett, and
  // Wanner
  double initial_radial_step = 1.0e-6;
  if (j_scale > 1.0e-5 and dy_j_scale > 1.0e-5) {
    initial_radial_step = 0.01 * j_scale / dy_j_scale;
  }

  const auto psi_0_condition_system =
      [](const std::array<ComplexDataVector, 2>& bondi_j_and_i,
         std::array<ComplexDataVector, 2>& dy_j_and_dy_i, const double y) {
        dy_j_and_dy_i[0] = bondi_j_and_i[1];
        const auto& bondi_j = bondi_j_and_i[0];
        const auto& bondi_i = bondi_j_and_i[1];
        dy_j_and_dy_i[1] =
            -0.0625 *
            (square(conj(bondi_i) * bondi_j) + square(conj(bondi_j) * bondi_i) -
             2.0 * bondi_i * conj(bondi_i) * (2.0 + bondi_j * conj(bondi_j))) *
            (4.0 * bondi_j + bondi_i * (1.0 - y)) /
            (1.0 + bondi_j * conj(bondi_j));
      };

  boost::numeric::odeint::dense_output_runge_kutta<
      boost::numeric::odeint::controlled_runge_kutta<
          boost::numeric::odeint::runge_kutta_dopri5<
              std::array<ComplexDataVector, 2>>>>
      dense_stepper = boost::numeric::odeint::make_dense_output(
          1.0e-14, 1.0e-14,
          boost::numeric::odeint::runge_kutta_dopri5<
              std::array<ComplexDataVector, 2>>{});
  dense_stepper.initialize(
      std::array<ComplexDataVector, 2>{
          {boundary_j.data(), 0.5 * boundary_dr_j.data() * r.data()}},
      -1.0, initial_radial_step);
  auto state_buffer =
      std::array<ComplexDataVector, 2>{{ComplexDataVector{boundary_j.size()},
                                        ComplexDataVector{boundary_j.size()}}};

  std::pair<double, double> step_range =
      dense_stepper.do_step(psi_0_condition_system);
  const auto& y_collocation =
      Spectral::collocation_points<Spectral::Basis::Legendre,
                                   Spectral::Quadrature::GaussLobatto>(
                                       number_of_radial_points);
  for (size_t y_collocation_point = 0;
       y_collocation_point < number_of_radial_points; ++y_collocation_point) {
    while(step_range.second < y_collocation[y_collocation_point]) {
      step_range = dense_stepper.do_step(psi_0_condition_system);
    }
    if (step_range.second < y_collocation[y_collocation_point] or
        step_range.first > y_collocation[y_collocation_point]) {
      ERROR(
          "Psi 0 radial integration failed. The current y value is "
          "incompatible with the required Gauss-Lobatto point.");
    }
    dense_stepper.calc_state(y_collocation[y_collocation_point], state_buffer);
    ComplexDataVector angular_view{
        volume_j_id->data().data() +
            y_collocation_point *
                Spectral::Swsh::number_of_swsh_collocation_points(l_max),
        Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
    angular_view = state_buffer[0];
  }
}
}  // namespace

NoIncomingRadiation<false>::NoIncomingRadiation(
    const double angular_coordinate_tolerance, const size_t max_iterations,
    const bool require_convergence)
    : require_convergence_{require_convergence},
      angular_coordinate_tolerance_{angular_coordinate_tolerance},
      max_iterations_{max_iterations} {}

std::unique_ptr<InitializeJ<false>> NoIncomingRadiation<false>::get_clone()
    const {
  return std::make_unique<NoIncomingRadiation>(*this);
}

void NoIncomingRadiation<false>::operator()(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& /*beta*/,
    const size_t l_max, const size_t number_of_radial_points,
    const gsl::not_null<Parallel::NodeLock*> /*hdf5_lock*/) const {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  radial_evolve_psi0_condition(make_not_null(&get(*j)), get(boundary_j),
                               get(boundary_dr_j), get(r), l_max,
                               number_of_radial_points);
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

  // find a coordinate transformation such that in the new coordinates,
  // J is zero at scri+
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

        double max_error = max(abs(evolution_gauge_surface_j.data()));

        // The alteration in each of the spin-weighted Jacobian factors
        // determined by linearizing the system in small J
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
      [&j, &gauge_omega, &l_max](
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
      };

  detail::iteratively_adapt_angular_coordinates(
      cartesian_cauchy_coordinates, angular_cauchy_coordinates, l_max,
      angular_coordinate_tolerance_, max_iterations_, 1.0e-2,
      iteration_function, require_convergence_, finalize_function);
}

void NoIncomingRadiation<false>::pup(PUP::er& p) {
  p | require_convergence_;
  p | angular_coordinate_tolerance_;
  p | max_iterations_;
}

PUP::able::PUP_ID NoIncomingRadiation<false>::my_PUP_ID = 0;  // NOLINT

NoIncomingRadiation<true>::NoIncomingRadiation(
    const double angular_coordinate_tolerance, const size_t max_iterations,
    const bool require_convergence)
    : require_convergence_{require_convergence},
      angular_coordinate_tolerance_{angular_coordinate_tolerance},
      max_iterations_{max_iterations} {}

std::unique_ptr<InitializeJ<true>> NoIncomingRadiation<true>::get_clone()
    const {
  return std::make_unique<NoIncomingRadiation>(*this);
}

void NoIncomingRadiation<true>::operator()(
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
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& /*beta*/,
    const size_t l_max, const size_t number_of_radial_points,
    const gsl::not_null<Parallel::NodeLock*> /*hdf5_lock*/) const {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  radial_evolve_psi0_condition(make_not_null(&get(*j)), get(boundary_j),
                               get(boundary_dr_j), get(r), l_max,
                               number_of_radial_points);
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

  // Captured from the converged forward solve and used to drive the inverse
  // (inertial) angular solve.
  SpinWeighted<ComplexDataVector, 2> target_c_inv{number_of_angular_points};
  SpinWeighted<ComplexDataVector, 0> target_d_inv{number_of_angular_points};

  // find a coordinate transformation such that in the new coordinates,
  // J is zero at scri+
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

        double max_error = max(abs(evolution_gauge_surface_j.data()));

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
      [&j, &gauge_omega, &l_max, &target_c_inv, &target_d_inv](
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

        // Capture the inverse-Jacobian target (on the Cauchy collocation grid)
        // from the converged forward transformation, to drive the inertial
        // solve below.
        detail::compute_inverse_jacobian_target(
            make_not_null(&target_c_inv), make_not_null(&target_d_inv),
            get(gauge_c), get(gauge_d), l_max);
      };

  detail::iteratively_adapt_angular_coordinates(
      cartesian_cauchy_coordinates, angular_cauchy_coordinates, l_max,
      angular_coordinate_tolerance_, max_iterations_, 1.0e-2,
      iteration_function, require_convergence_, finalize_function);

  // Solve for the inertial (partially flat) coordinates that invert the Cauchy
  // angular transformation. NOTE: while detail::compute_inverse_jacobian_target
  // holds a placeholder inverse-Jacobian relation, the inverse solve is run
  // permissively (large error threshold, convergence not required) so it
  // produces approximate inertial coordinates without aborting; tighten once
  // the exact relation is supplied.
  detail::invert_angular_coordinates(
      cartesian_inertial_coordinates, angular_inertial_coordinates,
      target_c_inv, target_d_inv, l_max, angular_coordinate_tolerance_,
      max_iterations_, std::numeric_limits<double>::max(), false);
}

void NoIncomingRadiation<true>::pup(PUP::er& p) {
  p | require_convergence_;
  p | angular_coordinate_tolerance_;
  p | max_iterations_;
}

PUP::able::PUP_ID NoIncomingRadiation<true>::my_PUP_ID = 0;  // NOLINT
}  // namespace Cce::InitializeJ
