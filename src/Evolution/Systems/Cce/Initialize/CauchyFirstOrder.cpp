// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/Initialize/CauchyFirstOrder.hpp"

#include <cstddef>
#include <memory>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "Parallel/NodeLock.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace Cce::InitializeJ {

CauchyFirstOrder::CauchyFirstOrder(const double angular_coordinate_tolerance,
                                   const size_t max_iterations,
                                   const bool require_convergence)
    : require_convergence_{require_convergence},
      angular_coordinate_tolerance_{angular_coordinate_tolerance},
      max_iterations_{max_iterations} {}

std::unique_ptr<InitializeJ<false>> CauchyFirstOrder::get_clone() const {
  return std::make_unique<CauchyFirstOrder>(*this);
}

void CauchyFirstOrder::operator()(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& r, const size_t l_max,
    const size_t number_of_radial_points,
    const gsl::not_null<Parallel::NodeLock*> /*hdf5_lock*/) const {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  // Build the volume J as the linear polynomial in (1 - y) that matches J and
  // dr_j at the worldtube: J = A + B (1 - y) with A = J + R dr_j and
  // B = -(R / 2) dr_j. This is the CauchySecondOrder ansatz with the dy^2 J
  // contribution dropped, so the H hypersurface equation is never solved.
  const DataVector one_minus_y_collocation =
      1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::GaussLobatto>(
                number_of_radial_points);

  for (size_t i = 0; i < number_of_radial_points; ++i) {
    ComplexDataVector angular_view_j{
        get(*j).data().data() + get(boundary_j).size() * i,  // NOLINT
        get(boundary_j).size()};
    const auto constant_term =
        get(boundary_j).data() + get(r).data() * get(boundary_dr_j).data();
    const auto one_minus_y_coefficient =
        -0.5 * get(r).data() * get(boundary_dr_j).data();

    angular_view_j =
        constant_term + one_minus_y_collocation[i] * one_minus_y_coefficient;
  }

  // Iteratively adjust the angular coordinates so that J vanishes at scri+
  // (identical to the procedure in NoIncomingRadiation and CauchySecondOrder).
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
      angular_coordinate_tolerance_, max_iterations_, max_angular_solve_error,
      iteration_function, require_convergence_, finalize_function);
}

void CauchyFirstOrder::pup(PUP::er& p) {
  p | require_convergence_;
  p | angular_coordinate_tolerance_;
  p | max_iterations_;
}

PUP::able::PUP_ID CauchyFirstOrder::my_PUP_ID = 0;  // NOLINT
}  // namespace Cce::InitializeJ
