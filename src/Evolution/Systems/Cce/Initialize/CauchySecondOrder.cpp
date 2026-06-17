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
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Parallel/NodeLock.hpp"
#include "Utilities/Gsl.hpp"

namespace Cce::InitializeJ {

CauchySecondOrder::CauchySecondOrder(const double angular_coordinate_tolerance,
                                     const size_t max_iterations,
                                     const bool require_convergence)
    : require_convergence_{require_convergence},
      angular_coordinate_tolerance_{angular_coordinate_tolerance},
      max_iterations_{max_iterations} {}

std::unique_ptr<InitializeJ<false>> CauchySecondOrder::get_clone() const {
  return std::make_unique<CauchySecondOrder>(*this);
}

void CauchySecondOrder::operator()(
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
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_h,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_du_dy_j,
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
      boundary_beta, boundary_q, boundary_h, boundary_dr_j, boundary_du_dy_j,
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

  detail::iteratively_adapt_angular_coordinates(
      cartesian_cauchy_coordinates, angular_cauchy_coordinates, l_max,
      angular_coordinate_tolerance_, max_iterations_, 1.0e-2,
      iteration_function, require_convergence_, finalize_function);
}

void CauchySecondOrder::pup(PUP::er& p) {
  p | require_convergence_;
  p | angular_coordinate_tolerance_;
  p | max_iterations_;
}

PUP::able::PUP_ID CauchySecondOrder::my_PUP_ID = 0;
}  // namespace Cce::InitializeJ
