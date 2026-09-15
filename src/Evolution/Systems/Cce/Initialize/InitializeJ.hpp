// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/GaugeTransformBoundaryData.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshDerivatives.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshInterpolation.hpp"
#include "Options/String.hpp"
#include "Parallel/NodeLock.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class ComplexDataVector;
namespace Cce::Solutions::LinearizedBondiSachs_detail::InitializeJ {
struct LinearizedBondiSachs;
}  // namespace Cce::Solutions::LinearizedBondiSachs_detail::InitializeJ
/// \endcond

namespace Cce {
/// Contains utilities and \ref DataBoxGroup mutators for generating data for
/// \f$J\f$ on the initial CCE hypersurface.
namespace InitializeJ {

namespace detail {
// used to provide a default for the finalize functor in
// `iteratively_adapt_angular_coordinates`
struct NoOpFinalize {
  void operator()(
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& /*gauge_c*/,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& /*gauge_d*/,
      const tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>&
      /*angular_cauchy_coordinates*/,
      const Spectral::Swsh::SwshInterpolator& /*interpolator*/) const {}
};

// perform an iterative solve for the set of angular coordinates. The iteration
// callable `iteration_function` must have function signature:
//
// double iteration_function(
//     const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
//         gauge_c_step,
//     const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
//         gauge_d_step,
//     const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
//     const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
//     const Spectral::Swsh::SwshInterpolator& iteration_interpolator);
//
// but need not be a function pointer -- a callable class or lambda will also
// suffice.
// For each step specified by the iteration function, the coordinates are
// updated via \hat \eth \delta x^i = \delta c \eth x^i|_{x^i=\hat x^i}
//                        + \delta \bar d (\eth x^i)|_{x^i=\hat x^i}
// This coordinate update is exact, and comes from expanding the chain rule to
// determine Jacobian factors. However, the result is not guaranteed to
// produce the desired Jacobian c and d, because \delta c and \delta d are
// not necessarily consistent with the underlying coordinates.
// We then update the x^i by inverting \hat \eth, which is also exact, but
// assumes a no l=0 contribution to the transformation.
// Depending on the choice of approximations used to specify
// `iteration_function`, though, the method can be slow to converge.

// However, the iterations are typically fast, and the computation is for
// initial data that needs to be computed only once during a simulation, so it
// is not currently an optimization priority. If this function becomes a
// bottleneck, the numerical procedure of the iterative method or the choice of
// approximation used for `iteration_function` should be revisited.
template <typename IterationFunctor, typename FinalizeFunctor = NoOpFinalize>
double iteratively_adapt_angular_coordinates(
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const size_t l_max, const double tolerance, const size_t max_steps,
    const double error_threshold, const IterationFunctor& iteration_function,
    const bool require_convergence,
    const FinalizeFunctor finalize_function = NoOpFinalize{},
    const bool initialize_coordinates = true) {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  // `initialize_coordinates = false` keeps whatever map the caller supplied, so
  // the solve can be seeded -- with the map another solve has already reached,
  // for instance -- instead of always starting from the identity.
  if (initialize_coordinates) {
    Spectral::Swsh::create_angular_and_cartesian_coordinates(
        cartesian_cauchy_coordinates, angular_cauchy_coordinates, l_max);
  }

  Variables<tmpl::list<
      // cartesian coordinates
      ::Tags::TempSpinWeightedScalar<0, 0>,
      ::Tags::TempSpinWeightedScalar<1, 0>,
      ::Tags::TempSpinWeightedScalar<2, 0>,
      // eth of cartesian coordinates
      ::Tags::TempSpinWeightedScalar<3, 1>,
      ::Tags::TempSpinWeightedScalar<4, 1>,
      ::Tags::TempSpinWeightedScalar<5, 1>,
      // eth of gauge-transformed cartesian coordinates
      ::Tags::TempSpinWeightedScalar<6, 1>,
      ::Tags::TempSpinWeightedScalar<7, 1>,
      ::Tags::TempSpinWeightedScalar<8, 1>,
      // gauge Jacobians
      ::Tags::TempSpinWeightedScalar<9, 2>,
      ::Tags::TempSpinWeightedScalar<10, 0>,
      // gauge Jacobians on next iteration
      ::Tags::TempSpinWeightedScalar<11, 2>,
      ::Tags::TempSpinWeightedScalar<12, 0>,
      // cartesian coordinates steps
      ::Tags::TempSpinWeightedScalar<13, 0>,
      ::Tags::TempSpinWeightedScalar<14, 0>,
      ::Tags::TempSpinWeightedScalar<15, 0>>>
      computation_buffers{number_of_angular_points};

  auto& x = get(get<::Tags::TempSpinWeightedScalar<0, 0>>(computation_buffers));
  auto& y = get(get<::Tags::TempSpinWeightedScalar<1, 0>>(computation_buffers));
  auto& z = get(get<::Tags::TempSpinWeightedScalar<2, 0>>(computation_buffers));

  x.data() =
      std::complex<double>(1.0, 0.0) * get<0>(*cartesian_cauchy_coordinates);
  y.data() =
      std::complex<double>(1.0, 0.0) * get<1>(*cartesian_cauchy_coordinates);
  z.data() =
      std::complex<double>(1.0, 0.0) * get<2>(*cartesian_cauchy_coordinates);

  auto& eth_x =
      get(get<::Tags::TempSpinWeightedScalar<3, 1>>(computation_buffers));
  auto& eth_y =
      get(get<::Tags::TempSpinWeightedScalar<4, 1>>(computation_buffers));
  auto& eth_z =
      get(get<::Tags::TempSpinWeightedScalar<5, 1>>(computation_buffers));

  Spectral::Swsh::angular_derivatives<
      tmpl::list<Spectral::Swsh::Tags::Eth, Spectral::Swsh::Tags::Eth,
                 Spectral::Swsh::Tags::Eth>>(l_max, 1, make_not_null(&eth_x),
                                             make_not_null(&eth_y),
                                             make_not_null(&eth_z), x, y, z);

  auto& evolution_gauge_eth_x_step =
      get(get<::Tags::TempSpinWeightedScalar<6, 1>>(computation_buffers));
  auto& evolution_gauge_eth_y_step =
      get(get<::Tags::TempSpinWeightedScalar<7, 1>>(computation_buffers));
  auto& evolution_gauge_eth_z_step =
      get(get<::Tags::TempSpinWeightedScalar<8, 1>>(computation_buffers));

  auto& gauge_c =
      get<::Tags::TempSpinWeightedScalar<9, 2>>(computation_buffers);
  auto& gauge_d =
      get<::Tags::TempSpinWeightedScalar<10, 0>>(computation_buffers);

  auto& gauge_c_step =
      get<::Tags::TempSpinWeightedScalar<11, 2>>(computation_buffers);
  auto& gauge_d_step =
      get<::Tags::TempSpinWeightedScalar<12, 0>>(computation_buffers);

  auto& x_step =
      get(get<::Tags::TempSpinWeightedScalar<13, 0>>(computation_buffers));
  auto& y_step =
      get(get<::Tags::TempSpinWeightedScalar<14, 0>>(computation_buffers));
  auto& z_step =
      get(get<::Tags::TempSpinWeightedScalar<15, 0>>(computation_buffers));

  double max_error = 1.0;
  size_t number_of_steps = 0;
  Spectral::Swsh::SwshInterpolator iteration_interpolator;
  while (true) {
    GaugeUpdateAngularFromCartesian<
        Tags::CauchyAngularCoords,
        Tags::CauchyCartesianCoords>::apply(angular_cauchy_coordinates,
                                            cartesian_cauchy_coordinates);

    iteration_interpolator = Spectral::Swsh::SwshInterpolator{
        get<0>(*angular_cauchy_coordinates),
        get<1>(*angular_cauchy_coordinates), l_max};

    GaugeUpdateJacobianFromCoordinates<
        Tags::PartiallyFlatGaugeC, Tags::PartiallyFlatGaugeD,
        Tags::CauchyAngularCoords,
        Tags::CauchyCartesianCoords>::apply(make_not_null(&gauge_c),
                                            make_not_null(&gauge_d),
                                            *angular_cauchy_coordinates,
                                            *cartesian_cauchy_coordinates,
                                            l_max);

    max_error = iteration_function(make_not_null(&gauge_c_step),
                                   make_not_null(&gauge_d_step), gauge_c,
                                   gauge_d, iteration_interpolator);

    if (max_error > error_threshold) {
      ERROR(
          "Iterative solve for surface coordinates of initial data failed. The "
          "strain is too large to be fully eliminated by a well-behaved "
          "alteration of the spherical mesh. This could be an indication that "
          "there is an issue with the worldtube data. If you are confident "
          "the worldtube data is correct, then please use an alternative "
          "initial data generator such as `InverseCubic`. If that fails, "
          "please double check that your spherical harmonic modes are decaying "
          "correctly with increasing (l,m).\nError: "
          << max_error << "\nError threshold: " << error_threshold);
    }
    ++number_of_steps;
    if (max_error < tolerance or number_of_steps > max_steps) {
      break;
    }
    // using the evolution_gauge_.._step as temporary buffers for the
    // interpolation results
    iteration_interpolator.interpolate(
        make_not_null(&evolution_gauge_eth_x_step), eth_x);
    iteration_interpolator.interpolate(
        make_not_null(&evolution_gauge_eth_y_step), eth_y);
    iteration_interpolator.interpolate(
        make_not_null(&evolution_gauge_eth_z_step), eth_z);

    evolution_gauge_eth_x_step =
        0.5 * ((get(gauge_c_step)) * conj(evolution_gauge_eth_x_step) +
               conj((get(gauge_d_step))) * evolution_gauge_eth_x_step);
    evolution_gauge_eth_y_step =
        0.5 * ((get(gauge_c_step)) * conj(evolution_gauge_eth_y_step) +
               conj((get(gauge_d_step))) * evolution_gauge_eth_y_step);
    evolution_gauge_eth_z_step =
        0.5 * ((get(gauge_c_step)) * conj(evolution_gauge_eth_z_step) +
               conj((get(gauge_d_step))) * evolution_gauge_eth_z_step);

    Spectral::Swsh::angular_derivatives<tmpl::list<
        Spectral::Swsh::Tags::InverseEth, Spectral::Swsh::Tags::InverseEth,
        Spectral::Swsh::Tags::InverseEth>>(
        l_max, 1, make_not_null(&x_step), make_not_null(&y_step),
        make_not_null(&z_step), evolution_gauge_eth_x_step,
        evolution_gauge_eth_y_step, evolution_gauge_eth_z_step);

    get<0>(*cartesian_cauchy_coordinates) += real(x_step.data());
    get<1>(*cartesian_cauchy_coordinates) += real(y_step.data());
    get<2>(*cartesian_cauchy_coordinates) += real(z_step.data());
  }

  finalize_function(gauge_c, gauge_d, *angular_cauchy_coordinates,
                    iteration_interpolator);

  if (tolerance < max_error) {
    if (require_convergence) {
      ERROR(
          "Initial data iterative angular solve did not reach "
          "target tolerance "
          << tolerance << ".\n"
          << "Exited after " << max_steps
          << " iterations, achieving final\n"
             "maximum over collocation points deviation of J from target of "
          << max_error);
    } else {
      Parallel::printf(
          "Warning: iterative angular solve did not reach "
          "target tolerance %e.\n"
          "Exited after %zu iterations, achieving final maximum over "
          "collocation points for deviation from target of %e\n"
          "Proceeding with evolution using the partial result from partial "
          "angular solve.\n",
          tolerance, max_steps, max_error);
    }
  }
  return max_error;
}

// Solve for the partially flat angular coordinates from the closed-form
// Jacobian, using a single potential for the coordinate variation.
//
// `iteratively_adapt_angular_coordinates` above prescribes BOTH Jacobian
// variations, \delta c and \delta d -- four real functions on the sphere -- for
// a map that has only two. A general such pair is not the Jacobian of any map,
// and the inconsistent part is discarded when \eth^{-1} is inverted and the
// real part taken. Measured on binary and head-on worldtube data, that
// projection realises a fraction \lambda ~ 0.54-0.65 of whatever change is
// requested, so
// the loop contracts at 1 - \lambda per sweep at best, and no choice of step
// rule recovers it.
//
// Here the displacement is generated instead by a single complex
// spin-weight-0 potential \zeta, so that it is the variation of a map by
// construction:
//
//     \eta = \eth \zeta,      \delta \hat x^i = Re(\eta conj(\eth \hat x^i)).
//
// Writing the Jacobian definition \hat a = \hat q^{\hat A} \partial_{\hat A}
// \phi^A q_A at the identity, where \hat a = q^A q_A = 0, and displacing
// \phi^A -> x^A + v^A with \eta = q^A v_A gives
//
//     \delta \hat a = q^A q^B \nabla_A v_B = \eth (q^B v_B) = \eth \eta,
//
// so \delta \hat d is an OUTPUT rather than something prescribed, and there is
// nothing to project. Only \hat c is constrained by the gauge condition, which
// is all `target_function` has to supply. The step is therefore one application
// of \eth^{-1},
//
//     \eta = \eth^{-1}(\hat c_* - \hat c),
//
// with \hat c_* the target Jacobian. The kernel of \eth^{-1} on spin weight 2
// is l < 2, which is exactly the residual conformal freedom of the problem, so
// discarding it selects a representative rather than approximating anything.
//
// Because \hat c_* is the exact root of the gauge condition rather than a
// linearisation of it, each pass composes the current map with the first-order
// solution for the Beltrami coefficient that remains: the residual falls
// geometrically at a rate set by the asymptotic shear itself, and two or three
// passes replace the O(10^2) sweeps the linearised iteration needs.
//
// `target_function` must have the signature
//
// double target_function(
//     const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
//         gauge_c_target,
//     const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
//     const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
//     const Spectral::Swsh::SwshInterpolator& iteration_interpolator);
//
// setting `gauge_c_target` to the Jacobian the gauge condition demands and
// returning the current residual. As above, a callable class or lambda will do.
//
// The solve returns the best map it evaluated, not the one the exit test
// happened to land on. The two trailing parameters are for a caller that chains
// this with another solve: `number_of_coordinate_updates`, when not null,
// receives the passes this solve performed (the ones the rewind discarded
// included), so a shared iteration budget can be charged for them, and
// `warn_if_not_converged` silences the "did not reach its target tolerance"
// message for a stage whose outcome the caller decides on itself.
template <typename TargetFunctor, typename FinalizeFunctor = NoOpFinalize>
double adapt_angular_coordinates_via_potential(
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const size_t l_max, const double tolerance, const size_t max_steps,
    const double error_threshold, const TargetFunctor& target_function,
    const bool require_convergence,
    const FinalizeFunctor finalize_function = NoOpFinalize{},
    const bool initialize_coordinates = true, const double plateau_factor = 0.9,
    size_t* const number_of_coordinate_updates = nullptr,
    const bool warn_if_not_converged = true) {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  if (initialize_coordinates) {
    Spectral::Swsh::create_angular_and_cartesian_coordinates(
        cartesian_cauchy_coordinates, angular_cauchy_coordinates, l_max);
  }

  Variables<tmpl::list<
      // cartesian coordinates
      ::Tags::TempSpinWeightedScalar<0, 0>,
      ::Tags::TempSpinWeightedScalar<1, 0>,
      ::Tags::TempSpinWeightedScalar<2, 0>,
      // eth of the cartesian coordinates
      ::Tags::TempSpinWeightedScalar<3, 1>,
      ::Tags::TempSpinWeightedScalar<4, 1>,
      ::Tags::TempSpinWeightedScalar<5, 1>,
      // ... interpolated onto the current map
      ::Tags::TempSpinWeightedScalar<10, 1>,
      ::Tags::TempSpinWeightedScalar<11, 1>,
      ::Tags::TempSpinWeightedScalar<12, 1>,
      // gauge Jacobians
      ::Tags::TempSpinWeightedScalar<6, 2>,
      ::Tags::TempSpinWeightedScalar<7, 0>,
      // the Jacobian the gauge condition demands, then the step to it
      ::Tags::TempSpinWeightedScalar<8, 2>,
      // the potential's gradient, eta = eth zeta
      ::Tags::TempSpinWeightedScalar<9, 1>>>
      computation_buffers{number_of_angular_points};

  auto& x = get(get<::Tags::TempSpinWeightedScalar<0, 0>>(computation_buffers));
  auto& y = get(get<::Tags::TempSpinWeightedScalar<1, 0>>(computation_buffers));
  auto& z = get(get<::Tags::TempSpinWeightedScalar<2, 0>>(computation_buffers));
  auto& eth_x =
      get(get<::Tags::TempSpinWeightedScalar<3, 1>>(computation_buffers));
  auto& eth_y =
      get(get<::Tags::TempSpinWeightedScalar<4, 1>>(computation_buffers));
  auto& eth_z =
      get(get<::Tags::TempSpinWeightedScalar<5, 1>>(computation_buffers));
  auto& gauge_c =
      get<::Tags::TempSpinWeightedScalar<6, 2>>(computation_buffers);
  auto& gauge_d =
      get<::Tags::TempSpinWeightedScalar<7, 0>>(computation_buffers);
  auto& gauge_c_target =
      get<::Tags::TempSpinWeightedScalar<8, 2>>(computation_buffers);
  auto& eta =
      get(get<::Tags::TempSpinWeightedScalar<9, 1>>(computation_buffers));
  auto& interpolated_eth_x =
      get(get<::Tags::TempSpinWeightedScalar<10, 1>>(computation_buffers));
  auto& interpolated_eth_y =
      get(get<::Tags::TempSpinWeightedScalar<11, 1>>(computation_buffers));
  auto& interpolated_eth_z =
      get(get<::Tags::TempSpinWeightedScalar<12, 1>>(computation_buffers));

  // eth of the Cartesian coordinates is taken ONCE, on the grid the solve
  // started from, and interpolated onto the current map at each pass -- exactly
  // as in `iteratively_adapt_angular_coordinates`. Recomputing it from the
  // displaced coordinates instead is equally correct to first order and
  // converges to the same map, but differentiating an already-displaced
  // coordinate field aliases, and the residual then plateaus one to two orders
  // higher.
  x.data() =
      std::complex<double>(1.0, 0.0) * get<0>(*cartesian_cauchy_coordinates);
  y.data() =
      std::complex<double>(1.0, 0.0) * get<1>(*cartesian_cauchy_coordinates);
  z.data() =
      std::complex<double>(1.0, 0.0) * get<2>(*cartesian_cauchy_coordinates);
  Spectral::Swsh::angular_derivatives<
      tmpl::list<Spectral::Swsh::Tags::Eth, Spectral::Swsh::Tags::Eth,
                 Spectral::Swsh::Tags::Eth>>(l_max, 1, make_not_null(&eth_x),
                                             make_not_null(&eth_y),
                                             make_not_null(&eth_z), x, y, z);

  double max_error = 1.0;
  double previous_max_error = std::numeric_limits<double>::max();
  size_t number_of_steps = 0;
  Spectral::Swsh::SwshInterpolator iteration_interpolator;

  // Score the map currently held in `cartesian_cauchy_coordinates`: bring the
  // angular coordinates and the Jacobians into step with it, then ask
  // `target_function` for the residual and the Jacobian the gauge condition
  // wants. Factored out because the rewind below has to repeat it.
  const auto evaluate_current_map = [&]() {
    GaugeUpdateAngularFromCartesian<
        Tags::CauchyAngularCoords,
        Tags::CauchyCartesianCoords>::apply(angular_cauchy_coordinates,
                                            cartesian_cauchy_coordinates);

    iteration_interpolator = Spectral::Swsh::SwshInterpolator{
        get<0>(*angular_cauchy_coordinates),
        get<1>(*angular_cauchy_coordinates), l_max};

    GaugeUpdateJacobianFromCoordinates<
        Tags::PartiallyFlatGaugeC, Tags::PartiallyFlatGaugeD,
        Tags::CauchyAngularCoords,
        Tags::CauchyCartesianCoords>::apply(make_not_null(&gauge_c),
                                            make_not_null(&gauge_d),
                                            *angular_cauchy_coordinates,
                                            *cartesian_cauchy_coordinates,
                                            l_max);

    return target_function(make_not_null(&gauge_c_target), gauge_c, gauge_d,
                           iteration_interpolator);
  };

  // The best map seen so far. The stopping rule below can only recognize the
  // minimum one pass after the fact, so the solve keeps a copy to rewind to.
  auto best_cartesian_cauchy_coordinates = *cartesian_cauchy_coordinates;
  double best_max_error = std::numeric_limits<double>::max();

  while (true) {
    max_error = evaluate_current_map();

    if (max_error < best_max_error) {
      best_max_error = max_error;
      best_cartesian_cauchy_coordinates = *cartesian_cauchy_coordinates;
    }

    if (max_error > error_threshold) {
      ERROR(
          "Iterative solve for surface coordinates of initial data failed. The "
          "strain is too large to be fully eliminated by a well-behaved "
          "alteration of the spherical mesh. This could be an indication that "
          "there is an issue with the worldtube data. If you are confident "
          "the worldtube data is correct, then please use an alternative "
          "initial data generator such as `InverseCubic`. If that fails, "
          "please double check that your spherical harmonic modes are decaying "
          "correctly with increasing (l,m).\nError: "
          << max_error << "\nError threshold: " << error_threshold);
    }
    ++number_of_steps;
    if (max_error < tolerance or number_of_steps > max_steps) {
      break;
    }
    // Stop on the last pass that still improves the residual. The early passes
    // divide it by a factor of order \|\mu\|_\infty -- three orders of
    // magnitude for a typical worldtube -- after which it bottoms out and
    // slowly creeps back up, each further pass adding a displacement the size
    // of the noise it is responding to. A `plateau_factor` just below one stops
    // within a pass of that minimum without cutting off a slow but genuine
    // descent; the caller can then fall back to a method that grinds lower.
    if (plateau_factor > 0.0 and number_of_steps > 2 and
        max_error > plateau_factor * previous_max_error) {
      break;
    }
    previous_max_error = max_error;

    // the step to the target Jacobian, then the potential that generates it
    get(gauge_c_target).data() -= get(gauge_c).data();
    Spectral::Swsh::angular_derivatives<
        tmpl::list<Spectral::Swsh::Tags::InverseEth>>(
        l_max, 1, make_not_null(&eta), get(gauge_c_target));

    iteration_interpolator.interpolate(make_not_null(&interpolated_eth_x),
                                       eth_x);
    iteration_interpolator.interpolate(make_not_null(&interpolated_eth_y),
                                       eth_y);
    iteration_interpolator.interpolate(make_not_null(&interpolated_eth_z),
                                       eth_z);

    // dx^i = Re(eta conj(eth x^i)); this is v^A \partial_A x^i written with the
    // dyad completeness relation, and is automatically tangent to the sphere
    // because x^i eth x^i = (1/2) eth(x^i x^i) = 0. The residual term
    // proportional to x^i in eth(dx^i) is radial and is removed by the
    // `GaugeUpdateAngularFromCartesian` call at the top of the next pass.
    get<0>(*cartesian_cauchy_coordinates) +=
        real(eta.data() * conj(interpolated_eth_x.data()));
    get<1>(*cartesian_cauchy_coordinates) +=
        real(eta.data() * conj(interpolated_eth_y.data()));
    get<2>(*cartesian_cauchy_coordinates) +=
        real(eta.data() * conj(interpolated_eth_z.data()));
  }

  // Hand back the minimum rather than whichever pass the exit test landed on.
  // The plateau rule fires on the first pass that fails to improve, so without
  // this the map that leaves here -- and seeds whatever solve runs next -- is
  // the one pass *past* the minimum.
  if (best_max_error < max_error) {
    *cartesian_cauchy_coordinates = best_cartesian_cauchy_coordinates;
    max_error = evaluate_current_map();
  }
  // Reported as the work done, including the passes the rewind discarded, so a
  // caller sharing an iteration budget with a following solve charges for them.
  if (number_of_coordinate_updates != nullptr) {
    *number_of_coordinate_updates =
        number_of_steps > 0 ? number_of_steps - 1 : 0;
  }

  finalize_function(gauge_c, gauge_d, *angular_cauchy_coordinates,
                    iteration_interpolator);

  if (tolerance < max_error) {
    if (require_convergence) {
      ERROR(
          "Initial data potential angular solve did not reach "
          "target tolerance "
          << tolerance << ".\n"
          << "Exited after " << max_steps
          << " iterations, achieving final\n"
             "maximum over collocation points deviation of J from target of "
          << max_error);
    } else if (warn_if_not_converged) {
      Parallel::printf(
          "Warning: potential angular solve did not reach "
          "target tolerance %e.\n"
          "Exited after %zu iterations, achieving final maximum over "
          "collocation points for deviation from target of %e\n"
          "Proceeding with evolution using the partial result from partial "
          "angular solve.\n",
          tolerance, max_steps, max_error);
    }
  }
  return max_error;
}

double adjust_angular_coordinates_for_j(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> volume_j,
    gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const SpinWeighted<ComplexDataVector, 2>& surface_j, size_t l_max,
    double tolerance, size_t max_steps, bool adjust_volume_gauge);
}  // namespace detail

/*!
 * \brief Apply a radius-independent angular gauge transformation to a volume
 * \f$J\f$, for use with initial data generation.
 *
 * \details Performs the gauge transformation to \f$\hat J\f$,
 *
 * \f{align*}{
 * \hat J = \frac{1}{4 \hat{\omega}^2} \left( \bar{\hat d}^2  J(\hat x^{\hat A})
 *  + \hat c^2 \bar J(\hat x^{\hat A})
 *  + 2 \hat c \bar{\hat d} K(\hat x^{\hat A}) \right).
 * \f}
 *
 * Where \f$\hat c\f$ and \f$\hat d\f$ are the spin-weighted angular Jacobian
 * factors computed by `GaugeUpdateJacobianFromCoords`, and \f$\hat \omega\f$ is
 * the conformal factor associated with the angular coordinate transformation.
 * Note that the right-hand sides with explicit \f$\hat x^{\hat A}\f$ dependence
 * must be interpolated and that \f$K = \sqrt{1 + J \bar J}\f$.
 */
struct GaugeAdjustInitialJ {
  using boundary_tags =
      tmpl::list<Tags::PartiallyFlatGaugeC, Tags::PartiallyFlatGaugeD,
                 Tags::PartiallyFlatGaugeOmega, Tags::CauchyAngularCoords,
                 Spectral::Swsh::Tags::LMax>;
  using return_tags = tmpl::list<Tags::BondiJ>;
  using argument_tags = tmpl::append<boundary_tags>;

  static void apply(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> volume_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_omega,
      const tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>&
          cauchy_angular_coordinates,
      const Spectral::Swsh::SwshInterpolator& interpolator, size_t l_max);
};

/// \cond
struct NoIncomingRadiation;
struct ZeroNonSmooth;
template <bool evolve_ccm>
struct InverseCubic;
template <bool evolve_ccm>
struct InitializeJ;
struct ConformalFactor;
struct CauchySecondOrder;
/// \endcond

/*!
 * \brief Abstract base class for an initial hypersurface data generator for
 * Cce, when the partially flat Bondi-like coordinates are evolved.
 *
 * \details The algorithm is same as `InitializeJ<false>`, but with an
 * additional initialization for the partially flat Bondi-like coordinates. The
 * functions that are required to be overriden in the derived classes are:
 * - `InitializeJ::get_clone()`: should return a
 * `std::unique_ptr<InitializeJ<true>>` with cloned state.
 * - `InitializeJ::operator() const`: should take as arguments, first a
 * set of `gsl::not_null` pointers represented by `mutate_tags`, followed by a
 * set of `const` references to quantities represented by `argument_tags`. \note
 * The `InitializeJ::operator()` should be const, and therefore not alter
 * the internal state of the generator. This is compatible with all known
 * use-cases and permits the `InitializeJ` generator to be placed in the
 * `GlobalCache`.
 */
template <>
struct InitializeJ<true> : public PUP::able {
  using boundary_tags = tmpl::list<Tags::BoundaryValue<Tags::BondiJ>,
                                   Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>,
                                   Tags::BoundaryValue<Tags::BondiR>,
                                   Tags::BoundaryValue<Tags::BondiBeta>>;

  using mutate_tags =
      tmpl::list<Tags::BondiJ, Tags::CauchyCartesianCoords,
                 Tags::CauchyAngularCoords, Tags::PartiallyFlatCartesianCoords,
                 Tags::PartiallyFlatAngularCoords>;
  using return_tags = mutate_tags;
  using argument_tags =
      tmpl::push_back<boundary_tags, Tags::LMax, Tags::NumberOfRadialPoints>;

  // The evolution of inertial coordinates are allowed only when InverseCubic is
  // used
  using creatable_classes = tmpl::list<InverseCubic<true>>;

  InitializeJ() = default;
  explicit InitializeJ(CkMigrateMessage* /*msg*/) {}

  WRAPPED_PUPable_abstract(InitializeJ);  // NOLINT

  virtual std::unique_ptr<InitializeJ<true>> get_clone() const = 0;

  /// \brief The interpolator this generator wants used to build the worldtube
  /// boundary value of \f$\partial_u \partial_r J\f$, or `nullptr` to use the
  /// evolution's `H5Interpolator`.
  ///
  /// \details That boundary value is consumed by the initial data alone, so a
  /// generator is free to choose its own time-interpolation order for it
  /// without affecting the evolution.
  virtual std::unique_ptr<intrp::SpanInterpolator> du_dr_j_interpolator()
      const {
    return nullptr;
  }

  // Each derived class declares its own `return_tags` and `argument_tags` and
  // implements a non-virtual `operator()`; the dispatch below picks the
  // correct dynamic type and forwards through `db::mutate_apply`.
  template <typename DbTags>
  void operator()(const gsl::not_null<db::DataBox<DbTags>*> box,
                  const gsl::not_null<Parallel::NodeLock*> hdf5_lock) const {
    call_with_dynamic_type<void, creatable_classes>(
        this, [&](auto* const derived) {
          db::mutate_apply(*derived, box, hdf5_lock);
        });
  }
};

/*!
 * \brief Abstract base class for an initial hypersurface data generator for
 * Cce, when the partially flat Bondi-like coordinates are not evolved.
 *
 * \details The functions that are required to be overriden in the derived
 * classes are:
 * - `InitializeJ::get_clone()`: should return a
 * `std::unique_ptr<InitializeJ<false>>` with cloned state.
 * - `InitializeJ::operator() const`: should take as arguments, first a
 * set of `gsl::not_null` pointers represented by `mutate_tags`, followed by a
 * set of `const` references to quantities represented by `argument_tags`. \note
 * The `InitializeJ::operator()` should be const, and therefore not alter
 * the internal state of the generator. This is compatible with all known
 * use-cases and permits the `InitializeJ` generator to be placed in the
 * `GlobalCache`.
 */
template <>
struct InitializeJ<false> : public PUP::able {
  // Default boundary/mutate/argument tags used by the simple derived classes
  // (ConformalFactor, InverseCubic<false>, NoIncomingRadiation, ZeroNonSmooth).
  // A derived class can shadow these with its own list (see CauchySecondOrder),
  // in which case the per-class list is used during dispatch.
  using boundary_tags = tmpl::list<Tags::BoundaryValue<Tags::BondiJ>,
                                   Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>,
                                   Tags::BoundaryValue<Tags::BondiR>,
                                   Tags::BoundaryValue<Tags::BondiBeta>>;

  using mutate_tags = tmpl::list<Tags::BondiJ, Tags::CauchyCartesianCoords,
                                 Tags::CauchyAngularCoords>;
  using return_tags = mutate_tags;
  using argument_tags =
      tmpl::push_back<boundary_tags, Tags::LMax, Tags::NumberOfRadialPoints>;

  using creatable_classes =
      tmpl::list<ConformalFactor, InverseCubic<false>, NoIncomingRadiation,
                 ZeroNonSmooth, CauchySecondOrder,
                 ::Cce::Solutions::LinearizedBondiSachs_detail::InitializeJ::
                     LinearizedBondiSachs>;

  InitializeJ() = default;
  explicit InitializeJ(CkMigrateMessage* /*msg*/) {}

  WRAPPED_PUPable_abstract(InitializeJ);  // NOLINT

  virtual std::unique_ptr<InitializeJ<false>> get_clone() const = 0;

  /// \brief The interpolator this generator wants used to build the worldtube
  /// boundary value of \f$\partial_u \partial_r J\f$, or `nullptr` to use the
  /// evolution's `H5Interpolator`.
  ///
  /// \details That boundary value is consumed by the initial data alone, so a
  /// generator is free to choose its own time-interpolation order for it
  /// without affecting the evolution.
  virtual std::unique_ptr<intrp::SpanInterpolator> du_dr_j_interpolator()
      const {
    return nullptr;
  }

  // Each derived class declares its own `return_tags` and `argument_tags` and
  // implements a non-virtual `operator()` whose signature matches those tags.
  // The dispatch below picks the correct dynamic type and forwards through
  // `db::mutate_apply` so the per-class tag lists are honored.
  template <typename DbTags>
  void operator()(const gsl::not_null<db::DataBox<DbTags>*> box,
                  const gsl::not_null<Parallel::NodeLock*> hdf5_lock) const {
    call_with_dynamic_type<void, creatable_classes>(
        this, [&](auto* const derived) {
          db::mutate_apply(*derived, box, hdf5_lock);
        });
  }
};
}  // namespace InitializeJ
}  // namespace Cce

#include "Evolution/Systems/Cce/AnalyticSolutions/LinearizedBondiSachsInitializeJ.hpp"
#include "Evolution/Systems/Cce/Initialize/CauchySecondOrder.hpp"
#include "Evolution/Systems/Cce/Initialize/ConformalFactor.hpp"
#include "Evolution/Systems/Cce/Initialize/InverseCubic.hpp"
#include "Evolution/Systems/Cce/Initialize/NoIncomingRadiation.hpp"
#include "Evolution/Systems/Cce/Initialize/ZeroNonSmooth.hpp"
