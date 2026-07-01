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
    const FinalizeFunctor finalize_function = NoOpFinalize{}) {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  Spectral::Swsh::create_angular_and_cartesian_coordinates(
      cartesian_cauchy_coordinates, angular_cauchy_coordinates, l_max);

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

double adjust_angular_coordinates_for_j(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> volume_j,
    gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const SpinWeighted<ComplexDataVector, 2>& surface_j, size_t l_max,
    double tolerance, size_t max_steps, bool adjust_volume_gauge);

/*!
 * \brief Given the converged forward (Cauchy) gauge Jacobians, produce the
 * target inverse Jacobians `(target_c_inv, target_d_inv)` that the inertial
 * (PartiallyFlat) angular solve must reproduce.
 *
 * \details This is the single place the inverse-transformation physics lives.
 * The body encodes the exact inverse-Jacobian relation (Moxon2020 Eq. 4.18),
 * \f$c_{\mathrm{inv}} = -c / \omega^2\f$ and
 * \f$d_{\mathrm{inv}} = \bar d / \omega^2\f$ with
 * \f$\omega^2 = (1/4)(d \bar d - c \bar c)\f$. See the definition in
 * `InitializeJ.cpp`.
 */
void compute_inverse_jacobian_target(
    gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> target_c_inv,
    gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> target_d_inv,
    const SpinWeighted<ComplexDataVector, 2>& forward_gauge_c,
    const SpinWeighted<ComplexDataVector, 0>& forward_gauge_d, size_t l_max);

/*!
 * \brief Iteration heuristic that drives the current inverse-solve gauge
 * Jacobians `(gauge_c, gauge_d)` toward the interpolated targets, used by
 * `invert_angular_coordinates`.
 */
void jacobian_match_heuristic(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> gauge_c_step,
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> gauge_d_step,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
    const SpinWeighted<ComplexDataVector, 2>& target_c,
    const SpinWeighted<ComplexDataVector, 0>& target_d, size_t l_max);

/*!
 * \brief Solve for the inertial ("PartiallyFlat") angular coordinates that
 * invert the Cauchy angular-coordinate transformation.
 *
 * \details Reuses `iteratively_adapt_angular_coordinates` (operating on the
 * supplied inertial coordinate buffers), driving the inverse-solve gauge
 * Jacobians toward `(target_c_inv, target_d_inv)` (the inverse Jacobians on the
 * Cauchy collocation grid, typically produced by
 * `compute_inverse_jacobian_target` in the forward solve's finalize hook). The
 * targets are interpolated through the current inverse-solve interpolator each
 * iteration. Returns the achieved error.
 */
inline double invert_angular_coordinates(
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_inertial_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_inertial_coordinates,
    const SpinWeighted<ComplexDataVector, 2>& target_c_inv,
    const SpinWeighted<ComplexDataVector, 0>& target_d_inv, const size_t l_max,
    const double tolerance, const size_t max_steps,
    const double error_threshold, const bool require_convergence) {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  SpinWeighted<ComplexDataVector, 2> interpolated_target_c{
      number_of_angular_points};
  SpinWeighted<ComplexDataVector, 0> interpolated_target_d{
      number_of_angular_points};
  const auto iteration_function =
      [&target_c_inv, &target_d_inv, &interpolated_target_c,
       &interpolated_target_d,
       &l_max](const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
                   gauge_c_step,
               const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
                   gauge_d_step,
               const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
               const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
               const Spectral::Swsh::SwshInterpolator& iteration_interpolator) {
        iteration_interpolator.interpolate(
            make_not_null(&interpolated_target_c), target_c_inv);
        iteration_interpolator.interpolate(
            make_not_null(&interpolated_target_d), target_d_inv);
        jacobian_match_heuristic(gauge_c_step, gauge_d_step, gauge_c, gauge_d,
                                 interpolated_target_c, interpolated_target_d,
                                 l_max);
        return max(abs(get(gauge_c).data() - interpolated_target_c.data()) +
                   abs(get(gauge_d).data() - interpolated_target_d.data()));
      };
  return iteratively_adapt_angular_coordinates(
      cartesian_inertial_coordinates, angular_inertial_coordinates, l_max,
      tolerance, max_steps, error_threshold, iteration_function,
      require_convergence);
}
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
template <bool evolve_ccm>
struct NoIncomingRadiation;
template <bool evolve_ccm>
struct ZeroNonSmooth;
template <bool evolve_ccm>
struct InverseCubic;
template <bool evolve_ccm>
struct InitializeJ;
template <bool evolve_ccm>
struct ConformalFactor;
template <bool evolve_ccm>
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

  // Generators that additionally produce the inertial (partially flat)
  // coordinates required when the partially flat Bondi-like coordinates are
  // evolved (`evolve_ccm = true`).
  using creatable_classes =
      tmpl::list<InverseCubic<true>, NoIncomingRadiation<true>,
                 ZeroNonSmooth<true>, ConformalFactor<true>,
                 CauchySecondOrder<true>>;

  InitializeJ() = default;
  explicit InitializeJ(CkMigrateMessage* /*msg*/) {}

  WRAPPED_PUPable_abstract(InitializeJ);  // NOLINT

  virtual std::unique_ptr<InitializeJ<true>> get_clone() const = 0;

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
      tmpl::list<ConformalFactor<false>, InverseCubic<false>,
                 NoIncomingRadiation<false>, ZeroNonSmooth<false>,
                 CauchySecondOrder<false>,
                 ::Cce::Solutions::LinearizedBondiSachs_detail::InitializeJ::
                     LinearizedBondiSachs>;

  InitializeJ() = default;
  explicit InitializeJ(CkMigrateMessage* /*msg*/) {}

  WRAPPED_PUPable_abstract(InitializeJ);  // NOLINT

  virtual std::unique_ptr<InitializeJ<false>> get_clone() const = 0;

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
