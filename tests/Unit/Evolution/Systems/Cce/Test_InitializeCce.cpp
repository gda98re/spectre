// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/Cce/GaugeTransformBoundaryData.hpp"
#include "Evolution/Systems/Cce/Initialize/CauchySecondOrder.hpp"
#include "Evolution/Systems/Cce/Initialize/ComputeSecondOrderRadialDerivativeJ.hpp"
#include "Evolution/Systems/Cce/Initialize/ConformalFactor.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Evolution/Systems/Cce/Initialize/InverseCubic.hpp"
#include "Evolution/Systems/Cce/Initialize/NoIncomingRadiation.hpp"
#include "Evolution/Systems/Cce/Initialize/ZeroNonSmooth.hpp"
#include "Evolution/Systems/Cce/LinearOperators.hpp"
#include "Evolution/Systems/Cce/LinearSolve.hpp"
#include "Evolution/Systems/Cce/NewmanPenrose.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/PreSwshDerivatives.hpp"
#include "Evolution/Systems/Cce/PrecomputeCceDependencies.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTestHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/CubicSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/LinearSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshFiltering.hpp"
#include "Parallel/NodeLock.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace Cce {

template <template <typename> typename BoundaryTag, typename DbTags>
void check_boundary_and_asymptotic_j(
    const gsl::not_null<db::DataBox<DbTags>*> box_to_initialize,
    const size_t number_of_radial_points, const size_t l_max) {
  // The goal for this initial data is that it should:
  // - match the value of J and its first derivative on the boundary
  // - have vanishing value and second derivative at scri+
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  const SpinWeighted<ComplexDataVector, 2> boundary_slice_dy_j;
  make_const_view(make_not_null(&boundary_slice_dy_j),
                  get(db::get<Tags::Dy<Tags::BondiJ>>(*box_to_initialize)), 0,
                  number_of_angular_points);

  const SpinWeighted<ComplexDataVector, 2> boundary_slice_j;
  const SpinWeighted<ComplexDataVector, 2> scri_slice_j;
  make_const_view(make_not_null(&boundary_slice_j),
                  get(db::get<Tags::BondiJ>(*box_to_initialize)), 0,
                  number_of_angular_points);
  make_const_view(make_not_null(&scri_slice_j),
                  get(db::get<Tags::BondiJ>(*box_to_initialize)),
                  number_of_angular_points * (number_of_radial_points - 1),
                  number_of_angular_points);

  const SpinWeighted<ComplexDataVector, 2> scri_slice_dy_dy_j;
  make_const_view(
      make_not_null(&scri_slice_dy_dy_j),
      get(db::get<Tags::Dy<Tags::Dy<Tags::BondiJ>>>(*box_to_initialize)),
      number_of_angular_points * (number_of_radial_points - 1),
      number_of_angular_points);

  Approx cce_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e4)
          .scale(1.0);

  CHECK_ITERABLE_CUSTOM_APPROX(
      get(db::get<BoundaryTag<Tags::BondiJ>>(*box_to_initialize)).data(),
      boundary_slice_j, cce_approx);
  const auto boundary_slice_dr_j =
      (2.0 /
       get(db::get<BoundaryTag<Tags::BondiR>>(*box_to_initialize)).data()) *
      boundary_slice_dy_j.data();
  CHECK_ITERABLE_CUSTOM_APPROX(
      boundary_slice_dr_j,
      get(db::get<BoundaryTag<Tags::Dr<Tags::BondiJ>>>(*box_to_initialize))
          .data(),
      cce_approx);
  const ComplexDataVector scri_plus_zeroes{
      Spectral::Swsh::number_of_swsh_collocation_points(l_max), 0.0};
  CHECK_ITERABLE_CUSTOM_APPROX(scri_slice_j, scri_plus_zeroes, cce_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(scri_slice_dy_dy_j.data(), scri_plus_zeroes,
                               cce_approx);
}

// Fill a worldtube boundary value with filtered random spin-weighted data of
// the requested spin weight. Used to populate the additional boundary
// quantities consumed by the `CauchySecondOrder` generator.
template <int Spin, typename Generator, typename Distribution>
void assign_random_swsh_boundary_value(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, Spin>>*> field,
    const gsl::not_null<Generator*> generator,
    const gsl::not_null<Distribution*> distribution, const size_t l_max) {
  SpinWeighted<ComplexModalVector, Spin> generated_modes{
      Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
  Spectral::Swsh::TestHelpers::generate_swsh_modes<Spin>(
      make_not_null(&generated_modes.data()), generator, distribution, 1,
      l_max);
  get(*field) =
      Spectral::Swsh::inverse_swsh_transform(l_max, 1, generated_modes);
  Spectral::Swsh::filter_swsh_boundary_quantity(make_not_null(&get(*field)),
                                                l_max, l_max / 2);
}

template <typename DbTags>
void test_initialize_j_inverse_cubic(
    const gsl::not_null<db::DataBox<DbTags>*> box_to_initialize,
    const size_t l_max, const size_t number_of_radial_points) {
  auto node_lock = Parallel::NodeLock{};
  db::mutate_apply<InitializeJ::InitializeJ<true>::mutate_tags,
                   InitializeJ::InitializeJ<true>::argument_tags>(
      InitializeJ::InverseCubic<true>{}, box_to_initialize,
      make_not_null(&node_lock));
  db::mutate_apply<PreSwshDerivatives<Tags::Dy<Tags::BondiJ>>>(
      box_to_initialize);
  db::mutate_apply<PreSwshDerivatives<Tags::Dy<Tags::Dy<Tags::BondiJ>>>>(
      box_to_initialize);
  check_boundary_and_asymptotic_j<Tags::BoundaryValue>(
      box_to_initialize, number_of_radial_points, l_max);
}

template <typename DbTags>
void test_initialize_j_zero_nonsmooth(
    const gsl::not_null<db::DataBox<DbTags>*> box_to_initialize,
    const size_t /*l_max*/, const size_t /*number_of_radial_points*/) {
  // The iterative procedure can reach error levels better than 1.0e-8, but it
  // is difficult to do so reliably and quickly for randomly generated data.
  auto node_lock = Parallel::NodeLock{};
  db::mutate_apply<InitializeJ::InitializeJ<false>::mutate_tags,
                   InitializeJ::InitializeJ<false>::argument_tags>(
      InitializeJ::ZeroNonSmooth{1.0e-8, 400}, box_to_initialize,
      make_not_null(&node_lock));

  // note we want to copy here to compare against the next version of the
  // computation
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  const auto initialized_j = db::get<Tags::BondiJ>(*box_to_initialize);

  const auto initializer = InitializeJ::ZeroNonSmooth{1.0e-8, 400};
  const auto serialized_and_deserialized_initializer =
      serialize_and_deserialize(initializer);

  db::mutate_apply<InitializeJ::InitializeJ<false>::mutate_tags,
                   InitializeJ::InitializeJ<false>::argument_tags>(
      serialized_and_deserialized_initializer, box_to_initialize,
      make_not_null(&node_lock));
  const auto& initialized_j_from_serialized_and_deserialized =
      db::get<Tags::BondiJ>(*box_to_initialize);

  CHECK_ITERABLE_APPROX(
      get(initialized_j).data(),
      get(initialized_j_from_serialized_and_deserialized).data());

  // generate the extra gauge quantities and verify that the boundary value for
  // J is indeed within the tolerance.
  db::mutate_apply<GaugeUpdateAngularFromCartesian<
      Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>(
      box_to_initialize);
  db::mutate_apply<GaugeUpdateJacobianFromCoordinates<
      Tags::PartiallyFlatGaugeC, Tags::PartiallyFlatGaugeD,
      Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>(
      box_to_initialize);
  db::mutate_apply<GaugeUpdateInterpolator<Tags::CauchyAngularCoords>>(
      box_to_initialize);
  db::mutate_apply<
      GaugeUpdateOmega<Tags::PartiallyFlatGaugeC, Tags::PartiallyFlatGaugeD,
                       Tags::PartiallyFlatGaugeOmega>>(box_to_initialize);

  db::mutate_apply<GaugeAdjustedBoundaryValue<Tags::BondiJ>>(box_to_initialize);

  const auto& gauge_adjusted_boundary_j =
      db::get<Tags::EvolutionGaugeBoundaryValue<Tags::BondiJ>>(
          *box_to_initialize);
  for (auto val : get(gauge_adjusted_boundary_j).data()) {
    CHECK(real(val) < 1.0e-8);
    CHECK(imag(val) < 1.0e-8);
  }

  for (auto val : get(initialized_j).data()) {
    CHECK(real(val) < 1.0e-8);
    CHECK(imag(val) < 1.0e-8);
  }
}

template <typename DbTags>
void test_zero_non_smooth_error(
    const gsl::not_null<db::DataBox<DbTags>*> box_to_initialize,
    const size_t /*l_max*/, const size_t /*number_of_radial_points*/) {
  auto node_lock = Parallel::NodeLock{};
  db::mutate_apply<InitializeJ::InitializeJ<false>::mutate_tags,
                   InitializeJ::InitializeJ<false>::argument_tags>(
      InitializeJ::ZeroNonSmooth{1.0e-12, 1, true}, box_to_initialize,
      make_not_null(&node_lock));
}

template <typename DbTags>
void test_initialize_j_no_radiation(
    const gsl::not_null<db::DataBox<DbTags>*> box_to_initialize,
    const size_t l_max, const size_t /*number_of_radial_points*/) {
  // The iterative procedure can reach error levels better than 1.0e-8, but it
  // is difficult to do so reliably and quickly for randomly generated data.
  auto node_lock = Parallel::NodeLock{};
  db::mutate_apply<InitializeJ::InitializeJ<false>::mutate_tags,
                   InitializeJ::InitializeJ<false>::argument_tags>(
      InitializeJ::NoIncomingRadiation{1.0e-8, 400}, box_to_initialize,
      make_not_null(&node_lock));

  // note we want to copy here to compare against the next version of the
  // computation
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  const auto initialized_j = db::get<Tags::BondiJ>(*box_to_initialize);

  const auto initializer = InitializeJ::NoIncomingRadiation{1.0e-8, 400};
  const auto serialized_and_deserialized_initializer =
      serialize_and_deserialize(initializer);

  db::mutate_apply<InitializeJ::InitializeJ<false>::mutate_tags,
                   InitializeJ::InitializeJ<false>::argument_tags>(
      serialized_and_deserialized_initializer, box_to_initialize,
      make_not_null(&node_lock));
  const auto& initialized_j_from_serialized_and_deserialized =
      db::get<Tags::BondiJ>(*box_to_initialize);

  CHECK_ITERABLE_APPROX(
      get(initialized_j).data(),
      get(initialized_j_from_serialized_and_deserialized).data());

  db::mutate_apply<GaugeUpdateAngularFromCartesian<
      Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>(
      box_to_initialize);
  db::mutate_apply<GaugeUpdateJacobianFromCoordinates<
      Tags::PartiallyFlatGaugeC, Tags::PartiallyFlatGaugeD,
      Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>(
      box_to_initialize);
  db::mutate_apply<GaugeUpdateInterpolator<Tags::CauchyAngularCoords>>(
      box_to_initialize);
  db::mutate_apply<
      GaugeUpdateOmega<Tags::PartiallyFlatGaugeC, Tags::PartiallyFlatGaugeD,
                       Tags::PartiallyFlatGaugeOmega>>(box_to_initialize);

  db::mutate_apply<PrecomputeCceDependencies<Tags::EvolutionGaugeBoundaryValue,
                                             Tags::OneMinusY>>(
      box_to_initialize);
  db::mutate_apply<GaugeAdjustedBoundaryValue<Tags::BondiJ>>(box_to_initialize);

  // check that the gauge-transformed boundary data matches up.
  const auto& boundary_gauge_j =
      db::get<Tags::EvolutionGaugeBoundaryValue<Tags::BondiJ>>(
          *box_to_initialize);
  for (size_t i = 0;
       i < Spectral::Swsh::number_of_swsh_collocation_points(l_max); ++i) {
    CHECK(approx(real(get(initialized_j).data()[i])) ==
          real(get(boundary_gauge_j).data()[i]));
    CHECK(approx(imag(get(initialized_j).data()[i])) ==
          imag(get(boundary_gauge_j).data()[i]));
  }

  db::mutate_apply<GaugeAdjustedBoundaryValue<Tags::BondiR>>(box_to_initialize);

  db::mutate_apply<PrecomputeCceDependencies<Tags::EvolutionGaugeBoundaryValue,
                                             Tags::BondiR>>(box_to_initialize);
  db::mutate_apply<PrecomputeCceDependencies<Tags::EvolutionGaugeBoundaryValue,
                                             Tags::BondiK>>(box_to_initialize);
  db::mutate_apply<PreSwshDerivatives<Tags::Dy<Tags::BondiJ>>>(
      box_to_initialize);
  db::mutate_apply<PreSwshDerivatives<Tags::Dy<Tags::Dy<Tags::BondiJ>>>>(
      box_to_initialize);

  db::mutate_apply<VolumeWeyl<Tags::Psi0>>(box_to_initialize);

  Approx cce_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e5)
          .scale(1.0);
  // check that the psi_0 condition holds to acceptable precision -- note the
  // result of this involves multiple numerical derivatives, so needs to be
  // slightly loose.
  for (auto val : get(db::get<Tags::Psi0>(*box_to_initialize)).data()) {
    CHECK(cce_approx(real(val)) == 0.0);
    CHECK(cce_approx(imag(val)) == 0.0);
  }
}

// The interpolator `CauchySecondOrder` hands to the worldtube data manager for
// the Du(Dr(J)) boundary value. These tests supply that boundary value directly
// rather than through a manager, so it is only carried along and serialized.
std::unique_ptr<intrp::SpanInterpolator> test_du_dr_j_interpolator() {
  return std::make_unique<intrp::BarycentricRationalSpanInterpolator>(2_st,
                                                                      2_st);
}

// The interpolator the generator hands to the worldtube data manager has to
// survive the trip into the GlobalCache and back out of a checkpoint.
void test_cauchy_second_order_interpolator_round_trip() {
  const InitializeJ::CauchySecondOrder with_interpolator{
      1.0e-10, 400, true, 1.0e-1, test_du_dr_j_interpolator()};
  REQUIRE(with_interpolator.du_dr_j_interpolator() != nullptr);
  CHECK(with_interpolator.du_dr_j_interpolator()
            ->required_number_of_points_before_and_after() == 2);

  const auto clone = with_interpolator.get_clone();
  REQUIRE(clone->du_dr_j_interpolator() != nullptr);
  CHECK(clone->du_dr_j_interpolator()
            ->required_number_of_points_before_and_after() == 2);

  const auto round_tripped = serialize_and_deserialize(with_interpolator);
  REQUIRE(round_tripped.du_dr_j_interpolator() != nullptr);
  CHECK(round_tripped.du_dr_j_interpolator()
            ->required_number_of_points_before_and_after() == 2);

  // A generator that asks for nothing leaves the manager on `H5Interpolator`.
  const InitializeJ::CauchySecondOrder without_interpolator{1.0e-10, 400, true,
                                                            1.0e-1, nullptr};
  CHECK(without_interpolator.du_dr_j_interpolator() == nullptr);
  CHECK(without_interpolator.get_clone()->du_dr_j_interpolator() == nullptr);
  CHECK(InitializeJ::InverseCubic<false>{}.du_dr_j_interpolator() == nullptr);
}

template <typename DbTags>
void test_initialize_j_cauchy_second_order(
    const gsl::not_null<db::DataBox<DbTags>*> box_to_initialize,
    const size_t l_max, const size_t number_of_radial_points) {
  auto node_lock = Parallel::NodeLock{};
  // The angular coordinates are adapted iteratively (as in NoIncomingRadiation
  // and ConformalFactor). For randomly generated data the linearized solve
  // occasionally needs more than a few hundred iterations to reach 1e-10, so we
  // allow up to 1000 iterations (the option maximum) to reliably converge with
  // `require_convergence = true`.
  const auto initializer = InitializeJ::CauchySecondOrder{
      1.0e-10, 1000, true, 1.0e-1, test_du_dr_j_interpolator()};
  db::mutate_apply<InitializeJ::CauchySecondOrder::return_tags,
                   InitializeJ::CauchySecondOrder::argument_tags>(
      initializer, box_to_initialize, make_not_null(&node_lock));

  // note we want to copy here to compare against the next version of the
  // computation
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  const auto initialized_j = db::get<Tags::BondiJ>(*box_to_initialize);
  const auto serialized_and_deserialized_initializer =
      serialize_and_deserialize(initializer);
  db::mutate_apply<InitializeJ::CauchySecondOrder::return_tags,
                   InitializeJ::CauchySecondOrder::argument_tags>(
      serialized_and_deserialized_initializer, box_to_initialize,
      make_not_null(&node_lock));
  CHECK_ITERABLE_APPROX(get(initialized_j).data(),
                        get(db::get<Tags::BondiJ>(*box_to_initialize)).data());

  // generate the gauge quantities so the boundary data can be compared in the
  // evolution gauge.
  db::mutate_apply<GaugeUpdateAngularFromCartesian<
      Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>(
      box_to_initialize);
  db::mutate_apply<GaugeUpdateJacobianFromCoordinates<
      Tags::PartiallyFlatGaugeC, Tags::PartiallyFlatGaugeD,
      Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>(
      box_to_initialize);
  db::mutate_apply<GaugeUpdateInterpolator<Tags::CauchyAngularCoords>>(
      box_to_initialize);
  db::mutate_apply<
      GaugeUpdateOmega<Tags::PartiallyFlatGaugeC, Tags::PartiallyFlatGaugeD,
                       Tags::PartiallyFlatGaugeOmega>>(box_to_initialize);
  db::mutate_apply<GaugeAdjustedBoundaryValue<Tags::BondiJ>>(box_to_initialize);

  // The polynomial construction matches J and its first radial derivative at
  // the worldtube, so the volume J on the boundary should equal the
  // gauge-transformed boundary value of J.
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  const auto& boundary_gauge_j =
      db::get<Tags::EvolutionGaugeBoundaryValue<Tags::BondiJ>>(
          *box_to_initialize);
  for (size_t i = 0; i < number_of_angular_points; ++i) {
    CHECK(approx(real(get(initialized_j).data()[i])) ==
          real(get(boundary_gauge_j).data()[i]));
    CHECK(approx(imag(get(initialized_j).data()[i])) ==
          imag(get(boundary_gauge_j).data()[i]));
  }

  // The cubic-in-(1 - y) construction forces the second radial derivative of J
  // to vanish at scri+ in the numerical gauge, and the angular gauge transform
  // then violates that by its own nonlinear part alone. Eq. (51b) of the
  // initial-data paper puts that violation at
  // ||Delta Jbreve^(2)|| ~ ||mu (Jtilde^(1))^2||, which in the numerical radial
  // coordinate is ||J^(0)|| ||B||^2 with B the (1 - y) coefficient of the
  // ansatz -- the same estimate the generator's guard is built on. Checking the
  // two agree is a far sharper statement than "close to zero", and it is what
  // makes that guard meaningful rather than an arbitrary number.
  db::mutate_apply<PreSwshDerivatives<Tags::Dy<Tags::BondiJ>>>(
      box_to_initialize);
  db::mutate_apply<PreSwshDerivatives<Tags::Dy<Tags::Dy<Tags::BondiJ>>>>(
      box_to_initialize);
  const SpinWeighted<ComplexDataVector, 2> scri_slice_dy_dy_j;
  make_const_view(
      make_not_null(&scri_slice_dy_dy_j),
      get(db::get<Tags::Dy<Tags::Dy<Tags::BondiJ>>>(*box_to_initialize)),
      number_of_angular_points * (number_of_radial_points - 1),
      number_of_angular_points);

  Scalar<SpinWeighted<ComplexDataVector, 2>> boundary_dy2_j{
      number_of_angular_points};
  InitializeJ::CauchySecondOrder_detail::compute_dy_dy_j(
      make_not_null(&boundary_dy2_j),
      db::get<Tags::BoundaryValue<Tags::BondiJ>>(*box_to_initialize),
      db::get<Tags::BoundaryValue<Tags::BondiU>>(*box_to_initialize),
      db::get<Tags::BoundaryValue<Tags::BondiW>>(*box_to_initialize),
      db::get<Tags::BoundaryValue<Tags::BondiBeta>>(*box_to_initialize),
      db::get<Tags::BoundaryValue<Tags::BondiQ>>(*box_to_initialize),
      db::get<Tags::BoundaryValue<Tags::Du<Tags::BondiJ>>>(*box_to_initialize),
      db::get<Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>>(*box_to_initialize),
      db::get<Tags::BoundaryValue<Tags::Du<Tags::Dr<Tags::BondiJ>>>>(
          *box_to_initialize),
      db::get<Tags::BoundaryValue<Tags::Du<Tags::BondiR>>>(*box_to_initialize),
      db::get<Tags::BoundaryValue<Tags::BondiR>>(*box_to_initialize), l_max);
  const ComplexDataVector r_dr_j =
      get(db::get<Tags::BoundaryValue<Tags::BondiR>>(*box_to_initialize))
          .data() *
      get(db::get<Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>>(
              *box_to_initialize))
          .data();
  const double asymptotic_j = max(
      abs(get(db::get<Tags::BoundaryValue<Tags::BondiJ>>(*box_to_initialize))
              .data() +
          r_dr_j + (4.0 / 3.0) * get(boundary_dy2_j).data()));
  const double one_minus_y_coefficient =
      max(abs(0.5 * r_dr_j + get(boundary_dy2_j).data()));
  const double predicted_scri_dy_dy_j =
      asymptotic_j * square(one_minus_y_coefficient);
  const double measured_scri_dy_dy_j = max(abs(scri_slice_dy_dy_j.data()));
  INFO("second partially flat violation: "
       << measured_scri_dy_dy_j << " measured against "
       << predicted_scri_dy_dy_j << " predicted by Eq. (51b)");
  // Two sided, and generously, since the estimate is an order-of-magnitude one;
  // the round-off floor is the noise of differentiating the ansatz twice.
  const double round_off_floor = 1.0e-14;
  CHECK(measured_scri_dy_dy_j <
        100.0 * (predicted_scri_dy_dy_j + round_off_floor));
  CHECK(measured_scri_dy_dy_j >
        0.01 * predicted_scri_dy_dy_j - round_off_floor);
}

template <typename DbTags>
void test_cauchy_second_order_angular_solve_threshold(
    const gsl::not_null<db::DataBox<DbTags>*> box_to_initialize) {
  // Before the angular solve the constructed J is generically nonzero at scri+
  // at the scale of the strain, so an unachievably small `MaxAngularSolveError`
  // trips the pre-solve guard even on uncorrupted worldtube data. This checks
  // that the threshold is honored from the option rather than hard-coded.
  auto node_lock = Parallel::NodeLock{};
  db::mutate_apply<InitializeJ::CauchySecondOrder::return_tags,
                   InitializeJ::CauchySecondOrder::argument_tags>(
      InitializeJ::CauchySecondOrder{1.0e-10, 400, true, 1.0e-14,
                                     test_du_dr_j_interpolator()},
      box_to_initialize, make_not_null(&node_lock));
}

template <typename DbTags>
void test_cauchy_second_order_asymptotic_j_error(
    const gsl::not_null<db::DataBox<DbTags>*> box_to_initialize) {
  // Inflate the worldtube J without the compensating radial derivative so the
  // asymptotic value of the constructed Cauchy-coordinate J is far too large
  // for the angular solve to eliminate, tripping the pre-solve guard. This
  // corrupts the boundary data, so it must run after the other checks.
  db::mutate<Tags::BoundaryValue<Tags::BondiJ>>(
      [](const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
             boundary_j) { get(*boundary_j).data() += 1.0; },
      box_to_initialize);
  auto node_lock = Parallel::NodeLock{};
  db::mutate_apply<InitializeJ::CauchySecondOrder::return_tags,
                   InitializeJ::CauchySecondOrder::argument_tags>(
      InitializeJ::CauchySecondOrder{1.0e-10, 400, true, 1.0e-1,
                                     test_du_dr_j_interpolator()},
      box_to_initialize, make_not_null(&node_lock));
}

template <typename DbTags>
void test_initialize_j_conformal_factor(
    const gsl::not_null<db::DataBox<DbTags>*> box_to_initialize,
    const bool optimize_l_0_mode, const bool use_beta_integral_estimate,
    const bool use_input_modes, const bool read_modes_from_file,
    const ::Cce::InitializeJ::ConformalFactorIterationHeuristic
        iteration_heuristic,
    const size_t l_max, const size_t number_of_radial_points) {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  CAPTURE(optimize_l_0_mode);
  CAPTURE(use_beta_integral_estimate);
  CAPTURE(use_input_modes);
  CAPTURE(read_modes_from_file);
  auto node_lock = Parallel::NodeLock{};
  InitializeJ::ConformalFactor initialize_j_conformal_factor;
  MAKE_GENERATOR(generator);
  UniformCustomDistribution<double> dist(1.0e-4, 1.0e-3);
  const std::string filename = "ConformalFactorInputModes.h5";

  std::vector<double> input_mode_data(2 * square(l_max + 1));
  for (size_t i = 0; i < 8; ++i) {
    input_mode_data[i] = 0.0;
  }
  for (size_t i = 8; i < input_mode_data.size(); ++i) {
    // exponentially decay higher l-modes
    input_mode_data[i] = dist(generator) * exp(-1.0 * sqrt(i / 2));
  }
  std::vector<std::complex<double>> input_modes(square(l_max + 1));
  for (size_t i = 0; i < input_modes.size(); ++i) {
    input_modes[i] = std::complex<double>(input_mode_data[i * 2],
                                          input_mode_data[i * 2 + 1]);
  }
  if (use_input_modes) {
    if (read_modes_from_file) {
      if (file_system::check_if_file_exists(filename)) {
        file_system::rm(filename, true);
      }
      h5::H5File<h5::AccessType::ReadWrite> input_h5_modes{filename};

      std::vector<std::string> file_legend{};
      for (int l = 0; l <= static_cast<int>(l_max); ++l) {
        for (int m = -l; m <= l; ++m) {
          file_legend.push_back("Real Y_" + std::to_string(l) + "," +
                                std::to_string(m));
          file_legend.push_back("Imag Y_" + std::to_string(l) + "," +
                                std::to_string(m));
        }
      }
      auto& dataset =
          input_h5_modes.try_insert<h5::Dat>("/InitialJ", file_legend, 0);
      dataset.append(input_mode_data);
      input_h5_modes.close_current_object();
    }
  }
  if (read_modes_from_file) {
    InitializeJ::ConformalFactor initialize_j_constructed{
        1.0e-8,
        400,
        true,
        optimize_l_0_mode,
        use_beta_integral_estimate,
        iteration_heuristic,
        use_input_modes,
        filename};
    initialize_j_conformal_factor =
        serialize_and_deserialize(initialize_j_constructed);
  } else {
    InitializeJ::ConformalFactor initialize_j_constructed{
        1.0e-8,
        400,
        true,
        optimize_l_0_mode,
        use_beta_integral_estimate,
        iteration_heuristic,
        use_input_modes,
        input_modes};
    const auto initialize_j_cloned = initialize_j_constructed.get_clone();
    initialize_j_conformal_factor =
        *dynamic_cast<::Cce::InitializeJ::ConformalFactor*>(
            initialize_j_cloned.get());
  }

  db::mutate_apply<InitializeJ::InitializeJ<false>::mutate_tags,
                   InitializeJ::InitializeJ<false>::argument_tags>(
      initialize_j_conformal_factor, box_to_initialize,
      make_not_null(&node_lock));
  Approx iterative_solve_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e5)
          .scale(1.0);

  // perform gauge transforms on the boundary
  db::mutate_apply<GaugeUpdateAngularFromCartesian<
      Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>(
      box_to_initialize);
  db::mutate_apply<GaugeUpdateJacobianFromCoordinates<
      Tags::PartiallyFlatGaugeC, Tags::PartiallyFlatGaugeD,
      Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>(
      box_to_initialize);
  db::mutate_apply<GaugeUpdateInterpolator<Tags::CauchyAngularCoords>>(
      box_to_initialize);
  db::mutate_apply<
      GaugeUpdateOmega<Tags::PartiallyFlatGaugeC, Tags::PartiallyFlatGaugeD,
                       Tags::PartiallyFlatGaugeOmega>>(box_to_initialize);
  db::mutate_apply<GaugeAdjustedBoundaryValue<Tags::BondiBeta>>(
      box_to_initialize);
  db::mutate_apply<GaugeAdjustedBoundaryValue<Tags::BondiR>>(box_to_initialize);
  db::mutate_apply<GaugeAdjustedBoundaryValue<Tags::BondiJ>>(box_to_initialize);
  db::mutate_apply<GaugeAdjustedBoundaryValue<Tags::Dr<Tags::BondiJ>>>(
      box_to_initialize);
  const ComplexDataVector surface_zeroes{
      Spectral::Swsh::number_of_swsh_collocation_points(l_max), 0.0};
  db::mutate_apply<
      PrecomputeCceDependencies<Tags::BoundaryValue, Tags::OneMinusY>>(
      box_to_initialize);
  db::mutate_apply<PreSwshDerivatives<Tags::Dy<Tags::BondiJ>>>(
      box_to_initialize);
  db::mutate_apply<PreSwshDerivatives<Tags::Dy<Tags::Dy<Tags::BondiJ>>>>(
      box_to_initialize);

  if (use_beta_integral_estimate) {
    // check the conformal factor on scri
    db::mutate_apply<ComputeBondiIntegrand<Tags::Integrand<Tags::BondiBeta>>>(
        box_to_initialize);
    db::mutate_apply<RadialIntegrateBondi<Tags::EvolutionGaugeBoundaryValue,
                                          Tags::BondiBeta>>(box_to_initialize);
    Approx beta_estimate_approx = Approx::custom().epsilon(5.0e-8).scale(1.0);
    auto mutable_beta_copy = get(db::get<Tags::BondiBeta>(*box_to_initialize));
    SpinWeighted<ComplexDataVector, 0> scri_slice_beta{ComplexDataVector{
        mutable_beta_copy.data().data() +
            (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points}};
    if (optimize_l_0_mode) {
      CHECK_ITERABLE_CUSTOM_APPROX(scri_slice_beta, surface_zeroes,
                                   beta_estimate_approx);
    } else {
      Spectral::Swsh::filter_swsh_boundary_quantity(
          make_not_null(&scri_slice_beta), l_max, 1_st, l_max);
      CHECK_ITERABLE_CUSTOM_APPROX(scri_slice_beta, surface_zeroes,
                                   beta_estimate_approx);
    }
  } else {
    // When not using the beta integral estimate, the conformal factor target on
    // the boundary is chosen to minimize the value of beta
    if (optimize_l_0_mode) {
      CHECK_ITERABLE_CUSTOM_APPROX(
          get(db::get<Tags::EvolutionGaugeBoundaryValue<Tags::BondiBeta>>(
                  *box_to_initialize))
              .data(),
          surface_zeroes, iterative_solve_approx);
    } else {
      auto filtered_beta =
          get(db::get<Tags::EvolutionGaugeBoundaryValue<Tags::BondiBeta>>(
              *box_to_initialize));
      Spectral::Swsh::filter_swsh_boundary_quantity(
          make_not_null(&filtered_beta), l_max, 1_st, l_max);
      CHECK_ITERABLE_CUSTOM_APPROX(filtered_beta, surface_zeroes,
                                   iterative_solve_approx);
    }
  }

  check_boundary_and_asymptotic_j<Tags::EvolutionGaugeBoundaryValue>(
      box_to_initialize, number_of_radial_points, l_max);
  if (use_input_modes) {
    // check the correctness of the initial data:
    // - if using input modes, the 1/r part of j should match those modes.
    const SpinWeighted<ComplexDataVector, 2> scri_slice_dy_j;
    make_const_view(make_not_null(&scri_slice_dy_j),
                    get(db::get<Tags::Dy<Tags::BondiJ>>(*box_to_initialize)),
                    (number_of_radial_points - 1) * number_of_angular_points,
                    number_of_angular_points);
    // asymptotically, we have J = J^{(1)}/r = J^{(1)} * (1 - y) / 2 R,
    SpinWeighted<ComplexDataVector, 2> inverse_r_part_of_asymptotic_j =
        -2.0 * scri_slice_dy_j *
        get(db::get<Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>>(
            *box_to_initialize));
    auto inverse_r_asymptotic_modes =
        Spectral::Swsh::libsharp_to_goldberg_modes(
            Spectral::Swsh::swsh_transform(l_max, 1_st,
                                           inverse_r_part_of_asymptotic_j),
            l_max);
    for (size_t i = 0; i < square(l_max + 1); ++i) {
      CAPTURE(i);
      CHECK(approx(real(inverse_r_asymptotic_modes.data()[i])) ==
            real(input_modes[i]));
      CHECK(approx(imag(inverse_r_asymptotic_modes.data()[i])) ==
            imag(input_modes[i]));
    }
  }
  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }

  const auto spin_weight_1_created = TestHelpers::test_creation<
      ::Cce::InitializeJ::ConformalFactorIterationHeuristic>(
      "SpinWeight1CoordPerturbation");
  CHECK(spin_weight_1_created ==
        ::Cce::InitializeJ::ConformalFactorIterationHeuristic::
            SpinWeight1CoordPerturbation);
  const auto only_vary_gauge_d_created = TestHelpers::test_creation<
      ::Cce::InitializeJ::ConformalFactorIterationHeuristic>("OnlyVaryGaugeD");
  CHECK(only_vary_gauge_d_created ==
        ::Cce::InitializeJ::ConformalFactorIterationHeuristic::OnlyVaryGaugeD);
  const std::string spin_weight_1_streamed =
      MakeString{} << ::Cce::InitializeJ::ConformalFactorIterationHeuristic::
          SpinWeight1CoordPerturbation;
  CHECK(spin_weight_1_streamed == "SpinWeight1CoordPerturbation");
  const std::string only_vary_gauge_d_streamed =
      MakeString{}
      << ::Cce::InitializeJ::ConformalFactorIterationHeuristic::OnlyVaryGaugeD;
  CHECK(only_vary_gauge_d_streamed == "OnlyVaryGaugeD");
}

// [[TimeOut, 10]]
// [[TimeOut, 10]]

// The two angular-coordinate solves target the same condition, Jhat^(0) = 0,
// and the solution of that condition is unique up to the l < 2 conformal
// freedom -- which both of them discard in the same place, the kernel of the
// inverse eth. They must therefore converge to the SAME map, and the
// potential-based solve must get there in far fewer passes: it steps to the
// exact root of the gauge condition rather than a linearisation of it, and its
// coordinate update is integrable by construction, so nothing is projected
// away.
void test_angular_coordinate_solves_agree() {
  // This test sets its own resolution rather than taking the surrounding test
  // case's l_max of 5-6: the residual floor of BOTH solves is set by how much
  // angular content the interpolation onto the displaced points has to carry,
  // so the field has to be resolved with room to spare for either of them to
  // reach a tolerance worth comparing at.
  const size_t l_max = 12;
  const size_t filter_max_l = 4;
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  // A smooth spin-weight-2 J at scri+, of a size the linearised iteration can
  // still handle (the generators refuse |J| > 1e-2).
  // A fixed seed, so that the comparison between the two solves is a
  // controlled one: what is being checked is that they agree, and a field that
  // varied run to run would vary the resolution-limited residual floor with it.
  MAKE_GENERATOR(gen, 8675309);
  UniformCustomDistribution<double> dist{-1.0, 1.0};
  SpinWeighted<ComplexDataVector, 2> j_at_scri{number_of_angular_points};
  fill_with_random_values(make_not_null(&j_at_scri.data()), make_not_null(&gen),
                          make_not_null(&dist));
  Spectral::Swsh::filter_swsh_boundary_quantity(make_not_null(&j_at_scri),
                                                l_max, filter_max_l);
  j_at_scri.data() *= 1.0e-3 / max(abs(j_at_scri.data()));

  const double tolerance = 1.0e-10;
  const size_t max_steps = 2000;

  SpinWeighted<ComplexDataVector, 2> surface_j{number_of_angular_points};
  SpinWeighted<ComplexDataVector, 2> j_hat{number_of_angular_points};
  SpinWeighted<ComplexDataVector, 0> interpolated_k{number_of_angular_points};
  SpinWeighted<ComplexDataVector, 0> omega{number_of_angular_points};

  // shared by both functors: interpolate J onto the current points and form the
  // residual Jhat^(0) of Eq. (4.11) of \cite Moxon2020
  const auto residual =
      [&surface_j, &j_hat, &interpolated_k, &omega, &j_at_scri](
          const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
          const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
          const Spectral::Swsh::SwshInterpolator& interpolator) {
        interpolator.interpolate(make_not_null(&surface_j), j_at_scri);
        interpolated_k.data() =
            sqrt(1.0 + surface_j.data() * conj(surface_j.data()));
        omega.data() =
            0.5 * sqrt(get(gauge_d).data() * conj(get(gauge_d).data()) -
                       get(gauge_c).data() * conj(get(gauge_c).data()));
        j_hat.data() = 0.25 *
                       (square(conj(get(gauge_d).data())) * surface_j.data() +
                        square(get(gauge_c).data()) * conj(surface_j.data()) +
                        2.0 * get(gauge_c).data() * conj(get(gauge_d).data()) *
                            interpolated_k.data()) /
                       square(omega.data());
        return max(abs(j_hat.data()));
      };

  // the linearised step this generator used before, kept here as the reference
  size_t linearized_steps = 0;
  const auto linearized_step =
      [&residual, &j_hat, &interpolated_k, &omega, &linearized_steps](
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
              gauge_c_step,
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
              gauge_d_step,
          const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
          const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
          const Spectral::Swsh::SwshInterpolator& interpolator) {
        const double max_error = residual(gauge_c, gauge_d, interpolator);
        get(*gauge_c_step).data() =
            -0.5 * j_hat.data() * square(omega.data()) /
            (get(gauge_d).data() * interpolated_k.data());
        get(*gauge_d_step).data() = get(*gauge_c_step).data() *
                                    conj(get(gauge_c).data()) /
                                    conj(get(gauge_d).data());
        ++linearized_steps;
        return max_error;
      };

  // the closed-form root of the same condition
  size_t potential_steps = 0;
  const auto closed_form_target =
      [&residual, &surface_j, &interpolated_k, &potential_steps](
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
              gauge_c_target,
          const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
          const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
          const Spectral::Swsh::SwshInterpolator& interpolator) {
        const double max_error = residual(gauge_c, gauge_d, interpolator);
        get(*gauge_c_target).data() = -conj(get(gauge_d).data()) *
                                      surface_j.data() /
                                      (1.0 + interpolated_k.data());
        ++potential_steps;
        return max_error;
      };

  tnsr::i<DataVector, 3> linearized_coordinates{number_of_angular_points};
  tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>
      linearized_angular{number_of_angular_points};
  const double linearized_residual =
      InitializeJ::detail::iteratively_adapt_angular_coordinates(
          make_not_null(&linearized_coordinates),
          make_not_null(&linearized_angular), l_max, tolerance, max_steps,
          1.0e-2, linearized_step, true);

  tnsr::i<DataVector, 3> potential_coordinates{number_of_angular_points};
  tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>
      potential_angular{number_of_angular_points};
  const double potential_residual =
      InitializeJ::detail::adapt_angular_coordinates_via_potential(
          make_not_null(&potential_coordinates),
          make_not_null(&potential_angular), l_max, tolerance, max_steps,
          1.0e-2, closed_form_target, true);

  INFO("linearised: " << linearized_steps << " sweeps, residual "
                      << linearized_residual
                      << "; potential: " << potential_steps
                      << " passes, residual " << potential_residual);
  CHECK(linearized_residual < tolerance);
  CHECK(potential_residual < tolerance);

  // The two maps agree, but only up to the conformal freedom that the gauge
  // condition does not fix, and they discard that freedom in different places:
  // the linearised update inverts eth on a SPIN-1 field, whose kernel is
  // l < 1, while the potential update inverts it on a SPIN-2 field, whose
  // kernel is l < 2. The potential solve therefore cannot generate l = 1
  // coordinate content and the linearised one can, so the two can drift apart
  // by an l = 1 conformal transformation -- which is a change of BMS frame, not
  // a different solution of the same problem.
  //
  // An l = 1 potential displaces the Cartesian components through
  // Re(eta conj(eth x^i)), a product of two l = 1 spin-1 fields, so it shows up
  // in l = 0, 1 AND 2 of the difference. Everything at l >= 3 is genuinely the
  // non-conformal part of the map, and there the two agree at round-off.
  double conformal_part = 0.0;
  double non_conformal_part = 0.0;
  SpinWeighted<ComplexDataVector, 0> difference{number_of_angular_points};
  for (size_t i = 0; i < 3; ++i) {
    difference.data() =
        std::complex<double>(1.0, 0.0) *
        (linearized_coordinates.get(i) - potential_coordinates.get(i));
    conformal_part = std::max(conformal_part, max(abs(difference.data())));
    Spectral::Swsh::filter_swsh_boundary_quantity(make_not_null(&difference),
                                                  l_max, 3_st, l_max);
    non_conformal_part =
        std::max(non_conformal_part, max(abs(difference.data())));
  }
  INFO("map difference: " << conformal_part << " in total, "
                          << non_conformal_part << " with l < 3 removed");
  CHECK(non_conformal_part < tolerance);

  // and an order of magnitude fewer passes to get there
  CHECK(potential_steps * 10 < linearized_steps);
}

// The residual of the potential solve falls for a few passes and then creeps
// back up, and the stopping rule can only recognize that one pass after the
// fact. This checks that what comes back is the minimum itself -- the map as
// well as the number -- and not the pass the rule stopped on.
void test_potential_solve_stops_at_minimum() {
  const size_t l_max = 12;
  const size_t filter_max_l = 4;
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  // Same field as `test_angular_coordinate_solves_agree`: fixed seed, so the
  // pass at which the residual bottoms out does not move between runs.
  MAKE_GENERATOR(gen, 8675309);
  UniformCustomDistribution<double> dist{-1.0, 1.0};
  SpinWeighted<ComplexDataVector, 2> j_at_scri{number_of_angular_points};
  fill_with_random_values(make_not_null(&j_at_scri.data()), make_not_null(&gen),
                          make_not_null(&dist));
  Spectral::Swsh::filter_swsh_boundary_quantity(make_not_null(&j_at_scri),
                                                l_max, filter_max_l);
  j_at_scri.data() *= 1.0e-3 / max(abs(j_at_scri.data()));

  SpinWeighted<ComplexDataVector, 2> surface_j{number_of_angular_points};
  SpinWeighted<ComplexDataVector, 2> j_hat{number_of_angular_points};
  SpinWeighted<ComplexDataVector, 0> interpolated_k{number_of_angular_points};
  SpinWeighted<ComplexDataVector, 0> omega{number_of_angular_points};
  const auto closed_form_target =
      [&surface_j, &j_hat, &interpolated_k, &omega, &j_at_scri](
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
              gauge_c_target,
          const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
          const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
          const Spectral::Swsh::SwshInterpolator& interpolator) {
        interpolator.interpolate(make_not_null(&surface_j), j_at_scri);
        interpolated_k.data() =
            sqrt(1.0 + surface_j.data() * conj(surface_j.data()));
        omega.data() =
            0.5 * sqrt(get(gauge_d).data() * conj(get(gauge_d).data()) -
                       get(gauge_c).data() * conj(get(gauge_c).data()));
        j_hat.data() = 0.25 *
                       (square(conj(get(gauge_d).data())) * surface_j.data() +
                        square(get(gauge_c).data()) * conj(surface_j.data()) +
                        2.0 * get(gauge_c).data() * conj(get(gauge_d).data()) *
                            interpolated_k.data()) /
                       square(omega.data());
        get(*gauge_c_target).data() = -conj(get(gauge_d).data()) *
                                      surface_j.data() /
                                      (1.0 + interpolated_k.data());
        return max(abs(j_hat.data()));
      };

  // A tolerance no map can reach, so the solve is never cut short by success
  // and the stopping rule is the only thing under test.
  const double unreachable_tolerance = 1.0e-30;
  const size_t max_passes = 12;
  const double plateau_factor = 0.9;

  // Each prefix is a fresh solve from the identity map with the stopping rule
  // switched off, so `prefix(k)` is the map after exactly `k` updates.
  const auto prefix = [&closed_form_target, &unreachable_tolerance, &l_max,
                       number_of_angular_points](
                          const size_t steps,
                          const gsl::not_null<tnsr::i<DataVector, 3>*> coords) {
    tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>> angular{
        number_of_angular_points};
    size_t updates = std::numeric_limits<size_t>::max();
    const double residual =
        InitializeJ::detail::adapt_angular_coordinates_via_potential(
            coords, make_not_null(&angular), l_max, unreachable_tolerance,
            steps, 1.0e-2, closed_form_target, false,
            InitializeJ::detail::NoOpFinalize{}, true, 0.0, &updates, false);
    CHECK(updates == steps);
    return residual;
  };

  std::vector<double> prefix_residual(max_passes + 1);
  for (size_t steps = 0; steps <= max_passes; ++steps) {
    tnsr::i<DataVector, 3> coords{number_of_angular_points};
    prefix_residual[steps] = prefix(steps, make_not_null(&coords));
  }
  const size_t best_pass = static_cast<size_t>(
      std::min_element(prefix_residual.begin(), prefix_residual.end()) -
      prefix_residual.begin());
  const double best_residual = prefix_residual[best_pass];
  // The premise of the stopping rule: the residual has an interior minimum, so
  // running to the cap is worse than stopping.
  CHECK(best_pass < max_passes);

  tnsr::i<DataVector, 3> default_coords{number_of_angular_points};
  tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>> default_angular{
      number_of_angular_points};
  size_t default_updates = std::numeric_limits<size_t>::max();
  const double default_residual =
      InitializeJ::detail::adapt_angular_coordinates_via_potential(
          make_not_null(&default_coords), make_not_null(&default_angular),
          l_max, unreachable_tolerance, max_passes, 1.0e-2, closed_form_target,
          false, InitializeJ::detail::NoOpFinalize{}, true, plateau_factor,
          &default_updates, false);

  REQUIRE(default_updates <= max_passes);
  // The rule stops as soon as a pass improves the residual by less than
  // `plateau_factor`, so it can stop just short of the true minimum; what it
  // must return is the best of the passes it did evaluate.
  const auto evaluated =
      std::min_element(prefix_residual.begin(),
                       prefix_residual.begin() +
                           static_cast<std::ptrdiff_t>(default_updates) + 1);
  const size_t rewound_pass =
      static_cast<size_t>(evaluated - prefix_residual.begin());
  INFO("minimum " << best_residual << " at pass " << best_pass
                  << "; default stopping rule ran " << default_updates
                  << " passes and returned " << default_residual
                  << ", the pass-" << rewound_pass << " map");
  CHECK(default_residual == approx(*evaluated));
  // ... which is no worse than the pass it stopped on, i.e. the rewind happened
  CHECK(default_residual <= prefix_residual[default_updates]);
  // ... and lands within one pass of the true minimum, which is what the
  // stopping factor buys over stopping at the first slow pass.
  CHECK(default_residual <= best_residual / plateau_factor);

  // The map that comes back is the one that produced that residual, not merely
  // some map with the same residual.
  tnsr::i<DataVector, 3> rewound_coords{number_of_angular_points};
  prefix(rewound_pass, make_not_null(&rewound_coords));
  CHECK_ITERABLE_APPROX(default_coords, rewound_coords);
}

// `MaxIterations` is the budget for the whole angular solve, so the passes the
// potential stage spends come out of what the linearised sweeps may use.
template <typename DbTags>
void test_cauchy_second_order_iteration_budget(
    const gsl::not_null<db::DataBox<DbTags>*> box_to_initialize) {
  auto node_lock = Parallel::NodeLock{};
  db::mutate_apply<InitializeJ::CauchySecondOrder::return_tags,
                   InitializeJ::CauchySecondOrder::argument_tags>(
      InitializeJ::CauchySecondOrder{1.0e-14, 10, true, 1.0e-1,
                                     test_du_dr_j_interpolator()},
      box_to_initialize, make_not_null(&node_lock));
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.InitializeJ", "[Unit][Cce]") {
  // `CauchySecondOrder` holds an interpolator, so serializing it needs the
  // derived span interpolators registered.
  register_derived_classes_with_charm<intrp::SpanInterpolator>();
  MAKE_GENERATOR(generator);
  UniformCustomDistribution<size_t> sdist{5, 6};
  const size_t l_max = sdist(generator);
  const size_t number_of_radial_points = sdist(generator);

  using boundary_variables_tag = ::Tags::Variables<tmpl::push_back<
      InitializeJ::InverseCubic<true>::boundary_tags, Tags::PartiallyFlatGaugeC,
      Tags::PartiallyFlatGaugeD, Tags::PartiallyFlatGaugeOmega,
      Spectral::Swsh::Tags::Derivative<Tags::PartiallyFlatGaugeOmega,
                                       Spectral::Swsh::Tags::Eth>,
      Tags::EvolutionGaugeBoundaryValue<Tags::BondiJ>,
      Tags::EvolutionGaugeBoundaryValue<Tags::Dr<Tags::BondiJ>>,
      Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>,
      Tags::EvolutionGaugeBoundaryValue<Tags::BondiBeta>,
      Tags::BoundaryValue<Tags::BondiU>, Tags::BoundaryValue<Tags::BondiW>,
      Tags::BoundaryValue<Tags::BondiQ>,
      Tags::BoundaryValue<Tags::Du<Tags::BondiJ>>,
      Tags::BoundaryValue<Tags::Du<Tags::Dr<Tags::BondiJ>>>,
      Tags::BoundaryValue<Tags::Du<Tags::BondiR>>>>;
  using pre_swsh_derivatives_variables_tag = ::Tags::Variables<tmpl::list<
      Tags::BondiJ, Tags::Dy<Tags::BondiJ>, Tags::Dy<Tags::Dy<Tags::BondiJ>>,
      Tags::BondiK, Tags::BondiR, Tags::Integrand<Tags::BondiBeta>,
      Tags::BondiBeta, Tags::OneMinusY, Tags::Psi0>>;
  using tensor_variables_tag = ::Tags::Variables<tmpl::list<
      Tags::CauchyCartesianCoords, Tags::CauchyAngularCoords,
      Tags::PartiallyFlatCartesianCoords, Tags::PartiallyFlatAngularCoords>>;

  const size_t number_of_boundary_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  const size_t number_of_volume_points =
      number_of_boundary_points * number_of_radial_points;
  auto box_to_initialize = db::create<db::AddSimpleTags<
      boundary_variables_tag, pre_swsh_derivatives_variables_tag,
      tensor_variables_tag, Tags::LMax, Tags::NumberOfRadialPoints,
      Spectral::Swsh::Tags::SwshInterpolator<Tags::CauchyAngularCoords>>>(
      typename boundary_variables_tag::type{number_of_boundary_points},
      typename pre_swsh_derivatives_variables_tag::type{
          number_of_volume_points},
      typename tensor_variables_tag::type{number_of_boundary_points}, l_max,
      number_of_radial_points, Spectral::Swsh::SwshInterpolator{});

  // generate some random values for the boundary data. Mode magnitudes are
  // roughly representative of typical strains seen in simulations, and are of a
  // scale that can be fully solved by the iterative procedure used in the more
  // elaborate initial data generators.
  UniformCustomDistribution<double> dist(1.0e-5, 1.0e-4);
  db::mutate<Tags::BoundaryValue<Tags::BondiR>,
             Tags::BoundaryValue<Tags::BondiBeta>,
             Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>,
             Tags::BoundaryValue<Tags::BondiJ>>(
      [&generator, &dist, &l_max](
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
              boundary_r,
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
              boundary_beta,
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
              boundary_dr_j,
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
              boundary_j) {
        SpinWeighted<ComplexModalVector, 2> generated_modes{
            Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
        Spectral::Swsh::TestHelpers::generate_swsh_modes<2>(
            make_not_null(&generated_modes.data()), make_not_null(&generator),
            make_not_null(&dist), 1, l_max);

        get(*boundary_j) =
            Spectral::Swsh::inverse_swsh_transform(l_max, 1, generated_modes);
        Spectral::Swsh::filter_swsh_boundary_quantity(
            make_not_null(&get(*boundary_j)), l_max, l_max / 2);

        SpinWeighted<ComplexModalVector, 0> generated_r_modes{
            Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
        Spectral::Swsh::TestHelpers::generate_swsh_modes<0>(
            make_not_null(&generated_modes.data()), make_not_null(&generator),
            make_not_null(&dist), 1, l_max);

        get(*boundary_r) = Spectral::Swsh::inverse_swsh_transform(
                               l_max, 1, generated_r_modes) +
                           100.0;
        Spectral::Swsh::filter_swsh_boundary_quantity(
            make_not_null(&get(*boundary_r)), l_max, l_max / 2);
        get(*boundary_beta) =
            Spectral::Swsh::inverse_swsh_transform(l_max, 1, generated_r_modes);
        Spectral::Swsh::filter_swsh_boundary_quantity(
            make_not_null(&get(*boundary_r)), l_max, l_max / 2);

        get(*boundary_dr_j) = -get(*boundary_j) / get(*boundary_r);
      },
      make_not_null(&box_to_initialize));
  // The CauchySecondOrder generator additionally consumes the worldtube values
  // of U, W, Q, Du(J), Du(Dr(J)), and Du(R) to evaluate the H hypersurface
  // equation for dy^2 J. These enter through factors of the (large) worldtube
  // radius R, so to keep the resulting dy^2 J at the same (strain) scale as J -
  // and thus small enough for the angular coordinate solve to converge - they
  // are drawn from a correspondingly smaller distribution.
  UniformCustomDistribution<double> second_order_dist(1.0e-10, 1.0e-9);
  db::mutate<Tags::BoundaryValue<Tags::BondiU>,
             Tags::BoundaryValue<Tags::BondiW>,
             Tags::BoundaryValue<Tags::BondiQ>,
             Tags::BoundaryValue<Tags::Du<Tags::BondiJ>>,
             Tags::BoundaryValue<Tags::Du<Tags::Dr<Tags::BondiJ>>>,
             Tags::BoundaryValue<Tags::Du<Tags::BondiR>>>(
      [&generator, &second_order_dist, &l_max](
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
              boundary_u,
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
              boundary_w,
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
              boundary_q,
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
              boundary_du_j,
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
              boundary_du_dr_j,
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
              boundary_du_r) {
        assign_random_swsh_boundary_value<1>(
            boundary_u, make_not_null(&generator),
            make_not_null(&second_order_dist), l_max);
        assign_random_swsh_boundary_value<0>(
            boundary_w, make_not_null(&generator),
            make_not_null(&second_order_dist), l_max);
        assign_random_swsh_boundary_value<1>(
            boundary_q, make_not_null(&generator),
            make_not_null(&second_order_dist), l_max);
        assign_random_swsh_boundary_value<2>(
            boundary_du_j, make_not_null(&generator),
            make_not_null(&second_order_dist), l_max);
        assign_random_swsh_boundary_value<2>(
            boundary_du_dr_j, make_not_null(&generator),
            make_not_null(&second_order_dist), l_max);
        assign_random_swsh_boundary_value<0>(
            boundary_du_r, make_not_null(&generator),
            make_not_null(&second_order_dist), l_max);
      },
      make_not_null(&box_to_initialize));
  {
    INFO("Check inverse cubic initial data generator");
    test_initialize_j_inverse_cubic(make_not_null(&box_to_initialize), l_max,
                                    number_of_radial_points);
  }
  {
    INFO("Check zero nonsmooth initial data generator");
    test_initialize_j_zero_nonsmooth(make_not_null(&box_to_initialize), l_max,
                                     number_of_radial_points);
  }
  CHECK_THROWS_WITH(
      (test_zero_non_smooth_error(make_not_null(&box_to_initialize), l_max,
                                  number_of_radial_points)),
      Catch::Matchers::ContainsSubstring(
          "Initial data iterative angular solve"));
  {
    INFO("Check no incoming radiation initial data generator");
    test_initialize_j_no_radiation(make_not_null(&box_to_initialize), l_max,
                                   number_of_radial_points);
  }
  {
    INFO("Check conformal factor initial data generator");
    test_initialize_j_conformal_factor(
        make_not_null(&box_to_initialize), false, false, false, false,
        ::Cce::InitializeJ::ConformalFactorIterationHeuristic::
            SpinWeight1CoordPerturbation,
        l_max, number_of_radial_points);
    test_initialize_j_conformal_factor(
        make_not_null(&box_to_initialize), false, true, false, false,
        ::Cce::InitializeJ::ConformalFactorIterationHeuristic::OnlyVaryGaugeD,
        l_max, number_of_radial_points);
    test_initialize_j_conformal_factor(
        make_not_null(&box_to_initialize), true, false, true, false,
        ::Cce::InitializeJ::ConformalFactorIterationHeuristic::
            SpinWeight1CoordPerturbation,
        l_max, number_of_radial_points);
    test_initialize_j_conformal_factor(
        make_not_null(&box_to_initialize), true, true, true, true,
        ::Cce::InitializeJ::ConformalFactorIterationHeuristic::
            SpinWeight1CoordPerturbation,
        l_max, number_of_radial_points);
  }
  {
    INFO("Check second-order initial data generator");
    test_initialize_j_cauchy_second_order(make_not_null(&box_to_initialize),
                                          l_max, number_of_radial_points);
  }
  {
    INFO("Check the potential-based angular solve against the linearised one");
    test_angular_coordinate_solves_agree();
  }
  {
    INFO("Check the potential solve returns its residual minimum");
    test_potential_solve_stops_at_minimum();
  }
  {
    INFO(
        "Check the second-order generator's Du(Dr(J)) interpolator survives "
        "cloning and serialization");
    test_cauchy_second_order_interpolator_round_trip();
  }
  CHECK_THROWS_WITH(test_cauchy_second_order_iteration_budget(
                        make_not_null(&box_to_initialize)),
                    Catch::Matchers::ContainsSubstring("did not reach"));
  CHECK_THROWS_WITH(test_cauchy_second_order_angular_solve_threshold(
                        make_not_null(&box_to_initialize)),
                    Catch::Matchers::ContainsSubstring(
                        "set by the MaxAngularSolveError option"));
  CHECK_THROWS_WITH(
      test_cauchy_second_order_asymptotic_j_error(
          make_not_null(&box_to_initialize)),
      Catch::Matchers::ContainsSubstring(
          "The asymptotic value of the initial J in Cauchy coordinates has "
          "magnitude"));
}
}  // namespace Cce
