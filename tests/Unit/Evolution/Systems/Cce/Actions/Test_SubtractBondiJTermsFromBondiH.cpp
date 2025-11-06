// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <utility>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/Actions/SubtractBondiJTermsFromBondiH.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/Components/CharacteristicEvolution.hpp"
#include "Evolution/Systems/Cce/IntegrandInputSteps.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "Parallel/Phase.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {

namespace {
template <typename Metavariables>
struct mock_characteristic_evolution {
  using component_being_mocked = CharacteristicEvolution<Metavariables>;
  using replace_these_simple_actions = tmpl::list<>;
  using with_these_simple_actions = tmpl::list<>;

  using simple_tags =
      db::AddSimpleTags<Tags::LMax, Tags::NumberOfRadialPoints,
                        ::Tags::Variables<tmpl::list<
                            Tags::BondiH, Tags::BondiJ,
                            Tags::Dy<Tags::Dy<Tags::BondiJ>>, Tags::OneMinusY>>>;
  using compute_tags = db::AddComputeTags<>;

  using initialize_action_list =
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags, compute_tags>>;
  using simple_tags_from_options =
      Parallel::get_simple_tags_from_options<initialize_action_list>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             initialize_action_list>,
      Parallel::PhaseActions<
          Parallel::Phase::Evolve,
          tmpl::list<Actions::SubtractBondiJTermsFromBondiH>>>;
  using const_global_cache_tags =
      Parallel::get_const_global_cache_tags_from_actions<
          phase_dependent_action_list>;
};

struct metavariables {
  using component_list =
      tmpl::list<mock_characteristic_evolution<metavariables>>;
};
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Cce.Actions.SubtractBondiJTermsFromBondiH",
    "[Unit][Cce]") {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<size_t> sdist{7, 10};
  const size_t l_max = sdist(gen);
  UniformCustomDistribution<double> value_distribution{-2.0, 2.0};
  const size_t number_of_radial_points = 5;
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  const size_t number_of_grid_points =
      number_of_angular_points * number_of_radial_points;

  CAPTURE(l_max);

  // Create test data
  Scalar<SpinWeighted<ComplexDataVector, 2>> bondi_h;
  Scalar<SpinWeighted<ComplexDataVector, 2>> bondi_j;
  Scalar<SpinWeighted<ComplexDataVector, 2>> dy_dy_bondi_j;
  Scalar<SpinWeighted<ComplexDataVector, 0>> one_minus_y;

  get(bondi_h).data() = make_with_random_values<ComplexDataVector>(
      make_not_null(&gen), make_not_null(&value_distribution),
      number_of_grid_points);
  get(bondi_j).data() = make_with_random_values<ComplexDataVector>(
      make_not_null(&gen), make_not_null(&value_distribution),
      number_of_grid_points);
  get(dy_dy_bondi_j).data() = make_with_random_values<ComplexDataVector>(
      make_not_null(&gen), make_not_null(&value_distribution),
      number_of_grid_points);

  // Set up OneMinusY using collocation points
  const DataVector one_minus_y_collocation =
      1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                          Spectral::Quadrature::GaussLobatto>(
                number_of_radial_points);
  get(one_minus_y).data() =
      ComplexDataVector{number_of_grid_points, 0.0};
  for (size_t i = 0; i < number_of_radial_points; ++i) {
    ComplexDataVector angular_view{
        get(one_minus_y).data().data() + number_of_angular_points * i,
        number_of_angular_points};
    angular_view = one_minus_y_collocation[i];
  }

  // Store original H for comparison
  const auto original_h = bondi_h;

  // Extract J and d²J/dy² at y=1 (first radial point)
  const SpinWeighted<ComplexDataVector, 2> j_at_y_equals_1{
      {get(bondi_j).data().data(), number_of_angular_points}};
  const SpinWeighted<ComplexDataVector, 2> dy_dy_j_at_y_equals_1{
      {get(dy_dy_bondi_j).data().data(), number_of_angular_points}};

  // Compute expected result
  Scalar<SpinWeighted<ComplexDataVector, 2>> expected_h = original_h;
  for (size_t radial_index = 0; radial_index < number_of_radial_points;
       ++radial_index) {
    ComplexDataVector angular_view_h{
        get(expected_h).data().data() + number_of_angular_points * radial_index,
        number_of_angular_points};
    ComplexDataVector angular_view_one_minus_y{
        get(one_minus_y).data().data() +
            number_of_angular_points * radial_index,
        number_of_angular_points};

    const ComplexDataVector one_minus_y_squared =
        square(angular_view_one_minus_y);

    angular_view_h -= j_at_y_equals_1.data();
    angular_view_h -= dy_dy_j_at_y_equals_1.data() * one_minus_y_squared;
  }

  // Set up the component and run the action
  Variables<tmpl::list<Tags::BondiH, Tags::BondiJ,
                       Tags::Dy<Tags::Dy<Tags::BondiJ>>, Tags::OneMinusY>>
      component_variables{number_of_grid_points};
  get<Tags::BondiH>(component_variables) = bondi_h;
  get<Tags::BondiJ>(component_variables) = bondi_j;
  get<Tags::Dy<Tags::Dy<Tags::BondiJ>>>(component_variables) = dy_dy_bondi_j;
  get<Tags::OneMinusY>(component_variables) = one_minus_y;

  using component = mock_characteristic_evolution<metavariables>;
  ActionTesting::MockRuntimeSystem<metavariables> runner{
      tuples::tagged_tuple_from_typelist<
          Parallel::get_const_global_cache_tags<metavariables>>{}};

  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0, {l_max, number_of_radial_points,
                   std::move(component_variables)});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Evolve);
  ActionTesting::next_action<component>(make_not_null(&runner), 0);

  const auto& result_h =
      ActionTesting::get_databox_tag<component, Tags::BondiH>(runner, 0);

  CHECK_ITERABLE_APPROX(get(result_h), get(expected_h));
}
}  // namespace Cce
