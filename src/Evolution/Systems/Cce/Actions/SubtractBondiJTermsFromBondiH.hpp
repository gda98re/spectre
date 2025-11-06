// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Cce {
namespace Actions {

/*!
 * \ingroup ActionsGroup
 * \brief Subtracts J(y=1) + d²J/dy²(y=1) * (1-y)² from BondiH
 *
 * \details This action modifies BondiH by subtracting specific terms involving
 * BondiJ evaluated at the innermost radial point (y=1). The subtraction is:
 *
 * \f$H \rightarrow H - J(y=1) - \frac{d^2 J}{dy^2}(y=1) (1-y)^2\f$
 *
 * where y are the radial collocation points.
 *
 * Uses:
 * - DataBox:
 *   - `Cce::Tags::LMax`
 *   - `Cce::Tags::BondiH`
 *   - `Cce::Tags::BondiJ`
 *   - `Cce::Tags::Dy<Cce::Tags::Dy<Cce::Tags::BondiJ>>`
 *   - `Cce::Tags::OneMinusY`
 *
 * \ref DataBoxGroup changes:
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies: `Cce::Tags::BondiH`
 */
struct SubtractBondiJTermsFromBondiH {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const size_t l_max = db::get<Tags::LMax>(box);
    const size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);

    db::mutate<Tags::BondiH>(
        [&number_of_angular_points](
            const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
                bondi_h,
            const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
            const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_dy_bondi_j,
            const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y) {
          // Extract J and d²J/dy² at y=1 (first radial point, indices 0 to
          // number_of_angular_points-1)
          const SpinWeighted<ComplexDataVector, 2> j_at_y_equals_1{
              {get(bondi_j).data().data(), number_of_angular_points}};
          const SpinWeighted<ComplexDataVector, 2> dy_dy_j_at_y_equals_1{
              {get(dy_dy_bondi_j).data().data(), number_of_angular_points}};

          // Compute the subtraction for all radial points
          const size_t number_of_radial_points =
              get(*bondi_h).size() / number_of_angular_points;

          for (size_t radial_index = 0; radial_index < number_of_radial_points;
               ++radial_index) {
            // Get view of angular data at this radial point
            ComplexDataVector angular_view_h{
                get(*bondi_h).data().data() +
                    number_of_angular_points * radial_index,
                number_of_angular_points};
            ComplexDataVector angular_view_one_minus_y{
                get(one_minus_y).data().data() +
                    number_of_angular_points * radial_index,
                number_of_angular_points};

            // Compute (1-y)²
            const ComplexDataVector one_minus_y_squared =
                square(angular_view_one_minus_y);

            // Subtract J(y=1) + d²J/dy²(y=1) * (1-y)²
            angular_view_h -= j_at_y_equals_1.data();
            angular_view_h -= dy_dy_j_at_y_equals_1.data() * one_minus_y_squared;
          }
        },
        make_not_null(&box), db::get<Tags::BondiJ>(box),
        db::get<Tags::Dy<Tags::Dy<Tags::BondiJ>>>(box),
        db::get<Tags::OneMinusY>(box));

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

}  // namespace Actions
}  // namespace Cce
