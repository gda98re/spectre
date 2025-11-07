// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Cce {  // NOLINT
namespace Actions {

/*!
 * \ingroup ActionsGroup
 * \brief Subtracts gamma0 * J(y=1) + 1/2 * gamma2 * d²J/dy²(y=1) * (1-y)² from
 * BondiH
 *
 * \details This action modifies BondiH by subtracting terms proportional to the
 * constraints, needed for constraint dumping of the constant and quadratic term
 * of the asymptotic expansion of J about scri+.
 *
 * \f$H \rightarrow H - \gamma_0 J(y=1) - \frac{1}{2} \gamma_2 \frac{d^2
 * J}{dy^2}(y=1) (1-y)^2\f$
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
struct PartiallyFlatConstraintsDumping {
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
            const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> h,
            const Scalar<SpinWeighted<ComplexDataVector, 2>>& j,
            const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_dy_j) {
          // Extract J and d²J/dy² at y=1 (first radial point, indices 0 to
          // number_of_angular_points-1)

          // Dumping constant gamma0 and gamma2

          const double gamma0 = 0.5;
          const double gamma2 = 0.5;
          Parallel::printf("Dumping constant terms: gamma0: %e, gamma2: %e\n",
                           gamma0, gamma2);

          const size_t number_of_radial_points =
              get(j).size() / number_of_angular_points;

          const DataVector one_minus_y_collocation =
              1.0 -
              Spectral::collocation_points<Spectral::Basis::Legendre,
                                           Spectral::Quadrature::GaussLobatto>(
                  number_of_radial_points);

          // Computes angular views of BondiJ and its second derivatives at
          // scri+
          const SpinWeighted<ComplexDataVector, 2> j_at_scri_view;
          make_const_view(
              make_not_null(&j_at_scri_view), get(j),
              (number_of_radial_points - 1) * number_of_angular_points,
              number_of_angular_points);

          const SpinWeighted<ComplexDataVector, 2> dy_dy_j_at_scri_view;
          make_const_view(
              make_not_null(&dy_dy_j_at_scri_view), get(dy_dy_j),
              (number_of_radial_points - 1) * number_of_angular_points,
              number_of_angular_points);

          for (size_t i = 0; i < number_of_radial_points; i++) {
            ComplexDataVector angular_view_h{
                get(*h).data().data() + j_at_scri_view.size() * i,  // NOLINT
                j_at_scri_view.size()};
            // auto is acceptable here as these two values are only used once in
            // the below computation. `auto` causes an expression template to be
            // generated, rather than allocating.
            angular_view_h -= gamma0 * j_at_scri_view.data();
            angular_view_h -= 0.5 * gamma2 *
                              square(one_minus_y_collocation[i]) *
                              dy_dy_j_at_scri_view.data();

            if (i == 0) {
              Parallel::printf(
                  "value of constraint dumping terms at scri+: "
                  "gamma0 * J(y=1): %e, 0.5 * gamma2 * d²J/dy²(y=1): "
                  "%e\n",
                  max(abs(j_at_scri_view.data())),
                  max(abs(0.5 * dy_dy_j_at_scri_view.data())));
            }
          }
        },
        make_not_null(&box), db::get<Tags::BondiJ>(box),
        db::get<Tags::Dy<Tags::Dy<Tags::BondiJ>>>(box));

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

}  // namespace Actions
}  // namespace Cce
