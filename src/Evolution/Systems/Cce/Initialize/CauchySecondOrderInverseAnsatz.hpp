// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class ComplexDataVector;
/// \endcond

namespace Cce::InitializeJ {

/*!
 * \brief Initialize \f$J\f$ on the first hypersurface using a second-order
 * matching at the worldtube and an inverse radial ansatz.
 *
 * \details This is a variant of `CauchySecondOrder`. As in that generator, the
 * worldtube value of \f$\partial_y^2 J\f$ is obtained from the H hypersurface
 * equation (reusing `CauchySecondOrder_detail::compute_dy_dy_j`) and a
 * Cauchy-gauge volume \f$J\f$ is built that matches \f$J\f$,
 * \f$\partial_r J\f$,
 * and \f$\partial_y^2 J\f$ at the worldtube. That Cauchy-gauge \f$J\f$ is used
 * only to drive the angular-coordinate iteration and is then discarded. The
 * worldtube \f$J\f$, \f$\partial_r J\f$ (and, for the quartic ansatz,
 * \f$\partial_y^2 J\f$) are transformed into the evolution gauge and the volume
 * \f$J\f$ is rebuilt from a radial ansatz with no constant term, so that
 * \f$J\f$ vanishes at scri+ by construction:
 * - `UseQuarticAnsatz = false`: inverse-cubic
 *   \f$J = A (1 - y) + B (1 - y)^3\f$, matched to the transformed \f$J\f$ and
 *   \f$\partial_r J\f$ (identical to `ConformalFactor` with inverse-cubic
 *   data).
 * - `UseQuarticAnsatz = true`: inverse-quartic
 *   \f$J = A (1 - y) + B (1 - y)^3 + C (1 - y)^4\f$, additionally matched to
 *   the transformed \f$\partial_y^2 J\f$, which also forces
 *   \f$\partial_y^2 J\f$ to vanish at scri+.
 *
 * As a safeguard, the initialization aborts if the second radial derivative of
 * \f$J\f$ at scri+ of the final solution exceeds `MaxScriSecondDerivative`.
 */
struct CauchySecondOrderInverseAnsatz : InitializeJ<false> {
  struct AngularCoordinateTolerance {
    using type = double;
    static std::string name() { return "AngularCoordTolerance"; }
    static constexpr Options::String help = {
        "Tolerance of initial angular coordinates for CCE"};
    static type lower_bound() { return 1.0e-14; }
    static type upper_bound() { return 1.0e-3; }
    static type suggested_value() { return 1.0e-10; }
  };

  struct MaxIterations {
    using type = size_t;
    static constexpr Options::String help = {
        "Number of linearized inversion iterations."};
    static type lower_bound() { return 10; }
    static type upper_bound() { return 1000; }
    static type suggested_value() { return 300; }
  };

  struct RequireConvergence {
    using type = bool;
    static constexpr Options::String help = {
        "If true, initialization will error if it hits MaxIterations"};
    static type suggested_value() { return true; }
  };

  struct MaxScriSecondDerivative {
    using type = double;
    static constexpr Options::String help = {
        "Abort initialization if the largest second radial derivative of J at "
        "scri+ of the final initial data exceeds this threshold. The "
        "second-order construction drives this derivative to (near) zero, so a "
        "large value indicates a poorly matched solution. Set to a large value "
        "to effectively disable the check."};
    static type lower_bound() { return 1.0e-14; }
    static type upper_bound() { return 1.0e2; }
    static type suggested_value() { return 1.0e-6; }
  };

  struct UseQuarticAnsatz {
    using type = bool;
    static constexpr Options::String help = {
        "If true, rebuild the volume J from the inverse-quartic ansatz "
        "A(1-y) + B(1-y)^3 + C(1-y)^4 matched to the gauge-transformed J, "
        "dr_j, and dy^2 J. If false (default), use the inverse-cubic ansatz "
        "A(1-y) + B(1-y)^3 matched to the gauge-transformed J and dr_j only."};
    static type suggested_value() { return false; }
  };

  using options =
      tmpl::list<AngularCoordinateTolerance, MaxIterations, RequireConvergence,
                 MaxScriSecondDerivative, UseQuarticAnsatz>;
  static constexpr Options::String help = {
      "Second-order initial data generator for the Cauchy CCE evolution that "
      "rebuilds the volume J from an inverse radial ansatz."};

  WRAPPED_PUPable_decl_template(CauchySecondOrderInverseAnsatz);  // NOLINT
  explicit CauchySecondOrderInverseAnsatz(CkMigrateMessage* /*unused*/) {}

  CauchySecondOrderInverseAnsatz(double angular_coordinate_tolerance,
                                 size_t max_iterations,
                                 bool require_convergence,
                                 double max_scri_second_derivative,
                                 bool use_quartic_ansatz);

  CauchySecondOrderInverseAnsatz() = default;

  std::unique_ptr<InitializeJ> get_clone() const override;

  // Per-class tag lists. The flexible dispatch in `InitializeJ<false>` reads
  // these via `call_with_dynamic_type` so this generator can request more
  // worldtube boundary values than the simpler sibling classes do.
  using return_tags = tmpl::list<Tags::BondiJ, Tags::CauchyCartesianCoords,
                                 Tags::CauchyAngularCoords>;
  using argument_tags = tmpl::list<
      Tags::BoundaryValue<Tags::BondiJ>, Tags::BoundaryValue<Tags::BondiU>,
      Tags::BoundaryValue<Tags::BondiW>, Tags::BoundaryValue<Tags::BondiBeta>,
      Tags::BoundaryValue<Tags::BondiQ>,
      Tags::BoundaryValue<Tags::Du<Tags::BondiJ>>,
      Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>,
      Tags::BoundaryValue<Tags::Du<Tags::Dr<Tags::BondiJ>>>,
      Tags::BoundaryValue<Tags::Du<Tags::BondiR>>,
      Tags::BoundaryValue<Tags::BondiR>, Tags::LMax,
      Tags::NumberOfRadialPoints>;

  void operator()(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
      gsl::not_null<
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
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r, size_t l_max,
      size_t number_of_radial_points,
      gsl::not_null<Parallel::NodeLock*> hdf5_lock) const;

  void pup(PUP::er& p) override;

 private:
  bool require_convergence_ = true;
  double angular_coordinate_tolerance_ = 1.0e-14;
  size_t max_iterations_ = 1000;
  double max_scri_second_derivative_ = 1.0e-6;
  bool use_quartic_ansatz_ = false;
};
}  // namespace Cce::InitializeJ
