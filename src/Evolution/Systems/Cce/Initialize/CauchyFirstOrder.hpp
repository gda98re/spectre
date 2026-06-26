// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
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
 * \brief Initialize \f$J\f$ on the first hypersurface using a first-order
 * matching at the worldtube.
 *
 * \details The volume \f$J\f$ is built as the linear polynomial in
 * \f$(1 - y)\f$
 *
 * \f{align}{
 *   J = A + B (1 - y), \quad A = J|_\Gamma + R\, \partial_r J|_\Gamma, \quad
 *   B = -\tfrac{1}{2} R\, \partial_r J|_\Gamma,
 * \f}
 *
 * that matches the worldtube values of \f$J\f$ and \f$\partial_r J\f$. This is
 * the `CauchySecondOrder` ansatz with the \f$\partial_y^2 J\f$ contribution
 * dropped, so it uses only \f$J\f$ and \f$\partial_r J\f$ (and the worldtube
 * radius \f$R\f$) and never solves the H hypersurface equation. The remaining
 * angular coordinates are determined iteratively to make \f$J\f$ vanish at
 * scri+, exactly as in `NoIncomingRadiation` and `CauchySecondOrder`.
 */
struct CauchyFirstOrder : InitializeJ<false> {
  struct AngularCoordinateTolerance {
    using type = double;
    static std::string name() { return "AngularCoordTolerance"; }
    static constexpr Options::String help = {
        "Tolerance of initial angular coordinates for CCE"};
    static type lower_bound() { return 1.0e-14; }
    static type upper_bound() { return 1.0e-3; }
    static type suggested_value() { return 1.0e-12; }
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

  using options =
      tmpl::list<AngularCoordinateTolerance, MaxIterations, RequireConvergence>;
  static constexpr Options::String help = {
      "First-order initial data generator for the Cauchy CCE evolution."};

  WRAPPED_PUPable_decl_template(CauchyFirstOrder);  // NOLINT
  explicit CauchyFirstOrder(CkMigrateMessage* /*unused*/) {}

  CauchyFirstOrder(double angular_coordinate_tolerance, size_t max_iterations,
                   bool require_convergence);

  CauchyFirstOrder() = default;

  std::unique_ptr<InitializeJ> get_clone() const override;

  using return_tags = tmpl::list<Tags::BondiJ, Tags::CauchyCartesianCoords,
                                 Tags::CauchyAngularCoords>;
  using argument_tags = tmpl::list<Tags::BoundaryValue<Tags::BondiJ>,
                                   Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>,
                                   Tags::BoundaryValue<Tags::BondiR>,
                                   Tags::LMax, Tags::NumberOfRadialPoints>;

  void operator()(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
      gsl::not_null<
          tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
          angular_cauchy_coordinates,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r, size_t l_max,
      size_t number_of_radial_points,
      gsl::not_null<Parallel::NodeLock*> hdf5_lock) const;

  void pup(PUP::er& p) override;

 private:
  bool require_convergence_ = false;
  double angular_coordinate_tolerance_ =
      std::numeric_limits<double>::signaling_NaN();
  size_t max_iterations_ = 0;
};
}  // namespace Cce::InitializeJ
