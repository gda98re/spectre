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

namespace Cce {
namespace InitializeJ {

/// \cond
template <bool EvolveCcm>
struct NoIncomingRadiation;
/// \endcond

namespace NoIncomingRadiation_detail {
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
}  // namespace NoIncomingRadiation_detail

/*!
 * \brief Initialize \f$J\f$ on the first hypersurface by constraining
 * \f$\Psi_0 = 0\f$.
 *
 * \details This algorithm first radially evolves the \f$\Psi_0 = 0\f$
 * condition, which can be converted to a second-order radial ODE for J. Then,
 * the initial data generator performs an iterative solve for the angular
 * coordinates necessary to ensure asymptotic flatness. The parameters for the
 * iterative procedure are determined by options
 * `AngularCoordinateTolerance` and `MaxIterations`.
 */
template <>
struct NoIncomingRadiation<false> : InitializeJ<false> {
  using AngularCoordinateTolerance =
      NoIncomingRadiation_detail::AngularCoordinateTolerance;
  using MaxIterations = NoIncomingRadiation_detail::MaxIterations;
  using RequireConvergence = NoIncomingRadiation_detail::RequireConvergence;

  using options =
      tmpl::list<AngularCoordinateTolerance, MaxIterations, RequireConvergence>;
  static constexpr Options::String help = {
      "Initialization process where J is set so Psi0 is vanishing\n"
      "(roughly a no incoming radiation condition)"};

  WRAPPED_PUPable_decl_template(NoIncomingRadiation);  // NOLINT
  explicit NoIncomingRadiation(CkMigrateMessage* /*unused*/) {}

  NoIncomingRadiation(double angular_coordinate_tolerance,
                      size_t max_iterations, bool require_convergence = false);

  NoIncomingRadiation() = default;

  std::unique_ptr<InitializeJ> get_clone() const override;

  void operator()(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
      gsl::not_null<
          tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
          angular_cauchy_coordinates,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& beta, size_t l_max,
      size_t number_of_radial_points,
      gsl::not_null<Parallel::NodeLock*> hdf5_lock) const;

  void pup(PUP::er& p) override;

 private:
  bool require_convergence_ = false;
  double angular_coordinate_tolerance_ = 1.0e-10;
  size_t max_iterations_ = 300;
};

/*!
 * \brief Initialize \f$J\f$ on the first hypersurface by constraining
 * \f$\Psi_0 = 0\f$, additionally producing the inertial (partially flat)
 * coordinates required to evolve the partially flat Bondi-like coordinates
 * (`evolve_ccm = true`).
 *
 * \details Identical to `NoIncomingRadiation<false>` for the Cauchy angular
 * solve, but also computes the inertial angular coordinates that invert the
 * Cauchy angular-coordinate transformation, via
 * `detail::invert_angular_coordinates`.
 */
template <>
struct NoIncomingRadiation<true> : InitializeJ<true> {
  using AngularCoordinateTolerance =
      NoIncomingRadiation_detail::AngularCoordinateTolerance;
  using MaxIterations = NoIncomingRadiation_detail::MaxIterations;
  using RequireConvergence = NoIncomingRadiation_detail::RequireConvergence;

  using options =
      tmpl::list<AngularCoordinateTolerance, MaxIterations, RequireConvergence>;
  static constexpr Options::String help = {
      "Initialization process where J is set so Psi0 is vanishing\n"
      "(roughly a no incoming radiation condition)"};

  WRAPPED_PUPable_decl_template(NoIncomingRadiation);  // NOLINT
  explicit NoIncomingRadiation(CkMigrateMessage* /*unused*/) {}

  NoIncomingRadiation(double angular_coordinate_tolerance,
                      size_t max_iterations, bool require_convergence = false);

  NoIncomingRadiation() = default;

  std::unique_ptr<InitializeJ> get_clone() const override;

  void operator()(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
      gsl::not_null<
          tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
          angular_cauchy_coordinates,
      gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_inertial_coordinates,
      gsl::not_null<
          tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
          angular_inertial_coordinates,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& beta, size_t l_max,
      size_t number_of_radial_points,
      gsl::not_null<Parallel::NodeLock*> hdf5_lock) const;

  void pup(PUP::er& p) override;

 private:
  bool require_convergence_ = false;
  double angular_coordinate_tolerance_ = 1.0e-10;
  size_t max_iterations_ = 300;
};
}  // namespace InitializeJ
}  // namespace Cce
