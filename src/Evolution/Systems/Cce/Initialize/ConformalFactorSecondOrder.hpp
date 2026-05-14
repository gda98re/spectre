// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/Initialize/ConformalFactorIterationHeuristic.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Options/Auto.hpp"
#include "Options/Options.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class ComplexDataVector;
/// \endcond

namespace Cce {
namespace InitializeJ {

/*!
 * \brief Like `ConformalFactor`, but matches the transformed worldtube
 * variables to second order in \f$(1 - y)\f$ on the initial hypersurface.
 *
 * \details The angular-coordinate iteration is identical to `ConformalFactor`
 * (selecting an angular conformal factor that compensates the worldtube value
 * of \f$\beta\f$). After the iteration converges, the worldtube values of
 * \f$\hat J\f$, \f$\partial_{\hat r} \hat J\f$ and
 * \f$\partial_{\hat r}^2 \hat J\f$ are computed in the partially flat gauge,
 * the latter obtained from the H hypersurface equation via
 * `compute_dy_dy_j` and gauge-transformed using
 * `GaugeAdjustedBoundaryValue<Tags::Dr<Tags::Dr<Tags::BondiJ>>>`. A polynomial
 * ansatz in \f$(1 - y)\f$ is then chosen so that \f$\hat J\f$ vanishes at
 * \f$\mathscr{I}^+\f$ and matches the worldtube values of \f$\hat J\f$,
 * \f$\partial_{\hat r} \hat J\f$ and \f$\partial_{\hat r}^2 \hat J\f$ at
 * \f$y = -1\f$.
 */
struct ConformalFactorSecondOrder : InitializeJ<false> {
  struct AngularCoordinateTolerance {
    using type = double;
    static std::string name() { return "AngularCoordTolerance"; }
    static constexpr Options::String help = {
        "Tolerance of initial angular coordinates for CCE"};
    static type lower_bound() { return 1.0e-14; }
    static type upper_bound() { return 1.0e-3; }
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
  struct OptimizeL0Mode {
    using type = bool;
    static constexpr Options::String help = {
        "If true, the average value of the conformal factor will be included "
        "during optimization; otherwise it will be omitted (filtered)."};
    static type suggested_value() { return false; }
  };
  struct UseBetaIntegralEstimate {
    using type = bool;
    static constexpr Options::String help = {
        "If true, the iterative algorithm will calculate an estimate of the "
        "asymptotic beta value using the 1/r part of the initial J."};
    static type suggested_value() { return true; }
  };
  struct ConformalFactorIterationHeuristic {
    using type = ::Cce::InitializeJ::ConformalFactorIterationHeuristic;
    static constexpr Options::String help = {
        "The heuristic method used to set the spin-weighted Jacobian factors "
        "when iterating to minimize the asymptotic conformal factor."};
    static type suggested_value() {
      return ::Cce::InitializeJ::ConformalFactorIterationHeuristic::
          SpinWeight1CoordPerturbation;
    }
  };
  struct UseInputModes {
    using type = bool;
    static constexpr Options::String help = {
        "If true, the 1/r part of J will be set using modes read from the "
        "input file, or from a specified h5 file. If false, the second-order "
        "ansatz determines the 1/r part of J."};
  };
  struct InputModesFromFile {
    using type = std::string;
    static constexpr Options::String help = {
        "A filename from which to retrieve a set of modes (from InitialJ.dat) "
        "to use to determine the 1/r part of J on the initial hypersurface. "
        "The modes are parsed in l-ascending, m-ascending, m-varies-fastest, "
        "real then imaginary part order."};
  };
  struct InputModes {
    using type = std::vector<std::complex<double>>;
    static constexpr Options::String help = {
        "An explicit list of modes to use to set the 1/r part of J on the "
        "initial hypersurface. They are parsed in l-ascending, m-ascending, "
        "m-varies-fastest order."};
  };

  using options =
      tmpl::list<AngularCoordinateTolerance, MaxIterations, RequireConvergence,
                 OptimizeL0Mode, UseBetaIntegralEstimate,
                 ConformalFactorIterationHeuristic, UseInputModes,
                 Options::Alternatives<tmpl::list<InputModesFromFile>,
                                       tmpl::list<InputModes>>>;
  static constexpr Options::String help = {
      "Like ConformalFactor, but matches the transformed worldtube variables "
      "to second order on the initial hypersurface."};

  WRAPPED_PUPable_decl_template(ConformalFactorSecondOrder);  // NOLINT
  explicit ConformalFactorSecondOrder(CkMigrateMessage* msg);

  ConformalFactorSecondOrder() = default;
  ConformalFactorSecondOrder(
      double angular_coordinate_tolerance, size_t max_iterations,
      bool require_convergence, bool optimize_l_0_mode,
      bool use_beta_integral_estimate,
      ::Cce::InitializeJ::ConformalFactorIterationHeuristic iteration_heuristic,
      bool use_input_modes, std::string input_mode_filename);

  ConformalFactorSecondOrder(
      double angular_coordinate_tolerance, size_t max_iterations,
      bool require_convergence, bool optimize_l_0_mode,
      bool use_beta_integral_estimate,
      ::Cce::InitializeJ::ConformalFactorIterationHeuristic iteration_heuristic,
      bool use_input_modes, std::vector<std::complex<double>> input_modes);

  std::unique_ptr<InitializeJ> get_clone() const override;

  // Per-class tag lists. Like CauchySecondOrder, this generator needs the
  // additional worldtube boundary values that feed `compute_dy_dy_j`.
  using return_tags = tmpl::list<Tags::BondiJ, Tags::CauchyCartesianCoords,
                                 Tags::CauchyAngularCoords>;
  using argument_tags = tmpl::list<
      Tags::BoundaryValue<Tags::BondiJ>, Tags::BoundaryValue<Tags::BondiU>,
      Tags::BoundaryValue<Tags::BondiW>, Tags::BoundaryValue<Tags::BondiBeta>,
      Tags::BoundaryValue<Tags::BondiQ>, Tags::BoundaryValue<Tags::BondiH>,
      Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>,
      Tags::BoundaryValue<Tags::Du<Tags::Dy<Tags::BondiJ>>>,
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
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_h,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_du_dy_j,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_du_r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r, size_t l_max,
      size_t number_of_radial_points,
      gsl::not_null<Parallel::NodeLock*> hdf5_lock) const;

  void pup(PUP::er& p) override;

 private:
  double angular_coordinate_tolerance_ = 1.0e-11;
  size_t max_iterations_ = 300;
  bool require_convergence_ = true;
  bool optimize_l_0_mode_ = false;
  bool use_beta_integral_estimate_ = true;
  ::Cce::InitializeJ::ConformalFactorIterationHeuristic iteration_heuristic_ =
      ::Cce::InitializeJ::ConformalFactorIterationHeuristic::
          SpinWeight1CoordPerturbation;
  bool use_input_modes_ = false;
  std::vector<std::complex<double>> input_modes_;
  std::optional<std::string> input_mode_filename_;
};
}  // namespace InitializeJ
}  // namespace Cce
