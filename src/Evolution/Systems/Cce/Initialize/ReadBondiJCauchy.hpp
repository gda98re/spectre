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

/// \cond
class ComplexDataVector;
/// \endcond

namespace Cce {  // NOLINT
namespace InitializeJ {

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
struct ReadBondiJCauchy : InitializeJ<false> {
  struct H5Filename {
    using type = std::string;
    static constexpr Options::String help = {
        "A filename from which to retrieve a set of modes for each radial "
        "collocation point of J"};
  };

  struct SubfileNameJ {
    using type = std::string;
    static constexpr Options::String help = {
        "The subfile name inside the H5 file for BondiJCauchy"};
  };

  struct StartTime {
    using type = double;
    static constexpr Options::String help = {
        "Start time at which read the input volume h5 file"};
  };

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

  using options =
      tmpl::list<H5Filename, SubfileNameJ, StartTime,
                 AngularCoordinateTolerance, MaxIterations, RequireConvergence>;

  static constexpr Options::String help = {
      "Generate CCE initial reading J in Cauchy coordinates and computing the "
      "angular coordinate transformation to partially flat gauge"};

  WRAPPED_PUPable_decl_template(ReadBondiJCauchy);  // NOLINT
  explicit ReadBondiJCauchy(CkMigrateMessage* /*unused*/) {}

  ReadBondiJCauchy(std::string input_filename, std::string input_subfile_name_j,
                   double start_time, double angular_coordinate_tolerance,
                   size_t max_iterations, bool require_convergence);

  ReadBondiJCauchy() = default;

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
  std::string input_filename_ =
      "/home/fs01/spec1187/CCE_initial_data/q1nospin/original/Lev3/"
      "InverseCubic/StartTime0.0/CharacteristicExtractVolume_R0512.h5";
  std::string input_subfile_name_j_ = "CceVolumeData/VolumeData";
  double start_time_ = 0.0;
  bool require_convergence_ = true;
  double angular_coordinate_tolerance_ = 1.0e-14;
  size_t max_iterations_ = 1000;
};
}  // namespace InitializeJ
}  // namespace Cce
