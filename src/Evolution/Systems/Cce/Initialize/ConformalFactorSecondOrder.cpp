// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/Initialize/ConformalFactorSecondOrder.hpp"

#include <cstddef>
#include <memory>
#include <mutex>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/GaugeTransformBoundaryData.hpp"
#include "Evolution/Systems/Cce/Initialize/ComputeSecondOrderRadialDerivativeJ.hpp"
#include "Evolution/Systems/Cce/Initialize/ConformalFactor.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshDerivatives.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshFiltering.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshInterpolation.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTags.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce::InitializeJ {
namespace {
// Helper functions reused from ConformalFactor's anonymous namespace. They are
// local to that translation unit, so we re-declare them here.
void read_modes_from_input_file(
    const gsl::not_null<ComplexModalVector*> input_modes,
    const std::string& input_filename) {
  Matrix initial_j_modes;
  {
    h5::H5File<h5::AccessType::ReadOnly> cce_data_file{input_filename};
    auto& dat_file = cce_data_file.get<h5::Dat>("/InitialJ");
    initial_j_modes = dat_file.get_data();
    cce_data_file.close_current_object();
  }
  for (size_t i = 0; i < input_modes->size(); ++i) {
    (*input_modes)[i] = std::complex<double>(initial_j_modes(0, 2 * i),
                                             initial_j_modes(0, 2 * i + 1));
  }
}

void spin_weight_1_coord_perturbation_heuristic(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> gauge_c_step,
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> gauge_d_step,
    const SpinWeighted<ComplexDataVector, 0>& full_omega,
    const SpinWeighted<ComplexDataVector, 0>& omega_filtered,
    const SpinWeighted<ComplexDataVector, 0>& target_omega,
    const SpinWeighted<ComplexDataVector, 2>& /*gauge_c*/,
    const SpinWeighted<ComplexDataVector, 0>& gauge_d, const size_t l_max) {
  SpinWeighted<ComplexDataVector, 1> jacobian_supplement_f{full_omega.size()};
  gauge_d_step->data() = full_omega.data() *
                         (target_omega.data() - omega_filtered.data()) /
                         gauge_d.data();
  Spectral::Swsh::angular_derivatives<
      tmpl::list<Spectral::Swsh::Tags::InverseEthbar>>(
      l_max, 1, make_not_null(&jacobian_supplement_f), *gauge_d_step);
  Spectral::Swsh::angular_derivatives<tmpl::list<Spectral::Swsh::Tags::Eth>>(
      l_max, 1, gauge_c_step, jacobian_supplement_f);
}

void only_vary_gauge_d_heuristic(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> gauge_c_step,
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> gauge_d_step,
    const SpinWeighted<ComplexDataVector, 0>& full_omega,
    const SpinWeighted<ComplexDataVector, 0>& omega_filtered,
    const SpinWeighted<ComplexDataVector, 0>& target_omega,
    const SpinWeighted<ComplexDataVector, 2>& /*gauge_c*/,
    const SpinWeighted<ComplexDataVector, 0>& gauge_d,
    const size_t /*l_max*/) {
  gauge_d_step->data() = full_omega.data() *
                         (target_omega.data() - omega_filtered.data()) /
                         gauge_d.data();
  gauge_c_step->data() = 0;
}
}  // namespace

ConformalFactorSecondOrder::ConformalFactorSecondOrder(CkMigrateMessage* msg)
    : InitializeJ<false>(msg) {}

ConformalFactorSecondOrder::ConformalFactorSecondOrder(
    const double angular_coordinate_tolerance, const size_t max_iterations,
    const bool require_convergence, const bool optimize_l_0_mode,
    const bool use_beta_integral_estimate,
    const ::Cce::InitializeJ::ConformalFactorIterationHeuristic
        iteration_heuristic,
    const bool use_input_modes, std::string input_mode_filename)
    : angular_coordinate_tolerance_{angular_coordinate_tolerance},
      max_iterations_{max_iterations},
      require_convergence_{require_convergence},
      optimize_l_0_mode_{optimize_l_0_mode},
      use_beta_integral_estimate_{use_beta_integral_estimate},
      iteration_heuristic_{iteration_heuristic},
      use_input_modes_{use_input_modes},
      input_mode_filename_{std::move(input_mode_filename)} {}

ConformalFactorSecondOrder::ConformalFactorSecondOrder(
    const double angular_coordinate_tolerance, const size_t max_iterations,
    const bool require_convergence, const bool optimize_l_0_mode,
    const bool use_beta_integral_estimate,
    const ::Cce::InitializeJ::ConformalFactorIterationHeuristic
        iteration_heuristic,
    const bool use_input_modes,
    std::vector<std::complex<double>> input_modes)
    : angular_coordinate_tolerance_{angular_coordinate_tolerance},
      max_iterations_{max_iterations},
      require_convergence_{require_convergence},
      optimize_l_0_mode_{optimize_l_0_mode},
      use_beta_integral_estimate_{use_beta_integral_estimate},
      iteration_heuristic_{iteration_heuristic},
      use_input_modes_{use_input_modes},
      input_modes_{std::move(input_modes)} {}

std::unique_ptr<InitializeJ<false>> ConformalFactorSecondOrder::get_clone()
    const {
  return std::make_unique<ConformalFactorSecondOrder>(*this);
}

void ConformalFactorSecondOrder::operator()(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<
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
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& r, const size_t l_max,
    const size_t number_of_radial_points,
    const gsl::not_null<Parallel::NodeLock*> hdf5_lock) const {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  Variables<tmpl::list<::Tags::TempSpinWeightedScalar<0, 2>,
                       ::Tags::TempSpinWeightedScalar<1, 2>,
                       ::Tags::TempSpinWeightedScalar<2, 2>,
                       ::Tags::TempSpinWeightedScalar<3, 0>,
                       ::Tags::TempSpinWeightedScalar<4, 0>,
                       ::Tags::TempSpinWeightedScalar<5, 0>,
                       ::Tags::TempSpinWeightedScalar<6, 0>,
                       ::Tags::TempSpinWeightedScalar<7, 0>,
                       ::Tags::TempSpinWeightedScalar<8, 0>,
                       ::Tags::TempSpinWeightedScalar<9, 2>,
                       ::Tags::TempSpinWeightedScalar<10, 2>,
                       ::Tags::TempSpinWeightedScalar<11, 2>,
                       ::Tags::TempSpinWeightedScalar<12, 2>,
                       ::Tags::TempSpinWeightedScalar<13, 2>,
                       ::Tags::TempSpinWeightedScalar<14, 2>>>
      buffers{number_of_angular_points};
  auto& surface_j_buffer = get<::Tags::TempSpinWeightedScalar<0, 2>>(buffers);
  auto& surface_dr_j_buffer =
      get<::Tags::TempSpinWeightedScalar<1, 2>>(buffers);
  auto& input_j_buffer =
      get(get<::Tags::TempSpinWeightedScalar<2, 2>>(buffers));
  auto& gauge_omega = get<::Tags::TempSpinWeightedScalar<4, 0>>(buffers);
  auto& filtered_gauge_omega =
      get(get<::Tags::TempSpinWeightedScalar<5, 0>>(buffers));
  auto& target_omega = get(get<::Tags::TempSpinWeightedScalar<6, 0>>(buffers));
  auto& interpolated_target_gauge_omega =
      get(get<::Tags::TempSpinWeightedScalar<7, 0>>(buffers));
  auto& surface_r_buffer = get<::Tags::TempSpinWeightedScalar<8, 0>>(buffers);

  auto& one_minus_y_coefficient =
      get(get<::Tags::TempSpinWeightedScalar<9, 2>>(buffers));
  auto& one_minus_y_cubed_coefficient =
      get(get<::Tags::TempSpinWeightedScalar<10, 2>>(buffers));
  auto& one_minus_y_fourth_coefficient =
      get(get<::Tags::TempSpinWeightedScalar<11, 2>>(buffers));
  auto& one_minus_y_fifth_coefficient =
      get(get<::Tags::TempSpinWeightedScalar<12, 2>>(buffers));

  // dy^2_J in the Cauchy gauge (from `compute_dy_dy_j`) and after the gauge
  // transform to the partially flat coordinates.
  auto& cauchy_dr_dr_j_buffer =
      get<::Tags::TempSpinWeightedScalar<13, 2>>(buffers);
  auto& surface_dr_dr_j_buffer =
      get<::Tags::TempSpinWeightedScalar<14, 2>>(buffers);

  Variables<tmpl::list<::Tags::ModalTempSpinWeightedScalar<0, 2>,
                       ::Tags::ModalTempSpinWeightedScalar<1, 0>>>
      modal_buffers{Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
  auto& input_j_libsharp_modes =
      get(get<::Tags::ModalTempSpinWeightedScalar<0, 2>>(modal_buffers));
  auto& gauge_omega_transform_buffer =
      get(get<::Tags::ModalTempSpinWeightedScalar<1, 0>>(modal_buffers));

  SpinWeighted<ComplexModalVector, 2> goldberg_modes{square(l_max + 1)};
  if (use_input_modes_) {
    if (input_mode_filename_.has_value()) {
      const std::lock_guard hold_lock(*hdf5_lock);
      read_modes_from_input_file(make_not_null(&(goldberg_modes.data())),
                                 input_mode_filename_.value());
    } else {
      ASSERT(input_modes_.size() <= goldberg_modes.size(),
             "The size of the input modes is too large. Specify at most  "
             "(l_max + 1)^2 modes in the input file.");
      std::fill(goldberg_modes.data().begin(), goldberg_modes.data().end(),
                0.0);
      std::copy(input_modes_.begin(), input_modes_.end(),
                goldberg_modes.data().begin());
    }
    Spectral::Swsh::goldberg_to_libsharp_modes(
        make_not_null(&input_j_libsharp_modes), goldberg_modes, l_max);
    Spectral::Swsh::inverse_swsh_transform(
        l_max, 1_st, make_not_null(&input_j_buffer), input_j_libsharp_modes);
  }

  // Solve the H hypersurface equation at y = -1 for dy^2 J in the Cauchy
  // gauge, then convert to dr^2 J using
  //   dy^2 J = (R^2 / 4) * dr^2 J + (R / 2) * dr J
  // so that the gauge transform below (which is naturally expressed in terms
  // of dr derivatives) can be applied.
  Scalar<SpinWeighted<ComplexDataVector, 2>> cauchy_dy_dy_j{
      number_of_angular_points};
  compute_dy_dy_j(make_not_null(&cauchy_dy_dy_j), boundary_j, boundary_u,
                  boundary_w, boundary_beta, boundary_q, boundary_h,
                  boundary_dr_j, boundary_du_dy_j, boundary_du_r, r, l_max);
  get(cauchy_dr_dr_j_buffer).data() =
      4.0 * get(cauchy_dy_dy_j).data() / square(get(r).data()) -
      2.0 * get(boundary_dr_j).data() / get(r).data();

  // The asymptotic value of beta is beta|_scri = beta|_\Gamma + \int_-1^1 dy
  // \partial_y \beta, and this estimates the second term (see ConformalFactor
  // for the full explanation).
  target_omega.data() = exp(2.0 * get(boundary_beta).data());
  if (not use_input_modes_ and use_beta_integral_estimate_) {
    get(surface_j_buffer) = 0.25 * (3.0 * get(boundary_j).data() +
                                    get(r).data() * get(boundary_dr_j).data());
    target_omega.data() /= pow(1.0 + 4.0 * get(surface_j_buffer).data() *
                                         conj(get(surface_j_buffer).data()),
                               0.125);
  }

  void (*iteration_heuristic_function)(
      const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*>,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*>,
      const SpinWeighted<ComplexDataVector, 0>&,
      const SpinWeighted<ComplexDataVector, 0>&,
      const SpinWeighted<ComplexDataVector, 0>&,
      const SpinWeighted<ComplexDataVector, 2>&,
      const SpinWeighted<ComplexDataVector, 0>&, size_t) = nullptr;
  if (iteration_heuristic_ ==
      ::Cce::InitializeJ::ConformalFactorIterationHeuristic::
          SpinWeight1CoordPerturbation) {
    iteration_heuristic_function = &spin_weight_1_coord_perturbation_heuristic;
  } else if (iteration_heuristic_ ==
             ::Cce::InitializeJ::ConformalFactorIterationHeuristic::
                 OnlyVaryGaugeD) {
    iteration_heuristic_function = &only_vary_gauge_d_heuristic;
  } else {  // LCOV_EXCL_LINE
    // LCOV_EXCL_START
    ERROR("Unknown ConformalFactorIterationHeuristic");
    // LCOV_EXCL_STOP
  }

  auto iteration_function =
      [&iteration_heuristic_function, &filtered_gauge_omega, &gauge_omega,
       &target_omega, &interpolated_target_gauge_omega,
       &gauge_omega_transform_buffer, &l_max, &surface_r_buffer,
       &input_j_buffer, &r,
       this](const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
                 gauge_c_step,
             const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
                 gauge_d_step,
             const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
             const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
             const Spectral::Swsh::SwshInterpolator& iteration_interpolator) {
        get(gauge_omega).data() =
            0.5 * sqrt(get(gauge_d).data() * conj(get(gauge_d).data()) -
                       get(gauge_c).data() * conj(get(gauge_c).data()));
        iteration_interpolator.interpolate(
            make_not_null(&interpolated_target_gauge_omega), target_omega);
        if (use_input_modes_ and use_beta_integral_estimate_) {
          iteration_interpolator.interpolate(
              make_not_null(&get(surface_r_buffer)), get(r));
          get(surface_r_buffer).data() *= get(gauge_omega).data();
          interpolated_target_gauge_omega.data() /= pow(
              1.0 + real(input_j_buffer.data() * conj(input_j_buffer.data()) /
                         (square(get(surface_r_buffer).data()))),
              0.125);
        }
        filtered_gauge_omega = get(gauge_omega);
        if (not optimize_l_0_mode_) {
          Spectral::Swsh::filter_swsh_boundary_quantity(
              make_not_null(&filtered_gauge_omega), l_max, 1_st, l_max,
              make_not_null(&gauge_omega_transform_buffer));
          Spectral::Swsh::filter_swsh_boundary_quantity(
              make_not_null(&interpolated_target_gauge_omega), l_max, 1_st,
              l_max, make_not_null(&gauge_omega_transform_buffer));
        }
        double max_error = max(abs(filtered_gauge_omega.data() -
                                   interpolated_target_gauge_omega.data()));
        iteration_heuristic_function(make_not_null(&get(*gauge_c_step)),
                                     make_not_null(&get(*gauge_d_step)),
                                     get(gauge_omega), filtered_gauge_omega,
                                     interpolated_target_gauge_omega,
                                     get(gauge_c), get(gauge_d), l_max);
        return max_error;
      };

  auto finalize_function =
      [&gauge_omega, &l_max, &surface_dr_j_buffer, &boundary_dr_j, &boundary_j,
       &surface_j_buffer, &surface_r_buffer, &surface_dr_dr_j_buffer,
       &cauchy_dr_dr_j_buffer,
       &r](const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
           const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
           const tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>&
           /*angular_cauchy_coordinates*/,
           const Spectral::Swsh::SwshInterpolator& interpolator) {
        get(gauge_omega).data() =
            0.5 * sqrt(get(gauge_d).data() * conj(get(gauge_d).data()) -
                       get(gauge_c).data() * conj(get(gauge_c).data()));
        GaugeAdjustedBoundaryValue<Tags::Dr<Tags::BondiJ>>::apply(
            make_not_null(&surface_dr_j_buffer), boundary_dr_j, boundary_j,
            gauge_c, gauge_d, gauge_omega, interpolator, l_max);
        GaugeAdjustedBoundaryValue<Tags::BondiJ>::apply(
            make_not_null(&surface_j_buffer), boundary_j, gauge_c, gauge_d,
            gauge_omega, interpolator);
        GaugeAdjustedBoundaryValue<Tags::BondiR>::apply(
            make_not_null(&surface_r_buffer), r, gauge_omega, interpolator);
        GaugeAdjustedBoundaryValue<Tags::Dr<Tags::Dr<Tags::BondiJ>>>::apply(
            make_not_null(&surface_dr_dr_j_buffer), cauchy_dr_dr_j_buffer,
            boundary_dr_j, boundary_j, gauge_c, gauge_d, gauge_omega,
            interpolator, l_max);
      };

  detail::iteratively_adapt_angular_coordinates(
      cartesian_cauchy_coordinates, angular_cauchy_coordinates, l_max,
      angular_coordinate_tolerance_, max_iterations_, 1.0e-2,
      iteration_function, require_convergence_, finalize_function);

  // Convert the partially-flat-gauge dr^2 J back to dy^2 J at y = -1, where
  //   dy^2 J = (hat_R^2 / 4) * dr^2 J + (hat_R / 2) * dr J.
  SpinWeighted<ComplexDataVector, 2> hat_dy_dy_j{number_of_angular_points};
  hat_dy_dy_j = 0.25 * get(surface_r_buffer) * get(surface_r_buffer) *
                    get(surface_dr_dr_j_buffer) +
                0.5 * get(surface_r_buffer) * get(surface_dr_j_buffer);

  const DataVector one_minus_y_collocation =
      1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::GaussLobatto>(
                number_of_radial_points);
  if (not use_input_modes_) {
    // Three-term ansatz J(y) = A (1 - y) + B (1 - y)^3 + C (1 - y)^4 chosen
    // so that J(y=1) = 0 and J, dr_J, dy^2 J at y = -1 match the gauge-
    // transformed worldtube values.
    one_minus_y_coefficient =
        get(surface_j_buffer) +
        0.5 * get(surface_r_buffer) * get(surface_dr_j_buffer) +
        hat_dy_dy_j / 3.0;
    one_minus_y_cubed_coefficient =
        -0.25 * (get(surface_j_buffer) +
                 get(surface_r_buffer) * get(surface_dr_j_buffer) +
                 hat_dy_dy_j);
    one_minus_y_fourth_coefficient =
        get(surface_j_buffer) / 16.0 +
        get(surface_r_buffer) * get(surface_dr_j_buffer) / 16.0 +
        hat_dy_dy_j / 12.0;
    for (size_t i = 0; i < number_of_radial_points; ++i) {
      ComplexDataVector angular_view_j{
          get(*j).data().data() + get(boundary_j).size() * i,  // NOLINT
          get(boundary_j).size()};
      angular_view_j = one_minus_y_collocation[i] *
                           one_minus_y_coefficient.data() +
                       pow<3>(one_minus_y_collocation[i]) *
                           one_minus_y_cubed_coefficient.data() +
                       pow<4>(one_minus_y_collocation[i]) *
                           one_minus_y_fourth_coefficient.data();
    }
  } else {
    // Four-term ansatz: J(y) = alpha (1-y) + beta (1-y)^3 + gamma (1-y)^4
    //   + delta (1-y)^5, with alpha fixed by the asymptotic input modes and
    // beta, gamma, delta chosen to match J, dr_J, dy^2 J at the worldtube.
    one_minus_y_coefficient = 0.5 * input_j_buffer / get(surface_r_buffer);
    const SpinWeighted<ComplexDataVector, 2> a_factor =
        0.5 * get(surface_r_buffer) * get(surface_dr_j_buffer);
    one_minus_y_cubed_coefficient = 1.25 * get(surface_j_buffer) + a_factor +
                                    0.25 * hat_dy_dy_j -
                                    1.5 * one_minus_y_coefficient;
    one_minus_y_fourth_coefficient = one_minus_y_coefficient -
                                     0.9375 * get(surface_j_buffer) -
                                     0.875 * a_factor - 0.25 * hat_dy_dy_j;
    one_minus_y_fifth_coefficient =
        hat_dy_dy_j / 16.0 +
        0.1875 *
            (get(surface_j_buffer) + a_factor - one_minus_y_coefficient);
    for (size_t i = 0; i < number_of_radial_points; ++i) {
      ComplexDataVector angular_view_j{
          get(*j).data().data() + get(boundary_j).size() * i,  // NOLINT
          get(boundary_j).size()};
      angular_view_j = one_minus_y_collocation[i] *
                           one_minus_y_coefficient.data() +
                       pow<3>(one_minus_y_collocation[i]) *
                           one_minus_y_cubed_coefficient.data() +
                       pow<4>(one_minus_y_collocation[i]) *
                           one_minus_y_fourth_coefficient.data() +
                       pow<5>(one_minus_y_collocation[i]) *
                           one_minus_y_fifth_coefficient.data();
    }
  }
}

void ConformalFactorSecondOrder::pup(PUP::er& p) {
  p | angular_coordinate_tolerance_;
  p | max_iterations_;
  p | require_convergence_;
  p | optimize_l_0_mode_;
  p | use_beta_integral_estimate_;
  p | iteration_heuristic_;
  p | use_input_modes_;
  p | input_modes_;
  p | input_mode_filename_;
}

PUP::able::PUP_ID ConformalFactorSecondOrder::my_PUP_ID = 0;
}  // namespace Cce::InitializeJ
