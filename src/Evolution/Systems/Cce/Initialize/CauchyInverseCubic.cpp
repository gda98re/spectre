// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/Initialize/CauchyInverseCubic.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/OdeIntegration/OdeIntegration.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCoefficients.hpp"
#include "Parallel/NodeLock.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce::InitializeJ {

CauchyInverseCubic::CauchyInverseCubic(
    std::string input_filename, std::string input_subfile_name_j,
    std::string input_subfile_name_coord, const double start_time,
    const bool get_j_from_file, const bool get_coord_from_file,
    const double angular_coordinate_tolerance, const size_t max_iterations,
    const bool require_convergence)
    : input_filename_{std::move(input_filename)},
      input_subfile_name_j_{std::move(input_subfile_name_j)},
      input_subfile_name_coord_{std::move(input_subfile_name_coord)},
      start_time_{start_time},
      get_j_from_file_{get_j_from_file},
      get_coord_from_file_{get_coord_from_file},
      require_convergence_{require_convergence},
      angular_coordinate_tolerance_{angular_coordinate_tolerance},
      max_iterations_{max_iterations} {}

std::unique_ptr<InitializeJ<false>> CauchyInverseCubic::get_clone() const {
  return std::make_unique<CauchyInverseCubic>(*this);
}

void CauchyInverseCubic::operator()(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& /*beta*/,
    const size_t l_max, const size_t number_of_radial_points,
    const gsl::not_null<Parallel::NodeLock*> /*hdf5_lock*/) const {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  const bool IL = false;

  // Initialize J in the original by reading CCM output

  // Reading volume data from "input_file_name_" h5
  // file path at the first observation id (initial time).
  if (get_j_from_file_) {
    Parallel::printf(
        "Initializing CCE Reading J from file: %s, subfile: %s at "
        "time: %f\n",
        input_filename_, input_subfile_name_j_, start_time_);
    {
      h5::H5File<h5::AccessType::ReadOnly> cce_data_file{input_filename_};
      auto& data_j = cce_data_file.get<h5::VolumeData>(input_subfile_name_j_);
      if (data_j.list_observation_ids().size() == 0) {
        ERROR("The observation IDs list is empty");
      }
      size_t target_obs_id_j = data_j.find_observation_id(start_time_);

      const auto& j_tensor_component =
          data_j.get_tensor_component(target_obs_id_j, "BondiJCauchy");

      if (not std::holds_alternative<DataVector>(j_tensor_component.data)) {
        ERROR("CCE initial J must be a DataVector");
      }

      // Construct a DataVector containing the real and imaginary parts of the
      // Goldberg modes of J for each radial collocation points

      const auto& modal_j_goldberg_interleaved =
          std::get<DataVector>(j_tensor_component.data);
      const size_t l_max_plus_one_squared = square(l_max + 1);
      if (modal_j_goldberg_interleaved.size() !=
          2 * number_of_radial_points * l_max_plus_one_squared) {
        ERROR(std::string(
            "Mismatch between l_max or number of radial points between "
            "J read from h5 and the input settings. j_data_size: "));
      }
      SpinWeighted<ComplexModalVector, 2> modal_j_goldberg{
          number_of_radial_points * l_max_plus_one_squared};
      for (size_t i = 0; i < modal_j_goldberg.size(); i++) {
        modal_j_goldberg.data()[i] =
            std::complex<double>(modal_j_goldberg_interleaved[2 * i],
                                 modal_j_goldberg_interleaved[(2 * i) + 1]);
      }

      Spectral::Swsh::goldberg_to_nodal(make_not_null(&get(*j)),
                                        modal_j_goldberg, l_max);
      cce_data_file.close();
    }
  } else {
    // Initialize J from J = A + B(1-y) + C(1-y)^3 matching, using boundary data
    // boundary_j, boundary_dr_j, r, and boundary_dy2_j (from ccm run) in Cauchy
    // coordinates
    // Read dy^2 J from file
    h5::H5File<h5::AccessType::ReadOnly> cce_data_file{input_filename_};
    auto& data_dy2j = cce_data_file.get<h5::VolumeData>(input_subfile_name_j_);
    if (data_dy2j.list_observation_ids().size() == 0) {
      ERROR("The observation IDs list is empty");
    }
    size_t target_obs_id_dy2j = data_dy2j.find_observation_id(start_time_);

    const auto& dy2j_tensor_component = data_dy2j.get_tensor_component(
        target_obs_id_dy2j, "Dy(Dy(BondiJCauchy))");

    if (not std::holds_alternative<DataVector>(dy2j_tensor_component.data)) {
      ERROR("CCE initial Dy(Dy(BondiJCauchy)) must be a DataVector");
    }

    // Construct a DataVector containing the real and imaginary parts of the
    // Goldberg modes of DyDyJ for each radial collocation points

    const auto& modal_dy2j_goldberg_interleaved =
        std::get<DataVector>(dy2j_tensor_component.data);
    const size_t l_max_plus_one_squared = square(l_max + 1);
    if (modal_dy2j_goldberg_interleaved.size() !=
        2 * number_of_radial_points * l_max_plus_one_squared) {
      ERROR(std::string(
          "Mismatch between l_max or number of radial points between "
          "J read from h5 and the input settings. j_data_size: "));
    }
    SpinWeighted<ComplexModalVector, 2> modal_dy2j_goldberg{
        number_of_radial_points * l_max_plus_one_squared};
    for (size_t i = 0; i < modal_dy2j_goldberg.size(); i++) {
      modal_dy2j_goldberg.data()[i] =
          std::complex<double>(modal_dy2j_goldberg_interleaved[2 * i],
                               modal_dy2j_goldberg_interleaved[(2 * i) + 1]);
    }

    SpinWeighted<ComplexDataVector, 2> volume_dy2_j{get(*j).data().size()};
    Spectral::Swsh::goldberg_to_nodal(make_not_null(&volume_dy2_j),
                                      modal_dy2j_goldberg, l_max);
    cce_data_file.close();

    // evaluate volume dy^2 j at the worldtube
    const SpinWeighted<ComplexDataVector, 2> boundary_dy2_j;
    make_const_view(make_not_null(&boundary_dy2_j), volume_dy2_j, 0,
                    number_of_angular_points);

    const DataVector one_minus_y_collocation =
        1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                           Spectral::Quadrature::GaussLobatto>(
                  number_of_radial_points);

    if (!IL) {
      Parallel::printf(
          "Initializing CCE Constructing J from MIC: %s, subfile: %s at "
          "time: %f\n",
          input_filename_, input_subfile_name_j_, start_time_);
      for (size_t i = 0; i < number_of_radial_points; i++) {
        ComplexDataVector angular_view_j{
            get(*j).data().data() + get(boundary_j).size() * i,  // NOLINT
            get(boundary_j).size()};
        // auto is acceptable here as these two values are only used once in the
        // below computation. `auto` causes an expression template to be
        // generated, rather than allocating.
        const auto constant_term = get(boundary_j).data() +
                                   get(r).data() * get(boundary_dr_j).data() +
                                   4 * boundary_dy2_j.data() / 3.0;
        const auto one_minus_y_coefficient =
            -(0.5 * get(r).data() * get(boundary_dr_j).data() +
              boundary_dy2_j.data());
        const auto one_minus_y_cubed_coefficient = boundary_dy2_j.data() / 12.;
        angular_view_j =
            constant_term +
            one_minus_y_collocation[i] * one_minus_y_coefficient +
            pow<3>(one_minus_y_collocation[i]) * one_minus_y_cubed_coefficient;
      }
    } else {
      Parallel::printf(
          "Initializing CCE Constructing J from MIL: %s, subfile: %s at "
          "time: %f\n",
          input_filename_, input_subfile_name_j_, start_time_);
      for (size_t i = 0; i < number_of_radial_points; i++) {
        ComplexDataVector angular_view_j{
            get(*j).data().data() + get(boundary_j).size() * i,  // NOLINT
            get(boundary_j).size()};
        // auto is acceptable here as these two values are only used once in the
        // below computation. `auto` causes an expression template to be
        // generated, rather than allocating.
        const auto constant_term =
            get(boundary_j).data() + get(r).data() * get(boundary_dr_j).data();
        const auto one_minus_y_coefficient =
            -0.5 * get(r).data() * get(boundary_dr_j).data();
        angular_view_j = constant_term +
                         one_minus_y_collocation[i] * one_minus_y_coefficient;
      }
    }
  }

  if (!get_coord_from_file_) {
    Parallel::printf(
        "Initializing CCE coordinates iteratively adjusting angular "
        "coordinates with "
        "tolerance: %1.1e, max iterations: %zu\n",
        angular_coordinate_tolerance_, max_iterations_);
    const SpinWeighted<ComplexDataVector, 2> j_at_scri_view;
    make_const_view(make_not_null(&j_at_scri_view), get(*j),
                    (number_of_radial_points - 1) * number_of_angular_points,
                    number_of_angular_points);

    Variables<tmpl::list<
        ::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                             std::integral_constant<int, 2>>,
        ::Tags::SpinWeighted<::Tags::TempScalar<1, ComplexDataVector>,
                             std::integral_constant<int, 0>>,
        ::Tags::SpinWeighted<::Tags::TempScalar<2, ComplexDataVector>,
                             std::integral_constant<int, 0>>>>
        iteration_buffers{number_of_angular_points};

    auto& evolution_gauge_surface_j =
        get(get<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                     std::integral_constant<int, 2>>>(
            iteration_buffers));
    auto& interpolated_k =
        get(get<::Tags::SpinWeighted<::Tags::TempScalar<1, ComplexDataVector>,
                                     std::integral_constant<int, 0>>>(
            iteration_buffers));
    auto& gauge_omega =
        get<::Tags::SpinWeighted<::Tags::TempScalar<2, ComplexDataVector>,
                                 std::integral_constant<int, 0>>>(
            iteration_buffers);

    // find a coordinate transformation such that in the new coordinates,
    // J is zero at scri+
    auto iteration_function =
        [&interpolated_k, &gauge_omega, &evolution_gauge_surface_j,
         &j_at_scri_view](
            const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
                gauge_c_step,
            const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
                gauge_d_step,
            const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
            const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
            const Spectral::Swsh::SwshInterpolator& iteration_interpolator) {
          iteration_interpolator.interpolate(
              make_not_null(&evolution_gauge_surface_j), j_at_scri_view);
          interpolated_k.data() =
              sqrt(1.0 + evolution_gauge_surface_j.data() *
                             conj(evolution_gauge_surface_j.data()));
          get(gauge_omega).data() =
              0.5 * sqrt(get(gauge_d).data() * conj(get(gauge_d).data()) -
                         get(gauge_c).data() * conj(get(gauge_c).data()));
          evolution_gauge_surface_j.data() =
              0.25 *
              (square(conj(get(gauge_d).data())) *
                   evolution_gauge_surface_j.data() +
               square(get(gauge_c).data()) *
                   conj(evolution_gauge_surface_j.data()) +
               2.0 * get(gauge_c).data() * conj(get(gauge_d).data()) *
                   interpolated_k.data()) /
              square(get(gauge_omega).data());

          double max_error = max(abs(evolution_gauge_surface_j.data()));

          // The alteration in each of the spin-weighted Jacobian factors
          // determined by linearizing the system in small J
          get(*gauge_c_step).data() =
              -0.5 * evolution_gauge_surface_j.data() *
              square(get(gauge_omega).data()) /
              (get(gauge_d).data() * interpolated_k.data());
          get(*gauge_d_step).data() = get(*gauge_c_step).data() *
                                      conj(get(gauge_c).data()) /
                                      conj(get(gauge_d).data());
          return max_error;
        };

    auto finalize_function =
        [&j, &gauge_omega, &l_max](
            const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
            const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
            const tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>&
                local_angular_cauchy_coordinates,
            const Spectral::Swsh::SwshInterpolator& interpolator) {
          get(gauge_omega).data() =
              0.5 * sqrt(get(gauge_d).data() * conj(get(gauge_d).data()) -
                         get(gauge_c).data() * conj(get(gauge_c).data()));

          GaugeAdjustInitialJ::apply(j, gauge_c, gauge_d, gauge_omega,
                                     local_angular_cauchy_coordinates,
                                     interpolator, l_max);
        };

    detail::iteratively_adapt_angular_coordinates(
        cartesian_cauchy_coordinates, angular_cauchy_coordinates, l_max,
        angular_coordinate_tolerance_, max_iterations_, 1.0e-2,
        iteration_function, require_convergence_, finalize_function);
  } else {
    // Read Cauchy Cartesian coordinates from file
    Parallel::printf(
        "Initializing CCE Reading Cauchy coordinates from file: %s, subfile: "
        "%s at time: %f\n",
        input_filename_, input_subfile_name_coord_, start_time_);
    {
      h5::H5File<h5::AccessType::ReadOnly> cce_data_file{input_filename_};
      auto& data_coord =
          cce_data_file.get<h5::VolumeData>(input_subfile_name_coord_);
      if (data_coord.list_observation_ids().size() == 0) {
        ERROR("The observation IDs list is empty");
      }
      size_t target_obs_id_coord = data_coord.find_observation_id(start_time_);

      const auto& coord_tensor_component_x = data_coord.get_tensor_component(
          target_obs_id_coord, "CauchyCartesianCoords_x");

      const auto& coord_tensor_component_y = data_coord.get_tensor_component(
          target_obs_id_coord, "CauchyCartesianCoords_y");

      const auto& coord_tensor_component_z = data_coord.get_tensor_component(
          target_obs_id_coord, "CauchyCartesianCoords_z");

      if ((not std::holds_alternative<DataVector>(
              coord_tensor_component_x.data)) or
          (not std::holds_alternative<DataVector>(
              coord_tensor_component_y.data)) or
          (not std::holds_alternative<DataVector>(
              coord_tensor_component_z.data))) {
        ERROR("CCE initial coord must be a DataVector");
      }

      get<0>(*cartesian_cauchy_coordinates) =
          std::get<DataVector>(coord_tensor_component_x.data);
      get<1>(*cartesian_cauchy_coordinates) =
          std::get<DataVector>(coord_tensor_component_y.data);
      get<2>(*cartesian_cauchy_coordinates) =
          std::get<DataVector>(coord_tensor_component_z.data);

      GaugeUpdateAngularFromCartesian<
          Tags::CauchyAngularCoords,
          Tags::CauchyCartesianCoords>::apply(angular_cauchy_coordinates,
                                              cartesian_cauchy_coordinates);
    }
  }
}

void CauchyInverseCubic::pup(PUP::er& p) {
  p | input_filename_;
  p | input_subfile_name_j_;
  p | input_subfile_name_coord_;
  p | start_time_;
  p | get_j_from_file_;
  p | get_coord_from_file_;
  p | require_convergence_;
  p | angular_coordinate_tolerance_;
  p | max_iterations_;
}

PUP::able::PUP_ID CauchyInverseCubic::my_PUP_ID = 0;
}  // namespace Cce::InitializeJ
