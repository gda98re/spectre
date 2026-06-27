// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/Initialize/ReadBondiJCauchy.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCoefficients.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTransform.hpp"
#include "Parallel/NodeLock.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Utilities/Gsl.hpp"

namespace Cce::InitializeJ {

ReadBondiJCauchy::ReadBondiJCauchy(std::string input_filename,
                                   std::string input_subfile_name_j,
                                   const double start_time,
                                   const double angular_coordinate_tolerance,
                                   const size_t max_iterations,
                                   const bool require_convergence)
    : input_filename_{std::move(input_filename)},
      input_subfile_name_j_{std::move(input_subfile_name_j)},
      start_time_{start_time},
      require_convergence_{require_convergence},
      angular_coordinate_tolerance_{angular_coordinate_tolerance},
      max_iterations_{max_iterations} {}

std::unique_ptr<InitializeJ<false>> ReadBondiJCauchy::get_clone() const {
  return std::make_unique<ReadBondiJCauchy>(*this);
}

void ReadBondiJCauchy::operator()(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& /*boundary_j*/,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& /*boundary_dr_j*/,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& /*r*/,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& /*beta*/,
    const size_t l_max, const size_t number_of_radial_points,
    const gsl::not_null<Parallel::NodeLock*> /*hdf5_lock*/) const {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  // Initialize J in the Cauchy coordinates by reading it from file
  Parallel::printf("ReadBondiJCauchy\n");
  Parallel::printf("start time: %.1e\n", start_time_);
  Parallel::printf("input file name: %s\n", input_filename_);

  h5::H5File<h5::AccessType::ReadOnly> cce_data_file{input_filename_};
  auto& data_ = cce_data_file.get<h5::VolumeData>(input_subfile_name_j_);
  if (data_.list_observation_ids().size() == 0) {
    ERROR("The observation IDs list is empty");
  }
  size_t target_obs_id = data_.find_observation_id(start_time_);
  const auto& j_tensor_component =
      data_.get_tensor_component(target_obs_id, "BondiJCauchy");

  if (not std::holds_alternative<DataVector>(j_tensor_component.data)) {
    ERROR("CCE initial BondiJCauchy must be a DataVector");
  }

  // Construct a DataVector containing the real and imaginary parts of the
  // Goldberg modes of DyDyJ for each radial collocation points

  const auto& modal_j_goldberg_interleaved =
      std::get<DataVector>(j_tensor_component.data);
  const size_t l_max_plus_one_squared = square(l_max + 1);
  if (modal_j_goldberg_interleaved.size() !=
      2 * number_of_radial_points * l_max_plus_one_squared) {
    ERROR(
        std::string("Mismatch between l_max or number of radial points between "
                    "J read from h5 and the input settings. j_data_size: "));
  }

  SpinWeighted<ComplexModalVector, 2> modal_j_goldberg{number_of_radial_points *
                                                       l_max_plus_one_squared};

  for (size_t i = 0; i < modal_j_goldberg.size(); i++) {
    modal_j_goldberg.data()[i] =
        std::complex<double>(modal_j_goldberg_interleaved[2 * i],
                             modal_j_goldberg_interleaved[(2 * i) + 1]);
  }

  const SpinWeighted<ComplexDataVector, 2> volume_j_cauchy =
      Spectral::Swsh::inverse_swsh_transform(
          l_max, number_of_radial_points,
          Spectral::Swsh::goldberg_to_libsharp_modes(modal_j_goldberg, l_max));

  cce_data_file.close();

  // Set volume j in Cauchy coordinates to be the read one
  get(*j).data() = volume_j_cauchy.data();

  const SpinWeighted<ComplexDataVector, 2> j_at_scri_view;
  make_const_view(make_not_null(&j_at_scri_view), get(*j),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  Variables<
      tmpl::list<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
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
}

void ReadBondiJCauchy::pup(PUP::er& p) {
  p | input_filename_;
  p | input_subfile_name_j_;
  p | start_time_;
  p | require_convergence_;
  p | angular_coordinate_tolerance_;
  p | max_iterations_;
}

PUP::able::PUP_ID ReadBondiJCauchy::my_PUP_ID = 0;
}  // namespace Cce::InitializeJ
