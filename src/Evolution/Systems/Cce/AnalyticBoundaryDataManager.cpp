// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/AnalyticBoundaryDataManager.hpp"

#include <cstddef>
#include <iomanip>
#include <utility>

#include "DataStructures/SpinWeighted.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/WorldtubeData.hpp"
#include "Utilities/MakeString.hpp"

namespace Cce {
AnalyticBoundaryDataManager::AnalyticBoundaryDataManager(
    const size_t l_max, const double extraction_radius,
    std::unique_ptr<Solutions::WorldtubeData> generator,
    const std::optional<std::string> output_file_prefix)
    : l_max_{l_max},
      generator_{std::move(generator)},
      extraction_radius_{extraction_radius},
      worldtube_mode_recorder_{std::nullopt} {
  if (output_file_prefix.has_value()) {
    const std::string filename = MakeString{}
                                 << output_file_prefix.value() << "CceR"
                                 << std::setw(4) << std::setfill('0')
                                 << static_cast<int>(extraction_radius_)
                                 << ".h5";
    worldtube_mode_recorder_ =
        std::make_unique<WorldtubeModeRecorder>(l_max_, filename);
  }
}

bool AnalyticBoundaryDataManager::populate_hypersurface_boundary_data(
    const gsl::not_null<Variables<
        Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>*>
        boundary_data_variables,
    const double time) const {
  const auto boundary_tuple = generator_->variables(
      l_max_, time,
      tmpl::list<gr::Tags::SpacetimeMetric<DataVector, 3>,
                 gh::Tags::Pi<DataVector, 3>, gh::Tags::Phi<DataVector, 3>>{});
  const auto& spacetime_metric =
      get<gr::Tags::SpacetimeMetric<DataVector, 3>>(boundary_tuple);
  const auto& pi = get<gh::Tags::Pi<DataVector, 3>>(boundary_tuple);
  const auto& phi = get<gh::Tags::Phi<DataVector, 3>>(boundary_tuple);
  create_bondi_boundary_data(boundary_data_variables, phi, pi, spacetime_metric,
                             extraction_radius_, l_max_);

  // Write boundary data to file if a recorder is present
  if (worldtube_mode_recorder_.has_value()) {
    tmpl::for_each<
        Tags::worldtube_boundary_tags_for_writing<Tags::BoundaryValue>>(
        [this, &boundary_data_variables, &time](auto tag_v) {
          using tag = typename decltype(tag_v)::type;
          const auto& nodal_data = get(get<tag>(*boundary_data_variables)).data();
          (*worldtube_mode_recorder_)
              ->append_modal_data<tag::tag::type::spin>(
                  dataset_label_for_tag<typename tag::tag>(), time, nodal_data,
                  l_max_);
        });
  }

  return true;
}

void AnalyticBoundaryDataManager::pup(PUP::er& p) {
  p | l_max_;
  p | extraction_radius_;
  p | generator_;
  // Note: WorldtubeModeRecorder is not serialized as it contains H5 file
  // handles that cannot be serialized. It will be reconstructed as needed.
  if (p.isUnpacking()) {
    worldtube_mode_recorder_ = std::nullopt;
  }
}
}  // namespace Cce
