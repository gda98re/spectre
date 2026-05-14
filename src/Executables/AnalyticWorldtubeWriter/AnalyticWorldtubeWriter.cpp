// Distributed under the MIT License.
// See LICENSE.txt for details.
//
// AnalyticWorldtubeWriter
//
// Evaluates one of the Cce analytic solutions on a worldtube of fixed
// extraction radius over a user-specified time grid, runs the analytic
// spacetime metric through `Cce::create_bondi_boundary_data`, and writes
// the resulting Bondi-Sachs boundary values to an h5 file in the same
// format that `CharacteristicExtract` expects from a BBH worldtube run.
//
// The point of this is to produce a "ground truth" worldtube file (e.g.
// `AnalyticBondiCceR0100.h5`) that is byte-compatible with the BBH
// workflow, so downstream comparison scripts and the standalone
// `CharacteristicExtract` executable can consume it unchanged.

#include <boost/program_options.hpp>
#include <cstddef>
#include <exception>
#include <memory>
#include <string>
#include <utility>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/AnalyticBoundaryDataManager.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/BouncingBlackHole.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/GaugeWave.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/LinearizedBondiSachs.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/RobinsonTrautman.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/RotatingSchwarzschild.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/TeukolskyWave.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/WorldtubeData.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/Cce/WorldtubeModeRecorder.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Options/String.hpp"
#include "Parallel/CreateFromOptions.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "Utilities/TMPL.hpp"

// Charm looks for this even though this is a plain main() executable.
extern "C" void CkRegisterMainModule(void) {}

namespace {
namespace OptionTags {
struct AnalyticSolution {
  using type = std::unique_ptr<Cce::Solutions::WorldtubeData>;
  static constexpr Options::String help{
      "Analytic worldtube data generator (e.g. TeukolskyWave). The "
      "ExtractionRadius set inside the generator is what determines the "
      "physical sphere on which the metric is evaluated."};
};

struct LMax {
  using type = size_t;
  static constexpr Options::String help{
      "Angular resolution used for the worldtube evaluation. Matches the "
      "`LMax` you intend to pass to `CharacteristicExtract`."};
  static type lower_bound() { return 2; }
};

struct ExtractionRadius {
  using type = double;
  static constexpr Options::String help{
      "Worldtube extraction radius written into the boundary h5 file. "
      "Should agree with the `ExtractionRadius` of the analytic solution."};
  static type lower_bound() { return 0.0; }
};

struct StartTime {
  using type = double;
  static constexpr Options::String help{
      "First time at which the boundary data is sampled."};
};

struct EndTime {
  using type = double;
  static constexpr Options::String help{
      "Last time at which the boundary data is sampled (inclusive)."};
};

struct TimeStep {
  using type = double;
  static constexpr Options::String help{
      "Uniform spacing of the time samples written to the worldtube h5."};
  static type lower_bound() { return 0.0; }
};

struct OutputH5File {
  using type = std::string;
  static constexpr Options::String help{
      "Path of the worldtube h5 file to write. `.h5` is appended if not "
      "present. The file layout matches what `CharacteristicExtract` "
      "expects from a BBH-style worldtube."};
};
}  // namespace OptionTags

using option_tags =
    tmpl::list<OptionTags::AnalyticSolution, OptionTags::LMax,
               OptionTags::ExtractionRadius, OptionTags::StartTime,
               OptionTags::EndTime, OptionTags::TimeStep,
               OptionTags::OutputH5File>;
using OptionTuple = tuples::tagged_tuple_from_typelist<option_tags>;

// Tag list of the Bondi-Sachs boundary quantities that the worldtube h5
// format stores. This is the same list `DumpBondiSachsOnWorldtube` uses
// when emitting a BBH worldtube file.
using bondi_dump_tags =
    Cce::Tags::worldtube_boundary_tags_for_writing<Cce::Tags::BoundaryValue>;

void write_bondi_data_to_disk(
    const gsl::not_null<Cce::WorldtubeModeRecorder*> recorder,
    const Variables<Cce::Tags::characteristic_worldtube_boundary_tags<
        Cce::Tags::BoundaryValue>>& boundary_data,
    const double time, const size_t l_max) {
  tmpl::for_each<bondi_dump_tags>([&](auto tag_v) {
    using tag = typename decltype(tag_v)::type;
    constexpr int spin = tag::tag::type::type::spin;
    const ComplexDataVector& nodal_data =
        get(get<tag>(boundary_data)).data();
    recorder->append_modal_data<spin>(
        Cce::dataset_label_for_tag<typename tag::tag>(), time, nodal_data,
        l_max);
  });
}

void write_worldtube_h5(
    const std::unique_ptr<Cce::Solutions::WorldtubeData>& generator,
    const size_t l_max, const double extraction_radius,
    const double start_time, const double end_time, const double time_step,
    const std::string& output_file) {
  if (end_time < start_time) {
    ERROR_NO_TRACE(
        "EndTime (" << end_time << ") must be >= StartTime (" << start_time
                    << ").");
  }

  Cce::AnalyticBoundaryDataManager data_manager{l_max, extraction_radius,
                                                generator->get_clone()};

  Variables<Cce::Tags::characteristic_worldtube_boundary_tags<
      Cce::Tags::BoundaryValue>>
      boundary_data{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

  Cce::WorldtubeModeRecorder recorder{l_max, output_file};

  // Walk the time grid inclusive of `end_time`. We add a tiny relative
  // tolerance so that a target like `end_time = start_time + N * dt`
  // (which can drift by a few ULPs after N additions) still emits the
  // final row.
  const double half_step_tolerance = 0.5 * time_step;
  size_t step = 0;
  while (true) {
    const double time = start_time + (static_cast<double>(step) * time_step);
    if (time > end_time + half_step_tolerance) {
      break;
    }
    data_manager.populate_hypersurface_boundary_data(
        make_not_null(&boundary_data), time);
    write_bondi_data_to_disk(make_not_null(&recorder), boundary_data, time,
                             l_max);
    ++step;
  }

  Parallel::printf("Wrote %zu time samples to %s\n", step, output_file);
}
}  // namespace

/*
 * Generate a Bondi-Sachs worldtube h5 file from a Cce analytic solution.
 *
 * Usage:
 *   AnalyticWorldtubeWriter --input-file <yaml>
 *
 * Yaml format (one file per radius):
 *
 *   AnalyticSolution:
 *     TeukolskyWave:
 *       ExtractionRadius: 100.0
 *       Amplitude: 1.0e-3
 *       Duration: 30.0
 *   LMax: 16
 *   ExtractionRadius: 100.0
 *   StartTime: 50.0
 *   EndTime: 250.0
 *   TimeStep: 0.5
 *   OutputH5File: AnalyticBondiCceR0100.h5
 */
int main(int argc, char** argv) {
  boost::program_options::options_description desc("Options");
  desc.add_options()("help,h,", "show this help message")(
      "input-file", boost::program_options::value<std::string>()->required(),
      "Name of YAML input file to use.");

  boost::program_options::variables_map vars;
  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
          .options(desc)
          .run(),
      vars);

  Options::Parser<option_tags> parser{
      "Evaluate a Cce analytic solution on a worldtube and write the "
      "resulting Bondi-Sachs boundary values to a BBH-style h5 file."};

  if (vars.contains("help")) {
    Parallel::printf("%s\n%s", desc, parser.help());
    return 0;
  }

  if (not vars.contains("input-file")) {
    Parallel::printf("Missing input file. Pass '--input-file'\n");
    return 1;
  }

  // Register the analytic-solution derived classes so the YAML factory can
  // construct `std::unique_ptr<Cce::Solutions::WorldtubeData>` from the
  // `AnalyticSolution` option.
  register_derived_classes_with_charm<Cce::Solutions::WorldtubeData>();

  try {
    const std::string input_yaml = vars["input-file"].as<std::string>();
    parser.parse_file(input_yaml, false);

    const OptionTuple options = parser.template apply<option_tags>(
        [](auto... args) { return OptionTuple(std::move(args)...); });

    std::string output_file =
        tuples::get<OptionTags::OutputH5File>(options);
    if (not output_file.ends_with(".h5")) {
      output_file += ".h5";
    }

    write_worldtube_h5(tuples::get<OptionTags::AnalyticSolution>(options),
                       tuples::get<OptionTags::LMax>(options),
                       tuples::get<OptionTags::ExtractionRadius>(options),
                       tuples::get<OptionTags::StartTime>(options),
                       tuples::get<OptionTags::EndTime>(options),
                       tuples::get<OptionTags::TimeStep>(options),
                       output_file);
  } catch (const std::exception& exception) {
    Parallel::printf("%s\n", exception.what());
    return 1;
  }
  return 0;
}
