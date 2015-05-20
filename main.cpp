#include "main.h"

#include <boost/program_options.hpp>

#include "source/utility/MeasureTimeRAII.h"

#include "LocalizerModel.h"

namespace opt {

boost::optional<CommandLineOptions> getCommandLineOptions(int argc, char **argv) {
	namespace po = boost::program_options;

	po::options_description desc("Allowed options");
	desc.add_options()("help", "produce help message")(
	    "data", po::value<std::string>(), "data folder")("n_init_samples",
	                                                     po::value<size_t>()->default_value(100))(
	    "n_iterations", po::value<size_t>()->default_value(500))(
	    "n_iter_relearn", po::value<size_t>()->default_value(25));

	po::positional_options_description p;
	p.add("data", 1);

	po::variables_map vm;
	po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
	po::notify(vm);

	if (!vm.count("data")) {
		std::cout << "Input data folder not specified." << std::endl
		          << std::endl;
		std::cout << desc << std::endl;

		return boost::optional<CommandLineOptions>();
	}

	CommandLineOptions options{vm["data"].as<std::string>(), vm["n_init_samples"].as<size_t>(),
	                           vm["n_iterations"].as<size_t>(), vm["n_iter_relearn"].as<size_t>()};

	return options;
}

task_vector_t getTasks(boost::filesystem::path dataFolder) {
	namespace fs = boost::filesystem;

	task_vector_t groundTruthByImagePaths;

	std::set<fs::path> files;
	std::copy(fs::directory_iterator(dataFolder), fs::directory_iterator(),
	          std::inserter(files, files.begin()));

	for (const fs::path entry : files) {
		if (fs::is_regular_file(entry)) {
			if (entry.extension() == ".jpeg") {
				fs::path groundTruthPath(entry);
				groundTruthPath.replace_extension(".tdat");
				if (fs::is_regular_file(groundTruthPath)) {
					groundTruthByImagePaths.emplace_back(entry, groundTruthPath);
				}
			}
		}
	}

	return groundTruthByImagePaths;
}

bopt_params getBoptParams(CommandLineOptions const &options) {
	bopt_params params = initialize_parameters_to_default();

	params.n_init_samples = options.n_init_samples;
	params.n_iterations = options.n_iterations;
	params.n_iter_relearn = options.n_iter_relearn;

	params.noise = 1e-14;
	params.init_method = 2;

	return params;
}

void optimizeParameters(const path_pair_t &task, const bopt_params &params) {
	Util::MeasureTimeRAII measureTime;

	LocalizerModel model(params, task);

	boost::numeric::ublas::vector<double> bestPoint(model.getNumDimensions());
	model.optimize(bestPoint);

	pipeline::settings::preprocessor_settings_t psettings = model.getPreprocessorSettings();
	pipeline::settings::localizer_settings_t lsettings;

	model.applyQueryToSettings(bestPoint, lsettings, psettings);

	const auto result = model.evaluate(lsettings, psettings);

	if (result) {
		std::cout << bestPoint << std::endl;
		psettings.print();
		lsettings.print();

		std::cout << "F2Score: " << result.get().fscore << std::endl;
		std::cout << "Recall: " << result.get().recall << std::endl;
		std::cout << "Precision: " << result.get().precision << std::endl;
	}
}
}

int main(int argc, char **argv) {
	using namespace opt;

	const boost::optional<CommandLineOptions> options = getCommandLineOptions(argc, argv);

	if (!options)
		return EXIT_FAILURE;

	if (!boost::filesystem::is_directory(options.get().data)) {
		std::cout << "Invalid input data path." << std::endl << std::endl;
		return EXIT_FAILURE;
	}

	task_vector_t tasks = getTasks(options.get().data);
	bopt_params boptParams = getBoptParams(options.get());

	for (const path_pair_t &task : tasks) {
		optimizeParameters(task, boptParams);
	}

	return EXIT_SUCCESS;
}
