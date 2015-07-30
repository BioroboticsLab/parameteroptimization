#include "main.h"

#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>

#include "source/utility/MeasureTimeRAII.h"

#include "LocalizerModel.h"
#include "EllipseFitterModel.h"
#include "GridFitterModel.h"
#include "StdioHandler.h"

namespace opt {

boost::optional<CommandLineOptions> getCommandLineOptions(int argc, char **argv) {
	namespace po = boost::program_options;

	po::options_description desc("Allowed options");
	desc.add_options()
			("help", "produce help message")
			("data", po::value<std::string>(), "data folder")
			("n_init_samples", po::value<size_t>()->default_value(100))
			("n_iterations", po::value<size_t>()->default_value(500))
			("n_iter_relearn", po::value<size_t>()->default_value(25));

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

	auto addOptionalFile = [](fs::path path, boost::optional<fs::path>& optional) {
		if (fs::is_regular_file(path)) {
			optional = path;
		}
	};

	for (const fs::path entry : files) {
		if (fs::is_regular_file(entry)) {
			if (entry.extension() == ".jpeg") {
				fs::path groundTruthPath(entry);
				groundTruthPath.replace_extension(".tdat");
				if (fs::is_regular_file(groundTruthPath)) {
					fs::path folder = entry.parent_path() / getDateTime();
					if (fs::create_directory(folder)) {
						fs::path logfile = folder / "output.log";

						path_struct_t pstruct {entry, groundTruthPath, folder, logfile};
						addOptionalFile(entry.parent_path() / "psettings.json", pstruct.preprocessorSettings);
						addOptionalFile(entry.parent_path() / "lsettings.json", pstruct.localizerSettings);
						addOptionalFile(entry.parent_path() / "esettings.json", pstruct.ellipseFitterSettings);
						addOptionalFile(entry.parent_path() / "gsettings.json", pstruct.gridFitterSettings);

						groundTruthByImagePaths.push_back(pstruct);
					} else {
						std::cerr << "Unable to create output directory: " << folder.string() << std::endl;
						exit(EXIT_FAILURE);
					}
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

#include <fstream>

void optimizeParameters(const path_struct_t &task, const CommandLineOptions &options,
						const bopt_params &params)
{
	std::ofstream logging(task.logfile.string());

	// capture BayesOpt logging output
	StdErrHandler err([&](const char* line){
		std::cerr << line << std::endl;
		logging << getDateTime() << " - ERROR: " << line << std::endl;
	});
	StdOutHandler out([&](const char* line){
		std::cout << line << std::endl;
		logging << getDateTime() << " - INFO: " << line << std::endl;
	});

	auto optimizeLocalizer = [&]() {
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

		return std::make_pair(psettings, lsettings);
	};

	pipeline::settings::preprocessor_settings_t psettings;
	pipeline::settings::localizer_settings_t lsettings;

	if (!task.preprocessorSettings || !task.localizerSettings) {
		std::tie(psettings, lsettings) = optimizeLocalizer();
	} else {
		psettings.loadFromJson(task.preprocessorSettings.get().string());
		lsettings.loadFromJson(task.localizerSettings.get().string());

		std::cout << "Using preprocessor settings from: " << task.preprocessorSettings.get() << std::endl;
		psettings.print();
		std::cout << "Using localizer settings from: " << task.localizerSettings.get() << std::endl;
		lsettings.print();
	}

	psettings.writeToJson((task.outputFolder / "psettings.json").string());
	lsettings.writeToJson((task.outputFolder / "lsettings.json").string());

	auto optimizeEllipseFitter = [&]() {
		Util::MeasureTimeRAII measureTime;

		pipeline::Preprocessor preprocessor;
		preprocessor.loadSettings(psettings);
		pipeline::Localizer localizer;
		localizer.loadSettings(lsettings);

        cv::Mat image = cv::imread(task.image.string(), CV_LOAD_IMAGE_GRAYSCALE);
		cv::Mat imagePreprocessed = preprocessor.process(image);
		taglist_t taglist = localizer.process(std::move(image), std::move(imagePreprocessed));

		EllipseFitterModel model(params, task, taglist);

		boost::numeric::ublas::vector<double> bestPoint(model.getNumDimensions());
		model.optimize(bestPoint);

		pipeline::settings::ellipsefitter_settings_t esettings;

		model.applyQueryToSettings(bestPoint, esettings);

		const auto result = model.evaluate(esettings);

		if (result) {
			std::cout << bestPoint << std::endl;
			esettings.print();

			std::cout << "F0.5Score: " << result.get().fscore << std::endl;
			std::cout << "Recall: " << result.get().recall << std::endl;
			std::cout << "Precision: " << result.get().precision << std::endl;
		}

		return esettings;
	};

	pipeline::settings::ellipsefitter_settings_t esettings;

	if (!task.ellipseFitterSettings) {
		esettings = optimizeEllipseFitter();
	} else {
		esettings.loadFromJson(task.ellipseFitterSettings.get().string());

		std::cout << "Using ellipseFitter settings from: " << task.ellipseFitterSettings.get() << std::endl;
		esettings.print();
	}

	esettings.writeToJson((task.outputFolder / "esettings.json").string());

	auto optimizeGridFitter = [&]() {
		Util::MeasureTimeRAII measureTime;

		pipeline::Preprocessor preprocessor;
		preprocessor.loadSettings(psettings);
		pipeline::Localizer localizer;
		localizer.loadSettings(lsettings);
		pipeline::EllipseFitter ellipseFitter;
		ellipseFitter.loadSettings(esettings);

        cv::Mat image = cv::imread(task.image.string(), CV_LOAD_IMAGE_GRAYSCALE);
		cv::Mat imagePreprocessed = preprocessor.process(image);
		taglist_t taglistLocalizer = localizer.process(std::move(image), std::move(imagePreprocessed));
		taglist_t taglist = taglistLocalizer;

		taglist = ellipseFitter.process(std::move(taglist));
		taglist.erase(
					std::remove_if(taglist.begin(), taglist.end(),
								   [](pipeline::Tag const& tag) { return tag.getCandidatesConst().empty(); }));
		GridfitterModel model(params, task, taglistLocalizer, taglist);

		boost::numeric::ublas::vector<double> bestPoint(model.getNumDimensions());
		model.optimize(bestPoint);

		pipeline::settings::gridfitter_settings_t gsettings;

		model.applyQueryToSettings(bestPoint, gsettings);

		const auto result = model.evaluate(gsettings);

		if (result) {
			std::cout << bestPoint << std::endl;
			gsettings.print();

			std::cout << "Avg.Hamming: " << result.get().score << std::endl;
		}

		return gsettings;
	};

	pipeline::settings::gridfitter_settings_t gsettings;

	if (!task.gridFitterSettings) {
		gsettings = optimizeGridFitter();
	} else {
		gsettings.loadFromJson(task.gridFitterSettings.get().string());

		std::cout << "Using gridFitter settings from: " << task.gridFitterSettings.get() << std::endl;
		gsettings.print();
	}

	gsettings.writeToJson((task.outputFolder / "gsettings.json").string());

	boost::property_tree::ptree pt;
	psettings.addToPTree(pt);
	lsettings.addToPTree(pt);
	esettings.addToPTree(pt);
	gsettings.addToPTree(pt);
	boost::property_tree::write_json((task.outputFolder / "settings.json").string(), pt);
}

std::string getDateTime()
{
	const auto now = std::chrono::system_clock::now();
	const auto in_time_t = std::chrono::system_clock::to_time_t(now);

	std::stringstream ss;
	ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
	return ss.str();
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

	for (const path_struct_t &task : tasks) {
		optimizeParameters(task, options.get(), boptParams);
	}

	return EXIT_SUCCESS;
}
