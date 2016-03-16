#include "main.h"

#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>

//#include "source/utility/MeasureTimeRAII.h"

#include "LocalizerModel.h"
#include "EllipseFitterModel.h"
#include "GridFitterModel.h"
#include "StdioHandler.h"

#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>

#include <biotracker/serialization/SerializationData.h>

namespace opt {

boost::optional<CommandLineOptions> getCommandLineOptions(int argc, char **argv) {
	namespace po = boost::program_options;

	po::options_description desc("Allowed options");
	desc.add_options()
			("help", "produce help message")
			("data", po::value<std::string>(), "data folder")
			("n_init_samples", po::value<size_t>()->default_value(100))
			("n_iterations", po::value<size_t>()->default_value(500))
            ("n_iter_relearn", po::value<size_t>()->default_value(25))
            ("optimize_mean", po::value<bool>()->default_value(false), "optimize mean of scores for all files")
            ("deeplocalizer_model_path", po::value<std::string>())
            ("deeplocalizer_param_path", po::value<std::string>());

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

    boost::optional<DeepLocalizerPaths> deeplocalizerPaths;
    if (vm.count("deeplocalizer_model_path") && vm.count("deeplocalizer_param_path")) {
        deeplocalizerPaths = { vm["deeplocalizer_model_path"].as<std::string>(),
                               vm["deeplocalizer_param_path"].as<std::string>() };
    }

	CommandLineOptions options{vm["data"].as<std::string>(), vm["n_init_samples"].as<size_t>(),
                               vm["n_iterations"].as<size_t>(), vm["n_iter_relearn"].as<size_t>(),
                               deeplocalizerPaths, vm["optimize_mean"].as<bool>()};

	return options;
}

multiple_path_struct_t getTasks(boost::filesystem::path dataFolder) {
	namespace fs = boost::filesystem;

	std::set<fs::path> files;
    std::copy(fs::recursive_directory_iterator(dataFolder), fs::recursive_directory_iterator(),
	          std::inserter(files, files.begin()));

	auto addOptionalFile = [](fs::path path, boost::optional<fs::path>& optional) {
		if (fs::is_regular_file(path)) {
			optional = path;
		}
	};


    multiple_path_struct_t pstruct;
    pstruct.outputFolder = dataFolder;

    for (const fs::path entry : files) {

		if (fs::is_regular_file(entry)) {

            if (entry.extension() == ".tdat") {
				fs::path groundTruthPath(entry);

				if (fs::is_regular_file(groundTruthPath)) {
                    Serialization::Data data;
                    {
                        std::ifstream is(groundTruthPath.string());
                        cereal::JSONInputArchive ar(is);

                        // load serialized data
                        ar(data);
                    }

                    std::vector<std::string> const& fileNames = data.getFilenames();
                    std::vector<fs::path> filePaths;

                    for (std::string const& path : fileNames) {
                        fs::path imagePath = groundTruthPath.parent_path() / path;
                        imagePath.replace_extension(".jpeg");

                        if (fs::is_regular_file(imagePath)) {
                            filePaths.push_back(imagePath);
                        }
                    }

                    pstruct.imageFilesByGroundTruthFile.insert({groundTruthPath, filePaths});
                }
            }
        }
    }

    const fs::path folder = dataFolder / getDateTime();

    if (fs::create_directory(folder)) {
            pstruct.logfile = folder / "output.log";

            addOptionalFile(dataFolder / "psettings.json", pstruct.preprocessorSettings);
            addOptionalFile(dataFolder / "lsettings.json", pstruct.localizerSettings);
            addOptionalFile(dataFolder / "esettings.json", pstruct.ellipseFitterSettings);
            addOptionalFile(dataFolder / "gsettings.json", pstruct.gridFitterSettings);

    } else {
            std::cerr << "Unable to create output directory: " << folder.string() << std::endl;
            exit(EXIT_FAILURE);
    }

    return pstruct;
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

void optimizeParameters(const multiple_path_struct_t &task, const CommandLineOptions &options,
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
        // TODO!
        //Util::MeasureTimeRAII measureTime;

        LocalizerModel model(params, task, options.deeplocalizer_paths);

		boost::numeric::ublas::vector<double> bestPoint(model.getNumDimensions());
		model.optimize(bestPoint);

		pipeline::settings::preprocessor_settings_t psettings = model.getPreprocessorSettings();
        pipeline::settings::localizer_settings_t lsettings = model.getLocalizerSettings();

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
        //Util::MeasureTimeRAII measureTime;

		pipeline::Preprocessor preprocessor;
		preprocessor.loadSettings(psettings);
		pipeline::Localizer localizer;
		localizer.loadSettings(lsettings);

        OptimizationModel::TaglistByImage taglistByImage;
        for (auto const& groundTruthImagePair : task.imageFilesByGroundTruthFile) {
            const auto& imagePathVector = groundTruthImagePair.second;

            for (boost::filesystem::path const& imagePath : imagePathVector) {
                cv::Mat image = cv::imread(imagePath.string(), CV_LOAD_IMAGE_GRAYSCALE);
                pipeline::PreprocessorResult preprocessed = preprocessor.process(image);
                taglist_t taglist = localizer.process(std::move(preprocessed));

                taglistByImage.insert({imagePath, taglist});
            }
        }

        EllipseFitterModel model(params, task, taglistByImage);

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
        //Util::MeasureTimeRAII measureTime;

		pipeline::Preprocessor preprocessor;
		preprocessor.loadSettings(psettings);
		pipeline::Localizer localizer;
		localizer.loadSettings(lsettings);
		pipeline::EllipseFitter ellipseFitter;
		ellipseFitter.loadSettings(esettings);

        OptimizationModel::TaglistByImage taglistByImage;
        for (auto const& groundTruthImagePair : task.imageFilesByGroundTruthFile) {
            const auto& imagePathVector = groundTruthImagePair.second;

            for (boost::filesystem::path const& imagePath : imagePathVector) {
                cv::Mat image = cv::imread(imagePath.string(), CV_LOAD_IMAGE_GRAYSCALE);
                pipeline::PreprocessorResult preprocessed = preprocessor.process(image);
                taglist_t taglistLocalizer = localizer.process(std::move(preprocessed));

                taglist_t taglist = taglistLocalizer;
                taglist = ellipseFitter.process(std::move(taglist));
                taglist.erase(
                            std::remove_if(taglist.begin(), taglist.end(),
                                           [](pipeline::Tag const& tag) { return tag.getCandidatesConst().empty(); }));

                taglistByImage.insert({imagePath, taglist});
            }
        }

        GridfitterModel model(params, task, taglistByImage);

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

    bopt_params boptParams = getBoptParams(options.get());

    if ((*options).optimize_mean) {
        multiple_path_struct_t task = getTasks(options.get().data);

        optimizeParameters(task, options.get(), boptParams);
    } else {
        // TODO: refactor duplicate code

        namespace fs = boost::filesystem;

        std::set<fs::path> files;
        std::copy(fs::recursive_directory_iterator(options.get().data), fs::recursive_directory_iterator(),
                  std::inserter(files, files.begin()));

        for (const fs::path entry : files) {

            if (fs::is_regular_file(entry)) {

                if (entry.extension() == ".tdat") {
                    fs::path groundTruthPath(entry);

                    if (fs::is_regular_file(groundTruthPath)) {
                        std::cout << "Optimizing: " << groundTruthPath.string() << std::endl;

                        multiple_path_struct_t task = getTasks(groundTruthPath.parent_path());

                        optimizeParameters(task, options.get(), boptParams);
                    }
                }
            }
        }
    }

	return EXIT_SUCCESS;
}
