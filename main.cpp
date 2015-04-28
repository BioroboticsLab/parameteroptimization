#include <chrono>
#include <functional>

#include <boost/filesystem.hpp>
#include <boost/optional.hpp>
#include <boost/program_options.hpp>

#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>

#include <opencv2/opencv.hpp>

#include "source/tracking/algorithm/BeesBook/BeesBookImgAnalysisTracker/GroundTruthEvaluator.h"

#include "source/tracking/algorithm/BeesBook/BeesBookImgAnalysisTracker/pipeline/Preprocessor.h"
#include "source/tracking/algorithm/BeesBook/BeesBookImgAnalysisTracker/pipeline/Localizer.h"
#include "source/tracking/algorithm/BeesBook/BeesBookImgAnalysisTracker/pipeline/EllipseFitter.h"
#include "source/tracking/algorithm/BeesBook/BeesBookImgAnalysisTracker/pipeline/GridFitter.h"
#include "source/tracking/algorithm/BeesBook/BeesBookImgAnalysisTracker/pipeline/Decoder.h"
#include "source/tracking/algorithm/BeesBook/BeesBookImgAnalysisTracker/pipeline/datastructure/settings.h"
#include "source/tracking/algorithm/BeesBook/BeesBookImgAnalysisTracker/pipeline/datastructure/PipelineGrid.h"
#include "source/tracking/algorithm/BeesBook/BeesBookImgAnalysisTracker/pipeline/datastructure/PipelineGrid.impl.h"

#include "source/tracking/serialization/SerializationData.h"

#include "source/utility/util.h"

namespace {
typedef std::pair<boost::filesystem::path, boost::filesystem::path> path_pair_t;
typedef std::vector<path_pair_t> task_vector_t;
}

class MeasureTimeRAII {
public:
	MeasureTimeRAII()
		: _start(std::chrono::steady_clock::now())
	{}

	~MeasureTimeRAII() {
		const auto end = std::chrono::steady_clock::now();
		const auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - _start).count();
		std::stringstream message;
		message << "finished in " << dur << "ms.";
		std::cout << message.str() << std::endl;
	}
private:
	const std::chrono::steady_clock::time_point _start;
};

template <typename Settings>
struct OptimizationResult
{
	const double fscore;
	const double recall;
	const double precision;

	Settings settings;
};

template <typename Settings>
bool operator<(const OptimizationResult<Settings>& a, const OptimizationResult<Settings>& b)
{
	return a.fscore > b.fscore;
}

typedef OptimizationResult<pipeline::settings::localizer_settings_t> LocalizerResult;

boost::optional<std::string> getCommandLineOptions(int argc, char** argv) {
	namespace po = boost::program_options;

	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("data", po::value<std::string>(), "data folder")
	;

	po::positional_options_description p;
	p.add("data", 1);

	po::variables_map vm;
	po::store(po::command_line_parser(argc, argv).
			  options(desc).positional(p).run(), vm);
	po::notify(vm);

	if (!vm.count("data")) {
		std::cout << "Input data folder not specified." << std::endl << std::endl;
		std::cout << desc << std::endl;

		return boost::optional<std::string>();
	}

	return vm["data"].as<std::string>();
}

task_vector_t getTasks(boost::filesystem::path dataFolder) {
	namespace fs = boost::filesystem;

	task_vector_t groundTruthByImagePaths;

	std::set<fs::path> files;
	std::copy(fs::directory_iterator(dataFolder), fs::directory_iterator(), std::inserter(files, files.begin()));

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

double getFScore(const double recall, const double precision, const double beta) {
	return ((1 + std::pow(beta, 2)) * ((precision * recall) / (std::pow(beta, 2) * precision + recall)));
}

boost::optional<LocalizerResult> getLocalizerResult(const GroundTruth::LocalizerEvaluationResults& results,
													const pipeline::settings::localizer_settings_t& settings)
{
	static const double beta = 2.;

	const size_t numGroundTruth    = results.taggedGridsOnFrame.size();
	const size_t numTruePositives  = results.truePositives.size();
	const size_t numFalsePositives = results.falsePositives.size();

	const double recall    = numGroundTruth ?
				(static_cast<double>(numTruePositives) / static_cast<double>(numGroundTruth)) : 0.;
	const double precision = (numTruePositives + numFalsePositives) ?
				(static_cast<double>(numTruePositives) / static_cast<double>(numTruePositives + numFalsePositives)) : 0.;

	const double fscore = getFScore(recall, precision, beta);

	if (std::isnan(fscore)) {
		return boost::optional<LocalizerResult>();
	} else {
		LocalizerResult result { fscore, recall, precision, settings };
		return result;
	}
}

class apply_t
{
public:
	apply_t(pipeline::Preprocessor& preprocessor, pipeline::Localizer& localizer, cv::Mat& imgOrig,
			GroundTruthEvaluation& evaluation, std::set<LocalizerResult>& localizerResults)
		: _preprocessor(preprocessor)
		, _localizer(localizer)
		, _imgOrig(imgOrig)
		, _evaluation(evaluation)
		, _localizerResults(localizerResults)
	{}

	inline void operator()() {
		MeasureTimeRAII measure;

		cv::Mat img(_imgOrig.get());

		cv::Mat imgPreprocessed = _preprocessor.get().process(img);
		taglist_t taglist       = _localizer.get().process(std::move(img), std::move(imgPreprocessed));

		_evaluation.get().evaluateLocalizer(0, taglist);

		_localizer.get().getSettings().print(_localizer.get().getSettings().getPTree());

		boost::optional<LocalizerResult> result =
				getLocalizerResult(_evaluation.get().getLocalizerResults(), _localizer.get().getSettings());

		if (result) {
			std::cout << "F-Score: " << result.get().fscore << std::endl << std::endl;
			_localizerResults.get().insert(result.get());
		} else {
			std::cout << "Invalid results" << std::endl << std::endl;
		}

		_evaluation.get().reset();
	}

private:
	std::reference_wrapper<pipeline::Preprocessor> _preprocessor;
	std::reference_wrapper<pipeline::Localizer> _localizer;
	std::reference_wrapper<cv::Mat> _imgOrig;
	std::reference_wrapper<GroundTruthEvaluation> _evaluation;
	std::reference_wrapper<std::set<LocalizerResult>> _localizerResults;
};


template <typename Setter, typename Range, typename RecurseFunctor>
class recurse_t
{
public:
	recurse_t(pipeline::settings::localizer_settings_t& settings,
			pipeline::Localizer& localizer,
			Setter&& setter,
			Range&& range,
			RecurseFunctor&& recurseFunctor)
		: _settings(settings)
		, _localizer(localizer)
		, _setter(std::move(setter))
		, _range(std::move(range))
		, _recurseFunctor(std::move(recurseFunctor))
	{}

	inline void operator()() {
		for (const auto elem : _range) {
			_setter(elem);
			_localizer.get().loadSettings(_settings.get());

			_recurseFunctor();
		}
	}

private:
	std::reference_wrapper<pipeline::settings::localizer_settings_t> _settings;
	std::reference_wrapper<pipeline::Localizer> _localizer;
	Setter _setter;
	Range _range;
	RecurseFunctor _recurseFunctor;
};

template <typename T, typename RecurseFunctor>
recurse_t<std::function<void(T)>, std::vector<T>, RecurseFunctor>
		getRecurseFunctor(std::vector<T>&& range, std::string const& param, RecurseFunctor&& recurseFunctor,
							pipeline::settings::localizer_settings_t& settings,
							pipeline::Localizer::Localizer& localizer)
{
	std::function<void(T)> paramSetter = std::bind(&pipeline::settings::localizer_settings_t::_setValue<T>, &settings,
								  param, std::placeholders::_1);
	recurse_t<std::function<void(T)>, std::vector<T>, RecurseFunctor>
			functor(settings, localizer, std::move(paramSetter), std::move(range), std::move(recurseFunctor));

	return functor;
}

void optimizeParameters(const path_pair_t& task) {
	cv::Mat imgOrig = cv::imread(task.first.string());

	Serialization::Data data;
	{
		std::ifstream is(task.second.string());
		cereal::JSONInputArchive ar(is);

		// load serialized data into member .data
		ar(data);
	}

	GroundTruthEvaluation evaluation(std::move(data));
	pipeline::Preprocessor preprocessor;
	pipeline::Localizer localizer;

	pipeline::settings::preprocessor_settings_t preprocessorSettings;
	pipeline::settings::localizer_settings_t localizerSettings;

	std::set<LocalizerResult> localizerResults;

	namespace settingspreprocessor = pipeline::settings::Preprocessor::Params;
	preprocessorSettings._setValue(settingspreprocessor::COMB_ENABLED, true);
	preprocessorSettings._setValue(settingspreprocessor::HONEY_ENABLED, true);

	apply_t applicator(preprocessor, localizer, imgOrig, evaluation, localizerResults);

	auto binaryThresholdFunctor = getRecurseFunctor(util::linspace<int>(25, 30, 2), pipeline::settings::Localizer::Params::BINARY_THRESHOLD,
													std::move(applicator), localizerSettings, localizer);

	auto numIterationsFunctor = getRecurseFunctor(util::linspace<unsigned int>(3, 5, 2), pipeline::settings::Localizer::Params::FIRST_DILATION_NUM_ITERATIONS,
													std::move(binaryThresholdFunctor), localizerSettings, localizer);

	auto firstDilationSizeFunctor = getRecurseFunctor(util::linspace<unsigned int>(2, 3, 2), pipeline::settings::Localizer::Params::FIRST_DILATION_SIZE,
													std::move(numIterationsFunctor), localizerSettings, localizer);

	auto erosionSizeFunctor = getRecurseFunctor(util::linspace<unsigned int>(23, 26, 2), pipeline::settings::Localizer::Params::EROSION_SIZE,
													std::move(firstDilationSizeFunctor), localizerSettings, localizer);

	auto secondDilationSizeFunctor = getRecurseFunctor(util::linspace<unsigned int>(2, 3, 2), pipeline::settings::Localizer::Params::SECOND_DILATION_SIZE,
													std::move(erosionSizeFunctor), localizerSettings, localizer);

	auto maxTagSizeFunctor = getRecurseFunctor(util::linspace<unsigned int>(200, 300, 2), pipeline::settings::Localizer::Params::MAX_TAG_SIZE,
													std::move(secondDilationSizeFunctor), localizerSettings, localizer);

	auto minBoundingBoxSizeFunctor = getRecurseFunctor(util::linspace<int>(300, 400, 2), pipeline::settings::Localizer::Params::MIN_BOUNDING_BOX_SIZE,
													std::move(maxTagSizeFunctor), localizerSettings, localizer);

	minBoundingBoxSizeFunctor();


	if (!localizerResults.empty()) {
		LocalizerResult bestResult = *localizerResults.begin();

		std::cout << "Best Result: " << std::endl;
		bestResult.settings.print(bestResult.settings.getPTree());
		std::cout << "Recall: " << bestResult.recall << std::endl;
		std::cout << "Precision: " << bestResult.precision << std::endl;
		std::cout << "F-Score: " << bestResult.fscore << std::endl;
	}

////		static const int BINARY_THRESHOLD = 29;
////static const unsigned int FIRST_DILATION_NUM_ITERATIONS = 4;
////static const unsigned int FIRST_DILATION_SIZE = 2;
////static const unsigned int EROSION_SIZE = 25;
////static const unsigned int SECOND_DILATION_SIZE = 2;
////static const unsigned int MAX_TAG_SIZE = 250;
////static const int MIN_BOUNDING_BOX_SIZE = 100;


//		for (int binary_threshold = 0; binary_threshold < 100; ++binary_threshold) {

//			MeasureTimeRAII measure;

//			cv::Mat img(imgOrig);

//			pipeline::settings::preprocessor_settings_t preprocessorSettings;
//			pipeline::settings::localizer_settings_t localizerSettings;

//			preprocessorSettings._setValue(settingspreprocessor::COMB_ENABLED, true);
//			preprocessorSettings._setValue(settingspreprocessor::HONEY_ENABLED, true);

////			localizerSettings._setValue(pipeline::settings::Localizer::Params::BINARY_THRESHOLD, binary_threshold);

//			auto binThrSetter = std::bind(&pipeline::settings::localizer_settings_t::_setValue<int>, &localizerSettings,
//										  pipeline::settings::Localizer::Params::BINARY_THRESHOLD, std::placeholders::_1);
//			const auto range = util::linspace(1, 100, 100);
//			binThrSetter(binary_threshold);

//			pipeline::Preprocessor preprocessor;
//			pipeline::Localizer localizer;

//			preprocessor.setOptions(preprocessorSettings);
//			localizer.loadSettings(localizerSettings);

//			cv::Mat imgOut = preprocessor.process(img);
//			taglist = localizer.process(std::move(img), std::move(imgOut));

//			evaluation.evaluateLocalizer(0, taglist);

//			std::cout << getLocalizerScore(evaluation.getLocalizerResults());

//			evaluation.reset();
//		}
//	}
}

int main(int argc, char** argv) {
	const boost::optional<std::string> dataFolder = getCommandLineOptions(argc, argv);

	if (!dataFolder) return EXIT_FAILURE;

	if (!boost::filesystem::is_directory(dataFolder.get())) {
		std::cout << "Invalid input data path." << std::endl << std::endl;

		return EXIT_FAILURE;
	}

	task_vector_t tasks = getTasks(dataFolder.get());

	for (const path_pair_t& task : tasks) {
		optimizeParameters(task);
	}

	return EXIT_SUCCESS;
}
