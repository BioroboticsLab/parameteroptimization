#include "LocalizerModel.h"

#include "StdioHandler.h"

#include <pipeline/util/ThreadPool.h>
#include <pipeline/datastructure/Tag.h>

namespace opt {

LocalizerModel::LocalizerModel(bopt_params param, const multiple_path_struct_t &task,
                               const boost::optional<DeepLocalizerPaths> &deeplocalizerPaths,
                               const ParameterMaps &parameterMaps)
    : OptimizationModel(param, task, parameterMaps, getNumDimensions())
    , _preprocessor(std::make_unique<pipeline::Preprocessor>())
    , _localizer(std::make_unique<pipeline::Localizer>())
{

	namespace settingspreprocessor = pipeline::settings::Preprocessor::Params;
    _preprocessorSettings.setValue(settingspreprocessor::COMB_ENABLED, true);
    _preprocessorSettings.setValue(settingspreprocessor::OPT_USE_CONTRAST_STRETCHING, true);
    _preprocessorSettings.setValue(settingspreprocessor::OPT_USE_EQUALIZE_HISTOGRAM, false);
    _preprocessorSettings.setValue(settingspreprocessor::HONEY_ENABLED, true);

    namespace settingslocalizer = pipeline::settings::Localizer::Params;
    if (deeplocalizerPaths) {
        _localizerSettings.setValue(settingslocalizer::DEEPLOCALIZER_FILTER, true);
        _localizerSettings.setValue(settingslocalizer::DEEPLOCALIZER_MODEL_FILE, (*deeplocalizerPaths).model_path);
        _localizerSettings.setValue(settingslocalizer::DEEPLOCALIZER_PARAM_FILE, (*deeplocalizerPaths).param_path);
        _localizerSettings.setValue(settingslocalizer::TAG_SIZE, 100u);
    }

    _localizer->loadSettings(_localizerSettings);
}

LocalizerModel::LocalizerModel(bopt_params param, const multiple_path_struct_t &task, const boost::optional<DeepLocalizerPaths> &deeplocalizerPaths)
    : LocalizerModel(param, task, deeplocalizerPaths, getDefaultLimits())
{}

OptimizationModel::ParameterMaps LocalizerModel::getDefaultLimits() {
    ParameterMaps parameterMaps;

    auto addLimitToParameterWrapper = [&](const std::string& paramName, limits_t limits)
    {
        this->addLimitToParameter(paramName, limits, parameterMaps);
    };

    {
		using namespace pipeline::settings::Localizer;
        addLimitToParameterWrapper(Params::BINARY_THRESHOLD, {45, 55});
        addLimitToParameterWrapper(Params::FIRST_DILATION_NUM_ITERATIONS, {4, 6});
        addLimitToParameterWrapper(Params::FIRST_DILATION_SIZE, {1, 3});
        addLimitToParameterWrapper(Params::EROSION_SIZE, {16, 20});
        addLimitToParameterWrapper(Params::SECOND_DILATION_SIZE, {1, 2});
        addLimitToParameterWrapper(Params::MIN_NUM_PIXELS, {1, 1});
        addLimitToParameterWrapper(Params::MAX_NUM_PIXELS, {7862, 7862});
        addLimitToParameterWrapper(Params::DEEPLOCALIZER_PROBABILITY_THRESHOLD, {0.9, 1.});
    }

	{
		using namespace pipeline::settings::Preprocessor;
        addLimitToParameterWrapper(Params::OPT_FRAME_SIZE, {500, 500});
        addLimitToParameterWrapper(Params::OPT_AVERAGE_CONTRAST_VALUE, {254.98, 254.98});
        addLimitToParameterWrapper(Params::COMB_MIN_SIZE, {0, 0});
        addLimitToParameterWrapper(Params::COMB_MAX_SIZE, {0, 0});
        addLimitToParameterWrapper(Params::COMB_THRESHOLD, {254.952, 254.952});
        addLimitToParameterWrapper(Params::HONEY_STD_DEV, {0.0474217, 0.0474217});
        addLimitToParameterWrapper(Params::HONEY_FRAME_SIZE, {5, 5});
        addLimitToParameterWrapper(Params::HONEY_AVERAGE_VALUE, {0.20471, 0.20471});
	}

    assert(parameterMaps.queryIdxByParameter.size() == getNumDimensions());

    return parameterMaps;
}

void LocalizerModel::applyQueryToSettings(const boost::numeric::ublas::vector<double> &query,
										  pipeline::settings::localizer_settings_t &lsettings,
										  pipeline::settings::preprocessor_settings_t &psettings) {
	{
		using namespace pipeline::settings::Localizer;
        setValueFromQuery<int>(lsettings, Params::BINARY_THRESHOLD, query);
		setValueFromQuery<unsigned int>(lsettings, Params::FIRST_DILATION_NUM_ITERATIONS,
                                        query);
        setValueFromQuery<unsigned int>(lsettings, Params::FIRST_DILATION_SIZE, query);
        setValueFromQuery<unsigned int>(lsettings, Params::EROSION_SIZE, query);
        setValueFromQuery<unsigned int>(lsettings, Params::SECOND_DILATION_SIZE, query);
        setValueFromQuery<unsigned int>(lsettings, Params::MIN_NUM_PIXELS, query);
        setValueFromQuery<unsigned int>(lsettings, Params::MAX_NUM_PIXELS, query);
        setValueFromQuery<double>(lsettings, Params::DEEPLOCALIZER_PROBABILITY_THRESHOLD, query);
    }
	{
		using namespace pipeline::settings::Preprocessor;
        setValueFromQuery<unsigned int>(psettings, Params::OPT_FRAME_SIZE, query);
        setValueFromQuery<double>(psettings, Params::OPT_AVERAGE_CONTRAST_VALUE, query);
        setValueFromQuery<unsigned int>(psettings, Params::COMB_MIN_SIZE, query);
        setValueFromQuery<unsigned int>(psettings, Params::COMB_MAX_SIZE, query);
        setValueFromQuery<double>(psettings, Params::COMB_THRESHOLD, query);
        setValueFromQuery<double>(psettings, Params::HONEY_STD_DEV, query);
        setValueFromQuery<unsigned int>(psettings, Params::HONEY_FRAME_SIZE, query);
        setValueFromQuery<double>(psettings, Params::HONEY_AVERAGE_VALUE, query);
	}
}

boost::optional<LocalizerResult>
LocalizerModel::evaluate(pipeline::settings::localizer_settings_t &lsettings,
                         pipeline::settings::preprocessor_settings_t &psettings) {
    std::vector<OptimizationResult> results;

    for (auto const& imagesByEvaluator : _imagesByEvaluator)
    {
        GroundTruthEvaluation* evaluator = imagesByEvaluator.first.get();
        const std::vector<boost::filesystem::path>& imagesPaths = imagesByEvaluator.second;

        _preprocessor->loadSettings(psettings);
        _localizer->loadSettings(lsettings);

        size_t frameNumber = 0;
        for (const boost::filesystem::path& imagePath : imagesPaths)
        {
            cv::Mat img(_imageByPath[imagePath]);

            pipeline::PreprocessorResult preprocessed = _preprocessor->process(img);
            taglist_t taglist = _localizer->process(std::move(preprocessed));

            evaluator->evaluateLocalizer(frameNumber, taglist);

            const auto localizerResult = evaluator->getLocalizerResults();

            const size_t numGroundTruth    = localizerResult.taggedGridsOnFrame.size();
            const size_t numTruePositives  = localizerResult.truePositives.size();
            const size_t numFalsePositives = localizerResult.falsePositives.size();

            results.push_back(getOptimizationResult(numGroundTruth, numTruePositives, numFalsePositives, 2.));

            ++frameNumber;

            evaluator->reset();
        }
    }

    return LocalizerResult(results, psettings, lsettings);
}

double LocalizerModel::evaluateSample(const boost::numeric::ublas::vector<double> &query) {
	applyQueryToSettings(query, _localizerSettings, _preprocessorSettings);

	const auto result = evaluate(_localizerSettings, _preprocessorSettings);

	double score = 0.;
	if (result) {
        std::cout << "Recall: " << result.get().recall << std::endl;
        std::cout << "Precision: " << result.get().precision << std::endl;
        std::cout << "F-Score: " << result.get().fscore << std::endl << std::endl;
		score = result.get().fscore;
	} else {
		std::cout << "Invalid results" << std::endl
		          << std::endl;
	}

	return (1 - score);
}

bool LocalizerModel::checkReachability(const boost::numeric::ublas::vector<double> &query)
{
    OptimizationModel::ParameterMaps::queryIdxByParam &queryIdxByParameter = _parameterMaps.queryIdxByParameter;

    return query[queryIdxByParameter[pipeline::settings::Localizer::Params::MIN_NUM_PIXELS]] <=
            query[queryIdxByParameter[pipeline::settings::Localizer::Params::MAX_NUM_PIXELS]];
}

size_t LocalizerModel::getNumDimensions()
{
    return 16;
}


}
