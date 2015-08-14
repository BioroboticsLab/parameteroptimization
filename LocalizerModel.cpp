#include "LocalizerModel.h"

#include "source/tracking/algorithm/BeesBook/ImgAnalysisTracker/pipeline/datastructure/Tag.h"

namespace opt {

LocalizerModel::LocalizerModel(bopt_params param, const path_struct_t &task,
                               const boost::optional<DeepLocalizerPaths> &deeplocalizerPaths,
                               const ParameterMaps &parameterMaps)
    : OptimizationModel(param, task, parameterMaps, getNumDimensions()) {

	namespace settingspreprocessor = pipeline::settings::Preprocessor::Params;
    _preprocessorSettings.setValue(settingspreprocessor::COMB_ENABLED, true);
    _preprocessorSettings.setValue(settingspreprocessor::HONEY_ENABLED, true);

#ifdef USE_DEEPLOCALIZER
    namespace settingslocalizer = pipeline::settings::Localizer::Params;
    if (deeplocalizerPaths) {
        _localizerSettings.setValue(settingslocalizer::DEEPLOCALIZER_FILTER, true);
        _localizerSettings.setValue(settingslocalizer::DEEPLOCALIZER_MODEL_FILE, (*deeplocalizerPaths).model_path);
        _localizerSettings.setValue(settingslocalizer::DEEPLOCALIZER_PARAM_FILE, (*deeplocalizerPaths).param_path);
        _localizerSettings.setValue(settingslocalizer::TAG_SIZE, 100u);
    }
#endif
}

LocalizerModel::LocalizerModel(bopt_params param, const path_struct_t &task, const boost::optional<DeepLocalizerPaths> &deeplocalizerPaths)
    : LocalizerModel(param, task, deeplocalizerPaths, getDefaultLimits())
{}

OptimizationModel::ParameterMaps LocalizerModel::getDefaultLimits() {
    ParameterMaps parameterMaps;

    auto addLimitToParameterWrapper = [this, &parameterMaps](const std::string& paramName, limits_t limits)
    {
        this->addLimitToParameter(paramName, limits, parameterMaps);
    };

    {
		using namespace pipeline::settings::Localizer;
        addLimitToParameterWrapper(Params::BINARY_THRESHOLD, {10, 50});
        addLimitToParameterWrapper(Params::FIRST_DILATION_NUM_ITERATIONS, {1, 5});
        addLimitToParameterWrapper(Params::FIRST_DILATION_SIZE, {1, 10});
        addLimitToParameterWrapper(Params::EROSION_SIZE, {10, 40});
        addLimitToParameterWrapper(Params::SECOND_DILATION_SIZE, {1, 5});
        addLimitToParameterWrapper(Params::MIN_NUM_PIXELS, {1, 200});
        addLimitToParameterWrapper(Params::MAX_NUM_PIXELS, {1, 200});
#ifdef USE_DEEPLOCALIZER
        addLimitToParameterWrapper(Params::DEEPLOCALIZER_PROBABILITY_THRESHOLD, {0., 1.});
#endif
    }

	{
		using namespace pipeline::settings::Preprocessor;
        addLimitToParameterWrapper(Params::OPT_FRAME_SIZE, {25, 500});
        addLimitToParameterWrapper(Params::OPT_AVERAGE_CONTRAST_VALUE, {0, 255});
        addLimitToParameterWrapper(Params::OPT_AVERAGE_CONTRAST_VALUE, {0, 255});
        addLimitToParameterWrapper(Params::COMB_MIN_SIZE, {0, 150});
        addLimitToParameterWrapper(Params::COMB_MAX_SIZE, {0, 150});
        addLimitToParameterWrapper(Params::COMB_THRESHOLD, {0, 255});
        addLimitToParameterWrapper(Params::HONEY_STD_DEV, {0, 255});
        addLimitToParameterWrapper(Params::HONEY_FRAME_SIZE, {5, 50});
        addLimitToParameterWrapper(Params::HONEY_AVERAGE_VALUE, {0, 255});
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
#ifdef USE_DEEPLOCALIZER
        setValueFromQuery<double>(lsettings, Params::DEEPLOCALIZER_PROBABILITY_THRESHOLD, query);
#endif
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
	_preprocessor.loadSettings(psettings);
	_localizer.loadSettings(lsettings);

	cv::Mat img(_image);

	cv::Mat imgPreprocessed = _preprocessor.process(img);
	taglist_t taglist = _localizer.process(std::move(img), std::move(imgPreprocessed));

	_evaluation->evaluateLocalizer(0, taglist);
	const auto localizerResult = _evaluation->getLocalizerResults();

	const size_t numGroundTruth    = localizerResult.taggedGridsOnFrame.size();
	const size_t numTruePositives  = localizerResult.truePositives.size();
	const size_t numFalsePositives = localizerResult.falsePositives.size();

	const auto optimizationResult = getOptimizationResult(numGroundTruth, numTruePositives,
														  numFalsePositives, 2.);

	_evaluation->reset();

	if (optimizationResult) {
		return LocalizerResult(optimizationResult.get(), psettings, lsettings);
	}

	return boost::optional<LocalizerResult>();
}

double LocalizerModel::evaluateSample(const boost::numeric::ublas::vector<double> &query) {
	applyQueryToSettings(query, _localizerSettings, _preprocessorSettings);

	const auto result = evaluate(_localizerSettings, _preprocessorSettings);

	double score = 0.;
	if (result) {
		std::cout << "F-Score: " << result.get().fscore << std::endl
		          << std::endl;
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
#ifndef USE_DEEPLOCALIZER
    return 15;
#else
    return 16;
#endif
}
}
