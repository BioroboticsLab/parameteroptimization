#include "LocalizerModel.h"

#include "source/tracking/algorithm/BeesBook/ImgAnalysisTracker/pipeline/datastructure/Tag.h"

namespace opt {

LocalizerModel::LocalizerModel(bopt_params param, const path_struct_t &task,
                               const LocalizerModel::limitsByParam &limitsByParameter)
    : OptimizationModel(param, task, limitsByParameter, getNumDimensions()) {
	namespace settingspreprocessor = pipeline::settings::Preprocessor::Params;
	_preprocessorSettings._setValue(settingspreprocessor::COMB_ENABLED, true);
	_preprocessorSettings._setValue(settingspreprocessor::HONEY_ENABLED, true);
}

LocalizerModel::LocalizerModel(bopt_params param, const path_struct_t &task)
	: LocalizerModel(param, task, getDefaultLimits())
{}

OptimizationModel::limitsByParam LocalizerModel::getDefaultLimits() const {
	OptimizationModel::limitsByParam limitsByParameter;
	{
		using namespace pipeline::settings::Localizer;
		limitsByParameter[Params::BINARY_THRESHOLD] = {10, 50};
		limitsByParameter[Params::FIRST_DILATION_NUM_ITERATIONS] = {1, 5};
		limitsByParameter[Params::FIRST_DILATION_SIZE] = {1, 10};
		limitsByParameter[Params::EROSION_SIZE] = {10, 40};
		limitsByParameter[Params::SECOND_DILATION_SIZE] = {1, 5};
	}

	{
		using namespace pipeline::settings::Preprocessor;
		limitsByParameter[Params::OPT_FRAME_SIZE] = {25, 500};
		limitsByParameter[Params::OPT_AVERAGE_CONTRAST_VALUE] = {0, 255};
		limitsByParameter[Params::OPT_AVERAGE_CONTRAST_VALUE] = {0, 255};
		limitsByParameter[Params::COMB_MIN_SIZE] = {0, 150};
		limitsByParameter[Params::COMB_MAX_SIZE] = {0, 150};
		limitsByParameter[Params::COMB_THRESHOLD] = {0, 255};
		limitsByParameter[Params::HONEY_STD_DEV] = {0, 255};
		limitsByParameter[Params::HONEY_FRAME_SIZE] = {5, 50};
		limitsByParameter[Params::HONEY_AVERAGE_VALUE] = {0, 255};
	}

	return limitsByParameter;
}

void LocalizerModel::applyQueryToSettings(const boost::numeric::ublas::vector<double> &query,
										  pipeline::settings::localizer_settings_t &lsettings,
										  pipeline::settings::preprocessor_settings_t &psettings) {
	size_t idx = 0;
	{
		using namespace pipeline::settings::Localizer;
		setValueFromQuery<int>(lsettings, Params::BINARY_THRESHOLD, query[idx++]);
		setValueFromQuery<unsigned int>(lsettings, Params::FIRST_DILATION_NUM_ITERATIONS,
										query[idx++]);
		setValueFromQuery<unsigned int>(lsettings, Params::FIRST_DILATION_SIZE, query[idx++]);
		setValueFromQuery<unsigned int>(lsettings, Params::EROSION_SIZE, query[idx++]);
		setValueFromQuery<unsigned int>(lsettings, Params::SECOND_DILATION_SIZE, query[idx++]);
	}

	{
		using namespace pipeline::settings::Preprocessor;
		setValueFromQuery<unsigned int>(psettings, Params::OPT_FRAME_SIZE, query[idx++]);
		setValueFromQuery<double>(psettings, Params::OPT_AVERAGE_CONTRAST_VALUE, query[idx++]);
		setValueFromQuery<unsigned int>(psettings, Params::COMB_MIN_SIZE, query[idx++]);
		setValueFromQuery<unsigned int>(psettings, Params::COMB_MAX_SIZE, query[idx++]);
		setValueFromQuery<double>(psettings, Params::COMB_THRESHOLD, query[idx++]);
		setValueFromQuery<double>(psettings, Params::HONEY_STD_DEV, query[idx++]);
		setValueFromQuery<unsigned int>(psettings, Params::HONEY_FRAME_SIZE, query[idx++]);
		setValueFromQuery<double>(psettings, Params::HONEY_AVERAGE_VALUE, query[idx++]);
	}

	assert(idx == getNumDimensions());
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
	return true;
}
}
