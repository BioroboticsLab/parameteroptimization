#include "LocalizerModel.h"

#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>

#include "source/tracking/serialization/SerializationData.h"
#include "source/tracking/algorithm/BeesBook/ImgAnalysisTracker/pipeline/datastructure/Tag.h"

namespace opt {

LocalizerModel::LocalizerModel(bopt_params param, const path_pair_t &task,
                               const LocalizerModel::limitsByParam &limitsByParameter)
    : bayesopt::ContinuousModel(_numDimensions, param)
    , _limitsByParameter(limitsByParameter) {
	_image = cv::imread(task.first.string());

	Serialization::Data data;
	{
		std::ifstream is(task.second.string());
		cereal::JSONInputArchive ar(is);

		// load serialized data into member .data
		ar(data);
	}

	_evaluation = std::make_unique<GroundTruthEvaluation>(std::move(data));

	namespace settingspreprocessor = pipeline::settings::Preprocessor::Params;
	_preprocessorSettings._setValue(settingspreprocessor::COMB_ENABLED, true);
	_preprocessorSettings._setValue(settingspreprocessor::HONEY_ENABLED, true);
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

	assert(idx == (_numDimensions - 1));
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
	const auto result = getLocalizerResult(_evaluation->getLocalizerResults());
	_evaluation->reset();

	if (result) {
		return LocalizerResult(result.get(), psettings, lsettings);
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

bool LocalizerModel::checkReachability(const boost::numeric::ublas::vector<double> &) {
	return true;
}
}
