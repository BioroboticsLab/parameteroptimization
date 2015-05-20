#pragma once

#include "Common.h"

#include <bayesopt/bayesopt.hpp>

#include "source/tracking/algorithm/BeesBook/ImgAnalysisTracker/pipeline/datastructure/settings.h"
#include "source/tracking/algorithm/BeesBook/ImgAnalysisTracker/pipeline/Preprocessor.h"
#include "source/tracking/algorithm/BeesBook/ImgAnalysisTracker/pipeline/Localizer.h"

namespace opt {

struct LocalizerResult : OptimizationResult {
	LocalizerResult(double fscore, double recall, double precision,
	                pipeline::settings::preprocessor_settings_t const &psettings,
	                pipeline::settings::localizer_settings_t const &lsettings)
	    : OptimizationResult(fscore, recall, precision)
	    , psettings(psettings)
	    , lsettings(lsettings) {}

	LocalizerResult(OptimizationResult const &oresult,
	                pipeline::settings::preprocessor_settings_t const &psettings,
	                pipeline::settings::localizer_settings_t const &lsettings)
	    : OptimizationResult(oresult.fscore, oresult.recall, oresult.precision)
	    , psettings(psettings)
	    , lsettings(lsettings) {}

	pipeline::settings::preprocessor_settings_t psettings;
	pipeline::settings::localizer_settings_t lsettings;
};

class LocalizerModel : public bayesopt::ContinuousModel {
  public:
	static const size_t _numDimensions = 13;

	typedef std::map<std::string, limits_t> limitsByParam;

	LocalizerModel(bopt_params param, path_pair_t const &task,
	               limitsByParam const &limitsByParameter);

	template <typename ParamType, typename Settings>
	void setValueFromQuery(Settings &settings, std::string const &paramName, double value) {
		settings.template _setValue<ParamType>(
		    paramName, _limitsByParameter[paramName].getVal<ParamType>(value));
	}

	void applyQueryToSettings(const boost::numeric::ublas::vector<double> &query,
	                          pipeline::settings::localizer_settings_t &lsettings,
	                          pipeline::settings::preprocessor_settings_t &psettings);

	boost::optional<LocalizerResult>
	evaluate(pipeline::settings::localizer_settings_t &lsettings,
	         pipeline::settings::preprocessor_settings_t &psettings);

	double evaluateSample(const boost::numeric::ublas::vector<double> &query);

	bool checkReachability(const boost::numeric::ublas::vector<double> &);

	pipeline::settings::preprocessor_settings_t getPreprocessorSettings() const {
		return _preprocessorSettings;
	}

  private:
	pipeline::settings::preprocessor_settings_t _preprocessorSettings;
	pipeline::Preprocessor _preprocessor;

	pipeline::settings::localizer_settings_t _localizerSettings;
	pipeline::Localizer _localizer;

	cv::Mat _image;

	std::unique_ptr<GroundTruthEvaluation> _evaluation;

	limitsByParam _limitsByParameter;
};
}
