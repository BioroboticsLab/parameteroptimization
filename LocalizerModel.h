#pragma once

#include "Common.h"
#include "OptimizationModel.h"

#include "source/tracking/algorithm/BeesBook/ImgAnalysisTracker/pipeline/settings/LocalizerSettings.h"
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

class LocalizerModel : public OptimizationModel {
  public:
	LocalizerModel(bopt_params param, path_struct_t const &task,
	               limitsByParam const &limitsByParameter);

	LocalizerModel(bopt_params param, const path_struct_t &task);

	virtual limitsByParam getDefaultLimits() const override;

	void applyQueryToSettings(const boost::numeric::ublas::vector<double> &query,
	                          pipeline::settings::localizer_settings_t &lsettings,
	                          pipeline::settings::preprocessor_settings_t &psettings);

	boost::optional<LocalizerResult>
	evaluate(pipeline::settings::localizer_settings_t &lsettings,
	         pipeline::settings::preprocessor_settings_t &psettings);

	virtual double evaluateSample(const boost::numeric::ublas::vector<double> &query) override;
	virtual bool checkReachability(const boost::numeric::ublas::vector<double> &query) override;

	static size_t getNumDimensions() { return 13; }

	pipeline::settings::preprocessor_settings_t getPreprocessorSettings() const {
		return _preprocessorSettings;
	}
	pipeline::settings::localizer_settings_t getLocalizerSettings() const {
		return _localizerSettings;
	}

  private:
	pipeline::settings::preprocessor_settings_t _preprocessorSettings;
	pipeline::Preprocessor _preprocessor;

	pipeline::settings::localizer_settings_t _localizerSettings;
	pipeline::Localizer _localizer;
};
}
