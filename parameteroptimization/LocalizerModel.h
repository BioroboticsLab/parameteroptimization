#pragma once

#include "Common.h"
#include "OptimizationModel.h"

#include "source/tracking/algorithm/BeesBook/pipeline/settings/LocalizerSettings.h"
#include "source/tracking/algorithm/BeesBook/pipeline/Preprocessor.h"
#include "source/tracking/algorithm/BeesBook/pipeline/Localizer.h"

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

    LocalizerResult(std::vector<OptimizationResult> const &oresults,
                    pipeline::settings::preprocessor_settings_t const &psettings,
                    pipeline::settings::localizer_settings_t const &lsettings)
        : OptimizationResult(getMeanFscore(oresults),
                             getMeanRecall(oresults),
                             getMeanPrecision(oresults))
        , psettings(psettings)
        , lsettings(lsettings)
    {}

	pipeline::settings::preprocessor_settings_t psettings;
	pipeline::settings::localizer_settings_t lsettings;
};

class LocalizerModel : public OptimizationModel {
  public:
    LocalizerModel(bopt_params param, multiple_path_struct_t const &task,
                   boost::optional<DeepLocalizerPaths> const &deeplocalizerPaths,
                   ParameterMaps const &limitsByParameter);

    LocalizerModel(bopt_params param, const multiple_path_struct_t &task,
                   boost::optional<DeepLocalizerPaths> const &deeplocalizerPaths);

    virtual ParameterMaps getDefaultLimits() override;

	void applyQueryToSettings(const boost::numeric::ublas::vector<double> &query,
	                          pipeline::settings::localizer_settings_t &lsettings,
	                          pipeline::settings::preprocessor_settings_t &psettings);

	boost::optional<LocalizerResult>
	evaluate(pipeline::settings::localizer_settings_t &lsettings,
	         pipeline::settings::preprocessor_settings_t &psettings);

	virtual double evaluateSample(const boost::numeric::ublas::vector<double> &query) override;
	virtual bool checkReachability(const boost::numeric::ublas::vector<double> &query) override;

    static size_t getNumDimensions();

	pipeline::settings::preprocessor_settings_t getPreprocessorSettings() const {
		return _preprocessorSettings;
	}
	pipeline::settings::localizer_settings_t getLocalizerSettings() const {
		return _localizerSettings;
	}

  private:
    pipeline::settings::preprocessor_settings_t _preprocessorSettings;
    pipeline::settings::localizer_settings_t _localizerSettings;

    std::unique_ptr<pipeline::Preprocessor> _preprocessor;
    std::unique_ptr<pipeline::Localizer> _localizer;
};
}
