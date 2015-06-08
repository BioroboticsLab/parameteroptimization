#pragma once

#include "Common.h"
#include "OptimizationModel.h"

#include "source/tracking/algorithm/BeesBook/ImgAnalysisTracker/pipeline/datastructure/settings.h"
#include "source/tracking/algorithm/BeesBook/ImgAnalysisTracker/pipeline/datastructure/Tag.h"
#include "source/tracking/algorithm/BeesBook/ImgAnalysisTracker/pipeline/EllipseFitter.h"

namespace opt {

struct EllipseFitterResult : OptimizationResult {
	EllipseFitterResult(double fscore, double recall, double precision,
					pipeline::settings::ellipsefitter_settings_t const &settings)
		: OptimizationResult(fscore, recall, precision)
		, settings(settings) {}

	EllipseFitterResult(OptimizationResult const &oresult,
					pipeline::settings::ellipsefitter_settings_t const &settings)
		: OptimizationResult(oresult.fscore, oresult.recall, oresult.precision)
		, settings(settings) {}

	pipeline::settings::ellipsefitter_settings_t settings;
};

class EllipseFitterModel : public OptimizationModel {
public:
	EllipseFitterModel(bopt_params param, path_struct_t const &task,
					   std::vector<pipeline::Tag> const &taglist,
					   limitsByParam const &limitsByParameter);

	EllipseFitterModel(bopt_params param, const path_struct_t &task,
					   std::vector<pipeline::Tag> const &taglist);

	virtual limitsByParam getDefaultLimits() const override;

	void applyQueryToSettings(const boost::numeric::ublas::vector<double> &query,
							  pipeline::settings::ellipsefitter_settings_t &settings);

	boost::optional<EllipseFitterResult>
	evaluate(pipeline::settings::ellipsefitter_settings_t &settings);

	virtual double evaluateSample(const boost::numeric::ublas::vector<double> &query) override;

	virtual bool checkReachability(const boost::numeric::ublas::vector<double> &query) override;

	static size_t getNumDimensions() { return 11; }

	pipeline::settings::ellipsefitter_settings_t getEllipseFitterSettings() const {
		return _settings;
	}

  private:
	pipeline::settings::ellipsefitter_settings_t _settings;
	pipeline::EllipseFitter _ellipseFitter;
	std::vector<pipeline::Tag> _taglist;
};
}
