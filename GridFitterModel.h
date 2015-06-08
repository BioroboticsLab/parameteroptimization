#pragma once

#include "Common.h"
#include "OptimizationModel.h"

#include "source/tracking/algorithm/BeesBook/ImgAnalysisTracker/pipeline/datastructure/settings.h"
#include "source/tracking/algorithm/BeesBook/ImgAnalysisTracker/pipeline/GridFitter.h"
#include "source/tracking/algorithm/BeesBook/ImgAnalysisTracker/pipeline/Decoder.h"

namespace opt {

struct GridfitterResult {
	GridfitterResult(double score,
					pipeline::settings::gridfitter_settings_t const settings)
		: score(score)
		, settings(settings) {}

	GridfitterResult(GridfitterResult const &oresult,
					pipeline::settings::gridfitter_settings_t const &settings)
		: GridfitterResult(oresult.score, settings)
	{}

	double score;
	pipeline::settings::gridfitter_settings_t settings;
};

class GridfitterModel : public OptimizationModel {
  public:
	GridfitterModel(bopt_params param, path_struct_t const &task,
				   std::vector<pipeline::Tag> const &taglistLocalizer,
				   std::vector<pipeline::Tag> const &taglistEllipseFitter,
				   limitsByParam const &limitsByParameter);

	GridfitterModel(bopt_params param, const path_struct_t &task,
				   std::vector<pipeline::Tag> const &taglistLocalizer,
				   std::vector<pipeline::Tag> const &taglistEllipseFitter);

	virtual limitsByParam getDefaultLimits() const override;

	void applyQueryToSettings(const boost::numeric::ublas::vector<double> &query,
							  pipeline::settings::gridfitter_settings_t &settings);

	boost::optional<GridfitterResult>
	evaluate(pipeline::settings::gridfitter_settings_t &settings);

	virtual double evaluateSample(const boost::numeric::ublas::vector<double> &query) override;
	virtual bool checkReachability(const boost::numeric::ublas::vector<double> &query) override;

	static size_t getNumDimensions() { return 12; }

  private:
	pipeline::settings::gridfitter_settings_t _settings;
	pipeline::GridFitter _gridfitter;
	pipeline::Decoder _decoder;
	std::vector<pipeline::Tag> _taglistLocalizer;
	std::vector<pipeline::Tag> _taglistEllipseFitter;
};
}
