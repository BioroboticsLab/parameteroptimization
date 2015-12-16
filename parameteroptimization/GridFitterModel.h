#pragma once

#include "Common.h"
#include "OptimizationModel.h"

#include <pipeline/settings/GridFitterSettings.h>
#include <pipeline/GridFitter.h>
#include <pipeline/Decoder.h>

namespace opt {

struct GridfitterResult;
double getMeanScore(const std::vector<GridfitterResult> &results);

struct GridfitterResult {
	GridfitterResult(double score,
					pipeline::settings::gridfitter_settings_t const settings)
		: score(score)
		, settings(settings) {}

	GridfitterResult(GridfitterResult const &oresult,
					pipeline::settings::gridfitter_settings_t const &settings)
		: GridfitterResult(oresult.score, settings)
	{}

    GridfitterResult(std::vector<GridfitterResult> const & results,
                    pipeline::settings::gridfitter_settings_t const &settings)
        : GridfitterResult(getMeanScore(results),
                           settings)
    {}

	double score;
	pipeline::settings::gridfitter_settings_t settings;
};

class GridfitterModel : public OptimizationModel {
  public:
    GridfitterModel(bopt_params param, multiple_path_struct_t const &task,
                   TaglistByImage const &taglistEllipseFitter,
                   ParameterMaps const &limitsByParameter);

    GridfitterModel(bopt_params param, const multiple_path_struct_t &task,
                   TaglistByImage const &taglistEllipseFitter);

    virtual ParameterMaps getDefaultLimits() override;

	void applyQueryToSettings(const boost::numeric::ublas::vector<double> &query,
							  pipeline::settings::gridfitter_settings_t &settings);

	boost::optional<GridfitterResult>
	evaluate(pipeline::settings::gridfitter_settings_t &settings);

	virtual double evaluateSample(const boost::numeric::ublas::vector<double> &query) override;
	virtual bool checkReachability(const boost::numeric::ublas::vector<double> &query) override;

    static size_t getNumDimensions() { return 13; }

  private:
	pipeline::settings::gridfitter_settings_t _settings;
	pipeline::GridFitter _gridfitter;
    pipeline::Decoder _decoder;

    TaglistByImage _taglistEllipseFitter;
};
}
