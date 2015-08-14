#include "GridFitterModel.h"

#include "source/tracking/algorithm/BeesBook/ImgAnalysisTracker/pipeline/datastructure/Tag.h"

namespace opt {

GridfitterModel::GridfitterModel(bopt_params param, const path_struct_t &task, const std::vector<pipeline::Tag> &taglistLocalizer, const std::vector<pipeline::Tag> &taglistEllipseFitter, const ParameterMaps &parameterMaps)
    : OptimizationModel(param, task, parameterMaps, getNumDimensions())
	, _taglistLocalizer(taglistLocalizer)
	, _taglistEllipseFitter(taglistEllipseFitter)
{}

GridfitterModel::GridfitterModel(bopt_params param, const path_struct_t &task, const std::vector<pipeline::Tag> &taglistLocalizer, const std::vector<pipeline::Tag> &taglistEllipseFitter)
	: GridfitterModel(param, task, taglistLocalizer, taglistEllipseFitter, getDefaultLimits())
{}

OptimizationModel::ParameterMaps GridfitterModel::getDefaultLimits()
{
    OptimizationModel::ParameterMaps parameterMaps;
    // TODO: use queryIdxByParameter
    OptimizationModel::ParameterMaps::limitsByParam &limitsByParameter = parameterMaps.limitsByParameter;

	using namespace pipeline::settings::Gridfitter;
	limitsByParameter[Params::ERR_FUNC_ALPHA_INNER] = {0., 1.};
	limitsByParameter[Params::ERR_FUNC_ALPHA_OUTER] = {0., 1.};
	limitsByParameter[Params::ERR_FUNC_ALPHA_VARIANCE] = {0., 1.};
	limitsByParameter[Params::ERR_FUNC_ALPHA_OUTER_EDGE] = {0., 1.};
	limitsByParameter[Params::ERR_FUNC_ALPHA_INNER_EDGE] = {0., 1.};

	limitsByParameter[Params::ADAPTIVE_BLOCK_SIZE] = {3., 61.};
	limitsByParameter[Params::ADAPTIVE_C] = {0., 255.};

	limitsByParameter[Params::GRADIENT_ERROR_THRESHOLD] = {0., 1.};

	limitsByParameter[Params::EPS_ANGLE] = {std::numeric_limits<double>::min(), 10.};
	limitsByParameter[Params::EPS_POS] = {1, 5};
	limitsByParameter[Params::EPS_SCALE] = {std::numeric_limits<double>::min(), 10.};
	limitsByParameter[Params::ALPHA] = {std::numeric_limits<double>::min(), 100.};

    return parameterMaps;
}

void GridfitterModel::applyQueryToSettings(const boost::numeric::ublas::vector<double> &query, pipeline::settings::gridfitter_settings_t &settings)
{
	size_t idx = 0;

	using namespace pipeline::settings::Gridfitter;
	setValueFromQuery<double>(settings, Params::ERR_FUNC_ALPHA_INNER, query[idx++]);
	setValueFromQuery<double>(settings, Params::ERR_FUNC_ALPHA_OUTER, query[idx++]);
	setValueFromQuery<double>(settings, Params::ERR_FUNC_ALPHA_VARIANCE, query[idx++]);
	setValueFromQuery<double>(settings, Params::ERR_FUNC_ALPHA_OUTER_EDGE, query[idx++]);
	setValueFromQuery<double>(settings, Params::ERR_FUNC_ALPHA_INNER_EDGE, query[idx++]);

	setOddValueFromQuery<int>(settings, Params::ADAPTIVE_BLOCK_SIZE, query[idx++]);
	setValueFromQuery<double>(settings, Params::ADAPTIVE_C, query[idx++]);

	setValueFromQuery<double>(settings, Params::GRADIENT_ERROR_THRESHOLD, query[idx++]);

	setValueFromQuery<double>(settings, Params::EPS_ANGLE, query[idx++]);
	setValueFromQuery<int>(settings, Params::EPS_POS, query[idx++]);
	setValueFromQuery<double>(settings, Params::EPS_SCALE, query[idx++]);
	setValueFromQuery<double>(settings, Params::ALPHA, query[idx++]);

	assert(idx == getNumDimensions());
}

boost::optional<GridfitterResult> GridfitterModel::evaluate(pipeline::settings::gridfitter_settings_t &settings)
{
	std::vector<pipeline::Tag> tagListCopy(_taglistEllipseFitter);

	_gridfitter.loadSettings(settings);

	// TODO: ref problem when using taglistLocalizer
	_evaluation->evaluateLocalizer(0, tagListCopy);
	_evaluation->evaluateEllipseFitter(tagListCopy);
	tagListCopy = _gridfitter.process(std::move(tagListCopy));
	_evaluation->evaluateGridFitter();
	tagListCopy = _decoder.process(std::move(tagListCopy));
	_evaluation->evaluateDecoder();

	const auto decoderResult = _evaluation->getDecoderResults();

	const boost::optional<double> avgHamming = decoderResult.getAverageHammingDistanceNormalized();

	_evaluation->reset();

	if (avgHamming) {
		return GridfitterResult(avgHamming.get(), settings);
	}

	return boost::optional<GridfitterResult>();
}

double GridfitterModel::evaluateSample(const boost::numeric::ublas::vector<double> &query)
{
	if (!checkReachability(query)) return std::numeric_limits<double>::max();

	applyQueryToSettings(query, _settings);

	_settings.print();

	const auto result = evaluate(_settings);

	if (result) {
		std::cout << "Avg. Hamming: " << result.get().score << std::endl << std::endl;
		return result.get().score;
	} else {
		std::cout << "Invalid results" << std::endl << std::endl;
		return std::numeric_limits<double>::max();
	}
}

bool GridfitterModel::checkReachability(const boost::numeric::ublas::vector<double> &query)
{
	return true;
	/*
	//TODO
	const int adC = _limitsByParameter[pipeline::settings::Gridfitter::Params::ADAPTIVE_BLOCK_SIZE].getVal<int>(query[5]);

	return (adC % 2);
	*/
}

}
