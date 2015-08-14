#include "EllipseFitterModel.h"

namespace opt {

EllipseFitterModel::EllipseFitterModel(bopt_params param, const path_struct_t &task, const std::vector<pipeline::Tag> &taglist, const ParameterMaps &parameterMaps)
    : OptimizationModel(param, task, parameterMaps, getNumDimensions())
	, _taglist(taglist)
{}

EllipseFitterModel::EllipseFitterModel(bopt_params param, const path_struct_t &task, const std::vector<pipeline::Tag> &taglist)
	: EllipseFitterModel(param, task, taglist, getDefaultLimits())
{}

OptimizationModel::ParameterMaps EllipseFitterModel::getDefaultLimits()
{
    OptimizationModel::ParameterMaps parameterMaps;
    // TODO: use queryIdxByParameter
    OptimizationModel::ParameterMaps::limitsByParam &limitsByParameter = parameterMaps.limitsByParameter;

	using namespace pipeline::settings::EllipseFitter;
	limitsByParameter[Params::CANNY_INITIAL_HIGH]    = {25, 150};
	limitsByParameter[Params::CANNY_VALUES_DISTANCE] = {10, 100};
	limitsByParameter[Params::CANNY_MEAN_MIN]        = {5, 12};
	limitsByParameter[Params::CANNY_MEAN_MAX]        = {13, 30};

	limitsByParameter[Params::MIN_MAJOR_AXIS] = {20, 45};
	limitsByParameter[Params::MAX_MAJOR_AXIS] = {46, 70};
	limitsByParameter[Params::MIN_MINOR_AXIS] = {20, 45};
	limitsByParameter[Params::MAX_MINOR_AXIS] = {46, 70};

	limitsByParameter[Params::THRESHOLD_EDGE_PIXELS] = {15, 50};
	limitsByParameter[Params::THRESHOLD_BEST_VOTE]   = {1500, 4000};
	limitsByParameter[Params::THRESHOLD_VOTE]        = {500, 1400};

    return parameterMaps;
}

void EllipseFitterModel::applyQueryToSettings(const boost::numeric::ublas::vector<double> &query, pipeline::settings::ellipsefitter_settings_t &settings)
{
	size_t idx = 0;

	using namespace pipeline::settings::EllipseFitter;
	setValueFromQuery<int>(settings, Params::CANNY_INITIAL_HIGH, query[idx++]);
	setValueFromQuery<int>(settings, Params::CANNY_VALUES_DISTANCE, query[idx++]);
	setValueFromQuery<int>(settings, Params::CANNY_MEAN_MIN, query[idx++]);
	setValueFromQuery<int>(settings, Params::CANNY_MEAN_MAX, query[idx++]);

	setValueFromQuery<int>(settings, Params::MIN_MAJOR_AXIS, query[idx++]);
	setValueFromQuery<int>(settings, Params::MAX_MAJOR_AXIS, query[idx++]);
	setValueFromQuery<int>(settings, Params::MIN_MINOR_AXIS, query[idx++]);
	setValueFromQuery<int>(settings, Params::MAX_MINOR_AXIS, query[idx++]);

	setValueFromQuery<int>(settings, Params::THRESHOLD_EDGE_PIXELS, query[idx++]);
	setValueFromQuery<int>(settings, Params::THRESHOLD_BEST_VOTE, query[idx++]);
	setValueFromQuery<int>(settings, Params::THRESHOLD_VOTE, query[idx++]);

	assert(idx == getNumDimensions());
}

boost::optional<EllipseFitterResult> EllipseFitterModel::evaluate(pipeline::settings::ellipsefitter_settings_t &settings)
{
	std::vector<pipeline::Tag> tagListCopy(_taglist);

	_ellipseFitter.loadSettings(settings);

	_evaluation->evaluateLocalizer(0, tagListCopy);
	tagListCopy = _ellipseFitter.process(std::move(tagListCopy));
	_evaluation->evaluateEllipseFitter(tagListCopy);

	const auto ellipseFitterResult = _evaluation->getEllipsefitterResults();

	const size_t numGroundTruth    = ellipseFitterResult.taggedGridsOnFrame.size();
	const size_t numTruePositives  = ellipseFitterResult.truePositives.size();
	const size_t numFalsePositives = ellipseFitterResult.falsePositives.size();

	std::cout << numGroundTruth << std::endl;
	std::cout << numTruePositives << std::endl;
	std::cout << numFalsePositives << std::endl;

	const auto optimizationResult = getOptimizationResult(numGroundTruth, numTruePositives,
														  numFalsePositives, 0.5);

	_evaluation->reset();

	if (optimizationResult) {
		return EllipseFitterResult(optimizationResult.get(), settings);
	}

	return boost::optional<EllipseFitterResult>();
}

double EllipseFitterModel::evaluateSample(const boost::numeric::ublas::vector<double> &query)
{
	// BayesOpt does not check reachability during initial sampling
	//if (!checkReachability(query)) return 0.;

	applyQueryToSettings(query, _settings);

	_settings.print();

	const auto result = evaluate(_settings);

	double score = result ? (1 - result.get().fscore) : 1;

	if (result) {
		std::cout << "F0.5-Score: " << (1 - score) << std::endl
				  << std::endl;
	} else {
		std::cout << "Invalid results" << std::endl
				  << std::endl;
	}

	return score;
}

bool EllipseFitterModel::checkReachability(const boost::numeric::ublas::vector<double> &query)
{
	//TODO: use transformed int values instead of doubles
	//std::cout << query << std::endl;
	return true;
	// CANNY_MEAN
	if ((query[2]) >= query[3]) return false;

	// MAJOR_AXIS
	if ((query[4]) >= query[5]) return false;

	// MINOR_AXIS
	if ((query[6]) >= query[7]) return false;

	return true;
}
}
