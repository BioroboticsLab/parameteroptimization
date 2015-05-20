#include "Common.h"

namespace opt {

bool operator<(const opt::OptimizationResult &a, const opt::OptimizationResult &b) {
	return a.fscore > b.fscore;
}

double getFScore(const double recall, const double precision, const double beta) {
	return ((1 + std::pow(beta, 2)) *
	        ((precision * recall) / (std::pow(beta, 2) * precision + recall)));
}

boost::optional<OptimizationResult>
getLocalizerResult(const GroundTruth::LocalizerEvaluationResults &results) {
	static const double beta = 2.;

	const size_t numGroundTruth = results.taggedGridsOnFrame.size();
	const size_t numTruePositives = results.truePositives.size();
	const size_t numFalsePositives = results.falsePositives.size();

	const double recall =
	    numGroundTruth
	        ? (static_cast<double>(numTruePositives) / static_cast<double>(numGroundTruth))
	        : 0.;
	const double precision = (numTruePositives + numFalsePositives)
	                             ? (static_cast<double>(numTruePositives) /
	                                static_cast<double>(numTruePositives + numFalsePositives))
	                             : 0.;

	const double fscore = getFScore(recall, precision, beta);

	if (std::isnan(fscore)) {
		return boost::optional<OptimizationResult>();
	} else {
		OptimizationResult result{fscore, recall, precision};
		return result;
	}
}
}
