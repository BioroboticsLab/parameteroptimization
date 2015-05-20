#pragma once

#include <vector>

#include <boost/filesystem.hpp>
#include <boost/optional.hpp>

#include "source/tracking/algorithm/BeesBook/ImgAnalysisTracker/GroundTruthEvaluator.h"

namespace opt {

typedef std::pair<boost::filesystem::path, boost::filesystem::path> path_pair_t;
typedef std::vector<path_pair_t> task_vector_t;

struct OptimizationResult {
	OptimizationResult(double fscore, double recall, double precision)
	    : fscore(fscore)
	    , recall(recall)
	    , precision(precision) {}

	const double fscore;
	const double recall;
	const double precision;
};

struct limits_t {
	double min;
	double max;

	double getVal(double val) const { return round(min + val * (max - min)); }

	template <typename T> T getVal(double val) const {
		return static_cast<T>(round(min + val * (max - min)));
	}
};

bool operator<(const OptimizationResult &a, const OptimizationResult &b);

double getFScore(const double recall, const double precision, const double beta);

boost::optional<OptimizationResult>
getLocalizerResult(const GroundTruth::LocalizerEvaluationResults &results);
}
