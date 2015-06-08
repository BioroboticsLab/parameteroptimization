#pragma once

#include <type_traits>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/optional.hpp>

#include "source/tracking/algorithm/BeesBook/ImgAnalysisTracker/GroundTruthEvaluator.h"
#include "source/tracking/algorithm/BeesBook/ImgAnalysisTracker/pipeline/util/Util.h"

namespace opt {

struct path_struct_t {
	boost::filesystem::path image;
	boost::filesystem::path groundTruth;
	boost::filesystem::path outputFolder;
	boost::filesystem::path logfile;

	boost::optional<boost::filesystem::path> preprocessorSettings;
	boost::optional<boost::filesystem::path> localizerSettings;
	boost::optional<boost::filesystem::path> ellipseFitterSettings;
	boost::optional<boost::filesystem::path> gridFitterSettings;

	path_struct_t(boost::filesystem::path const& image,
				  boost::filesystem::path const& groundTruth,
				  boost::filesystem::path const& outputFolder,
				  boost::filesystem::path const& logfile)
		: image(image), groundTruth(groundTruth), outputFolder(outputFolder), logfile(logfile)
	{}
};
typedef std::vector<path_struct_t> task_vector_t;

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

	template <typename T>
	typename std::enable_if<std::is_integral<T>::value, T>::type getVal(double val) const {
		return static_cast<T>(std::round(min + val * (max - min)));
	}

	template <typename T>
	typename std::enable_if<std::is_floating_point<T>::value, T>::type getVal(double val) const {
		return static_cast<T>(min + val * (max - min));
	}

	template <typename T>
	typename std::enable_if<std::is_integral<T>::value, T>::type getNearestOddVal(double val) const {
		const T rounded = getVal<T>(val);
		if (rounded % 2) return rounded;
		if (Util::sgn(static_cast<double>(rounded) - val) >= 0) {
			return rounded + 1;
		} else {
			return rounded - 1;
		}
	}
};

bool operator<(const OptimizationResult &a, const OptimizationResult &b);

double getFScore(const double recall, const double precision, const double beta);

boost::optional<OptimizationResult>
getOptimizationResult(const size_t numGroundTruth, const size_t numTruePositives, const size_t numFalsePositives, const double beta);
}
