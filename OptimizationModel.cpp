#include "OptimizationModel.h"

#include <fstream>

#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>

#include "source/tracking/serialization/SerializationData.h"

namespace opt {

OptimizationModel::OptimizationModel(bopt_params param, const path_struct_t &task,
                                     const ParameterMaps &parameterMaps, size_t numDimensions)
    : bayesopt::ContinuousModel(numDimensions, param)
    , _parameterMaps(parameterMaps) {
    _image = cv::imread(task.image.string(), CV_LOAD_IMAGE_GRAYSCALE);

	Serialization::Data data;
	{
		std::ifstream is(task.groundTruth.string());
		cereal::JSONInputArchive ar(is);

		// load serialized data into member .data
		ar(data);
	}

    _evaluation = std::make_unique<GroundTruthEvaluation>(std::move(data));
}

void OptimizationModel::addLimitToParameter(const std::string &param, limits_t limits,
                                            ParameterMaps& parameterMaps)
{
    if (!parameterMaps.limitsByParameter.count(param)) {
        parameterMaps.limitsByParameter[param] = limits;
        parameterMaps.queryIdxByParameter[param] = parameterMaps.queryIdxByParameter.size() - 1;
    }
}
}
