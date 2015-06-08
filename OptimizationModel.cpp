#include "OptimizationModel.h"

#include <fstream>

#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>

#include "source/tracking/serialization/SerializationData.h"

namespace opt {

OptimizationModel::OptimizationModel(bopt_params param, const path_struct_t &task,
                                     const limitsByParam &limitsByParameter, size_t numDimensions)
    : bayesopt::ContinuousModel(numDimensions, param)
    , _limitsByParameter(limitsByParameter) {
	_image = cv::imread(task.image.string());

	Serialization::Data data;
	{
		std::ifstream is(task.groundTruth.string());
		cereal::JSONInputArchive ar(is);

		// load serialized data into member .data
		ar(data);
	}

	_evaluation = std::make_unique<GroundTruthEvaluation>(std::move(data));
}
}
