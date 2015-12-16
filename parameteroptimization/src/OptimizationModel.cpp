#include "OptimizationModel.h"

#include <fstream>

#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>

#include <biotracker/serialization/SerializationData.h>

#include <pipeline/util/GroundTruthEvaluator.h>

#include "Grid3D.h"

namespace opt {

OptimizationModel::OptimizationModel(bopt_params param, const multiple_path_struct_t &task,
                                     const ParameterMaps &parameterMaps, size_t numDimensions)
    : bayesopt::ContinuousModel(numDimensions, param)
    , _parameterMaps(parameterMaps)
{

    for (auto const& keyValuePair : task.imageFilesByGroundTruthFile)
    {
        boost::filesystem::path groundTruthPath = keyValuePair.first;

        Serialization::Data data;
        {
            std::ifstream is(groundTruthPath.string());
            cereal::JSONInputArchive ar(is);

            // load serialized data into member .data
            ar(data);
        }

        for (boost::filesystem::path const& imagePath : keyValuePair.second)
        {
            const cv::Mat image = cv::imread(imagePath.string(), CV_LOAD_IMAGE_GRAYSCALE);

            _imageByPath.insert({imagePath, image});
        }

        GroundTruthEvaluation::ResultsByFrame resultsByFrame;
        for (TrackedObject const& object : data.getTrackedObjects()) {
            for (size_t frameNumber = 0; frameNumber <= object.getLastFrameNumber(); ++frameNumber) {
                std::vector<GroundTruthGridSPtr>& results = resultsByFrame[frameNumber];

                const std::shared_ptr<Grid3D> grid3d = object.maybeGet<Grid3D>(frameNumber);

                if (!grid3d) continue;

                // convert to PipelineGrid
                const auto grid = std::make_shared<PipelineGrid>(
                            grid3d->getCenter(), grid3d->getPixelRadius(),
                            grid3d->getZRotation(), grid3d->getYRotation(),
                            grid3d->getXRotation());
                grid->setIdArray(grid3d->getIdArray());
                grid->setSettable(grid3d->isSettable());
                grid->setHasBeenSet(grid3d->hasBeenBitToggled() == boost::logic::tribool::true_value);

                // insert in set
                if (grid) {
                    results.push_back(grid);
                }
            }
        }

        _imagesByEvaluator.insert(std::make_pair(
                               std::make_unique<GroundTruthEvaluation>(std::move(resultsByFrame)),
                               keyValuePair.second));
    }
}

void OptimizationModel::addLimitToParameter(const std::string &param, limits_t limits,
                                            ParameterMaps& parameterMaps)
{
    if (!parameterMaps.limitsByParameter.count(param)) {
        parameterMaps.limitsByParameter[param] = limits;
        parameterMaps.queryIdxByParameter[param] = parameterMaps.queryIdxByParameter.size() - 1;
    }
}

double getMeanFscore(const std::vector<OptimizationResult> &results) {
    const double sum = std::accumulate(results.begin(), results.end(), 0.,
                                       [](double& acc, OptimizationResult const& result)
    {
        std::cout << result.fscore << std::endl;
        return acc + result.fscore;
    });

    assert(!results.empty());
    return sum / results.size();
}

double getMeanPrecision(const std::vector<OptimizationResult> &results) {
    const double sum = std::accumulate(results.begin(), results.end(), 0.,
                                       [](double& acc, OptimizationResult const& result)
    {
        return acc + result.precision;
    });

    assert(!results.empty());
    return sum / results.size();
}

double getMeanRecall(const std::vector<OptimizationResult> &results) {
    const double sum = std::accumulate(results.begin(), results.end(), 0.,
                                       [](double& acc, OptimizationResult const& result)
    {
        return acc + result.recall;
    });

    assert(!results.empty());
    return sum / results.size();
}

}
