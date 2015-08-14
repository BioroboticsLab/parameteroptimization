#pragma once

#include "Common.h"

#include "source/utility/stdext.h"

#include <bayesopt/bayesopt.hpp>

namespace opt {
class OptimizationModel : public bayesopt::ContinuousModel {
  public:
    struct ParameterMaps {
        typedef std::map<std::string, limits_t> limitsByParam;
        typedef std::map<std::string, size_t> queryIdxByParam;

        limitsByParam limitsByParameter;
        queryIdxByParam queryIdxByParameter;
    };

	OptimizationModel(bopt_params param, path_struct_t const &task,
                      ParameterMaps const &limitsByParameter, size_t numDimensions);

    virtual ParameterMaps getDefaultLimits() = 0;

    void addLimitToParameter(std::string const& param, limits_t limits,
                             ParameterMaps &parameterMaps);

    template <typename ParamType, typename Settings>
    void setValueFromQuery(Settings &settings, std::string const &paramName, const boost::numeric::ublas::vector<double>& query) {
        settings.template setValue<ParamType>(
            paramName, _parameterMaps.limitsByParameter[paramName].getVal<ParamType>(
                        query[_parameterMaps.queryIdxByParameter[paramName]]));
    }

    template <typename ParamType, typename Settings>
	void setValueFromQuery(Settings &settings, std::string const &paramName, double value) {
        settings.template setValue<ParamType>(
            paramName, _parameterMaps.limitsByParameter[paramName].getVal<ParamType>(value));
	}

	template <typename ParamType, typename Settings>
	void setOddValueFromQuery(Settings &settings, std::string const &paramName, double value) {
        settings.template setValue<ParamType>(
            paramName, _parameterMaps.limitsByParameter[paramName].getNearestOddVal<ParamType>(value));
	}

	virtual double evaluateSample(const boost::numeric::ublas::vector<double> &query) override = 0;
	virtual bool checkReachability(const boost::numeric::ublas::vector<double> &query) override = 0;

  protected:
	cv::Mat _image;
	std::unique_ptr<GroundTruthEvaluation> _evaluation;
    ParameterMaps _parameterMaps;
};
}
