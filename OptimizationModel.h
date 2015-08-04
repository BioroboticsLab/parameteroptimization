#pragma once

#include "Common.h"

#include "source/utility/stdext.h"

#include <bayesopt/bayesopt.hpp>

namespace opt {
class OptimizationModel : public bayesopt::ContinuousModel {
  public:
	typedef std::map<std::string, limits_t> limitsByParam;

	OptimizationModel(bopt_params param, path_struct_t const &task,
	                  limitsByParam const &limitsByParameter, size_t numDimensions);

	virtual limitsByParam getDefaultLimits() const = 0;

	template <typename ParamType, typename Settings>
	void setValueFromQuery(Settings &settings, std::string const &paramName, double value) {
        settings.template setValue<ParamType>(
		    paramName, _limitsByParameter[paramName].getVal<ParamType>(value));
	}

	template <typename ParamType, typename Settings>
	void setOddValueFromQuery(Settings &settings, std::string const &paramName, double value) {
        settings.template setValue<ParamType>(
			paramName, _limitsByParameter[paramName].getNearestOddVal<ParamType>(value));
	}

	virtual double evaluateSample(const boost::numeric::ublas::vector<double> &query) override = 0;
	virtual bool checkReachability(const boost::numeric::ublas::vector<double> &query) override = 0;

  protected:
	cv::Mat _image;
	std::unique_ptr<GroundTruthEvaluation> _evaluation;
	limitsByParam _limitsByParameter;
};
}
