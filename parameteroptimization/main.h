#pragma once

#include <chrono>
#include <cstddef>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>

#include <bayesopt.hpp>

#include <boost/filesystem.hpp>
#include <boost/optional.hpp>

#include "Common.h"

namespace opt {

struct CommandLineOptions {
	std::string data;

	size_t n_init_samples;
	size_t n_iterations;
	size_t n_iter_relearn;

    boost::optional<DeepLocalizerPaths> deeplocalizer_paths;

    bool optimize_mean;

	CommandLineOptions(std::string const& data, size_t n_init_samples, size_t n_iterations,
                       size_t n_iter_relearn, boost::optional<DeepLocalizerPaths> deeplocalizer_paths,
                       bool optimize_mean)
		: data(data)
		, n_init_samples(n_init_samples)
		, n_iterations(n_iterations)
		, n_iter_relearn(n_iter_relearn)
        , deeplocalizer_paths(deeplocalizer_paths)
        , optimize_mean(optimize_mean)
	{}
};

std::string getDateTime();

boost::optional<CommandLineOptions> getCommandLineOptions(int argc, char **argv);

opt::multiple_path_struct_t getTasks(boost::filesystem::path dataFolder);

bopt_params getBoptParams(CommandLineOptions const &options);

void optimizeParameters(const multiple_path_struct_t &task, const CommandLineOptions &options, const bopt_params &params);
/*
void optimizeParameters(const path_struct_t &task, const CommandLineOptions &options, const bopt_params &params);
*/

}

int main(int argc, char **argv);
