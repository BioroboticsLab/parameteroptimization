#pragma once

#include <cstddef>
#include <string>

#include <bayesopt/bayesopt.hpp>

#include <boost/filesystem.hpp>
#include <boost/optional.hpp>

#include "Common.h"

namespace opt {

struct CommandLineOptions {
	std::string data;

	size_t n_init_samples;
	size_t n_iterations;
	size_t n_iter_relearn;
};

boost::optional<CommandLineOptions> getCommandLineOptions(int argc, char **argv);

task_vector_t getTasks(boost::filesystem::path dataFolder);

bopt_params getBoptParams(CommandLineOptions const &options);

void optimizeParameters(const path_pair_t &task, const bopt_params &params);
}

int main(int argc, char **argv);
