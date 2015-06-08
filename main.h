#pragma once

#include <chrono>
#include <cstddef>
#include <ctime>
#include <iomanip>
#include <sstream>
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

	CommandLineOptions(std::string const& data, size_t n_init_samples, size_t n_iterations,
					   size_t n_iter_relearn)
		: data(data)
		, n_init_samples(n_init_samples)
		, n_iterations(n_iterations)
		, n_iter_relearn(n_iter_relearn)
	{}
};

std::string getDateTime();

boost::optional<CommandLineOptions> getCommandLineOptions(int argc, char **argv);

task_vector_t getTasks(boost::filesystem::path dataFolder);

bopt_params getBoptParams(CommandLineOptions const &options);

void optimizeParameters(const path_struct_t &task, const CommandLineOptions &options, const bopt_params &params);
}

int main(int argc, char **argv);
