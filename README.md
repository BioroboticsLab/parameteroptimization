# Automatic parameter optimization for BeesBook

This project uses bayesian global optimization (via the [BayesOpt library](https://bitbucket.org/rmcantin/bayesopt))
to find a good set of parameters for the BeesBook image processing pipeline and a set of images with available
ground truth data files.

The ground truth data files can be created with the BioTracker module `BeesBook TagMatcher`.

## Dependencies

* GCC >= 4.8 or clang >= 3.6
* boost
* OpenCV
* Python

## Setup

### Clone, initialize build directory and compile project

```
git clone git@github.com:BioroboticsLab/parameteroptimization.git
mkdir parameteroptimization-build
cd parameteroptimization-build
cmake ../parameteroptimization
make
```

### Usage example
```
git clone git@github.com:BioroboticsLab/deeplocalizer_data.git
git clone git@github.com:BioroboticsLab/deeplocalizer_models.git
./pipelineParameterOptimization deeplocalizer_data/images/season_2015/cam2/ \
    --deeplocalizer_param_path deeplocalizer_models/models/conv12_conv48_fc1024_fc_2/model_iter_20000.caffemodel \
    --deeplocalizer_model_path deeplocalizer-data//models/conv12_conv48_fc1024_fc_2/deploy.prototxt \
    --optimize_mean true --n_init_samples 250 --n_iterations 499 --n_iter_relearn 100
```
    
Fair warning: It's probably advisable to get in touch with someone who's used the
parameteroptimization before if you intend to use it ;)

### Citation
> Wario, Fernando, et al. "Automatic methods for long-term tracking and the detection and decoding of communication dances in honeybees." Frontiers in Ecology and Evolution 3 (2015): 103.
