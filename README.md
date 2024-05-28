# Tracking objects that change in appearance with phase synchrony

## Generating *FeatureTracker*

In the repository `feature_tracker`, the files `utils_shapes.py`, `utils_colors.py`, and `utils_coordinates.py` respectively generate the trajectories of the objects in the shape, color, and position dimensions. `utils_drawing.py` assigns each object to a position in the frame.
`generate_data.py` creates the paths for each condition, generates the videos, and saves them.

The dataset can be generated using `gen_stim.sh`. This file sets and contains a description of each parameter of the dataset (including the length of the video, the number of distractors, the paths to save the videos, ...)

`gen_stim.sh` should be run twice to generate positive (NEGATIVE_SAMPLE=0) and negative samples (NEGATIVE_SAMPLE=1) for each subset.
Similarly, each subset is created by changing the path and creating three locations for *training* (designated as `batch_0`), *validation* (`batch_5`) and *test* (`batch_10`) subset. 

The necessary packages to run the code are listed in `stim_gen.yml`.

## Transforming Numpy videos into Tfr files

Once the videos are generated, the users should create the `pytorch_tracking` environment using the packages from `pytorch_tracking.yml`.
Then, for each condition and each subset, run the file `batch_0_numpy_3d_to_tfrecord.py`.
At lines 104 and 105, change the input directory and output directory according to the paths used during the generation.
Lines 385 and 390 detail the category of the subset. Use `-batch_0-` and `train` for the training subset, `-batch_5-` and `val` for the validation subset, and `-batch_10-` and `test` for the test subset.

## Training models on *FeatureTracker*

Using the environment `pytorch_tracking` (see `pytorch_tracking.yml` file), the models can be trained from the `featuretracker_models` directory. All the models are defined in the `models` folder (including CV-RNN). The `engine.py` contains the functions related to data pre-processing, models' initialization, ...
The training can be launched using `mainclean.py`, using the following command:
`python mainclean.py --model CVInT --print-freq 20 --learning-rate 3e-04 --epochs 2000 -b 64 --log True --length 32 --name CVRNN_feature_tracker --speed 1 --dist 10 --parallel True --data_dir feature_tracker --data_rep idshapes_idcolors --pretrained False`

Similarly, the command to test the models on the different conditions of *FeatureTracker*, is:
`python test_model.py --model CVInT -b 64 --log True --length 32 --name CVRNN_feature_tracker --speed 1 --dist 10 --parallel True --data_dir feature_tracker --data_rep oodcolors_oodshapes`
