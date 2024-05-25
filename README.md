# Tracking objects that change in appearance with phase synchrony

## Generating *FeatureTracker*

The files `utils_shapes.py`, `utils_colors.py`, and `utils_coordinates.py` respectively generate the trajectories of the objects in the shape, color, and position dimensions. `utils_drawing.py` assigns each object to a position in the frame.
`generate_data.py` creates the paths for each condition, generates the videos, and saves them.

The dataset can be generated using `gen_stim.sh`. This file sets and contains a description of each parameter of the dataset (including the length of the video, the number of distractors, the paths to save the videos, ...)
The necessary packages to run the code are listed in `stim_gen.yml`.

