#!/bin/sh

radius=1 #default=1 height and width of the squares in stimuli
num_samples=50000 #default=50000 number of samples to generate
num_distractors=10 #default=10 number of distractor paths
independent_distractors=0 #default=0
extra_dist=0 #4 #default=4 number of extra distractor paths
HUMAN_MODE=0 #default=0 Generate movie in human mode with lines [REMOVED]
skip_param=1 #default=1 number of coordinates to skip when generating coordinates. Increase speed/path length. MIN:1, MAX:5
path_length=32 #default=32 length of the trajectory, also equals the number of frames in the sequence
NEGATIVE_SAMPLE=1 #default=0 Generate a negative sample of movie
gif=0 #default=0 Generate a gif of movie as well in the same folder as path [REMOVED]
save_image=0 #default=0 save images of the generated sample as .png files
path="./10dist/32frames/batch_0/" #default=pwd path at which the stimuli should be stored
coordinates=0
echo $coordinates
outer_path="./"
start_sample=0
echo $start_sample

python -u generate_data.py -r $radius -n $num_samples -ss $start_sample -nd $num_distractors -ed $extra_dist -HM $HUMAN_MODE -pl $path_length -sp $skip_param -NS $NEGATIVE_SAMPLE --gif $gif -si $save_image --path $path --independent_distractors $independent_distractors #--path "$path/sample_$i"

