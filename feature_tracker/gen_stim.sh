#!/bin/sh

vEnv=stim_gen

# Activate environment
# conda activate $vEnv


radius=1 #default=1 height and width of the squares in stimuli
# start_sample=0
# last_sample=$(ls -lat | head -2 | tail -1 | awk '{print $9}' | cut -d "_" -f 2)
# echo $last_sample
# arrIN=(${last_sample//_/ })
# echo $arrIN
num_samples=50000 #0000
num_distractors=10 #10 #default=10 number of distractor paths
independent_distractors=0 #default=0
extra_dist=0 #4 #default=4 number of extra distractor paths
HUMAN_MODE=0 #default=0 Generate movie in human mode with lines [REMOVED]
skip_param=1 #default=1 number of coordinates to skip when generating coordinates. Increase speed/path length. MIN:1, MAX:5
path_length=32 #default=64 length of the trajectory, also equals the number of frames in the sequence
NEGATIVE_SAMPLE=1 #default=0 Generate a negative sample of movie
gif=0 #default=0 Generate a gif of movie as well in the same folder as path [REMOVED]
save_image=0 #default=0 save images of the generated sample as .png files
path="./10dist_64speed/32frames/batch_0/" #default=pwd path at which the stimuli should be stored
coordinates=0
echo $coordinates
outer_path="./" #outer path under which the pos and neg stim directories reside
# echo $path
# echo $outer_path
# last_sample=$(ls -lat | head -2 | tail -1 | awk '{print $9}' ) # | cut -d "_" -f 2)
#USE the below logic to restart the code to generate samples from where the last one left off.
#Possible reason for stalling would be not finding the right set of coordinates for randomizing trajectories.
# last_sample=$(ls -lshtr $path | tail -1 | awk '{print $10}' | cut -d "_" -f 3 | cut -d "." -f 1)
# last_sample=-1
# echo $last_sample
# start_sample=$(($last_sample-3))
start_sample=0
echo $start_sample
# echo "Bash version ${BASH_VERSION}..."

# for i in {"$start_sample".."$num_samples"}
#  echo "Welcome $i times"
# echo "$path/sample_$i/"
#  echo $radius
python -u generate_data.py -r $radius -n $num_samples -ss $start_sample -nd $num_distractors -ed $extra_dist -HM $HUMAN_MODE -pl $path_length -sp $skip_param -NS $NEGATIVE_SAMPLE --gif $gif -si $save_image --path $path --independent_distractors $independent_distractors #--path "$path/sample_$i"


# echo $radius
# echo $path
