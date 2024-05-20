import argparse
import os

# Basic commandline argument parsing
parser = argparse.ArgumentParser()

parser.add_argument('-r', '--radius', type=int, default=1,
                    help="int: height and width of the squares in stimuli [default 1]")
parser.add_argument('-n', '--num_samples', type=int, default=10,
                    help="int: number of samples to be generated [default 10]")
parser.add_argument('-ss', '--start_sample', type=int, default=0,
                    help="int: number at which to start the sample numbering [default 0]")
parser.add_argument('-nd', '--num_distractors', type=int, default=10,
                    help="int: number of distractor paths [default 10]")
parser.add_argument('-pl', '--path_length', type=int, default=64, help="int: length of the paths [default 64]")
parser.add_argument('-ed', '--extra_dist', type=int, default=4,
                    help="int: number of extra distractors to be added to the movie [defalut 4]")
parser.add_argument('-sp', '--skip_param', type=int, default=True,
                    help="int: (slice step) number of coordinates to skip from the generated set of full coordinates. Increases the speed of points, by increasing their path length (displacement) but keeping the number of frames same (time). MIN: 1. MAX: 5 [defalut 1]")
parser.add_argument('-HM', '--HUMAN_MODE', type=eval, nargs='?', const=True, default=False,
                    help="bool: Activate human mode. Show path lines for all paths and slow down the movie with extra frames [default False]")
parser.add_argument('-NS', '--NEGATIVE_SAMPLE', type=eval, nargs='?', const=True, default=True,
                    help="bool: Generate negative sample [default False]")
parser.add_argument('-g', '--gif', type=eval, const=True, nargs='?', default=False,
                    help="bool: Generate movie frames in gif files, and atore it with the frames [default False]")
parser.add_argument('-si', '--save_image', type=eval, const=True, nargs='?', default=True,
                    help="bool: Store individual frames on disk in individual directories at the specified path. [default False]")
parser.add_argument('-p', '--path', type=str, default='./tests/',
                    help="str: Path to store the folder structure of the generate samples [default current directory]")
parser.add_argument('-b', '--batch', type=str, default="batch_0",
                    help="str: Split of the dataset for testing and training [default batch_0]")
parser.add_argument('-c', '--saved_coordinates', type=eval, const=True, nargs='?', default=False,
                    help="bool: Whether to use already saved coordinates or generate new ones [default False]")
parser.add_argument('-id', '--independent_distractors', type=eval, const=True, nargs='?', default=False,
                    help="bool: Whether to use independent distractors or not [default False]")

args = parser.parse_args()