import time

import numpy as np
# import math
import random
import argparse
import os
import skimage  # try and use only the draw functions, and not the entire library
import imageio  # can be eliminated altogether if not writing images, just dumping np array
# from scipy.sparse import coo_matrix # make sparse matrix for individual frames, and then write to a bigger np.ndarray
import matplotlib.pyplot as plt
import matplotlib as mpl

from utils_color import generate_all_colors
import utils_shapes as us
import utils_coordinates as uc
import utils_drawing as ud
from opts import args

#fix all seeds
np.random.seed(0)
random.seed(0)


nPoints = 2  # no. of points to be fit using the bernstein polynomial (degree of the polynomial to fit)
nTimes = 22  # 10#16#150 #no. of points in the curve

# if using circle from visual stimulus, use radius
# radius=1#1.5#3#5
radius=args.radius
#dimensions of window
win_x=32-5#460
win_y=32-5#460

square_height = 2
square_width = 2
blob_height = 3
blob_width = 3
channels = 3

pt_rect_height = 2
pt_rect_width = 2

num_distractors = args.num_distractors
HUMAN_MODE = args.HUMAN_MODE
NEGATIVE_SAMPLE = args.NEGATIVE_SAMPLE
path_length = args.path_length  # same as number of frames. To be taken as argparse argument

start_sample = args.start_sample
num_samples = args.num_samples

# if args.save_image:
# check if the directory for sample exists, else make one for positive (1) or negative (0)
# make the directory in any case, given the bigger npz/tfrecords files would also be stored in their respective sample directories
path = args.path  # +str(int(not args.NEGATIVE_SAMPLE))
if not os.path.exists(path):
    os.makedirs(path)

# save coordinates in an individual directory
# add this to the args
path_to_save_coordinates = path + "/coordinates/" + str(args.batch) + "/"
if not os.path.exists(path_to_save_coordinates):
    os.makedirs(path_to_save_coordinates)

# make directories to save the npy files atleast
if not os.path.exists(path + "/pathtracker/samples/"):
    os.makedirs(path + "/pathtracker/samples/")
if not os.path.exists(path + "/pathtracker_color/samples/"):
    os.makedirs(path + "/pathtracker_color/samples/")
if not os.path.exists(path + "/one_pixel_manipulation_no_color/samples/"):
    os.makedirs(path + "/one_pixel_manipulation_no_color/samples/")
if not os.path.exists(path + "/one_pixel_manipulation_color/samples/"):
    os.makedirs(path + "/one_pixel_manipulation_color/samples/")


#generate colorspace
colorspace_variable = generate_all_colors(num_samples, path_length, args.num_distractors + args.extra_dist + 2, 1/64.0)
colorspace_fixed = np.zeros_like(colorspace_variable)
colorspace_fixed[:, :, :] = (0., 255., 0.)

#check if some values are below 1 or above 255
if (colorspace_variable < 0).any():
    print("Some values are below 0")
if (colorspace_variable > 255).any():
    print("Some values are above 255")

np.save(path + "/colorspace.npy", colorspace_variable)

for sample in range(start_sample, num_samples):
    # generate shapes
    start_state = us.start_shapes(args.num_distractors + args.extra_dist + 2)
    #randomly choose between idx=0,1,2
    idx = 0 #random.randint(0, 2)
    squares = us.pathtracker_shapes(args.num_distractors + args.extra_dist + 2, idx)
    # start=time.time()

    # make individual directory for the given sample
    if args.save_image:
        path_to_save_pathtracker = path + "pathtracker/" + str(int(not args.NEGATIVE_SAMPLE)) + "_sample_" + str(
            sample) + "/"
        path_to_save_pathtracker_color = path + "pathtracker_color/" + str(int(not args.NEGATIVE_SAMPLE)) + "_sample_" + str(
            sample) + "/"
        path_to_save_one_pixel_manipulation_no_color = path + "one_pixel_manipulation_no_color/" + str(
            int(not args.NEGATIVE_SAMPLE)) + "_sample_" + str(sample) + "/"
        path_to_save_one_pixel_manipulation_color = path + "one_pixel_manipulation_color/" + str(
            int(not args.NEGATIVE_SAMPLE)) + "_sample_" + str(sample) + "/"

        # print("Saving to: "+path_to_save)

        try:
            os.makedirs(path_to_save_pathtracker)
            os.makedirs(path_to_save_pathtracker_color)
            os.makedirs(path_to_save_one_pixel_manipulation_no_color)
            os.makedirs(path_to_save_one_pixel_manipulation_color)
        except OSError:
            print ("Creation of the directory %s failed" % path_to_save_pathtracker)


    # generate the coordinates for the path
    points = uc.get_points(nPoints)
    normalized = set([])


    if not args.saved_coordinates:
        coordinates_2, normalized, non_floor_coordinates_2 = uc.get_full_length_coordinates(
            nPoints, nTimes * 3, normalized, path_length, args.skip_param, win_x, win_y, positive=not args.NEGATIVE_SAMPLE) #, non_floor_coordinates_3, coordinates_3

        normalized = [0] * (
                args.num_distractors + args.extra_dist)  # DO NOT DISPLACE. CONSTRAIN THE COORDINATES TO BE IN THE DEFINED VISIBLE VISUAL SCREEN

        coordinates_d, normalized = uc.get_third_length_distractor_coordinates(nPoints, nTimes * 3, normalized,
                                                                            num_distractors, path_length, args.skip_param, win_x, win_y)
        #get distractor with closest end point to end of coordinates_2
        min_distance = 100
        for i in range(num_distractors):
            dist = np.sqrt(((coordinates_2[-1][0] - coordinates_d[i][-1][0]) ** 2) + ((coordinates_2[-1][1] - coordinates_d[i][-1][1]) ** 2))
            if dist < min_distance:
                min_distance = dist
                closest_distractor = i
        coordinates_3 = coordinates_d[closest_distractor] #, normalized[closest_distractor]

        # adding 4 extra distractors
        # 2 at the begining and end of the sequence
        extra_dist = args.extra_dist
        coordinates_e, normalized = uc.get_third_length_distractor_coordinates(nPoints, nTimes * 3, normalized, extra_dist,
                                                                            path_length, args.skip_param, win_x, win_y)
    else:
        coordinates = np.load(
            path + "/coordinates/" + str(args.batch) + "/" + str(int(not args.NEGATIVE_SAMPLE)) + "_sample_" + str(
                sample) + ".npy")
        coordinates_2 = coordinates[0]
        coordinates_3 = coordinates[1]
        coordinates_d = coordinates[2:2 + num_distractors]
        if args.extra_dist:
            coordinates_e = coordinates[2 + num_distractors:2 + num_distractors + args.extra_dist]
        else:
            coordinates_e = []

    images_pathtracker = np.zeros((path_length, 32, 32, channels), dtype=np.uint8)
    mask_pathtracker = np.zeros((path_length, 32, 32, channels), dtype=np.uint8)
    images_pathtracker_color = np.zeros((path_length, 32, 32, channels), dtype=np.uint8)
    mask_pathtracker_color = np.zeros((path_length, 32, 32, channels), dtype=np.uint8)
    images_one_pixel_manipulation_no_color = np.zeros((path_length, 32, 32, channels), dtype=np.uint8)
    mask_one_pixel_manipulation_no_color = np.zeros((path_length, 32, 32, channels), dtype=np.uint8)
    images_one_pixel_manipulation_color = np.zeros((path_length, 32, 32, channels), dtype=np.uint8)
    mask_one_pixel_manipulation_color = np.zeros((path_length, 32, 32, channels), dtype=np.uint8)

    images_pathtracker[0] = ud.plot_frame(sample, images_pathtracker[0], 0, squares, colorspace_fixed,
                                          coordinates_2, coordinates_3, coordinates_d, coordinates_e, not args.NEGATIVE_SAMPLE)
    mask_pathtracker[0] = ud.plot_frame_mask(mask_pathtracker[0], 0, squares,
                                              coordinates_2, coordinates_3, coordinates_d, coordinates_e,
                                             not args.NEGATIVE_SAMPLE, args.independent_distractors)
    images_pathtracker_color[0] = ud.plot_frame(sample, images_pathtracker_color[0], 0, squares, colorspace_variable,
                                                coordinates_2, coordinates_3, coordinates_d, coordinates_e, not args.NEGATIVE_SAMPLE)
    mask_pathtracker_color[0] = ud.plot_frame_mask(mask_pathtracker_color[0], 0, squares,
                                                    coordinates_2, coordinates_3, coordinates_d, coordinates_e,
                                                   not args.NEGATIVE_SAMPLE, args.independent_distractors)
    images_one_pixel_manipulation_no_color[0] = ud.plot_frame(sample, images_one_pixel_manipulation_no_color[0], 0,
                                                              start_state, colorspace_fixed, coordinates_2,
                                                              coordinates_3, coordinates_d, coordinates_e, not args.NEGATIVE_SAMPLE)
    mask_one_pixel_manipulation_no_color[0] = ud.plot_frame_mask(mask_one_pixel_manipulation_no_color[0], 0,
                                                                  start_state, coordinates_2,
                                                                  coordinates_3, coordinates_d, coordinates_e,
                                                                 not args.NEGATIVE_SAMPLE, args.independent_distractors)
    images_one_pixel_manipulation_color[0] = ud.plot_frame(sample, images_one_pixel_manipulation_color[0], 0,
                                                           start_state, colorspace_variable, coordinates_2,
                                                           coordinates_3, coordinates_d, coordinates_e,
                                                           not args.NEGATIVE_SAMPLE, args.independent_distractors)
    mask_one_pixel_manipulation_color[0] = ud.plot_frame_mask(mask_one_pixel_manipulation_color[0], 0,
                                                                start_state, coordinates_2,
                                                                coordinates_3, coordinates_d, coordinates_e,
                                                              not args.NEGATIVE_SAMPLE, args.independent_distractors)

    if args.save_image:
        imageio.imwrite(path_to_save_pathtracker + "/frame_" + str(0) + ".png", images_pathtracker[0])
        imageio.imwrite(path_to_save_pathtracker + "/mask_" + str(0) + ".png", mask_pathtracker[0])
        imageio.imwrite(path_to_save_pathtracker_color + "/frame_" + str(0) + ".png", images_pathtracker_color[0])
        imageio.imwrite(path_to_save_pathtracker_color + "/mask_" + str(0) + ".png", mask_pathtracker_color[0])
        imageio.imwrite(path_to_save_one_pixel_manipulation_no_color + "/frame_" + str(0) + ".png",
                        images_one_pixel_manipulation_no_color[0])
        imageio.imwrite(path_to_save_one_pixel_manipulation_no_color + "/mask_" + str(0) + ".png",
                        mask_one_pixel_manipulation_no_color[0])
        imageio.imwrite(path_to_save_one_pixel_manipulation_color + "/frame_" + str(0) + ".png",
                        images_one_pixel_manipulation_color[0])
        imageio.imwrite(path_to_save_one_pixel_manipulation_color + "/mask_" + str(0) + ".png",
                        mask_one_pixel_manipulation_color[0])

    frames_state = start_state
    for f in range(1, path_length-1):
        images_pathtracker[f] = ud.plot_frame(sample, images_pathtracker[f], f, squares,
                                               colorspace_fixed,
                                               coordinates_2, coordinates_3, coordinates_d, coordinates_e, not args.NEGATIVE_SAMPLE)
        mask_pathtracker[f] = ud.plot_frame_mask(mask_pathtracker[f], f, squares,
                                               coordinates_2, coordinates_3, coordinates_d, coordinates_e,
                                                 not args.NEGATIVE_SAMPLE, args.independent_distractors)
        images_pathtracker_color[f] = ud.plot_frame(sample, images_pathtracker_color[f], f, squares,
                                               colorspace_variable,
                                               coordinates_2, coordinates_3, coordinates_d, coordinates_e, not args.NEGATIVE_SAMPLE)
        mask_pathtracker_color[f] = ud.plot_frame_mask(mask_pathtracker_color[f], f, squares,
                                                  coordinates_2, coordinates_3, coordinates_d, coordinates_e,
                                                       not args.NEGATIVE_SAMPLE, args.independent_distractors)

        frames_state = us.frames_shapes(args.num_distractors + args.extra_dist + 2, frames_state)

        images_one_pixel_manipulation_no_color[f,:,:] = ud.plot_frame(sample, images_one_pixel_manipulation_no_color[f], f, frames_state,
                                               colorspace_fixed,
                                               coordinates_2, coordinates_3, coordinates_d, coordinates_e, not args.NEGATIVE_SAMPLE)
        mask_one_pixel_manipulation_no_color[f,:,:] = ud.plot_frame_mask(mask_one_pixel_manipulation_no_color[f], f, frames_state,
                                                  coordinates_2, coordinates_3, coordinates_d, coordinates_e, not args.NEGATIVE_SAMPLE, args.independent_distractors)

        images_one_pixel_manipulation_color[f,:,:] = ud.plot_frame(sample, images_one_pixel_manipulation_color[f], f, frames_state,
                                               colorspace_variable,
                                               coordinates_2, coordinates_3, coordinates_d, coordinates_e, not args.NEGATIVE_SAMPLE)
        mask_one_pixel_manipulation_color[f,:,:] = ud.plot_frame_mask(mask_one_pixel_manipulation_color[f], f, frames_state,
                                                    coordinates_2, coordinates_3, coordinates_d, coordinates_e, not args.NEGATIVE_SAMPLE, args.independent_distractors)

        if args.save_image:
            imageio.imwrite(path_to_save_pathtracker + "/frame_" + str(f) + ".png", images_pathtracker[f])
            imageio.imwrite(path_to_save_pathtracker + "/mask_" + str(f) + ".png", mask_pathtracker[f])
            imageio.imwrite(path_to_save_pathtracker_color + "/frame_" + str(f) + ".png", images_pathtracker_color[f])
            imageio.imwrite(path_to_save_pathtracker_color + "/mask_" + str(f) + ".png", mask_pathtracker_color[f])
            imageio.imwrite(path_to_save_one_pixel_manipulation_no_color + "/frame_" + str(f) + ".png", images_one_pixel_manipulation_no_color[f])
            imageio.imwrite(path_to_save_one_pixel_manipulation_no_color + "/mask_" + str(f) + ".png", mask_one_pixel_manipulation_no_color[f])
            imageio.imwrite(path_to_save_one_pixel_manipulation_color + "/frame_" + str(f) + ".png", images_one_pixel_manipulation_color[f])
            imageio.imwrite(path_to_save_one_pixel_manipulation_color + "/mask_" + str(f) + ".png", mask_one_pixel_manipulation_color[f])


    end_state = us.end_shapes(args.num_distractors + args.extra_dist + 2, frames_state)
    images_pathtracker[-1] = ud.plot_frame(sample, images_pathtracker[-1], -1, squares, colorspace_fixed,
                                           coordinates_2, coordinates_3, coordinates_d, coordinates_e,
                                           not args.NEGATIVE_SAMPLE)
    mask_pathtracker[-1] = ud.plot_frame_mask(mask_pathtracker[-1], -1, squares,
                                                coordinates_2, coordinates_3, coordinates_d, coordinates_e,
                                                not args.NEGATIVE_SAMPLE, args.independent_distractors)
    images_pathtracker_color[-1] = ud.plot_frame(sample, images_pathtracker_color[-1], -1, squares, colorspace_variable,
                                                 coordinates_2, coordinates_3, coordinates_d, coordinates_e,
                                                 not args.NEGATIVE_SAMPLE)
    mask_pathtracker_color[-1] = ud.plot_frame_mask(mask_pathtracker_color[-1], -1, squares,
                                                     coordinates_2, coordinates_3, coordinates_d, coordinates_e,
                                                     not args.NEGATIVE_SAMPLE, args.independent_distractors)
    images_one_pixel_manipulation_no_color[-1] = ud.plot_frame(sample, images_one_pixel_manipulation_no_color[-1], -1,
                                                               end_state, colorspace_fixed, coordinates_2,
                                                               coordinates_3, coordinates_d, coordinates_e,
                                                               not args.NEGATIVE_SAMPLE)
    mask_one_pixel_manipulation_no_color[-1] = ud.plot_frame_mask(mask_one_pixel_manipulation_no_color[-1], -1,
                                                                     end_state, coordinates_2,
                                                                     coordinates_3, coordinates_d, coordinates_e,
                                                                     not args.NEGATIVE_SAMPLE, args.independent_distractors)
    images_one_pixel_manipulation_color[-1] = ud.plot_frame(sample, images_one_pixel_manipulation_color[-1], -1,
                                                            end_state, colorspace_variable, coordinates_2,
                                                            coordinates_3, coordinates_d, coordinates_e,
                                                            not args.NEGATIVE_SAMPLE)
    mask_one_pixel_manipulation_color[-1] = ud.plot_frame_mask(mask_one_pixel_manipulation_color[-1], -1,
                                                                 end_state, coordinates_2,
                                                                 coordinates_3, coordinates_d, coordinates_e,
                                                                 not args.NEGATIVE_SAMPLE, args.independent_distractors)

    if args.save_image:
        imageio.imwrite(path_to_save_pathtracker + "/frame_" + str(path_length-1) + ".png", images_pathtracker[-1])
        imageio.imwrite(path_to_save_pathtracker + "/mask_" + str(path_length-1) + ".png", mask_pathtracker[-1])
        imageio.imwrite(path_to_save_pathtracker_color + "/frame_" + str(path_length-1) + ".png", images_pathtracker_color[-1])
        imageio.imwrite(path_to_save_pathtracker_color + "/mask_" + str(path_length-1) + ".png", mask_pathtracker_color[-1])
        imageio.imwrite(path_to_save_one_pixel_manipulation_no_color + "/frame_" + str(path_length-1) + ".png",
                        images_one_pixel_manipulation_no_color[-1])
        imageio.imwrite(path_to_save_one_pixel_manipulation_no_color + "/mask_" + str(path_length-1) + ".png",
                        mask_one_pixel_manipulation_no_color[-1])
        imageio.imwrite(path_to_save_one_pixel_manipulation_color + "/frame_" + str(path_length-1) + ".png",
                        images_one_pixel_manipulation_color[-1])
        imageio.imwrite(path_to_save_one_pixel_manipulation_color + "/mask_" + str(path_length-1) + ".png",
                        mask_one_pixel_manipulation_color[-1])

    images_pathtracker = np.stack((images_pathtracker, mask_pathtracker), axis=0)
    np.save(
        path + "pathtracker/samples/" + str(int(not args.NEGATIVE_SAMPLE)) + "_sample_" + str(
            sample), images_pathtracker, allow_pickle=False)
    images_pathtracker_color = np.stack((images_pathtracker_color, mask_pathtracker_color), axis=0)
    np.save(
        path + "pathtracker_color/samples/" + str(int(not args.NEGATIVE_SAMPLE)) + "_sample_" + str(
            sample), images_pathtracker_color, allow_pickle=False)
    images_one_pixel_manipulation_no_color = np.stack((images_one_pixel_manipulation_no_color, mask_one_pixel_manipulation_no_color), axis=0)
    np.save(
        path + "one_pixel_manipulation_no_color/samples/" + str(int(not args.NEGATIVE_SAMPLE)) + "_sample_" + str(
            sample), images_one_pixel_manipulation_no_color, allow_pickle=False)
    images_one_pixel_manipulation_color = np.stack((images_one_pixel_manipulation_color, mask_one_pixel_manipulation_color), axis=0)
    np.save(
        path + "one_pixel_manipulation_color/samples/" + str(int(not args.NEGATIVE_SAMPLE)) + "_sample_" + str(
            sample), images_one_pixel_manipulation_color, allow_pickle=False)

    # save coordinates to npy array
    coordinates_all = np.stack((np.array(coordinates_2), np.array(coordinates_3)))
    if len(coordinates_d) > 0:
        # if there are distractors in the movie
        coordinates_all = np.vstack((coordinates_all, np.array(coordinates_d)))
    if len(coordinates_e) > 0:
        # if there are extra distractors in the movie
        # this option exists to fine tune certain distractors other than the ones that are always present in the movie
        # something like this should always be present at the start/middle/end of the movie
        coordinates_all = np.vstack((coordinates_all, np.array(coordinates_e)))

    np.save(
        path + "coordinates/" + str(args.batch) + "/" + str(int(not args.NEGATIVE_SAMPLE)) + "_sample_" + str(sample),
        coordinates_all, allow_pickle=False)

