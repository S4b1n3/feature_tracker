import numpy as np
import random
import os
import imageio

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


radius=args.radius
#dimensions of window
win_x=32-5
win_y=32-5

#dimension of objects
square_height = 2
square_width = 2
blob_height = 3
blob_width = 3
channels = 3

pt_rect_height = 2
pt_rect_width = 2

#other parameters
num_distractors = args.num_distractors
HUMAN_MODE = args.HUMAN_MODE
NEGATIVE_SAMPLE = args.NEGATIVE_SAMPLE
path_length = args.path_length

start_sample = args.start_sample
num_samples = args.num_samples

#path to save the images
path = args.path
if not os.path.exists(path):
    os.makedirs(path)
path_to_save_coordinates = path + "/coordinates/" + str(args.batch) + "/"
path_to_save_coordinates_irregular = path + "/coordinates_irregular/" + str(args.batch) + "/"
if not os.path.exists(path_to_save_coordinates):
    os.makedirs(path_to_save_coordinates)
if not os.path.exists(path_to_save_coordinates_irregular):
    os.makedirs(path_to_save_coordinates_irregular)

# make directories to save the npy files atleast
if not os.path.exists(path + "/noshapes_nocolors/samples/"):
    os.makedirs(path + "/noshapes_nocolors/samples/")
if not os.path.exists(path + "/noshapes_idcolors/samples/"):
    os.makedirs(path + "/noshapes_idcolors/samples/")
if not os.path.exists(path + "/idshapes_nocolors/samples/"):
    os.makedirs(path + "/idshapes_nocolors/samples/")
if not os.path.exists(path + "/idshapes_idcolors/samples/"):
    os.makedirs(path + "/idshapes_idcolors/samples/")
if not os.path.exists(path + "/idshapes_oodcolors/samples/"):
    os.makedirs(path + "/idshapes_oodcolors/samples/")
if not os.path.exists(path + "/oodshapes_idcolors/samples/"):
    os.makedirs(path + "/oodshapes_idcolors/samples/")
if not os.path.exists(path + "/oodshapes_oodcolors/samples/"):
    os.makedirs(path + "/oodshapes_oodcolors/samples/")
if not os.path.exists(path + "/idshapes_irregularcolors/"):
    os.makedirs(path + "/idshapes_irregularcolors/")
if not os.path.exists(path + "/idshapes_idcolors_irregular/"):
    os.makedirs(path + "/idshapes_idcolors_irregular/")


#generate colorspace
colorspace_variable_id = generate_all_colors(num_samples, path_length, args.num_distractors + args.extra_dist + 2, True, 1/64.0)
colorspace_variable_ood = generate_all_colors(num_samples, path_length, args.num_distractors + args.extra_dist + 2, False, 1/64.0)
colorspace_variable_irregular = generate_all_colors(num_samples, path_length, args.num_distractors + args.extra_dist + 2, True, np.random.uniform(0.02, 0.05))
colorspace_fixed = np.zeros_like(colorspace_variable_id)
colorspace_fixed[:, :, :] = (0., 255., 0.)

#check if some values are below 1 or above 255
if (colorspace_variable_id < 0).any():
    print("Some values are below 0")
if (colorspace_variable_id > 255).any():
    print("Some values are above 255")

np.save(path + "/colorspace_id.npy", colorspace_variable_id)
np.save(path + "/colorspace_ood.npy", colorspace_variable_ood)
np.save(path + "/colorspace_irregular.npy", colorspace_variable_irregular)

#generate videos for each condition
for sample in range(start_sample, num_samples):
    # generate shapes
    start_state_id = us.start_shapes_id(args.num_distractors + args.extra_dist + 2)
    start_state_ood = us.start_shapes_ood(args.num_distractors + args.extra_dist + 2)
    squares = us.squares(args.num_distractors + args.extra_dist + 2)

    # make individual directory for the given sample
    if args.save_image:
        path_to_save_noshapes_nocolors = path + "noshapes_nocolors/" + str(int(not args.NEGATIVE_SAMPLE)) + "_sample_" + str(
            sample) + "/"
        path_to_save_noshapes_idcolors = path + "noshapes_idcolors/" + str(int(not args.NEGATIVE_SAMPLE)) + "_sample_" + str(
            sample) + "/"
        path_to_save_idshapes_nocolors = path + "idshapes_nocolors/" + str(
            int(not args.NEGATIVE_SAMPLE)) + "_sample_" + str(sample) + "/"
        path_to_save_idshapes_idcolors = path + "idshapes_idcolors/" + str(
            int(not args.NEGATIVE_SAMPLE)) + "_sample_" + str(sample) + "/"
        path_to_save_oodshapes_idcolors = path + "oodshapes_idcolors/" + str(
            int(not args.NEGATIVE_SAMPLE)) + "_sample_" + str(sample) + "/"
        path_to_save_idshapes_oodcolors = path + "idshapes_oodcolors/" + str(
            int(not args.NEGATIVE_SAMPLE)) + "_sample_" + str(sample) + "/"
        path_to_save_oodshapes_oodcolors = path + "oodshapes_oodcolors/" + str(
            int(not args.NEGATIVE_SAMPLE)) + "_sample_" + str(sample) + "/"
        path_to_save_idshapes_irregularcolors = path + "idshapes_irregularcolors/" + str(
            int(not args.NEGATIVE_SAMPLE)) + "_sample_" + str(sample) + "/"
        path_to_save_idshapes_idcolors_irregular = path + "idshapes_idcolors_irregular/" + str(
            int(not args.NEGATIVE_SAMPLE)) + "_sample_" + str(sample) + "/"

        # print("Saving to: "+path_to_save)

        try:
            os.makedirs(path_to_save_noshapes_nocolors)
            os.makedirs(path_to_save_noshapes_idcolors)
            os.makedirs(path_to_save_idshapes_nocolors)
            os.makedirs(path_to_save_idshapes_idcolors)
            os.makedirs(path_to_save_oodshapes_idcolors)
            os.makedirs(path_to_save_idshapes_oodcolors)
            os.makedirs(path_to_save_oodshapes_oodcolors)
            os.makedirs(path_to_save_idshapes_irregularcolors)
            os.makedirs(path_to_save_idshapes_idcolors_irregular)
        except OSError:
            print ("Creation of the directory %s failed" % path_to_save_idshapes_idcolors)


    # generate the coordinates for the path
    points = uc.get_points(nPoints)
    normalized = set([])
    normalized_irregular = set([])


    if not args.saved_coordinates:
        coordinates_2, normalized, non_floor_coordinates_2 = uc.get_full_length_coordinates(
            nPoints, nTimes * 3, normalized, path_length, args.skip_param, win_x, win_y) #, non_floor_coordinates_3, coordinates_3
        coordinates_2_irregular, normalized_irregular, non_floor_coordinates_2_irregular = uc.get_full_length_coordinates(
            nPoints, nTimes * 3, normalized_irregular, path_length, args.skip_param, win_x,
            win_y, angle_range=[10,170], delta_angle_max=180)  # , non_floor_coordinates_3, coordinates_3

        normalized = [0] * (args.num_distractors + args.extra_dist)  # DO NOT DISPLACE. CONSTRAIN THE COORDINATES TO BE IN THE DEFINED VISIBLE VISUAL SCREEN
        normalized_irregular = [0] * (args.num_distractors + args.extra_dist)

        coordinates_d, normalized = uc.get_third_length_distractor_coordinates(nPoints, nTimes * 3, normalized,
                                                        num_distractors, path_length, args.skip_param, win_x, win_y)
        coordinates_d_irregular, normalized_irregular = uc.get_third_length_distractor_coordinates(nPoints, nTimes * 3,
                                                        normalized_irregular, num_distractors, path_length,
                                                        args.skip_param, win_x, win_y, angle_range=[10, 170], delta_angle_max=180)

        #get distractor with closest end point to end of coordinates_2
        min_distance = 100
        min_distance_irregular = 100
        for i in range(num_distractors):
            dist = np.sqrt(((coordinates_2[-1][0] - coordinates_d[i][-1][0]) ** 2) + ((coordinates_2[-1][1] - coordinates_d[i][-1][1]) ** 2))
            dist_irregular = np.sqrt(((coordinates_2_irregular[-1][0] - coordinates_d_irregular[i][-1][0]) ** 2) + ((coordinates_2_irregular[-1][1] - coordinates_d_irregular[i][-1][1]) ** 2))
            if dist < min_distance:
                min_distance = dist
                closest_distractor = i
            if dist_irregular < min_distance_irregular:
                min_distance_irregular = dist_irregular
                closest_distractor_irregular = i
        coordinates_3 = coordinates_d[closest_distractor]
        coordinates_3_irregular = coordinates_d_irregular[closest_distractor_irregular]

        extra_dist = args.extra_dist
        coordinates_e, normalized = uc.get_third_length_distractor_coordinates(nPoints, nTimes * 3, normalized, extra_dist,
                                                                            path_length, args.skip_param, win_x, win_y)
        coordinates_e_irregular, normalized_irregular = uc.get_third_length_distractor_coordinates(nPoints, nTimes * 3,
                                                                            normalized_irregular, extra_dist, path_length,
                                                                            args.skip_param, win_x, win_y, angle_range=[10, 170], delta_angle_max=180)
    else:
        coordinates = np.load(
            path + "/coordinates/" + str(args.batch) + "/" + str(int(not args.NEGATIVE_SAMPLE)) + "_sample_" + str(
                sample) + ".npy")
        coordinates_irregular = np.load(
            path + "/coordinates_irregular/" + str(args.batch) + "/" + str(int(not args.NEGATIVE_SAMPLE)) + "_sample_" + str(
                sample) + ".npy")
        coordinates_2 = coordinates[0]
        coordinates_2_irregular = coordinates_irregular[0]
        coordinates_3 = coordinates[1]
        coordinates_3_irregular = coordinates_irregular[1]
        coordinates_d = coordinates[2:2 + num_distractors]
        coordinates_d_irregular = coordinates_irregular[2:2 + num_distractors]
        if args.extra_dist:
            coordinates_e = coordinates[2 + num_distractors:2 + num_distractors + args.extra_dist]
            coordinates_e_irregular = coordinates_irregular[2 + num_distractors:2 + num_distractors + args.extra_dist]
        else:
            coordinates_e = []
            coordinates_e_irregular = []

    images_noshapes_nocolors = np.zeros((path_length, 32, 32, channels), dtype=np.uint8)
    mask_noshapes_nocolors = np.zeros((path_length, 32, 32, channels), dtype=np.uint8)
    images_noshapes_idcolors = np.zeros((path_length, 32, 32, channels), dtype=np.uint8)
    mask_noshapes_idcolors = np.zeros((path_length, 32, 32, channels), dtype=np.uint8)
    images_idshapes_nocolors = np.zeros((path_length, 32, 32, channels), dtype=np.uint8)
    mask_idshapes_nocolors = np.zeros((path_length, 32, 32, channels), dtype=np.uint8)
    images_idshapes_idcolors = np.zeros((path_length, 32, 32, channels), dtype=np.uint8)
    mask_idshapes_idcolors = np.zeros((path_length, 32, 32, channels), dtype=np.uint8)
    images_idshapes_oodcolors = np.zeros((path_length, 32, 32, channels), dtype=np.uint8)
    mask_idshapes_oodcolors = np.zeros((path_length, 32, 32, channels), dtype=np.uint8)
    images_oodshapes_idcolors = np.zeros((path_length, 32, 32, channels), dtype=np.uint8)
    mask_oodshapes_idcolors = np.zeros((path_length, 32, 32, channels), dtype=np.uint8)
    images_oodshapes_oodcolors = np.zeros((path_length, 32, 32, channels), dtype=np.uint8)
    mask_oodshapes_oodcolors = np.zeros((path_length, 32, 32, channels), dtype=np.uint8)
    images_idshapes_irregularcolors = np.zeros((path_length, 32, 32, channels), dtype=np.uint8)
    mask_idshapes_irregularcolors = np.zeros((path_length, 32, 32, channels), dtype=np.uint8)
    images_idshapes_idcolors_irregular = np.zeros((path_length, 32, 32, channels), dtype=np.uint8)
    mask_idshapes_idcolors_irregular = np.zeros((path_length, 32, 32, channels), dtype=np.uint8)

    #draw the first frame
    images_noshapes_nocolors[0] = ud.plot_frame(sample, images_noshapes_nocolors[0], 0, squares, colorspace_fixed,
                                                                coordinates_2, coordinates_3, coordinates_d,
                                                                coordinates_e, not args.NEGATIVE_SAMPLE)
    mask_noshapes_nocolors[0] = ud.plot_frame_mask(mask_noshapes_nocolors[0], 0, squares,
                                                                coordinates_2, coordinates_3, coordinates_d, coordinates_e,
                                                                not args.NEGATIVE_SAMPLE, args.independent_distractors)
    images_noshapes_idcolors[0] = ud.plot_frame(sample, images_noshapes_idcolors[0], 0, squares, colorspace_variable_id,
                                                                coordinates_2, coordinates_3, coordinates_d,
                                                                coordinates_e, not args.NEGATIVE_SAMPLE)
    mask_noshapes_idcolors[0] = ud.plot_frame_mask(mask_noshapes_idcolors[0], 0, squares,
                                                                coordinates_2, coordinates_3, coordinates_d, coordinates_e,
                                                                not args.NEGATIVE_SAMPLE, args.independent_distractors)
    images_idshapes_nocolors[0] = ud.plot_frame(sample, images_idshapes_nocolors[0], 0,
                                                                start_state_id, colorspace_fixed, coordinates_2,
                                                                coordinates_3, coordinates_d, coordinates_e,
                                                                not args.NEGATIVE_SAMPLE)
    images_idshapes_nocolors[0] = ud.plot_frame_mask(images_idshapes_nocolors[0], 0,
                                                                start_state_id, coordinates_2,
                                                                coordinates_3, coordinates_d, coordinates_e,
                                                                not args.NEGATIVE_SAMPLE, args.independent_distractors)
    images_idshapes_idcolors[0] = ud.plot_frame(sample, images_idshapes_idcolors[0], 0,
                                                                start_state_id, colorspace_variable_id, coordinates_2,
                                                                coordinates_3, coordinates_d, coordinates_e,
                                                                not args.NEGATIVE_SAMPLE, args.independent_distractors)
    mask_idshapes_idcolors[0] = ud.plot_frame_mask(mask_idshapes_idcolors[0], 0,
                                                                start_state_id, coordinates_2,
                                                                coordinates_3, coordinates_d, coordinates_e,
                                                                not args.NEGATIVE_SAMPLE, args.independent_distractors)
    images_idshapes_oodcolors[0] = ud.plot_frame(sample, images_idshapes_oodcolors[0], 0,
                                                                start_state_id, colorspace_variable_ood, coordinates_2,
                                                                coordinates_3, coordinates_d, coordinates_e,
                                                                not args.NEGATIVE_SAMPLE, args.independent_distractors)
    mask_idshapes_oodcolors[0] = ud.plot_frame_mask(mask_idshapes_oodcolors[0], 0,
                                                                start_state_id, coordinates_2,
                                                                coordinates_3, coordinates_d, coordinates_e,
                                                                not args.NEGATIVE_SAMPLE, args.independent_distractors)
    images_oodshapes_idcolors[0] = ud.plot_frame(sample, images_oodshapes_idcolors[0], 0,
                                                                start_state_ood, colorspace_variable_id, coordinates_2,
                                                                coordinates_3, coordinates_d, coordinates_e,
                                                                not args.NEGATIVE_SAMPLE, args.independent_distractors)
    mask_oodshapes_idcolors[0] = ud.plot_frame_mask(mask_oodshapes_idcolors[0], 0,
                                                                start_state_ood, coordinates_2,
                                                                coordinates_3, coordinates_d, coordinates_e,
                                                                not args.NEGATIVE_SAMPLE, args.independent_distractors)
    images_oodshapes_oodcolors[0] = ud.plot_frame(sample, images_oodshapes_oodcolors[0], 0,
                                                                start_state_ood, colorspace_variable_ood, coordinates_2,
                                                                coordinates_3, coordinates_d, coordinates_e,
                                                                not args.NEGATIVE_SAMPLE, args.independent_distractors)
    mask_oodshapes_oodcolors[0] = ud.plot_frame_mask(mask_oodshapes_oodcolors[0], 0,
                                                                start_state_ood, coordinates_2,
                                                                coordinates_3, coordinates_d, coordinates_e,
                                                                not args.NEGATIVE_SAMPLE, args.independent_distractors)
    images_idshapes_irregularcolors[0] = ud.plot_frame(sample, images_idshapes_irregularcolors[0], 0,
                                                                start_state_id, colorspace_variable_irregular, coordinates_2,
                                                                coordinates_3, coordinates_d, coordinates_e,
                                                                not args.NEGATIVE_SAMPLE, args.independent_distractors)
    mask_idshapes_irregularcolors[0] = ud.plot_frame_mask(mask_idshapes_irregularcolors[0], 0,
                                                                start_state_id, coordinates_2,
                                                                coordinates_3, coordinates_d, coordinates_e,
                                                                not args.NEGATIVE_SAMPLE, args.independent_distractors)
    images_idshapes_idcolors_irregular[0] = ud.plot_frame(sample, images_idshapes_idcolors_irregular[0], 0,
                                                                start_state_id, colorspace_variable_id, coordinates_2_irregular,
                                                                coordinates_3_irregular, coordinates_d_irregular, coordinates_e_irregular,
                                                                not args.NEGATIVE_SAMPLE, args.independent_distractors)
    mask_idshapes_idcolors_irregular[0] = ud.plot_frame_mask(mask_idshapes_idcolors_irregular[0], 0,
                                                                start_state_id, coordinates_2_irregular,
                                                                coordinates_3_irregular, coordinates_d_irregular, coordinates_e_irregular,
                                                                not args.NEGATIVE_SAMPLE, args.independent_distractors)


    if args.save_image:
        imageio.imwrite(path_to_save_noshapes_nocolors + "/frame_" + str(0) + ".png", images_noshapes_nocolors[0])
        imageio.imwrite(path_to_save_noshapes_nocolors + "/mask_" + str(0) + ".png", mask_noshapes_nocolors[0])
        imageio.imwrite(path_to_save_noshapes_idcolors + "/frame_" + str(0) + ".png", images_noshapes_idcolors[0])
        imageio.imwrite(path_to_save_noshapes_idcolors + "/mask_" + str(0) + ".png", mask_noshapes_idcolors[0])
        imageio.imwrite(path_to_save_idshapes_nocolors + "/frame_" + str(0) + ".png", images_idshapes_nocolors[0])
        imageio.imwrite(path_to_save_idshapes_nocolors + "/mask_" + str(0) + ".png", mask_idshapes_nocolors[0])
        imageio.imwrite(path_to_save_idshapes_idcolors + "/frame_" + str(0) + ".png", images_idshapes_idcolors[0])
        imageio.imwrite(path_to_save_idshapes_idcolors + "/mask_" + str(0) + ".png", mask_idshapes_idcolors[0])
        imageio.imwrite(path_to_save_idshapes_oodcolors + "/frame_" + str(0) + ".png", images_idshapes_oodcolors[0])
        imageio.imwrite(path_to_save_idshapes_oodcolors + "/mask_" + str(0) + ".png", mask_idshapes_oodcolors[0])
        imageio.imwrite(path_to_save_oodshapes_idcolors + "/frame_" + str(0) + ".png", images_oodshapes_idcolors[0])
        imageio.imwrite(path_to_save_oodshapes_idcolors + "/mask_" + str(0) + ".png", mask_oodshapes_idcolors[0])
        imageio.imwrite(path_to_save_oodshapes_oodcolors + "/frame_" + str(0) + ".png", images_oodshapes_oodcolors[0])
        imageio.imwrite(path_to_save_oodshapes_oodcolors + "/mask_" + str(0) + ".png", mask_oodshapes_oodcolors[0])
        imageio.imwrite(path_to_save_idshapes_irregularcolors + "/frame_" + str(0) + ".png", images_idshapes_irregularcolors[0])
        imageio.imwrite(path_to_save_idshapes_irregularcolors + "/mask_" + str(0) + ".png", mask_idshapes_irregularcolors[0])
        imageio.imwrite(path_to_save_idshapes_idcolors_irregular + "/frame_" + str(0) + ".png", images_idshapes_idcolors_irregular[0])
        imageio.imwrite(path_to_save_idshapes_idcolors_irregular + "/mask_" + str(0) + ".png", mask_idshapes_idcolors_irregular[0])

    #draw the rest of the frames
    frames_state_id = start_state_id
    frames_state_ood = start_state_ood
    for f in range(1, path_length-1):
        images_noshapes_nocolors[f] = ud.plot_frame(sample, images_noshapes_nocolors[f], f, squares,
                                                colorspace_fixed,
                                                coordinates_2, coordinates_3, coordinates_d,
                                                coordinates_e, not args.NEGATIVE_SAMPLE)
        mask_noshapes_nocolors[f] = ud.plot_frame_mask(mask_noshapes_nocolors[f], f, squares,
                                                coordinates_2, coordinates_3, coordinates_d, coordinates_e,
                                                not args.NEGATIVE_SAMPLE, args.independent_distractors)
        images_noshapes_idcolors[f] = ud.plot_frame(sample, images_noshapes_idcolors[f], f, squares,
                                                colorspace_variable_id,
                                                coordinates_2, coordinates_3, coordinates_d, coordinates_e,
                                                not args.NEGATIVE_SAMPLE)
        mask_noshapes_idcolors[f] = ud.plot_frame_mask(mask_noshapes_idcolors[f], f, squares,
                                                coordinates_2, coordinates_3, coordinates_d, coordinates_e,
                                                not args.NEGATIVE_SAMPLE, args.independent_distractors)

        frames_state_id = us.frames_shapes(args.num_distractors + args.extra_dist + 2, frames_state_id)
        frames_state_ood = us.frames_shapes(args.num_distractors + args.extra_dist + 2, frames_state_ood)

        images_idshapes_nocolors[f,:,:] = ud.plot_frame(sample, images_idshapes_nocolors[f], f, frames_state_id,
                                                colorspace_fixed, coordinates_2, coordinates_3,
                                                coordinates_d, coordinates_e, not args.NEGATIVE_SAMPLE)
        mask_idshapes_nocolors[f,:,:] = ud.plot_frame_mask(mask_idshapes_nocolors[f], f, frames_state_id,
                                                coordinates_2, coordinates_3, coordinates_d,
                                                coordinates_e, not args.NEGATIVE_SAMPLE, args.independent_distractors)

        images_idshapes_idcolors[f,:,:] = ud.plot_frame(sample, images_idshapes_idcolors[f], f, frames_state_id,
                                                colorspace_variable_id, coordinates_2, coordinates_3,
                                                coordinates_d, coordinates_e, not args.NEGATIVE_SAMPLE)
        mask_idshapes_idcolors[f,:,:] = ud.plot_frame_mask(mask_idshapes_idcolors[f], f, frames_state_id,
                                                coordinates_2, coordinates_3, coordinates_d,
                                                coordinates_e, not args.NEGATIVE_SAMPLE, args.independent_distractors)
        images_oodshapes_idcolors[f,:,:] = ud.plot_frame(sample, images_oodshapes_idcolors[f], f, frames_state_ood,
                                                colorspace_variable_id, coordinates_2, coordinates_3,
                                                coordinates_d, coordinates_e, not args.NEGATIVE_SAMPLE)
        mask_oodshapes_idcolors[f,:,:] = ud.plot_frame_mask(mask_oodshapes_idcolors[f], f, frames_state_ood,
                                                coordinates_2, coordinates_3, coordinates_d,
                                                coordinates_e, not args.NEGATIVE_SAMPLE, args.independent_distractors)
        images_idshapes_oodcolors[f,:,:] = ud.plot_frame(sample, images_idshapes_oodcolors[f], f, frames_state_id,
                                                colorspace_variable_ood, coordinates_2, coordinates_3,
                                                coordinates_d, coordinates_e, not args.NEGATIVE_SAMPLE)
        mask_idshapes_oodcolors[f,:,:] = ud.plot_frame_mask(mask_idshapes_oodcolors[f], f, frames_state_id,
                                                coordinates_2, coordinates_3, coordinates_d,
                                                coordinates_e, not args.NEGATIVE_SAMPLE, args.independent_distractors)
        images_oodshapes_oodcolors[f,:,:] = ud.plot_frame(sample, images_oodshapes_oodcolors[f], f, frames_state_ood,
                                                colorspace_variable_ood, coordinates_2, coordinates_3,
                                                coordinates_d, coordinates_e, not args.NEGATIVE_SAMPLE)
        mask_oodshapes_oodcolors[f,:,:] = ud.plot_frame_mask(mask_oodshapes_oodcolors[f], f, frames_state_ood,
                                                coordinates_2, coordinates_3, coordinates_d,
                                                coordinates_e, not args.NEGATIVE_SAMPLE, args.independent_distractors)
        images_idshapes_irregularcolors[f,:,:] = ud.plot_frame(sample, images_idshapes_irregularcolors[f], f, frames_state_id,
                                                colorspace_variable_irregular, coordinates_2, coordinates_3,
                                                coordinates_d, coordinates_e, not args.NEGATIVE_SAMPLE)
        mask_idshapes_irregularcolors[f,:,:] = ud.plot_frame_mask(mask_idshapes_irregularcolors[f], f, frames_state_id,
                                                coordinates_2, coordinates_3, coordinates_d,
                                                coordinates_e, not args.NEGATIVE_SAMPLE, args.independent_distractors)
        images_idshapes_idcolors_irregular[f,:,:] = ud.plot_frame(sample, images_idshapes_idcolors_irregular[f], f, frames_state_id,
                                                colorspace_variable_id, coordinates_2_irregular, coordinates_3_irregular,
                                                coordinates_d_irregular, coordinates_e_irregular, not args.NEGATIVE_SAMPLE)
        mask_idshapes_idcolors_irregular[f,:,:] = ud.plot_frame_mask(mask_idshapes_idcolors_irregular[f], f, frames_state_id,
                                                coordinates_2_irregular, coordinates_3_irregular, coordinates_d_irregular,
                                                coordinates_e_irregular, not args.NEGATIVE_SAMPLE, args.independent_distractors)


        if args.save_image:
            imageio.imwrite(path_to_save_noshapes_nocolors + "/frame_" + str(f) + ".png", images_noshapes_nocolors[f])
            imageio.imwrite(path_to_save_noshapes_nocolors + "/mask_" + str(f) + ".png", mask_noshapes_nocolors[f])
            imageio.imwrite(path_to_save_noshapes_idcolors + "/frame_" + str(f) + ".png", images_noshapes_idcolors[f])
            imageio.imwrite(path_to_save_noshapes_idcolors + "/mask_" + str(f) + ".png", mask_noshapes_idcolors[f])
            imageio.imwrite(path_to_save_idshapes_nocolors + "/frame_" + str(f) + ".png", images_idshapes_nocolors[f])
            imageio.imwrite(path_to_save_idshapes_nocolors + "/mask_" + str(f) + ".png", mask_idshapes_nocolors[f])
            imageio.imwrite(path_to_save_idshapes_idcolors + "/frame_" + str(f) + ".png", images_idshapes_idcolors[f])
            imageio.imwrite(path_to_save_idshapes_idcolors + "/mask_" + str(f) + ".png", mask_idshapes_idcolors[f])
            imageio.imwrite(path_to_save_idshapes_oodcolors + "/frame_" + str(f) + ".png", images_idshapes_oodcolors[f])
            imageio.imwrite(path_to_save_idshapes_oodcolors + "/mask_" + str(f) + ".png", mask_idshapes_oodcolors[f])
            imageio.imwrite(path_to_save_oodshapes_idcolors + "/frame_" + str(f) + ".png", images_oodshapes_idcolors[f])
            imageio.imwrite(path_to_save_oodshapes_idcolors + "/mask_" + str(f) + ".png", mask_oodshapes_idcolors[f])
            imageio.imwrite(path_to_save_oodshapes_oodcolors + "/frame_" + str(f) + ".png", images_oodshapes_oodcolors[f])
            imageio.imwrite(path_to_save_oodshapes_oodcolors + "/mask_" + str(f) + ".png", mask_oodshapes_oodcolors[f])
            imageio.imwrite(path_to_save_idshapes_irregularcolors + "/frame_" + str(f) + ".png", images_idshapes_irregularcolors[f])
            imageio.imwrite(path_to_save_idshapes_irregularcolors + "/mask_" + str(f) + ".png", mask_idshapes_irregularcolors[f])
            imageio.imwrite(path_to_save_idshapes_idcolors_irregular + "/frame_" + str(f) + ".png", images_idshapes_idcolors_irregular[f])
            imageio.imwrite(path_to_save_idshapes_idcolors_irregular + "/mask_" + str(f) + ".png", mask_idshapes_idcolors_irregular[f])


    end_state_id = us.end_shapes(args.num_distractors + args.extra_dist + 2, frames_state_id)
    end_state_ood = us.end_shapes(args.num_distractors + args.extra_dist + 2, frames_state_ood)
    images_noshapes_nocolors[-1] = ud.plot_frame(sample, images_noshapes_nocolors[-1], -1, squares, colorspace_fixed,
                                            coordinates_2, coordinates_3, coordinates_d, coordinates_e,
                                            not args.NEGATIVE_SAMPLE)
    mask_noshapes_nocolors[-1] = ud.plot_frame_mask(mask_noshapes_nocolors[-1], -1, squares,
                                            coordinates_2, coordinates_3, coordinates_d, coordinates_e,
                                            not args.NEGATIVE_SAMPLE, args.independent_distractors)
    images_noshapes_idcolors[-1] = ud.plot_frame(sample, images_noshapes_idcolors[-1], -1, squares, colorspace_variable_id,
                                             coordinates_2, coordinates_3, coordinates_d, coordinates_e,
                                             not args.NEGATIVE_SAMPLE)
    mask_noshapes_idcolors[-1] = ud.plot_frame_mask(mask_noshapes_idcolors[-1], -1, squares,
                                             coordinates_2, coordinates_3, coordinates_d, coordinates_e,
                                             not args.NEGATIVE_SAMPLE, args.independent_distractors)
    images_idshapes_nocolors[-1] = ud.plot_frame(sample, images_idshapes_nocolors[-1], -1,
                                             end_state_id, colorspace_fixed, coordinates_2,
                                             coordinates_3, coordinates_d, coordinates_e,
                                             not args.NEGATIVE_SAMPLE)
    mask_idshapes_nocolors[-1] = ud.plot_frame_mask(mask_idshapes_nocolors[-1], -1,
                                             end_state_id, coordinates_2,
                                             coordinates_3, coordinates_d, coordinates_e,
                                             not args.NEGATIVE_SAMPLE, args.independent_distractors)
    images_idshapes_idcolors[-1] = ud.plot_frame(sample, images_idshapes_idcolors[-1], -1,
                                             end_state_id, colorspace_variable_id, coordinates_2,
                                             coordinates_3, coordinates_d, coordinates_e,
                                             not args.NEGATIVE_SAMPLE)
    mask_idshapes_idcolors[-1] = ud.plot_frame_mask(mask_idshapes_idcolors[-1], -1,
                                             end_state_id, coordinates_2,
                                             coordinates_3, coordinates_d, coordinates_e,
                                             not args.NEGATIVE_SAMPLE, args.independent_distractors)
    images_idshapes_oodcolors[-1] = ud.plot_frame(sample, images_idshapes_oodcolors[-1], -1,
                                             end_state_id, colorspace_variable_ood, coordinates_2,
                                             coordinates_3, coordinates_d, coordinates_e,
                                             not args.NEGATIVE_SAMPLE)
    mask_idshapes_oodcolors[-1] = ud.plot_frame_mask(mask_idshapes_oodcolors[-1], -1,
                                             end_state_id, coordinates_2,
                                             coordinates_3, coordinates_d, coordinates_e,
                                             not args.NEGATIVE_SAMPLE, args.independent_distractors)
    images_oodshapes_idcolors[-1] = ud.plot_frame(sample, images_oodshapes_idcolors[-1], -1,
                                             end_state_ood, colorspace_variable_id, coordinates_2,
                                             coordinates_3, coordinates_d, coordinates_e,
                                             not args.NEGATIVE_SAMPLE)
    mask_oodshapes_idcolors[-1] = ud.plot_frame_mask(mask_oodshapes_idcolors[-1], -1,
                                             end_state_ood, coordinates_2,
                                             coordinates_3, coordinates_d, coordinates_e,
                                             not args.NEGATIVE_SAMPLE, args.independent_distractors)
    images_oodshapes_oodcolors[-1] = ud.plot_frame(sample, images_oodshapes_oodcolors[-1], -1,
                                             end_state_ood, colorspace_variable_ood, coordinates_2,
                                             coordinates_3, coordinates_d, coordinates_e,
                                             not args.NEGATIVE_SAMPLE)
    mask_oodshapes_oodcolors[-1] = ud.plot_frame_mask(mask_oodshapes_oodcolors[-1], -1,
                                             end_state_ood, coordinates_2,
                                             coordinates_3, coordinates_d, coordinates_e,
                                             not args.NEGATIVE_SAMPLE, args.independent_distractors)
    images_idshapes_irregularcolors[-1] = ud.plot_frame(sample, images_idshapes_irregularcolors[-1], -1,
                                             end_state_id, colorspace_variable_irregular, coordinates_2,
                                             coordinates_3, coordinates_d, coordinates_e,
                                             not args.NEGATIVE_SAMPLE)
    mask_idshapes_irregularcolors[-1] = ud.plot_frame_mask(mask_idshapes_irregularcolors[-1], -1,
                                             end_state_id, coordinates_2,
                                             coordinates_3, coordinates_d, coordinates_e,
                                             not args.NEGATIVE_SAMPLE, args.independent_distractors)
    images_idshapes_idcolors_irregular[-1] = ud.plot_frame(sample, images_idshapes_idcolors_irregular[-1], -1,
                                             end_state_id, colorspace_variable_id, coordinates_2_irregular,
                                             coordinates_3_irregular, coordinates_d_irregular, coordinates_e_irregular,
                                             not args.NEGATIVE_SAMPLE)

    if args.save_image:
        imageio.imwrite(path_to_save_noshapes_nocolors + "/frame_" + str(path_length-1) + ".png", images_noshapes_nocolors[-1])
        imageio.imwrite(path_to_save_noshapes_nocolors + "/mask_" + str(path_length-1) + ".png", mask_noshapes_nocolors[-1])
        imageio.imwrite(path_to_save_noshapes_idcolors + "/frame_" + str(path_length-1) + ".png", images_noshapes_idcolors[-1])
        imageio.imwrite(path_to_save_noshapes_idcolors + "/mask_" + str(path_length-1) + ".png", mask_noshapes_idcolors[-1])
        imageio.imwrite(path_to_save_idshapes_nocolors + "/frame_" + str(path_length-1) + ".png", images_idshapes_nocolors[-1])
        imageio.imwrite(path_to_save_idshapes_nocolors + "/mask_" + str(path_length-1) + ".png", mask_idshapes_nocolors[-1])
        imageio.imwrite(path_to_save_idshapes_idcolors + "/frame_" + str(path_length-1) + ".png", images_idshapes_idcolors[-1])
        imageio.imwrite(path_to_save_idshapes_idcolors + "/mask_" + str(path_length-1) + ".png", mask_idshapes_idcolors[-1])
        imageio.imwrite(path_to_save_idshapes_oodcolors + "/frame_" + str(path_length-1) + ".png", images_idshapes_oodcolors[-1])
        imageio.imwrite(path_to_save_idshapes_oodcolors + "/mask_" + str(path_length-1) + ".png", mask_idshapes_oodcolors[-1])
        imageio.imwrite(path_to_save_oodshapes_idcolors + "/frame_" + str(path_length-1) + ".png", images_oodshapes_idcolors[-1])
        imageio.imwrite(path_to_save_oodshapes_idcolors + "/mask_" + str(path_length-1) + ".png", mask_oodshapes_idcolors[-1])
        imageio.imwrite(path_to_save_oodshapes_oodcolors + "/frame_" + str(path_length-1) + ".png", images_oodshapes_oodcolors[-1])
        imageio.imwrite(path_to_save_oodshapes_oodcolors + "/mask_" + str(path_length-1) + ".png", mask_oodshapes_oodcolors[-1])
        imageio.imwrite(path_to_save_idshapes_irregularcolors + "/frame_" + str(path_length-1) + ".png", images_idshapes_irregularcolors[-1])
        imageio.imwrite(path_to_save_idshapes_irregularcolors + "/mask_" + str(path_length-1) + ".png", mask_idshapes_irregularcolors[-1])
        imageio.imwrite(path_to_save_idshapes_idcolors_irregular + "/frame_" + str(path_length-1) + ".png", images_idshapes_idcolors_irregular[-1])
        imageio.imwrite(path_to_save_idshapes_idcolors_irregular + "/mask_" + str(path_length-1) + ".png", mask_idshapes_idcolors_irregular[-1])

    images_noshapes_nocolors = np.stack((images_noshapes_nocolors, mask_noshapes_nocolors), axis=0)
    np.save(path + "noshapes_nocolors/samples/" + str(int(not args.NEGATIVE_SAMPLE)) + "_sample_" + str(
            sample), images_noshapes_nocolors, allow_pickle=False)
    images_noshapes_idcolors = np.stack((images_noshapes_idcolors, mask_noshapes_idcolors), axis=0)
    np.save(path + "noshapes_idcolors/samples/" + str(int(not args.NEGATIVE_SAMPLE)) + "_sample_" + str(
            sample), images_noshapes_idcolors, allow_pickle=False)
    images_idshapes_nocolors = np.stack((images_idshapes_nocolors, mask_idshapes_nocolors), axis=0)
    np.save(path + "idshapes_nocolors/samples/" + str(int(not args.NEGATIVE_SAMPLE)) + "_sample_" + str(
            sample), images_idshapes_nocolors, allow_pickle=False)
    images_idshapes_idcolors = np.stack((images_idshapes_idcolors, mask_idshapes_idcolors), axis=0)
    np.save(path + "idshapes_idcolors/samples/" + str(int(not args.NEGATIVE_SAMPLE)) + "_sample_" + str(
            sample), images_idshapes_idcolors, allow_pickle=False)
    images_idshapes_oodcolors = np.stack((images_idshapes_oodcolors, mask_idshapes_oodcolors), axis=0)
    np.save(path + "idshapes_oodcolors/samples/" + str(int(not args.NEGATIVE_SAMPLE)) + "_sample_" + str(
            sample), images_idshapes_oodcolors, allow_pickle=False)
    images_oodshapes_idcolors = np.stack((images_oodshapes_idcolors, mask_oodshapes_idcolors), axis=0)
    np.save(path + "oodshapes_idcolors/samples/" + str(int(not args.NEGATIVE_SAMPLE)) + "_sample_" + str(
            sample), images_oodshapes_idcolors, allow_pickle=False)
    images_oodshapes_oodcolors = np.stack((images_oodshapes_oodcolors, mask_oodshapes_oodcolors), axis=0)
    np.save(path + "oodshapes_oodcolors/samples/" + str(int(not args.NEGATIVE_SAMPLE)) + "_sample_" + str(
            sample), images_oodshapes_oodcolors, allow_pickle=False)
    images_idshapes_irregularcolors = np.stack((images_idshapes_irregularcolors, mask_idshapes_irregularcolors), axis=0)
    np.save(path + "idshapes_irregularcolors/" + str(int(not args.NEGATIVE_SAMPLE)) + "_sample_" + str(
            sample), images_idshapes_irregularcolors, allow_pickle=False)
    images_idshapes_idcolors_irregular = np.stack((images_idshapes_idcolors_irregular, mask_idshapes_idcolors_irregular), axis=0)
    np.save(path + "idshapes_idcolors_irregular/" + str(int(not args.NEGATIVE_SAMPLE)) + "_sample_" + str(
            sample), images_idshapes_idcolors_irregular, allow_pickle=False)

    # save coordinates to npy array
    coordinates_all = np.stack((np.array(coordinates_2), np.array(coordinates_3)))
    coordinates_all_irregular = np.stack((np.array(coordinates_2_irregular), np.array(coordinates_3_irregular)))
    if len(coordinates_d) > 0:
        # if there are distractors in the movie
        coordinates_all = np.vstack((coordinates_all, np.array(coordinates_d)))
        coordinates_all_irregular = np.vstack((coordinates_all_irregular, np.array(coordinates_d_irregular)))
    if len(coordinates_e) > 0:
        # if there are extra distractors in the movie
        # this option exists to fine tune certain distractors other than the ones that are always present in the movie
        # something like this should always be present at the start/middle/end of the movie
        coordinates_all = np.vstack((coordinates_all, np.array(coordinates_e)))
        coordinates_all_irregular = np.vstack((coordinates_all_irregular, np.array(coordinates_e_irregular)))

    np.save(
        path + "coordinates/" + str(args.batch) + "/" + str(int(not args.NEGATIVE_SAMPLE)) + "_sample_" + str(sample),
        coordinates_all, allow_pickle=False)
    np.save(
        path + "coordinates_irregular/" + str(args.batch) + "/" + str(int(not args.NEGATIVE_SAMPLE)) + "_sample_" + str(sample),
        coordinates_all_irregular, allow_pickle=False)

