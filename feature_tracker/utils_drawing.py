import numpy as np
import skimage
import random

##########################################################################################
##                  Drawing functions
##########################################################################################

def rectangle(r0, c0, width, height):
    # original square drawing function
    # draws a filled in rectangle
    rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + height, c0 + height]

    return skimage.draw.polygon(rr, cc)


def rectangle_vertical(r0, c0, width, height):
    # draws a filled-in vertical rectangle
    # rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + height, c0 + height]
    rr, cc = [r0, r0 + width + 2, r0 + width + 2, r0], [c0, c0, c0 + height, c0 + height]

    return skimage.draw.polygon(rr, cc)


def rectangle_horizontal(r0, c0, width, height):
    # draws a filled-in horizontal rectangle
    rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + height + 2, c0 + height + 2]
    # rr, cc = [r0, r0 + width+2, r0 + width+2, r0], [c0, c0, c0 + height, c0 + height]

    return skimage.draw.polygon(rr, cc)


def inverted_L(r0, c0, width, height):
    # draws a filled-in inverted-L
    rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + height + 2, c0 + height + 2]
    rr += [r0, r0 + width + 2, r0 + width, r0 + 1]
    cc += [c0, c0, c0 + height, c0 + height]
    # rr, cc = [r0, r0 + width+2, r0 + width+2, r0], [c0, c0, c0 + height, c0 + height]

    return skimage.draw.polygon(rr, cc)


def inverted_L_E(r0, c0, width, height):
    # draws a filled-in inverted-L with a jutting out center part of E on the other side
    # rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + height+2, c0 + height+2]
    rr = [r0, r0 + width + 2, r0 + width + 2, r0]
    cc = [c0 + 1, c0 + 1, c0 + height + 1, c0 + height + 1]
    # untill here draws a centered vertical line

    rr += [r0 + 2, r0 + width, r0 + width, r0]
    cc += [c0, c0, c0 + height + 2, c0 + height + 2]
    # rr, cc = [r0, r0 + width+2, r0 + width+2, r0], [c0, c0, c0 + height, c0 + height]

    return skimage.draw.polygon(rr, cc)


def inverted_half_L(r0, c0, width, height):
    # draws a filled-in inverted-L with the lower line being half of the original L. This is centered.
    # rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + height+2, c0 + height+2]
    rr = [r0, r0 + width + 2, r0 + width + 2, r0]
    cc = [c0 + 1, c0 + 1, c0 + height + 1, c0 + height + 1]
    # untill here draws a centered vertical line

    rr += [r0 + 1, r0 + width, r0 + width, r0]
    cc += [c0, c0, c0 + height + 3, c0 + height + 2]
    # rr, cc = [r0, r0 + width+2, r0 + width+2, r0], [c0, c0, c0 + height, c0 + height]

    return skimage.draw.polygon(rr, cc)


def small_r(r0, c0, width, height):
    # draws a filled-in small 'r'
    # rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + height+2, c0 + height+2]
    rr = [r0 - 1, r0 + width + 2, r0 + width + 2, r0 - 1]
    cc = [c0 + 1, c0 + 1, c0 + height + 1, c0 + height + 1]
    # untill here draws a centered vertical line

    rr += [r0 + 1, r0 + width, r0 + width, r0 + 1]
    cc += [c0, c0, c0 + height + 3, c0 + height + 2]

    rr += [r0, r0 + width, r0 + width, r0]
    cc += [c0, c0, c0 + height + 2, c0 + height + 2]
    # rr, cc = [r0, r0 + width+2, r0 + width+2, r0], [c0, c0, c0 + height, c0 + height]
    return skimage.draw.polygon(rr, cc)


def cross(r0, c0, width, height):
    # draws a filled-in cross
    # rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + height+2, c0 + height+2]
    rr = [r0, r0 + width + 2, r0 + width + 2, r0]
    cc = [c0 + 1, c0 + 1, c0 + height + 1, c0 + height + 1]
    # untill here draws a centered vertical line

    rr1 = [r0 + 1, r0 + width + 1, r0 + width + 1, r0 + 1]
    cc1 = [c0, c0, c0 + height + 2, c0 + height + 2]
    # the above two lines draw a centered horizontal line

    abc = (skimage.draw.polygon(rr, cc), skimage.draw.polygon(rr1, cc1))
    ab = np.concatenate((abc[0][0], abc[1][0]))
    bc = np.concatenate((abc[0][1], abc[1][1]))
    return (ab, bc)


def capital_T(r0, c0, width, height):
    # draws a filled-in capital T
    # rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + height+2, c0 + height+2]
    rr = [r0, r0 + width + 2, r0 + width + 2, r0]
    cc = [c0 + 1, c0 + 1, c0 + height + 1, c0 + height + 1]
    # untill here draws a centered vertical line

    rr1 = [r0, r0 + width, r0 + width, r0]
    cc1 = [c0, c0, c0 + height + 2, c0 + height + 2]
    # the above two lines draw a horizontal line

    abc = (skimage.draw.polygon(rr, cc), skimage.draw.polygon(rr1, cc1))
    ab = np.concatenate((abc[0][0], abc[1][0]))
    bc = np.concatenate((abc[0][1], abc[1][1]))
    return (ab, bc)


def capital_C(r0, c0, width, height):
    # draws a filled-in capital C
    # rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + height+2, c0 + height+2]
    rr = [r0, r0 + width + 2, r0 + width + 2, r0]
    cc = [c0, c0, c0 + height, c0 + height]
    # untill here draws a centered vertical line

    rr1 = [r0, r0 + width, r0 + width, r0]
    cc1 = [c0, c0, c0 + height + 2, c0 + height + 2]
    # the above two lines draw a horizontal line

    rr2 = [r0 + 2, r0 + width + 2, r0 + width + 2, r0 + 2]
    cc2 = [c0, c0, c0 + height + 2, c0 + height + 2]
    # the above two lines draw a horizontal line shifted below

    abc = (skimage.draw.polygon(rr, cc), skimage.draw.polygon(rr1, cc1), skimage.draw.polygon(rr2, cc2))
    ab = np.concatenate((abc[0][0], abc[1][0], abc[2][0]))
    bc = np.concatenate((abc[0][1], abc[1][1], abc[2][1]))
    return (ab, bc)


def rectangle_perimeter(r0, c0, width, height, shape=None, clip=False):
    # draws only the perimeter of the rectangle
    rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + height, c0 + height]

    return skimage.draw.polygon_perimeter(rr, cc, shape=shape, clip=clip)


drawing_function_list = [rectangle_vertical, rectangle_horizontal, inverted_L, inverted_L_E, inverted_half_L, small_r,
                         cross, capital_T, capital_C]

# Pixel flickering drawing functions
# small and medium are for a 3x3 grid
# small, medium and large are for a 5x5 grid, showing a smooth transition

def center_small_rectangle(r0, c0, width, height):
    # pos 5
    rr = [r0 + 1, r0 + width + 1, r0 + width, r0]
    cc = [c0 + 1, c0 + 1, c0 + height + 1, c0 + height + 1]
    return skimage.draw.polygon(rr, cc)


def center_medium_rectangle(r0, c0, width, height):
    # original square drawing function
    # draws a filled in rectangle
    rr, cc = [r0, r0 + width + 2, r0 + width + 2, r0], [c0, c0, c0 + height + 2, c0 + height + 2]

    return skimage.draw.polygon(rr, cc)


def center_large_rectangle(r0, c0, width, height):
    # original square drawing function
    # draws a filled in rectangle increased by 1 pixel
    rr, cc = [r0 - 1, r0 + width + 3, r0 + width + 3, r0 - 1], [c0 - 1, c0 - 1, c0 + height + 3, c0 + height + 3]

    return skimage.draw.polygon(rr, cc)

flicker_5_list = [center_small_rectangle, center_medium_rectangle, center_large_rectangle, center_medium_rectangle]
flicker_3_list = [center_small_rectangle, center_medium_rectangle]

def blobs():
    blob = np.zeros((5,5), dtype=np.uint8)
    #fill the perimeter of the blob
    blob[0, :] = 1
    blob[-1, :] = 1
    blob[:, 0] = 1
    blob[:, -1] = 1
    return blob

def plot_objects(frame, shape, color, x, y):
    for i in range(len(shape)):
        for j in range(len(shape)):
            if shape[i,j] == 0:
                frame[x+i, y+j] += color.astype(np.uint8)*shape[i,j].astype(np.uint8)
            else:
                frame[x+i, y+j] = color*shape[i,j]
    #frame[x:x+len(shape), y:y+len(shape)] = color*shape[im_idx]
    return frame

def plot_frame(sample, frame, idx_frame, shape, color, coordinates_2, coordinates_3, coordinates_d, coordinates_e, positive=True, draw_shift=False):
    frame = plot_objects(frame, blobs(), np.array([255, 0, 0], dtype=np.uint8),
                         int(coordinates_2[0, 0]), int(coordinates_2[0, 1]))
    if not positive:
        # if draw_shift:
        #     shift_i = random.choice([5, 0, -5])
        #     shift_j = random.choice([5, 0, -5])
        #     while int(coordinates_2[-1, 0]) + shift_i < 0 or int(coordinates_2[-1, 0])+5 + shift_i > 31:
        #         shift_i = random.choice([5, 0, -5])
        #     while int(coordinates_2[-1, 1]) + shift_j < 0 or int(coordinates_2[-1, 1])+5 + shift_j > 31:
        #         shift_j = random.choice([5, 0, -5])
        # frame = plot_objects(frame, blobs(), np.array([0, 0, 255], dtype=np.uint8),
        #                      int(coordinates_2[-1, 0]) + shift_i, int(coordinates_2[-1, 1]) + shift_j)
        frame = plot_objects(frame, blobs(), np.array([0, 0, 255], dtype=np.uint8),
                             int(coordinates_3[-1][0]), int(coordinates_3[-1][1]))
    else:
        frame = plot_objects(frame, blobs(), np.array([0, 0, 255], dtype=np.uint8),
                             int(coordinates_2[-1, 0]), int(coordinates_2[-1, 1]))

    frame = plot_objects(frame, shape[1].grid, color[sample, 1, idx_frame],
                                    int(coordinates_3[idx_frame][0]), int(coordinates_3[idx_frame][1]))
    for i in range(len(coordinates_d)):
        frame = plot_objects(frame, shape[2+i].grid, color[sample, 2 + i, idx_frame],
                                        int(coordinates_d[i][idx_frame][0]), int(coordinates_d[i][idx_frame][1]))
    for i in range(len(coordinates_e)):
        frame = plot_objects(frame, shape[2+len(coordinates_d)+i].grid, color[sample, 2 + len(coordinates_d) + i, idx_frame],
                                        int(coordinates_e[i][idx_frame][0]), int(coordinates_e[i][idx_frame][1]))
    frame = plot_objects(frame, shape[0].grid, color[sample, 0, idx_frame],
                         int(coordinates_2[idx_frame, 0]), int(coordinates_2[idx_frame, 1]))

    return frame


def plot_frame_mask(frame, idx_frame, shape, coordinates_2, coordinates_3, coordinates_d, coordinates_e,
               positive=True, independent_distractors=False):
    frame = plot_objects(frame, blobs(), np.array([255, 0, 0], dtype=np.uint8),
                         int(coordinates_2[0, 0]), int(coordinates_2[0, 1]))
    if not positive:
        frame = plot_objects(frame, blobs(), np.array([255, 0, 0], dtype=np.uint8),
                             int(coordinates_3[-1][0]), int(coordinates_3[-1][1]))
        #
        # frame = plot_objects(frame, blobs(), np.array([255, 0, 0], dtype=np.uint8),
        #                      int(coordinates_2[-1, 0])+shift_i, int(coordinates_2[-1, 1])+shift_j)
    else:
        frame = plot_objects(frame, blobs(), np.array([255, 0, 0], dtype=np.uint8),
                             int(coordinates_2[-1, 0]), int(coordinates_2[-1, 1]))


    if independent_distractors:
        color_dist = np.array([0, 1, 0], dtype=np.uint8)
    else:
        color_dist = np.array([0, 255, 0], dtype=np.uint8)
    frame = plot_objects(frame, shape[1].grid, color_dist,
                         int(coordinates_3[idx_frame][0]), int(coordinates_3[idx_frame][1]))
    for i in range(len(coordinates_d)):
        if independent_distractors:
            color_dist = np.array([0, 1+i+1, 0], dtype=np.uint8)
        else:
            color_dist = np.array([0, 255, 0], dtype=np.uint8)
        frame = plot_objects(frame, shape[2 + i].grid, color_dist,
                             int(coordinates_d[i][idx_frame][0]), int(coordinates_d[i][idx_frame][1]))
    for i in range(len(coordinates_e)):
        if independent_distractors:
            color_dist = np.array([0, 1+len(coordinates_d)+i+1, 0], dtype=np.uint8)
        else:
            color_dist = np.array([0, 255, 0], dtype=np.uint8)
        frame = plot_objects(frame, shape[2 + len(coordinates_d) + i].grid,
                             color_dist,
                             int(coordinates_e[i][idx_frame][0]), int(coordinates_e[i][idx_frame][1]))
    frame = plot_objects(frame, shape[0].grid, np.array([0, 0, 255], dtype=np.uint8),
                         int(coordinates_2[idx_frame, 0]), int(coordinates_2[idx_frame, 1]))
    return frame


