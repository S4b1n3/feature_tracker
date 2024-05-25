import numpy as np
from spline_scaled_curve_generator_class import GenCoord
from utils_drawing import rectangle_perimeter

##########################################################################################
##                  Coordinate generation logic
##########################################################################################

def get_points(nPoints):
    # if seeded, it will return the exact same points for all curves
    # np.random.seed(999)

    return np.random.rand(nPoints, 2) * 200


def check_range_set(s, number, interval=2):
    '''
    Checks if the number is in the set
    along with the intervals on either
    side of the number line.
    For eg: if number is 10, and interval
    is 2, then this will check if the
    numbers 8,9,10,11,12 are in the given set
    or not.

    Returns: True if the range is in set
    False otherwise
    '''
    # print('in check_range_set')
    ret = False
    if interval:
        # for non zero intervals
        for i in range(number - interval, number + interval + 1):
            if i in s:
                ret = True
        return ret
    else:
        # for zero interval
        if number in s:
            ret = True
        return ret
    # return ret


def get_full_length_coordinates(nPoints, nTimes, normalized, path_length, skip_param, win_x, win_y, angle_range=[90, 100], delta_angle_max=90):
    '''
    Returns the floor of coordinates of full length for
    both the tracking curve, and the distractor
    curve of the same length.
    The coordinates are normalized to a different
    location of the screen, randomly.
    0 - no normalization (actually in the first quadrant)
    2 - normalized to the center of the screen (actually in the third quadrant)
    4 - normalized to the first quadrant of the screen (actually in the center of the display)
    6 - roughly in the center of the display
    8 - roughly in the center of the display

    The below two coordinate generation codes can be put in a loop.
    '''

    # normal = np.random.choice([0,2,4,6,8],2,replace=False)
    # generating unique normalization positions, which would be used to shift
    # the dots by certain number of pixels
    '''
    # DO NOT GENERATE NORMALIZED COORDINATES. NO NEED HERE. LET TARGETS BE WITHIN THE SCREEN RANGE 
    normal=[]
    for i in range(2):
        _normal = np.random.choice(range(5))#,2,replace=False) # *** changed from 5
        while check_range_set(normalized,_normal,interval=2):# _normal in normalized:
            _normal = np.random.choice(range(13))#,2,replace=False) # *** 13
        normal.append(_normal)
    #print('done with range mask\ngenerating coordinates')
    '''
    '''
    #Generate 3x the coordinates, and then sample path_length out of them
    '''
    cd_2 = GenCoord()
    coordinates_2 = cd_2.get_coordinates(length_curve=nTimes, angle_range=angle_range, distance_points=.002,
                                         delta_angle_max=delta_angle_max, wiggle_room=.95, rigidity=.95, x_min=0, x_max=win_x,
                                         y_min=0, y_max=win_y)

    # coordinates_2 = cd_2.get_coordinates(length_curve=nTimes, angle_range=[10, 170], distance_points=.002,
    #                                      delta_angle_max=180, wiggle_room=.95, rigidity=.95, x_min=0, x_max=win_x,
    #                                      y_min=0, y_max=win_y)

    # taking one-third points from a random location
    rn_l = np.random.randint(len(coordinates_2) - (path_length * skip_param) + 1)
    coordinates_2 = coordinates_2[rn_l:rn_l + (path_length * skip_param):skip_param]
    # compute the distance between start and end points
    start = coordinates_2[0]
    end = coordinates_2[-1]
    # compute distance
    dist_2 = np.sqrt(((start[0] - end[0]) ** 2) + ((start[1] - end[1]) ** 2))
    #print('dist_2: ', dist_2)

    # normalizing to an arbitrary location in the screen
    # NEW: shifting the dots by a random number of pixels
    normal = [0, 0]
    if normal[0]:
        # coordinates_2=[((coordinates_2[i][0]-(win_x/float(normal[0]))),(coordinates_2[i][1]-(win_y/float(normal[0])))) for i in range(0,len(coordinates_2))]
        coordinates_2 = [((coordinates_2[i][0] - (float(normal[0]))), (coordinates_2[i][1] - (float(normal[0])))) for i
                         in range(0, len(coordinates_2))]


    # add the normalization coefficients to the normalized set for further use
    normalized.add(normal[0])
    normalized.add(normal[1])

    return np.floor(coordinates_2), normalized, coordinates_2 #, coordinates_3, np.floor(coordinates_3)


def get_third_length_distractor_coordinates(nPoints, nTimes, normalized, num_distractors, path_length,
                                            skip_param, win_x, win_y, angle_range=[90, 100], delta_angle_max=90):
    '''
    Returns the coordinates of one-third length for
    specified number of distractor curves.
    The one-third length of the distractors is
    randomly sampled, from a random starting point to
    the next available 50 points.

    The coordinates are normalized to a different
    location of the screen, randomly, other than what
    is in normalized set already
    0 - no normalization (actually in the first quadrant)
    2 - normalized to the center of the screen (actually in the third quadrant)
    4 - normalized to the first quadrant of the screen (actually in the center of the display)
    '''

    dis_len = int(nTimes)  # length of the distractor. By default one-third for this function
    coord_d = []

    # generating unique normalization positions, which would be used to shift
    # the dots by certain number of pixels

    '''
    # DO NOT GENERATE RANDOM NUMBERS. USE THE ONES PASSED THROUGH NORMALIZED
    normal=[]
    for i in range(num_distractors):
        _normal = np.random.choice(range(-13,13))#,2,replace=False) # *** -13 to 13
        #while _normal in normalized:
        while check_range_set(normalized,_normal,interval=1):# _normal in normalized:
            _normal = np.random.choice(range(-13,13))#,2,replace=False) # *** -13 to 13
        #print(i)
        normal.append(_normal)
    '''
    normal = normalized

    for i in range(num_distractors):

        # generating coordinates for the negative distractor sample of same length for now
        # using the coordinates generator class
        '''
        #Generate 3x the coordinates, and then sample one-third out of them
        '''
        cd_d = GenCoord()
        coordinates_d = cd_d.get_coordinates(length_curve=nTimes, angle_range=angle_range, distance_points=.002,
                                             delta_angle_max=delta_angle_max, wiggle_room=.95, rigidity=.95, x_min=0, x_max=win_x,
                                             y_min=0, y_max=win_y)
        # coordinates_d = cd_d.get_coordinates(length_curve=nTimes, angle_range=[10, 170], distance_points=.002,
        #                                      delta_angle_max=180, wiggle_room=.95, rigidity=.95, x_min=0, x_max=win_x,
        #                                      y_min=0, y_max=win_y)

        # taking one-third points from a random location
        rn_l = np.random.randint(len(coordinates_d) - (path_length * skip_param) + 1)
        coordinates_d = coordinates_d[rn_l:rn_l + (path_length * skip_param):skip_param]
        # normalizing to an arbitrary location in the screen
        # NEW: shifting the dots by a random number of pixels
        if normal[i]:
            # coordinates_d=[((coordinates_d[j][0]-(win_x/float(normal[i]))),(coordinates_d[j][1]-(win_y/float(normal[i])))) for j in range(0,len(coordinates_d))]
            coordinates_d = [((coordinates_d[j][0] - (float(normal[i]))), (coordinates_d[j][1] - (float(normal[i]))))
                             for j in range(0, len(coordinates_d))]
        coord_d.append(coordinates_d)
        # normalized.add(normal[i])
        normalized = np.delete(normalized, 0)

    return coord_d, normalized


def draw_blobs(coordinates, height, width):
    '''
    Generic function to draw blobs, given the coordinates
    No need to pass the whole set of coordinates_2 and coordinates_3,
    just the starting and ending points depending on positive or negative sample.

    Condition to check positive/negative sample takes care of what needs to be parsed in this function.
    '''

    # rr, cc = rectangle_perimeter(np.floor(coordinates_2[0][0]), np.floor(coordinates_2[0][1]), 3, 3)
    rr, cc = rectangle_perimeter(coordinates[0], coordinates[1], height, width)

    return rr, cc