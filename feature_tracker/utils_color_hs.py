import colorsys
import numpy as np
from PIL import Image

def angle_to_hsv(angle, saturation=1.0):
    hue = angle / 360.0
    return (hue, saturation, 1.0)

def angle_to_hsv_half(angle, first_half=True):
    if first_half:
        hue = (angle%180) / 360.0
    else:
        hue = (angle % 180 + 180) / 360.0
    return (hue, 1.0, 1.0)

def generate_trajectory(num_frames, speed, start_angle):
    trajectory_hsv = []
    trajectory_rgb = []
    for frame in range(num_frames):
        speed = 0.05 #np.random.uniform(0.015, 0.04) #0.05 #

        import random
        # saturation_fluctuation = np.random.uniform(-0.2, 0.2) #random.randint(-1, 1) * 0.3 #
        # saturation = np.sin(frame / num_frames * np.pi) * 0.5 + 0.5 + saturation_fluctuation

        time = (frame * speed)*360 % 360+start_angle # Adjust the speed of color change by changing the denominator
        hsv = angle_to_hsv(time) #, saturation
        trajectory_hsv.append(hsv)
        rgb = [int(c * 255) for c in colorsys.hsv_to_rgb(*hsv)]
        trajectory_rgb.append(rgb)
    return trajectory_hsv, trajectory_rgb

def generate_all_colors(nb_images, nb_frames, nb_objects, speed=1/64.0):
    colors = np.zeros((nb_images, nb_objects, nb_frames, 3))
    for img in range(nb_images):
        for obj in range(nb_objects):
            start_angle = np.random.randint(0, 360)
            trajectory_hsv, trajectory_rgb = generate_trajectory(nb_frames, speed, start_angle)
            colors[img, obj] = trajectory_rgb
    return colors