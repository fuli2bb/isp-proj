import matplotlib.pyplot as plt
import cv2
import numpy as np
import math

height = 50
width = 50

def distance_euclid(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1]-point2[1])**2)

def create_lens_shading_correction_images(dark_current=0, flat_max=65535, flat_min=0, clip_range=[0, 65535]):
        # Objective: creates two images:
        #               dark_current_image and flat_field_image
        dark_current_image = dark_current * np.ones((height, width), dtype=np.float32)
        flat_field_image = np.empty((height, width), dtype=np.float32)

        center_pixel_pos = [height/2, width/2]
        max_distance = distance_euclid(center_pixel_pos, [height, width])

        for i in range(0, height):
            for j in range(0, width):
                flat_field_image[i, j] = (max_distance - distance_euclid(center_pixel_pos, [i, j])) / max_distance
                flat_field_image[i, j] = flat_min + flat_field_image[i, j] * (flat_max - flat_min)

        dark_current_image = np.clip(dark_current_image, clip_range[0], clip_range[1])
        flat_field_image = np.clip(flat_field_image, clip_range[0], clip_range[1])

        return dark_current_image, flat_field_image


a,b = create_lens_shading_correction_images()
cv2.imshow('1', b/np.max(b)*255) 
  
# waits for user to press any key 
# (this is necessary to avoid Python kernel form crashing) 
cv2.waitKey(0) 
  
# closing all open windows 
cv2.destroyAllWindows() 
