# import necessary libraries
import pyrealsense2 as rs
import numpy as np
import cv2
import os
# details of pyrealsens2 library:
# https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.html
# detiail of opencv-python library
# https://docs.opencv.org/4.x/

# configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# get camera details
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

# check whether camera is rgb
found_rgb = False
for sensor in device.sensors:
    if sensor.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if found_rgb:
    print("Color sensor exists.")
else:
    print("Color sensor DOES NOT exist.")
    
    # configure resolutions
# maximum resolution for D435 is 1280X720
width = 640
height = 360
config.enable_stream(stream_type=rs.stream.depth, 
                    width=width, height=height, 
                    format=rs.format.z16, framerate=30)
# color images
config.enable_stream(stream_type=rs.stream.color, 
                    width=width, height=height, 
                    format=rs.format.bgr8, framerate=30)

# create a directory to store image if doesn't exist
output_dir_train = "data/train/"
output_dir_test = "data/test/"
os.makedirs(name=output_dir_train, exist_ok=True)

# define label pair dictionary for saving image files
label_pair = {0: "fist", 1: "up", 2: "left", 3: "down", 4: "right"}

# start streaming
pipeline.start(config=config)
instruction = False     # control variable for keyboard input message

try:
    # infinite loop for streaming
    while True:
        # Wait for a set of frames for depth and color images
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        
        # convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # apply colormap on depth images
        # images must be converted to 8-bit per pixel
        # https://shimat.github.io/opencvsharp_docs/html/7633dbe6-e0b8-23bf-6966-dc9b4720e3d1.htm
        # alpha here is a scaling factor applied to the input image
        # any color maps can be used: https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html#gga9a805d8262bcbe273f16be9ea2055a65ab3f207661ddf74511b002b1acda5ec09
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)
        
        # set color map dimensions
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape
        
        # if color image and depth image do not have same dimensions,
        # resize color image to match depth image's dimension
        if depth_colormap_dim != color_colormap_dim:
            print("Depth and color images have different dimensions.")
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))
            
        # show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        # show details about keyboard inputs for user interface
        if not instruction:
            print('Press "q" or "esc" to stop streaming.')
            print('Press "s" or "space" to save the image frame.')
            instruction = True
            
        # wait for a key press (1 ms delay)
        # ref: https://www.asciitable.com/
        key = cv2.waitKey(1)
        # check keyboard input    
        if key == ord('q') or key == 27:
            break
        elif key == ord('s') or key == 32:
            # save image frame
            # check whether the image is in correct format
            if color_image[0, 0].dtype != 'uint8':
                color_image = color_image.astype(np.uint8)
            
            # reshape the image from (height, width, color) to (width, height, color)
            color_image_reshaped = color_image.transpose(1, 0, 2)
            
            # save the image in png format
            # ask the label based on the "label_pair" dictionary
            # 0: fist, 1: up, 2: left, 3: down, 4: right
            file_label = int(input("Enter the image label [0-9]: "))
            # file_label = 3    # activate this line for saving specific label
            # define the path for images based on the label
            file_dir_train = os.path.join(output_dir_train, label_pair[file_label])
            file_dir_test = os.path.join(output_dir_test, label_pair[file_label])
            os.makedirs(name=file_dir_train, exist_ok=True)
            # check how many images already existed for file names
            image_counter = len(os.listdir(file_dir_train)) + len(os.listdir(file_dir_test)) + 1
            file_name = os.path.join(file_dir_train, f"{file_label}_{image_counter:08d}.png")
            # save images in the defined directory
            cv2.imwrite(file_name, color_image)
            print(f"Image saved as: {file_name}")
                    
except:
    print("The streaming is not available.")
    
finally:
    # close all windows
    cv2.destroyAllWindows()
    
    # stop streaming
    pipeline.stop()