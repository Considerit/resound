# Initiating Prompt:

# You're given a "base video" and a "reaction video".  The reaction video has one or two 
# people who have recorded themselves watching the base video. Most often, the base video 
# is shown in some region of the reaction video. Sometimes, the base video is not shown, 
# most often at the beginning or end of the reaction video. The two most common 
# configurations are: (1) the base video is shown in its own rectangle on top of a 
# video of the reactors; (2) reactors are shown in a rectangle overlaid on top of 
# the base video playing.

# I'd like to write a python function that takes in a base video and a reaction video, 
# and then identifies each reactor's location within the reaction video, in the form 
# of coordinates of a square that approximately fits the area of the screen where 
# their faces typically are. 

# Can you help write that function? Additional requirements include:
#   1) Try not to identify the face of anyone shown in the base video. For example, 
#      if you generate a list of candidate reactor faces, you might then filter out 
#      the candidates that appear to also be present in the base video.
#   2) Reactors can move during the video. You can probably sample from different parts 
#      of the reaction video and find average locations for the reactors and use that. 
#      It is okay if sometimes a reactor moves out of the identified coordinates. 


# Here's a general outline of the steps involved in identifying reactor locations in a 
# reaction video:

# Face Detection: Use a face detection algorithm to detect faces in each frame of the 
# reaction video. There are several popular face detection libraries available, 
# such as OpenCV's Haar cascades or deep learning-based models like dlib or MTCNN.

# Base Video Exclusion: To exclude faces from the base video, you can either 
# perform face detection on the base video separately and compare the detected 
# faces with the reactor faces, or you can use techniques like background subtraction 
# or masking to exclude regions occupied by the base video from the subsequent face 
# detection steps.

# Face Tracking: Once you have detected the faces in the reaction video, you can 
# apply face tracking algorithms to track the detected faces across frames. This 
# helps in maintaining consistency and enables you to estimate the average 
# locations of each reactor's face over time. Popular face tracking algorithms 
# include Kalman filters, Lucas-Kanade, or correlation-based tracking.

# Coordinate Extraction: Analyze the tracked face locations to estimate the 
# coordinates of the region where each reactor's face typically appears. This 
# can be done by calculating the bounding box or fitting a square/rectangle 
# around the tracked face positions. You can calculate the average or representative 
# coordinates for each reactor based on the tracked locations.

# Filtering and Refinement: To improve the accuracy of the identified reactor 
# locations, you can apply additional filtering and refinement techniques. 
# This may include spatial filtering, temporal smoothing, or more sophisticated 
# algorithms to handle occlusions, lighting changes, or facial pose variations.




# Face Detection: To perform face detection in Python, we can use the OpenCV library, 
# which provides various pre-trained face detection models. Here's an example code 
# snippet to detect faces in a given video

# Base Video Exclusion: To exclude faces from the base video, I'll do a perceptual 
# hash into the base video frame at the same time frame as the reaction video 
# to see if it exists there too. 


import cv2
import numpy as np
import subprocess
import pygetwindow as gw
import os
from moviepy.editor import VideoFileClip, AudioFileClip

def replace_audio(video, audio_path):
    audio = AudioFileClip(audio_path)

    # Set the audio of the video clip
    video = video.set_audio(audio)

    return video


def crop_video(input_file, output_file, replacement_audio, x, y, w, h):
    # Load the video clip
    video = VideoFileClip(input_file)

    if w != h:
      w = h = min(w,h)

    if w % 2 > 0:
      w -= 1
      h -= 1

    # Crop the video clip
    cropped_video = video.crop(x1=x, y1=y, x2=x+w, y2=y+h)


    if w > 450: 
      w = h = 450
      cropped_video = cropped_video.resize(width=video.w//2*2, height=video.h//2*2)

    if replacement_audio:
      cropped_video = replace_audio(cropped_video, replacement_audio)

    # Write the cropped video to a file
    cropped_video.write_videofile(output_file, codec="libx264", audio_codec="aac", fps=30,
                                  ffmpeg_params=['-pix_fmt', 'yuv420p'])

    # Close the video clip
    video.close()

def create_reactor_view(react_path, base_path, replacement_audio=None, show_facial_recognition=False): 
  base_reaction_path, base_video_ext = os.path.splitext(react_path)

  output_files = []
  i = 0
  
  # Check if files exist with naming convention
  while os.path.exists(f"{base_reaction_path}-cropped-{i}{base_video_ext}"):
      output_files.append(f"{base_reaction_path}-cropped-{i}{base_video_ext}")
      i += 1

  # If no existing files found, proceed with face detection and cropping
  if len(output_files) == 0:
      reactors = detect_faces(react_path, base_path, show_facial_recognition)
      print(reactors)
      for i, (x,y,w,h) in enumerate(reactors): 
          output_file = f"{base_reaction_path}-cropped-{i}{base_video_ext}"
          crop_video(react_path, output_file, replacement_audio, int(x), int(y), int(w), int(h))
          output_files.append(output_file)

  return output_files



detector = None

def get_faces_from(img):
  global detector
  if detector is None: 
    from mtcnn import MTCNN
    detector = MTCNN()

  faces = detector.detect_faces(img)

  ret = [ (face['box'], face['keypoints']['nose']) for face in faces ]
  return ret


def detect_faces(react_path, base_path, show_facial_recognition=False):

    # Open the video file
    react_capture = cv2.VideoCapture(react_path)
    # base_capture = cv2.VideoCapture(base_path)

    # Iterate over each frame in the video
    face_matches = []

    width = int(react_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(react_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f'\nDetecting faces for {react_path}')

    while True:

        try: 
          current_react = react_capture.get(cv2.CAP_PROP_POS_FRAMES)
          # current_base = base_capture.get(cv2.CAP_PROP_POS_FRAMES)

          react_capture.set(cv2.CAP_PROP_POS_FRAMES, current_react + 500)
          # base_capture.set(cv2.CAP_PROP_POS_FRAMES, current_base + 500)

          ret, react_frame = react_capture.read()
          # Break the loop if no more frames
          if not ret:
              break

          # ret, base_frame = base_capture.read()
          # Break the loop if no more frames
          # if not ret:
          #     break
        except Exception as e: 
          print('exception', e)
          break

        # Convert the frame to grayscale for face detection
        react_gray = cv2.cvtColor(react_frame, cv2.COLOR_BGR2RGB)
        # base_gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2RGB)

        # Perform face detection
        faces = get_faces_from(react_gray)

        # print("FACES:", faces)
        # Draw rectangles around the detected faces
        for ((x, y, w, h), nose) in faces:
            # print( (x,y,w,h), nose)
            if w > .05 * width:
              # candidate_face = react_gray[y:y+h, x:x+w]  # Extract the candidate face region

              if show_facial_recognition:
                cv2.rectangle(react_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
              face_matches.append((x, y, w, h, nose))

        # print("matches:", face_matches)

        if show_facial_recognition:
          top, heat_map_color = find_top_candidates(face_matches, width, height)
          for tc in top:
            # print('TC', tc)

            (x,y,w,h) = expand_face(tc, width, height)

            # print(rect, center)

            rect = tc[2]
            center = tc[3]
            cv2.rectangle(react_frame, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 0, 255), 2)
            cv2.circle(react_frame,(int(center[0]), int(center[1])), 5, (0, 0, 255), 2)
            cv2.rectangle(react_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


        if show_facial_recognition:        
          # Display the resulting frame
          cv2.resize(react_frame, (960, 540))        
          cv2.imshow('Face Detection', react_frame)

          cv2.resize(heat_map_color, (960, 540))

          cv2.imshow('Heat map', heat_map_color)
          cv2.moveWindow('Heat map', 0, int(540) ) 

          # Exit if 'q' is pressed
          if cv2.waitKey(1) & 0xFF == ord('q'):
              break

    # cv2.waitKey(0)

    print('facematches:', face_matches)
    tc_final, _ = find_top_candidates(face_matches, width, height)
    

    # Release the video capture and close the window
    react_capture.release()
    # base_capture.release()

    cv2.destroyAllWindows()


    final_set = []
    for tc in tc_final:
      reactor = expand_face(tc, width, height) # (x,y,w,h)
      final_set.append(reactor)

    return final_set



def is_overlap(x1, y1, w1, h1, x2, y2, w2, h2):
    # Check if two rectangles overlap
    return (x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2)

def expand_face(group, width, height, expansion=.7, sidedness=.7):
  (faces, score, (x,y,w,h), center) = group


  if (center[0] > x + .65 * w): # facing right
    x = max(0, int(x - expansion * sidedness * w))

  elif (center[0] < x + .35 * w): # facing left
    x = max(0, int(x - expansion * (1 - sidedness) * w))

  else: # sorta in the middle
    x = max(0, int(x - expansion / 2 * w))


  y = max(0, int(y - expansion * sidedness * h))
  h = min(height, int(h + h * expansion))
  w = min(width, int(w + w * expansion))

  if h > w: 
    x -= int((h - w) / 2)
    w = h
    if x < 0: 
      x = 0

  if w > h: 
    y -= int((w - h) / 2)
    h = w
    if y < 0: 
      y = 0



  return (x,y,w,h)







# Face Tracking
# To perform face tracking, we can utilize the face landmarks detected in the initial face 
# detection step. By tracking the landmarks across subsequent frames, we can estimate 
# the movement of the face and update the face coordinates

# Example usage
# video_path = 'path/to/reaction_video.mp4'
# face_landmarks = [[(x1, y1, w1, h1), (x2, y2, w2, h2)], [(x3, y3, w3, h3), (x4, y4, w4, h4)], ...]  # List of face landmarks from the previous step
# track_faces(video_path, face_landmarks)



# I have generated a list of potential faces contained in a video. Each frame, I've identified potential faces and appended them to a list of matches, with each entry being (x,y,width,height). 

# I'd now like to parse this list of potential matches to find the top candidates. 

# The rough algorithm is as follows: 
#   - iterate over the potential faces to create groups of overlapping faces
#   - iterate over the groups. score each of the groups by the total area covered by each square in the group. 
#   - for each group, identify the smallest square that covers all the squares in the group  
#   - sort the groups by their score


def find_top_candidates(matches, wwidth, hheight):
    # Create a list to store the groups of overlapping faces
    face_groups = []

    # Iterate over the potential faces to create groups of overlapping faces
    try: 

      # Create an empty heat map
      heat_map = np.zeros((int(hheight) + 1, int(wwidth) + 1), dtype=int)

      # Iterate over the matches and update the heat map
      for match in matches:
          x, y, width, height, _ = match
          heat_map[y:y+height, x:x+width] += 1

      # Normalize the heat map to the range [0, 255]
      heat_map = (heat_map / np.max(heat_map)) * 255
      heat_map = heat_map.astype(np.uint8)

      # Apply a color map to the heat map
      heat_map_color = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)

      # Threshold
      _, binary = cv2.threshold(heat_map, 170, 255, cv2.THRESH_BINARY)

      # Find contours in the heat map
      contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

      # Filter contours based on area
      min_contour_area = int(width * .04 * height * .04)
      contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]

      # Draw bounding boxes around the hot areas
      for contour in contours:

        # Create a mask for the current contour
        contour_mask = np.zeros_like(heat_map)
        cv2.drawContours(contour_mask, [contour], 0, 255, thickness=cv2.FILLED)
        
        # Calculate the sum total of heat within the contour region
        heat_sum = np.sum(heat_map[contour_mask > 0])

        x, y, w, h = cv2.boundingRect(contour)
      
        cv2.rectangle(heat_map_color, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Identify faces within the contour
        group = [match for match in matches if x <= match[0] <= x+w and y <= match[1] <= y+h]

        face_groups.append( ((x, y, w, h), group, heat_sum)   )

    except Exception as e:
      import traceback
      traceback.print_exc()
      print("didn'twork!", e)





    # Calculate the score for each group based on the total area covered
    group_scores = []
    for kernel, group, heat in face_groups:
        total_area = heat # calculate_total_area(group)
        center = calculate_center(group)
        group_scores.append((group, total_area, kernel, center))

        # group_scores.append((group, total_area, find_smallest_square(group), center))

    # Sort the groups by their score in descending order
    sorted_groups = sorted(group_scores, key=lambda x: x[1], reverse=True)

    max_score = sorted_groups[0][1]
    accepted_groups = [grp for grp in sorted_groups if grp[1] >= .5 * max_score]

    print(sorted_groups)
    # Return the sorted groups
    return (accepted_groups, heat_map_color)

def calculate_center(group):
    if len(group) == 0:
      return (0,0)

    total_x = 0
    total_y = 0
    for face in group:
        x, y, w, h, center = face
        total_x += center[0]
        total_y += center[1]

    return (total_x / len(group), total_y / len(group))

def find_smallest_square(group):
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')

    # Iterate over the faces in the group to find the minimum and maximum coordinates
    for face in group:
        x, y, w, h, center = face
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

    # Calculate the dimensions of the smallest square
    square_size = max(max_x - min_x, max_y - min_y)

    # Calculate the coordinates of the top-left corner of the square
    square_x = min_x
    square_y = min_y

    return square_x, square_y, square_size, square_size


def not_outlier(face, group):
    x1, y1, w1, h1, center = face

    area_without = calculate_total_area(group)
    aaa = [face]
    aaa.extend(group)
    area_with = calculate_total_area(aaa)

    return (area_with - area_without) / area_with < .6




def is_overlapping(face, group):
    x1, y1, w1, h1, center = face

    for face2 in group:
        x2, y2, w2, h2, center = face2

        # Calculate the coordinates of the corners for the two faces
        top_left1 = (x1, y1)
        top_right1 = (x1 + w1, y1)
        bottom_left1 = (x1, y1 + h1)
        bottom_right1 = (x1 + w1, y1 + h1)

        top_left2 = (x2, y2)
        top_right2 = (x2 + w2, y2)
        bottom_left2 = (x2, y2 + h2)
        bottom_right2 = (x2 + w2, y2 + h2)

        # Check if any of the corner points of one face is inside the other face
        if is_point_inside_face(top_left1, face2) or is_point_inside_face(top_right1, face2) \
                or is_point_inside_face(bottom_left1, face2) or is_point_inside_face(bottom_right1, face2) \
                or is_point_inside_face(top_left2, face) or is_point_inside_face(top_right2, face) \
                or is_point_inside_face(bottom_left2, face) or is_point_inside_face(bottom_right2, face):
            return True

        # Check if the faces intersect horizontally or vertically
        if x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2:
            return True

    return False

def is_point_inside_face(point, face):
    x, y, w, h, center = face
    px, py = point

    if px >= x and px <= x + w and py >= y and py <= y + h:
        return True

    return False

def calculate_total_area(group):
    total_area = 0
    for face in group:
        x, y, w, h, center = face
        total_area += w * h

    return total_area












import cv2




# import tensorflow as tf


# # Loads the module from internet, unpacks it and initializes a Tensorflow saved model.
# def load_model(model_name):
#     model_url = 'http://download.tensorflow.org/models/object_detection/' + model_name + '.tar.gz'
    
#     model_dir = tf.keras.utils.get_file(
#         fname=model_name, 
#         origin=model_url,
#         untar=True,
#         cache_dir=pathlib.Path('.tmp').absolute()
#     )
#     model = tf.saved_model.load(model_dir + '/saved_model')
    
#     return model

# MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
# saved_model = load_model(MODEL_NAME)

# model = saved_model.signatures['serving_default']


# import cv2

# def image_contains_subimage(large_image_path, object_image_path, confidence_threshold=0.5):
#     # Load the pre-trained SSD model
#     model_file = 'path/to/ssd_model.pb'
#     config_file = 'path/to/ssd_config.pbtxt'
#     net = cv2.dnn.readNetFromTensorflow(model_file, config_file)

#     # Load the larger image and the object image
#     large_image = large_image_path #cv2.imread(large_image_path)
#     object_image = object_image_path #cv2.imread(object_image_path)

#     # Prepare the object image for matching
#     object_blob = cv2.dnn.blobFromImage(object_image, size=(300, 300), swapRB=True, crop=False)

#     # Set the input to the network for object matching
#     net.setInput(object_blob)

#     # Perform object detection on the object image
#     object_detections = net.forward()

#     # Prepare the larger image for object detection
#     large_blob = cv2.dnn.blobFromImage(large_image, size=(300, 300), swapRB=True, crop=False)

#     # Set the input to the network for object detection
#     net.setInput(large_blob)

#     # Perform object detection on the larger image
#     large_detections = net.forward()

#     # Loop over the detections in the larger image
#     for i in range(large_detections.shape[2]):
#         confidence = large_detections[0, 0, i, 2]
#         if confidence > confidence_threshold:
#             class_id = int(large_detections[0, 0, i, 1])
#             box = large_detections[0, 0, i, 3:7] * [large_image.shape[1], large_image.shape[0], large_image.shape[1], large_image.shape[0]]
#             (start_x, start_y, end_x, end_y) = box.astype(int)

#             # Extract the region of interest (ROI) from the larger image
#             roi = large_image[start_y:end_y, start_x:end_x]

#             # Prepare the ROI for matching
#             roi_blob = cv2.dnn.blobFromImage(roi, size=(300, 300), swapRB=True, crop=False)

#             # Set the input to the network for matching
#             net.setInput(roi_blob)

#             # Perform object matching on the ROI
#             roi_detections = net.forward()

#             # Loop over the detections in the ROI
#             for j in range(roi_detections.shape[2]):
#                 roi_confidence = roi_detections[0, 0, j, 2]
#                 if roi_confidence > confidence_threshold:
#                     roi_class_id = int(roi_detections[0, 0, j, 1])

#                     # Check if the detected class matches the object image class
#                     if roi_class_id == class_id:
#                         return True  # Object image found in the larger image

#     return False  # Object image not found in the larger image













# # Initialize SIFT detector
# sift = cv2.xfeatures2d.SIFT_create()
# # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
# bf = cv2.FlannBasedMatcher()

# def image_contains_subimage(large_image, small_image, threshold=1):
#     # Convert images to grayscale
#     large_gray = large_image # cv2.cvtColor(large_image, cv2.COLOR_BGR2GRAY)
#     small_gray = small_image # cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)

#     # Detect keypoints and compute descriptors for both images
#     kp1, des1 = sift.detectAndCompute(large_gray, None)
#     kp2, des2 = sift.detectAndCompute(small_gray, None)

#     if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
#       return True

#     # Match descriptors using FLANN matcher
#     try: 
#       # matches = bf.match(des1, des2)

#       matches = bf.knnMatch(des1, des2, k=2) # knnMatch is crucial

#     except Exception as e: 
#       print(e)      
#       print("MATCHES FAILED", des1, des2)
#       raise e


#     # Print distances of good matches
#     good_matches = []
#     for m in matches:  
#       if m[0].distance < 100:
#         if len(m) == 1: 
#           good_matches.append(m)
#         elif len(m) == 2:
#           (m1, m2) = m
#           if m1.distance < 0.7 * m2.distance:
#             good_matches.append(m1)
#             print(m1.distance)

#     if len(good_matches) > 0: 
#       print(f"\nGOOD MATCHES: {len(good_matches)} of {len(des2)} ({len(good_matches) / len(des2)}%)")
#       for m in good_matches:
#         print(m.distance)


#     # Check if the number of good matches is above the threshold
#     if len(good_matches) >= threshold:
#         print("FOUND IN BOTH IMAGES", len(good_matches) / len(des2))
#         return True
#     else:
        
#         return False



