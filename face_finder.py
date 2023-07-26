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



def crop_video(video_file, output_file, replacement_audio, x, y, w, h, centroids):

    # TODO: There is a new argument to this function, centroids, is an array of 
    #       (x,y) values. Each entry in the array is a frame of the video at video_file. 
    #       The below function currently crops the whole video at fixed centroid given
    #       by function parameters x and y. The function needs to be modified such that
    #       at each frame f the frame is cropped to the centroid given at centroids[f]. 

    # Load the video clip
    video = VideoFileClip(video_file)


    if w != h:
      w = h = min(w,h)

    if w % 2 > 0:
      w -= 1
      h -= 1

    if x < 0:
      x = 0

    if x + w > video.w:
      x -= x + w - video.w

    if y < 0: 
      y = 0

    if y + h > video.h:
      y -= y + h - video.h

    # Crop the video clip
    cropped_video = video.crop(x1=x, y1=y, x2=x+w, y2=y+h)

    if w > 450: 
      w = h = 450
      cropped_video = cropped_video.resize(width=w, height=h)


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
  
  # # Check if files exist with naming convention
  # while os.path.exists(f"{base_reaction_path}-cropped-{i}{base_video_ext}"):
  #     output_files.append(f"{base_reaction_path}-cropped-{i}{base_video_ext}")
  #     i += 1

  orientations = ["", "-center", "-left", "-right"]

  # Check if files exist with naming convention
  found = True    
  while found:
    found = False
    for orientation in orientations:
      f = f"{base_reaction_path}-cropped-{i}{orientation}{base_video_ext}"
      if os.path.exists(f):
        output_files.append(f)
        found = True

    i += 1

  # If no existing files found, proceed with face detection and cropping
  if len(output_files) == 0:
      reactors = detect_faces(react_path, base_path, show_facial_recognition = show_facial_recognition)
      for i, reactor in enumerate(reactors): 
          (x,y,w,h,orientation) = reactor[0]
          reactor_captures = reactor[1]
          # centroids = reactor[2]
          centroids = []
          output_file = f"{base_reaction_path}-cropped-{i}-{orientation}{base_video_ext}"
          crop_video(react_path, output_file, replacement_audio, int(x), int(y), int(w), int(h), centroids)
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


def detect_faces(react_path, base_path, frames_per_capture=500, show_facial_recognition=False):

    # Open the video file
    react_capture = cv2.VideoCapture(react_path)

    # Iterate over each frame in the video
    face_matches = []

    width = int(react_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(react_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f'\nDetecting faces for {react_path}')

    while True:

        try: 
          current_react = react_capture.get(cv2.CAP_PROP_POS_FRAMES)

          next_frame = current_react + frames_per_capture

          react_capture.set(cv2.CAP_PROP_POS_FRAMES, next_frame)

          ret, react_frame = react_capture.read()


          if not ret:
              break

        except Exception as e: 
          print('exception', e)
          break

        # Convert the frame to grayscale for face detection
        react_gray = cv2.cvtColor(react_frame, cv2.COLOR_BGR2RGB)

        # Perform face detection
        faces = get_faces_from(react_gray)

        # print("FACES:", faces)
        # Draw rectangles around the detected faces
        faces_this_frame = []
        for ((x, y, w, h), nose) in faces:
            # print( (x,y,w,h), nose)
            if w > .05 * width:
              if show_facial_recognition:
                cv2.rectangle(react_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
              faces_this_frame.append((x, y, w, h, nose))

        face_matches.append( (current_react + frames_per_capture, faces_this_frame))
        # print("matches:", face_matches)

        if show_facial_recognition:
          top, heat_map_color = find_top_candidates(face_matches, width, height)
          for tc in top:

            (x,y,w,h,o) = expand_face(tc, width, height)

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

    react_capture.release()
    cv2.destroyAllWindows()


    print('facematches:', face_matches)

    # Coarse-grained facial tracking
    # Now we're going to do some coarse matching to identify the most prominent
    # face locations. This will set the number of reactors we're dealing with, 
    # as well as their orientation and size. 
    coarse_reactors, _ = find_top_candidates(face_matches, width, height)
    reactors = []
    for reactor_captures in coarse_reactors:
      reactor = expand_face(reactor_captures, width, height) # (x,y,w,h,orientation)
      reactors.append( [reactor, reactor_captures] )

    # Fine-grained facial tracking
    # for (reactor, (group, total_area, kernel, center, avg_size) ) in reactors:
    #   #   - group is a list of the face matches in bounds (x,y,w,h)
    #   #   - total_area is the coverage area of the union of the bounds
    #   #   - kernel is the (x,y,w,h) bounding box
    #   #   - center is the (x,y) centroid
    #   #   - avg_size is the average (w,h) of candidate face matches in group

    #   centroids = find_reactor_centroids(face_matches, center, avg_size, kernel)
    #   centroids = smooth_and_interpolate_centroids(centroids)
    #   reactor.append(centroids)

    return reactors




#######
# Fine-grained face matching. 
# In the coarse-grained face matching step, we identified how many reactors there are
# and each of their anchor positions in the video. However, reactors often move their
# heads as they react. And, during production, sometimes reactors move their faces
# all over at different times, to create a more engaging experience. 
#
# We'd like to be able to track these movements so that we can crop the video in such 
# a way as to follow each reactors' face through the video. 
#
# This can be challenging because facial recognition on a frame-by-frame basis leads
# to all kinds of false positives and false negatives. Our coarse-grained algorithm 
# aggregated information in such a way that we could identify dominant facial positions
# and deal with false positives & negatives. If we get more fine-grained, we run the 
# risk of being deceived again.
#
# Luckily we can use the results of the coarse-grained algorithm to guide the fine-grained
# algorithm. Specifically, we know the average size (avg_width, avg_height) of the matched 
# reactor face, and the average centroid. For any given frame, we can examine the candidate 
# faces identified in a previous step (x,y,w,h), and score them based on their similarity to 
# the average size of and proximity to the centroid of the coarse matched reactor. We can
# then greedily select the best match for the reactor face for that frame. The idea is that
# most of the time, the reactor's face won't be moving far from the coarse centroid, and
# when it does (because of e.g. production), the size of the face region probably won't be
# that different, allowing us to find it. 
# 
# The find_reaction_centroids function is passed the following data: 
#    face_matches: An array of sampled frames with face-detection performed on them. Each 
#                  entry is a tuple (frame_number, faces), where faces is a list of 
#                  candidate faces (x,y,w,h,nose) }
#    center: The coarse-match centroid (x,y) for this reactor's face.
#    avg_size: The average size (width, height) of the facial matches that comprise the
#              coarse match.
#    kernel:   The (x,y,w,h) bounding box of the coarse_match. 
#    
def find_reactor_centroids(face_matches, center, avg_size, kernel):
  centroids = []
  for sampled_frame, faces in face_matches:

    best_score = 0
    best_centroid = None 

    for (x,y,w,h,nose) in faces:
      dx = center[0] - x
      dy = center[1] - y
      dist = sqrt( dx * dx + dy * dy )

      dw = avg_size[0] - w
      dh = avg_size[1] - h      
      size_diff = sqrt ( dw * dw + dh * dh )

      score = 1 / (1 + dist + size_diff)
      if score > best_score:
        best_score = score
        best_centroid = (x,y)

    centroids.append( (sampled_frame, best_centroid ) )

  return centroids



################
# Convolves and interpolates centroids
# sampled_centroids is an array of (frame, (x,y)), where frame is the 
# sampled frame number (doesn't increase by 1) and (x,y) is 
# the proposed centroid at that point. 
def smooth_and_interpolate_centroids(sampled_centroids):
  # TODO: Identify outliers in the sampled centroids. Replace the outliers
  #       with linearly interpolated centroid between the nearest non-outliers. 
  #       Ignore centroids with a value of None.
  # de_outliered_centroids = ...

  # TODO: Replace centroids with a value of None by linearly interpolating 
  #       between the closest defined centroid on either side. 
  # fully_valued_centroids = ...

  # TODO: smooth out the sampled centroids. Use a convolution that samples
  #       equally from the current frame and the 3 surrounding frames on 
  #       either side. 
  # smoothed_centroids = ...

  # TODO: interpolate smoothed_centroids. We want a centroid at each 
  # actual frame, not just each sampled frame. So fill in the values 
  # between each sampled centroid. The centroids can be linearly
  # interpolated between the two nearest sampled centroids, except 
  # for the first and last, which can just adopt the value of the 
  # nearest centroid.
  # interpolated_centroids = ...

  return interpolated_centroids


def is_overlap(x1, y1, w1, h1, x2, y2, w2, h2):
    # Check if two rectangles overlap
    return (x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2)

def expand_face(group, width, height, expansion=.9, sidedness=.7):
  (faces, score, (x,y,w,h), center, avg_size) = group


  if (center[0] > x + .65 * w): # facing right
    x = max(0, int(x - expansion * sidedness * w))
    orientation = 'right'

  elif (center[0] < x + .35 * w): # facing left
    x = max(0, int(x - expansion * (1 - sidedness) * w))
    orientation = 'left'

  else: # sorta in the middle
    x = max(0, int(x - expansion / 2 * w))
    orientation = 'center'


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



  return (x,y,w,h,orientation)










# Face Tracking
# To perform face tracking, we can utilize the face landmarks detected in the initial face 
# detection step. By tracking the landmarks across subsequent frames, we can estimate 
# the movement of the face and update the face coordinates

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
      for frame, faces in matches:
        for face in faces:
          x, y, width, height, _ = face
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
      min_contour_area = int(wwidth * .04 * hheight * .04)
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
        group = []
        for frame, faces in matches:
          for face in faces:
            if x <= face[0] <= x+w and y <= face[1] <= y+h:
              group.append(face)

        face_groups.append( ((x, y, w, h), group, heat_sum)   )

    except Exception as e:
      import traceback
      traceback.print_exc()
      print("didn'twork!", e)


    # Calculate the score for each group based on the total area covered
    group_scores = []
    for kernel, group, heat in face_groups:
        total_area = heat # calculate_total_area(group)
        center,avg_size = calculate_center(group)
        group_scores.append((group, total_area, kernel, center, avg_size))

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
    total_width = 0 
    total_height = 0 
    for face in group:
        x, y, w, h, center = face
        total_x += center[0]
        total_y += center[1]
        if len(center) > 2:
          total_width += center[2]
          total_height += center[3]

    return (total_x / len(group), total_y / len(group)), (total_width / len(group), total_height / len(group))



def calculate_total_area(group):
    total_area = 0
    for face in group:
        x, y, w, h, center = face
        total_area += w * h

    return total_area






