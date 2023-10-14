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
from moviepy.editor import VideoClip, VideoFileClip, AudioFileClip
from scipy import interpolate
from scipy.ndimage import convolve1d
import math
import matplotlib.pyplot as plt
import pickle

from utilities import conf



def create_reactor_view(reaction, show_facial_recognition=False, aside_video=None): 

  if aside_video is not None:
    react_path = aside_video
    replacement_audio = None
    frames_per_capture = 20
  else:
    react_path = reaction.get('aligned_path')
    replacement_audio = reaction.get("backchannel_audio")
    frames_per_capture = 45

  base_reaction_path, base_video_ext = os.path.splitext(react_path)

  output_files = []
  i = 0
  
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
      reactors = detect_faces(reaction, react_path, base_reaction_path, show_facial_recognition=show_facial_recognition, frames_per_capture=frames_per_capture)
      for i, reactor in enumerate(reactors): 
          (x,y,w,h,orientation) = reactor[0]
          reactor_captures = reactor[1]
          centroids = reactor[2]
          output_file = f"{base_reaction_path}-cropped-{i}-{orientation}{base_video_ext}"
          print("Going to crop the video now...")
          crop_video(react_path, output_file, replacement_audio, len(reactors), int(w), int(h), centroids)
          print("\t...Done cropping")
          output_files.append(output_file)

  cropped_reactors = []
  for file in output_files:
    cropped_reactors.append({
      'key': file,
      'priority': reaction.get('priority'), # circular reference      
      'clip': VideoFileClip(file),
      'orientation': get_orientation(file),
    })
  return cropped_reactors

def get_orientation(input_file):
    base_name, ext = os.path.splitext(input_file)
    parts = base_name.split('-')
    orientation = parts[-1] if len(parts) > 3 else 'center'
    return orientation



def replace_audio(video, audio_path, num_reactors):
    audio = AudioFileClip(audio_path)

    audio = audio.volumex(1.5) # so we hear the reactor backchannel better

    if num_reactors > 1: 
        # Divide the volume of the audio by the number of reactors
        audio = audio.volumex(1.0 / num_reactors)

    # Set the audio of the video clip
    video = video.set_audio(audio)

    return video


#       There is a new argument to the below crop_video function, centroids, is an array of 
#       (x,y) values. Each entry in the array is a frame of the video at video_file. 
#       The below function currently crops the whole video at fixed centroid given
#       by function parameters x and y. The function needs to be modified such that
#       at each frame f the frame is cropped to the centroid given at centroids[f]. 

def crop_video2(video_file, output_file, replacement_audio, x, y, w, h, centroids):

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
    cropped_video.write_videofile(output_file, codec="h264_videotoolbox", audio_codec="aac", fps=30,
                                  ffmpeg_params=['-pix_fmt', 'yuv420p','-q:v', '40'])



    # Close the video clip
    video.close()


def crop_video(video_file, output_file, replacement_audio, num_reactors, w, h, centroids):
    # Load the video clip
    print(f"\tloading the video {video_file}")
    video = VideoFileClip(video_file)

    if w != h:
      w = h = min(w,h)

    if w % 2 > 0:
      w -= 1
      h -= 1

    def make_frame(t):
        nonlocal w, h
        # Compute the current frame index
        frame_index = int(t * video.fps)
        # Retrieve the corresponding centroid and convert to integers

        if len(centroids) <= frame_index:
          x, y = map(int, centroids[len(centroids) - 1])          
        else:
          x, y = map(int, centroids[frame_index])

        x -= int(w / 2)
        y -= int(h / 2)

        if x < 0:
            x = 0
        if x + w > video.w:
            x -= x + w - video.w
        if y < 0: 
            y = 0
        if y + h > video.h:
            y -= y + h - video.h

        frame = video.get_frame(t)
        cropped_frame = frame[y:y+h, x:x+w]

        if w > 450: 
            cropped_frame = cv2.resize(cropped_frame, (450, 450))

        return cropped_frame

    print("\tmaking video clip")
    cropped_video = VideoClip(make_frame, duration=video.duration)

    if replacement_audio:
        print("\treplacing audio")
        cropped_video = replace_audio(cropped_video, replacement_audio, num_reactors)
    else: 
      cropped_video.audio = video.audio

    print("\twriting video")

    # Write the cropped video to a file
    cropped_video.write_videofile(output_file, codec="h264_videotoolbox", audio_codec="aac", fps=30,
                                ffmpeg_params=['-pix_fmt', 'yuv420p', '-q:v', '40'])

    # Close the video clip
    video.close()












#######
# Coarse-grained Face Tracking
#
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

detector = None

def get_faces_from(img):
  global detector
  from mtcnn import MTCNN
  detector = MTCNN()

  faces = detector.detect_faces(img)

  ret = [ (face['box'], face['keypoints']['nose']) for face in faces ]
  return ret



def detect_faces_in_frame(react_frame, show_facial_recognition, width, images_to_ignore, reduction_factor=0.5):
    # Resize the frame for faster face detection
    reduced_frame = cv2.resize(react_frame, (int(react_frame.shape[1] * reduction_factor), 
                                             int(react_frame.shape[0] * reduction_factor)))

    react_gray = cv2.cvtColor(reduced_frame, cv2.COLOR_BGR2RGB)

    # Perform face detection
    faces = get_faces_from(react_gray)

    # Scale the detected bounding boxes and keypoints back to original size
    scaled_faces = [((int(x/reduction_factor), int(y/reduction_factor), 
                      int(w/reduction_factor), int(h/reduction_factor)), 
                     (int(nose[0]/reduction_factor), int(nose[1]/reduction_factor))) for (x, y, w, h), nose in faces]

    # Draw rectangles around the detected faces
    faces_this_frame = []
    for ((x, y, w, h), nose) in scaled_faces:
        if w > .05 * width:
            faces_this_frame.append([x, y, w, h, nose])

    # compute histogram of grays for later fine-grained matching
    filtered_faces_this_frame = []
    ignored_faces_this_frame = []
    for face in faces_this_frame:
        react_frame_converted = cv2.cvtColor(react_frame, cv2.COLOR_BGR2RGB)

        should_ignore = False
        if len(images_to_ignore) > 0:
          candidate_face = cv2.cvtColor(get_face_img(react_frame_converted, face), cv2.COLOR_BGR2GRAY )  

          for image_to_ignore in images_to_ignore:
            fingerprint_found, distance = find_fingerprint_in_image(image_to_ignore, candidate_face)
            if fingerprint_found:
              should_ignore = True
              break
        
        if not should_ignore:
          hist = get_face_histogram(react_frame_converted, face)  # use the original frame here

          face.append(hist)
          filtered_faces_this_frame.append(face)

          if show_facial_recognition:
              x, y, w, h, nose, hist = face
              cv2.rectangle(react_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        else:
          if show_facial_recognition:
              x, y, w, h, nose = face
              cv2.rectangle(react_frame, (x, y), (x+w, y+h), (0, 40, 190), 2)
          ignored_faces_this_frame.append(face)
          print("IGNORING FACE!!!")

    return filtered_faces_this_frame, ignored_faces_this_frame


def detect_faces_in_frames(video, frames_to_read, show_facial_recognition=False, images_to_ignore=[]):

    # Iterate over each frame in the video
    face_matches = []

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))



    for current_react in frames_to_read: 

        try: 
          video.set(cv2.CAP_PROP_POS_FRAMES, current_react)
          ret, react_frame = video.read()

          if not ret:
            raise 
            break

        except Exception as e: 
          print('exception', e)
          break


        frame_faces, ignored_faces = detect_faces_in_frame(react_frame, show_facial_recognition, width, images_to_ignore)
        face_matches.append( (current_react, frame_faces, ignored_faces) )
        # print("matches:", face_matches)

        if show_facial_recognition and len(face_matches[-1][1]) > 0:
          top, heat_map_color = find_top_candidates(face_matches, width, height, None)
          for tc in top:

            try: 
              (x,y,w,h,o) = expand_face(tc, width, height)

              rect = tc[2]
              center = tc[3]
              cv2.rectangle(react_frame, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 0, 255), 2)
              cv2.circle(react_frame,(int(center[0]), int(center[1])), 5, (0, 0, 255), 2)
              cv2.rectangle(react_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            except: 
              print(f"Failed to plot {tc}")

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


    return face_matches


def detect_faces(reaction, react_path, base_reaction_path, frames_to_read=None, frames_per_capture=45, show_facial_recognition=False):

    # Open the video file
    video = cv2.VideoCapture(react_path)
    print(f'\nDetecting faces for {react_path}')

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frames_to_read is None:
      frames_to_read = []

      total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
      ts = 0
      while ts < total_frames - 1:
        frames_to_read.append(ts)
        ts += frames_per_capture
      frames_to_read.append(total_frames - 1)

    coarse_face_metadata = f"{base_reaction_path}-coarse_face_position_metadata.pckl"
    if os.path.exists(coarse_face_metadata):
      face_matches = read_coarse_face_metadata(coarse_face_metadata)
    else:

      base_dir = os.path.dirname(os.path.dirname(react_path))
      images_to_ignore = [ 
          cv2.imread(os.path.join(base_dir, f), cv2.IMREAD_GRAYSCALE)   
          for f in os.listdir(base_dir)
          if f.startswith("face-to-ignore")
      ]

      face_matches = detect_faces_in_frames(video, frames_to_read, show_facial_recognition=show_facial_recognition, images_to_ignore=images_to_ignore)
      output_coarse_face_metadata(coarse_face_metadata, face_matches)

    # print('facematches:', [(frame, x,y,w,h) for frame, (x,y,w,nose,hist) in face_matches])

    # Coarse-grained facial tracking
    # Now we're going to do some coarse matching to identify the most prominent
    # face locations. This will set the number of reactors we're dealing with, 
    # as well as their orientation and size. 

    num_reactors = reaction.get('num_reactors', None)
    print(f"LOOKING FOR {num_reactors}")

    coarse_reactors, _ = find_top_candidates(face_matches, width, height, num_reactors)
    reactors = []
    for reactor_captures in coarse_reactors:
      reactor = expand_face(reactor_captures, width, height) # (x,y,w,h,orientation)
      reactors.append( [reactor, reactor_captures] )

    # Fine-grained facial tracking
    for i, reactor in enumerate(reactors):
      
      if num_reactors is not None and i >= num_reactors:
        print(f"Skipping reactor because {reaction.get('channel')} is configured for only {num_reactors} reactors")
        break

      (coarse_reactor, (group, total_area, kernel, center, avg_size) ) = reactor

      #   - group is a list of the face matches in bounds (x,y,w,h)
      #   - total_area is the coverage area of the union of the bounds
      #   - kernel is the (x,y,w,h) bounding box
      #   - center is the (x,y) centroid
      #   - avg_size is the average (w,h) of candidate face matches in group

      centroids = find_reactor_centroids(face_matches, group, center, avg_size, kernel, width, height, len(reactors))

      centroids = smooth_and_interpolate_centroids(centroids)

      reactor.append(centroids)

    print(f"done detecting faces, found {len(reactors)}")

    video.release()
    cv2.destroyAllWindows()

    return reactors


def output_coarse_face_metadata(output_file, metadata):
    with open(output_file, 'wb') as f:
        pickle.dump(metadata, f)

def read_coarse_face_metadata(input_file):
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    return data




def expand_face(group, width, height, expansion=.8, sidedness=.7):
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


  print("Expanded face from ", group[2], " to ", (x,y,w,h))
  return (x,y,w,h,orientation)




def overlap_percentage(rect1, rect2):
    # Extract the x, y, width, and height for each rectangle
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # Find the x and y coordinates for the overlapping rectangle
    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

    # Calculate the area of the overlapping rectangle
    overlap_area = x_overlap * y_overlap

    # Calculate the areas of the input rectangles
    area1 = w1 * h1
    area2 = w2 * h2

    # Calculate the area of the union of the two rectangles
    total_area = area1 + area2 - overlap_area

    # Guard against division by zero
    if total_area == 0:
        return 0

    # Return the percentage overlap
    return (overlap_area / total_area) * 100



def create_heat_map_from_faces(faces_over_time, hheight, wwidth, best_ignored_area = None):




    # Create a list to store the groups of overlapping faces
    face_groups = []

    # Iterate over the potential faces to create groups of overlapping faces
    try: 

      # Create an empty heat map
      heat_map = np.zeros((int(hheight) + 1, int(wwidth) + 1), dtype=int)

      # Iterate over the matches and update the heat map
      for faces in faces_over_time:
        for face in faces:
          x, y, width, height, nose, hist = face


          # Don't add to the heat map any entries that overlap with best_ignored_area
          if best_ignored_area is not None:
            if overlap_percentage( best_ignored_area, (x, y, width, height) ) > 20:
              print('NOT ADDING FACE BECAUSE IN IGNORED AREA')
              continue


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
        for faces in faces_over_time:
          for face in faces:
            if x <= face[0] + face[2] / 2 <= x+w and y <= face[1] + face[3] / 2 <= y+h:
              group.append(face)

        face_groups.append( ((x, y, w, h), group, heat_sum)   )

    except Exception as e:
      import traceback
      traceback.print_exc()
      print("didn't work!", e)


    return face_groups, heat_map_color



def find_top_candidates(matches, wwidth, hheight, num_reactors):

    ignored_face_groups, __ = create_heat_map_from_faces( [f[2] for f in matches], hheight, wwidth )

    if len(ignored_face_groups) > 0:
      ignored_face_groups.sort(key=lambda x: x[2], reverse=True)
      best_ignored_area = ignored_face_groups[0][0]
    else:
      best_ignored_area = None

    face_groups, heat_map_color = create_heat_map_from_faces( [f[1] for f in matches], hheight, wwidth, best_ignored_area )


    # Calculate the score for each group based on the total area covered
    group_scores = []
    for kernel, group, heat in face_groups:
        total_area = heat 
        center,avg_size = calculate_center(group)
        group_scores.append((group, total_area, kernel, center, avg_size))

    # Sort the groups by their score in descending order
    sorted_groups = sorted(group_scores, key=lambda x: x[1], reverse=True)

    if num_reactors is None:
      max_score = sorted_groups[0][1]
      accepted_groups = [grp for grp in sorted_groups if grp[1] >= .5 * max_score]
    else: 
      accepted_groups = sorted_groups[0:num_reactors]
      
    # print(sorted_groups)
    # Return the sorted groups
    return (accepted_groups, heat_map_color)

def calculate_center(group):
    if len(group) == 0:
      return (0,0), (0,0)

    total_x = 0
    total_y = 0
    total_width = 0 
    total_height = 0 
    for face in group:
        x, y, w, h, center, hist = face
        total_x += center[0] + w / 2
        total_y += center[1] + h / 2
        total_width += w
        total_height += h

    return (total_x / len(group), total_y / len(group)), (total_width / len(group), total_height / len(group))






#######
# Fine-grained Face Tracking. 
#
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
def find_reactor_centroids(face_matches, coarse_matches, center, avg_size, kernel, video_width, video_height, num_reactors):
  centroids = []
  print(f"Finding reactor centroids {len(face_matches)}")

  assert(len(coarse_matches) > 0)

  # print(f"coarse_samples {len(coarse_matches)}", coarse_matches)

  if len(coarse_matches) > 3:
    coarse_samples = [0, int(len(coarse_matches)/2), len(coarse_matches) - 1 ]
    rep_histos = [coarse_matches[i][5] for i in coarse_samples]
  else:
    rep_histos = [match[5] for match in coarse_matches]


  moving_avg_centroid = None
  for i, (sampled_frame, faces, ignored_faces) in enumerate(face_matches):
    # print(f"\tmatch for frame {sampled_frame} {len(faces)} {center} {avg_size}")

    best_score = 0
    best_centroid = None 
    

    if i == 0 or len(faces) >= num_reactors: # we get bogus results if we don't have enough faces to 
                                             # choose from in a given frame

      if len(faces) > 1:
        for (x,y,w,h,nose,hist) in faces:
          dx = center[0] - (x + w/2)
          dy = center[1] - (y + h/2)
          dist = math.sqrt( dx * dx + dy * dy )
          dist /= video_width

          if moving_avg_centroid is None:
            dist_from_previous = 0
          else: 
            dx = moving_avg_centroid[0] - (x + w/2)
            dy = moving_avg_centroid[1] - (y + h/2)
            dist_from_previous = math.sqrt( dx * dx + dy * dy )
            dist_from_previous /= video_width


          dw = avg_size[0] - w
          dh = avg_size[1] - h      
          size_diff = math.sqrt ( dw * dw + dh * dh )
          size_diff /= video_height

          hist_similarity = 0
          for rep_histo in rep_histos:
            sim = compare_faces(hist, rep_histo)
            sim = (sim + 1) / 2 # convert from [-1, 1] to [0, 1]
            hist_similarity += sim

          hist_similarity /= len(rep_histos)

          score = hist_similarity / (1 + size_diff + 0.5 * dist + 0.5 * dist_from_previous)
          # print(f"\t\t{score} ({best_score}) dist={dist} size={size_diff} sim={hist_similarity} {(x,y,w,h)}")
          if score > best_score:
            best_score = score
            best_centroid = (x + w/2, y + h/2)

      elif len(faces) == 1:
        (x,y,w,h,nose,hist) = faces[0]
        best_centroid = (x + w/2, y + h/2)

      centroids.append( (sampled_frame, best_centroid ) )
      if moving_avg_centroid is None:
        moving_avg_centroid = best_centroid
      else:
        curr_x, curr_y = best_centroid
        ma_x, ma_y = moving_avg_centroid
        weight = .33
        moving_avg_centroid = (curr_x * weight + (1 - weight) * ma_x, curr_y * weight + (1 - weight) * ma_y )
    else: 
      centroids.append( (sampled_frame, centroids[-1][1]))

  print(f"Done finding centroids {len(centroids)}")
  return centroids






def get_face_img(image, bbox):
  # Get the face image from the bounding box
  x, y, w, h, nose = bbox
  face_img = image[y:y+h, x:x+w]
  return face_img

def preprocess_face(face): 
  face_color_shifted = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
  face_equalized = cv2.equalizeHist(face_color_shifted)
  face_resized = cv2.resize(face_equalized, (128, 128))  
  return face_resized

def compute_histogram(face):
  # Compute histogram of pixel intensities
  hist = cv2.calcHist([face],[0],None,[256],[0,256])
  cv2.normalize(hist, hist)
  hist = hist.astype(np.float32)  # Ensure the histogram is float32
  return hist

def get_face_histogram(image=None, face=None, face_img=None):
  if face:
    (x,y,w,h,nose) = face
    face_img = get_face_img(image, face)
  elif face_img is None:
    raise Exception("No face or face_img given")

  face_preprocessed = preprocess_face(face_img)
  hist = compute_histogram(face_preprocessed)
  return hist

def compare_faces(hist1, hist2):
  # Compare two faces using histogram correlation
  # hist1 = hist1.astype('float32')
  # hist2 = hist2.astype('float32')    
  correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
  return correlation






def smooth_and_interpolate_centroids(sampled_centroids):

    print("Smoothing and interpolating centroids...")
    # Unpack the frames and the centroids from the input
    frames, centroids = zip(*sampled_centroids)
    frames = np.array(frames)
    
    # Prepare x, y arrays filled with NaNs
    x = np.empty(len(frames))
    x.fill(np.nan)
    y = np.empty(len(frames))
    y.fill(np.nan)
    
    # Assign values from centroids to x and y, skipping None centroids
    for i, centroid in enumerate(centroids):
        if centroid is not None:
            x[i], y[i] = centroid

    # For plotting, create a copy before the processing
    original_x, original_y = np.copy(x), np.copy(y)


    # Convert the coordinates to numpy arrays and set outliers to NaN
    # (Identify outliers as points that are more than 3 standard deviations from the mean)
    x = np.array(x, dtype=np.float)
    y = np.array(y, dtype=np.float)
    x[np.abs(x - np.nanmean(x)) > 3 * np.nanstd(x)] = np.nan
    y[np.abs(y - np.nanmean(y)) > 3 * np.nanstd(y)] = np.nan

    # Replace outliers with linearly interpolated values
    nans_x, = np.where(np.isnan(x))
    nans_y, = np.where(np.isnan(y))
    if nans_x.size > 0:
        x[nans_x] = np.interp(frames[nans_x], frames[~np.isnan(x)], x[~np.isnan(x)])
    if nans_y.size > 0:
        y[nans_y] = np.interp(frames[nans_y], frames[~np.isnan(y)], y[~np.isnan(y)])

    # # Smooth out the sampled centroids with convolution
    kernel = np.array([1/32, 1/16, 1/8 + 1/32, 1/2, 1/8 + 1/32, 1/16, 1/32])
    x = convolve1d(x, kernel, mode='nearest')
    y = convolve1d(y, kernel, mode='nearest')

    # Interpolate to get a centroid for each frame
    interp_func_x = interpolate.interp1d(frames, x, kind='linear', fill_value="extrapolate")
    interp_func_y = interpolate.interp1d(frames, y, kind='linear', fill_value="extrapolate")

    # We assume actual frames are from the minimum sampled frame to the maximum sampled frame
    actual_frames = np.arange(min(frames), max(frames)+1)
    interpolated_x = interp_func_x(actual_frames)
    interpolated_y = interp_func_y(actual_frames)

    # Form the final list of interpolated centroids
    interpolated_centroids = list(zip(interpolated_x, interpolated_y))
    

    if False: 
      # Start the plot process
      plt.figure(figsize=(14,6))
      
      plt.subplot(2,1,1)
      plt.scatter(frames, original_x, color='red', alpha=0.5, label='Sampled X')
      plt.plot(actual_frames, interpolated_x, color='blue', alpha=0.5, label='Interpolated X')
      plt.title('X coordinates')
      plt.legend()
      
      plt.subplot(2,1,2)
      plt.scatter(frames, original_y, color='red', alpha=0.5, label='Sampled Y')
      plt.plot(actual_frames, interpolated_y, color='blue', alpha=0.5, label='Interpolated Y')
      plt.title('Y coordinates')
      plt.legend()

      plt.tight_layout()
      plt.show()

    print("\t...done")

    return interpolated_centroids






def find_fingerprint_in_image(fingerprint_image, target_image, scales=None, threshold=0.95):
    best_corr = -1  # Best correlation value
    best_scale = None  # Best scale factor
    best_loc = None  # Best match location

    if scales is None:
      scales = [0.05, 0.1, .2, 0.25, .35, 0.5,.625, 0.75, .85, 1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5, 6, 7, 8]
    
    max_scale_x = target_image.shape[1] / fingerprint_image.shape[1]
    max_scale_y = target_image.shape[0] / fingerprint_image.shape[0]
    max_scale = min(max_scale_x, max_scale_y)

    scales = [scale for scale in scales if scale <= max_scale]

    # Convert images to grayscale if they are not
    if len(fingerprint_image.shape) == 3:
        fingerprint_image = cv2.cvtColor(fingerprint_image, cv2.COLOR_BGR2GRAY)
    if len(target_image.shape) == 3:
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

    


    contains_fingerprint = False
    secondary_threshold = 0.95  # set your secondary threshold
    
    # Loop over each scale factor
    for scale in scales:
        # Resize the template according to the scale factor
        resized_fingerprint = cv2.resize(fingerprint_image, (0, 0), fx=scale, fy=scale)

        # Check if resized_fingerprint is smaller than target_image
        if resized_fingerprint.shape[0] > target_image.shape[0] or resized_fingerprint.shape[1] > target_image.shape[1]:
            continue  # Skip this iteration if the resized template is larger
        
        # Perform template matching
        result = cv2.matchTemplate(target_image, resized_fingerprint, cv2.TM_CCOEFF_NORMED)
        
        # Secondary check
        result_secondary = cv2.matchTemplate(target_image, resized_fingerprint, cv2.TM_CCORR)
        
        # Find points where both primary and secondary checks pass the thresholds
        match_locations_primary = np.where(result >= threshold)
        match_locations_secondary = np.where(result_secondary >= secondary_threshold)
        
        # Check if both metrics agree on a match
        for pt in zip(*match_locations_primary[::-1]):
            if pt in zip(*match_locations_secondary[::-1]):
                contains_fingerprint = True
                break
                
        if contains_fingerprint:
            break
    
    return contains_fingerprint, 100

