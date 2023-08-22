import cv2
import numpy as np
import subprocess
import os
import math
from PIL import Image, ImageDraw, ImageChops
from itertools import groupby
import colorsys


from moviepy.editor import ImageClip, CompositeVideoClip, CompositeAudioClip
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.video.VideoClip import VideoClip, ColorClip
from moviepy.editor import VideoFileClip
from moviepy.video.fx.all import crop

from utilities import conversion_frame_rate, conversion_audio_sample_rate


# I have a base video, and then any number of videos of people reacting to that base video. I would 
# like to use MoviePy to compose a video that has all of these videos together. 

# The most difficult part of this will probably be the layout, given that:
#   1) There can be any number of reaction videos
#   2) There won't always be a base video

# We want to find a visually attractive layout that:
#   1) When there is a base video, it should be shown prominently. Its size shouldn't depend on the 
#      number of reactions. Perhaps its size should be invariate to the total space allocated to 
#      the reactions, say at least 25% of the space allocated to reactions (or some other percentage). 
#      Its position can change though, depending on aesthetic considerations given the number of reactions. 
#   2) The individual reactions should be sized to fit the layout.  
#   3) There can be any number of layout patterns used. There could be a base video in the middle, with 
#      reactions surrounding it. The base video could be at the top, with reactions to the bottom, 
#      or the bottom and the sides. etc. etc. I'm particularly interested in using a honeycomb layout 
#      for the reactions, to create interlocking hexagon pattern. I'm open to other layout styles.  
#   4) Each reaction video will also provide data about the reactor's orientation in the video 
#      (like, looking to the right or left). The algorithm should use this information to try to 
#      put each reaction video in a place where it looks like reactor is looking toward the base 
#      video. This is a soft constraint. 

# The aspect ration of the resulting video should be approximately that of a laptop screen, and can 
# be any size up to the resolution of a modern macbook pro. These constraints are soft. 

def compose_reactor_compilation(song, base_video_path, reactions, output_path, output_size=(1792, 1120), fast_only=False):

    if os.path.exists(output_path):
      print("Compilation already exists", output_path)
      return

    print(f"Creating compilation for {output_path}")

    base_video = VideoFileClip(base_video_path)

    width, height = output_size
    base_size = int(width * .55)

    audio_clips = []  # list to hold all audio clips

    include_base_video = song['include_base_video']

    if include_base_video:

      base_video = base_video.resize(base_size / base_video.w)
      base_width, base_height = base_video.size

      if base_video.audio.fps != conversion_audio_sample_rate:
          base_video = base_video.set_audio(base_video.audio.set_fps(conversion_audio_sample_rate))

      x,y = (base_width / 2, base_height / 2)

      base_video = base_video.set_position((x - base_width / 2, y - base_height / 2))

      # upper left point of rectangle, lower right point of rectangle
      bounds = (x - base_width / 2, y - base_height / 2, x + base_width / 2, y + base_height / 2)
    else: 
      bounds = [0,0,width,0]

    audio_clips.append(base_video.audio)


    total_videos = len(reactions)

    # enough_grid_cells = False
    # while not enough_grid_cells: 
    #   hex_grid, cell_size = generate_hexagonal_grid(width, height, total_videos, bounds)
    #   cell_size = math.floor(cell_size)

    #   all_good = []
    #   add_to_end = []
    #   for cell in hex_grid: 
    #     dist = distance_from_region( cell, bounds )
    #     if dist <= 0: 
    #       x,y = cell
    #       dist = -overlap_percentage( ( x - cell_size / 2, y - cell_size / 2, x + cell_size / 2, y + cell_size / 2 ), bounds)   # -% overlap with bounds 
    #       if dist > -40:
    #         all_good.append( (dist, cell) )
    #     else:
    #       all_good.append( (dist, cell) )
    #   hex_grid = sorted(all_good, key=lambda cell:cell[0], reverse = False)
    #   add_to_end = sorted(add_to_end, key=lambda cell: cell[0], reverse = True)
    #   hex_grid.extend(add_to_end)
    #   hex_grid = [cell[1] for cell in hex_grid]


    #   print(f"len(hex_grid) = {len(hex_grid)}  total_videos={total_videos}")
    #   enough_grid_cells = len(hex_grid) >= total_videos
    #   if not enough_grid_cells:
    #     total_videos += 1

    hex_grid, cell_size = generate_hexagonal_grid(width, height, total_videos, bounds)
    hex_grid = sorted(hex_grid, key=lambda cell: distance_from_region( cell, bounds ), reverse = False)
    cell_size = math.floor(cell_size)
    print(f"len(hex_grid) = {len(hex_grid)}  total_videos={total_videos} cell_size={cell_size}")

    # print(f"total_videos = {total_videos}, cell_size={cell_size}, cells_returned={len(hex_grid)}, grid={hex_grid}")

    positions = assign_hex_cells_to_videos(width, height, hex_grid, cell_size, bounds, reactions)

    # Position reactions around the base video
    featured_clips = []
    other_clips = []

    base_audio_as_array = base_video.audio.to_soundarray()
    reactor_colors = generate_hsv_colors(len(positions), 1, .6)
    clip_length = 0
    for i, (reaction, pos) in enumerate(positions):
        featured = reaction['featured']

        clip = reaction['clip']
        volume_adjusted_audio = match_audio_peak(base_audio_as_array, clip.audio.to_soundarray(), factor=1.5)
        audio_clips.append(AudioArrayClip(volume_adjusted_audio, fps=clip.audio.fps))

        size = cell_size
        if featured: 
          size *= 1.15
          size = int(size)


        hsv_color = reactor_colors[i]
        color_func = create_color_func(hsv_color, clip)
        clip = create_masked_video(clip, color_func=color_func, border_thickness=10, width=size, height=size, as_circle=featured)

        x,y = pos

        clip = clip.set_position((x - size / 2, y - size / 2))
        if featured:
          featured_clips.append(clip)
        else: 
          other_clips.append(clip)

        if clip_length < clip.duration:
          clip_length = clip.duration



    duration = max(base_video.duration, clip_length)
    # duration = 30 


    # Create the composite video
    clips = other_clips + featured_clips
    if include_base_video:
        clips = [base_video] + clips
    final_clip = CompositeVideoClip(clips, size=output_size)
    final_clip = final_clip.set_duration(duration)


    final_audio = CompositeAudioClip(audio_clips)
    final_audio = final_audio.set_duration(duration)        
    final_clip = final_clip.set_audio(final_audio)
    final_clip.set_fps(30)

    # Save the result
    fast_path = output_path + "fast.mp4"
    if not os.path.exists(fast_path):
      final_clip.write_videofile(fast_path, 
                             codec='libx264', 
                             audio_codec="aac", 
                             threads=2, 
                             preset='ultrafast', 
                             profile='baseline', 
                             bitrate="500k")

    if not fast_only:
      final_clip.write_videofile(output_path, codec='libx264', audio_codec="aac", threads=2, )





# I want to create a hexagonal grid that fits a space of size width x height. There needs to be at 
# least n cells, each of which is fully visible. Cells should have equal height and width. 
# Cell size should be maximized given the constraints. Please write a function generate_hexagonal_grid 
# that is given a width and a height and a minimum number of cells and which returns a list of cell 
# centroids for this hexagonal grid, and the cell size. The centroids should be sorted by 
# distance to the center of the grid, with the first ones the closest. 

def generate_hexagonal_grid(width, height, min_cells, outside_bounds=None, center=None):
    # Calculate a starting size for the hexagons based on the width and height
    a = min(width / ((min_cells / 2) ** 0.5), height / ((min_cells / (2 * np.sqrt(3))) ** 0.5))
    a *= 2


    # Calculate the center of the grid
    if center is None:  
      main_padding = max(0, 180 - 8 * min_cells)
      center = (outside_bounds[2] + main_padding, outside_bounds[3] + main_padding)
    
    # Create an empty list to hold the hexagon center coordinates
    coords = []

    while len(coords) < min_cells:
      # Empty the list
      coords.clear()
      
      # Calculate the horizontal and vertical spacing for the hexagons
      dx = 2 * a
      dy = np.sqrt(3) * a

      # Adjust the vertical spacing: every second row is moved up by half of the hexagon's height
      dy_adj = dy * 7/8

      # Calculate the number of hexagons that can fit in the width and height
      nx = int(np.ceil(width / dx))
      ny = int(np.ceil(height / dy_adj))

      # Generate a dense grid of hexagons around the center
      for j in range(-ny, ny + 1):
        for i in range(-nx, nx + 1):
          x = center[0] + dx * (i + 0.5 * (j % 2))
          y = center[1] + dy_adj * j
          
          # Add the hexagon to the list if it's fully inside the width x height area
          if x - a >= 0 and y - a >= 0 and x + a <= width and y + a <= height and distance_from_region((x,y), outside_bounds) > 0:
            coords.append((x, y))

      # Reduce the size of the hexagons and try again if we don't have enough
      if len(coords) < min_cells:
        a *= 0.99

    # Sort hexagons by distance from the center
    coords.sort(key=lambda p: (p[0] - center[0])**2 + (p[1] - center[1])**2)

    assert len(coords) >= min_cells, f"Only {len(coords)} cells could be placed"
    
    return coords, 2 * a  # return 2 * a (diameter) to match with the circle representation


def assign_hex_cells_to_videos(width, height, grid_cells, cell_size, base_video, reaction_videos):
    # Assumes: grid_cells are sorted
    assignments = {}

    title_map = {}
    for r in reaction_videos:
      title_map[r['key']] = r

    # Define a helper function to assign a video to a cell
    def assign_video(video, cells, assignments):

        # Iterate over the cells from closest to farthest
        best_score = -9999999999
        best_spot = None 
        best_adjacent = None

        for cell in grid_cells:

            if cell in assignments.values():
              continue

            score = 0

            # Check the orientation constraint
            if (video['orientation'] == 'center'):

              if base_video[0] * .25 <= cell[0] and cell[0] <= base_video[0] * .75:
                score += 1
              else:
                score += 1 / abs(cell[0] - base_video[0] / 2)

            elif (video['orientation'] == 'right'):
              if cell[0] <= base_video[0] / 2:
                score += 1
              else: 
                score += 1 / abs(cell[0])
            else: 
              if cell[0] >= base_video[0]:
                score += 1
              else: 
                score += 1 / abs(cell[0] - base_video[0])


            # If the video is featured, ensure it is not on the edge and not adjacent to other featured videos
            if video.get('featured', False):
                # Check if cell is adjacent to another featured video
                for v in assignments.keys():
                  if title_map[v].get('featured', False):
                    if distance(cell, assignments[v]) <= 1.1*cell_size:
                      score -= 1
            


            if score > best_score: 
              if video.get('group', False):

                # Check if there is an open adjacent to the immediate left or right (y-difference is 0, x-difference~=cell_size)
                # If there is, assign the spot and return the open cell
                found_adjacent = False
                for adjacent_cell in cells:
                  if adjacent_cell in assignments.values():
                    continue
                  if abs(cell[1] - adjacent_cell[1]) <= 1 and abs(abs(cell[0] - adjacent_cell[0]) - cell_size) <= 1:
                    assignments[video['key']] = cell
                    best_adjacent = adjacent_cell
                    found_adjacent = True
                    break

                if not found_adjacent:
                  continue

              best_score = score
              best_spot = cell



            # if video.get('featured', False):
            #   print(f"Evaluating {cell}={score} for {video['key']}")


        assignments[video['key']] = best_spot
        return best_adjacent


    # Partition the reaction videos into featured, connected (not featured), and the rest
    featured_videos = [v for v in reaction_videos if v.get('featured', False)]
    connected_videos = [v for v in reaction_videos if v.get('group', None) and not v.get('featured', False)]
    # other_videos = [v for v in reaction_videos if not v.get('featured', False) and 'connection' not in v]
    other_videos = [v for v in reaction_videos if v.get('group', None) is None and not v.get('featured', False) and v.get('orientation') == 'right'] + \
                   [v for v in reaction_videos if v.get('group', None) is None and not v.get('featured', False) and v.get('orientation') == 'left'] + \
                   [v for v in reaction_videos if v.get('group', None) is None and not v.get('featured', False) and v.get('orientation') == 'center']

    print(f"feat={len(featured_videos)}  connected={len(connected_videos)}   other={len(other_videos)}")

    # First, assign the featured videos to the cells bordering the base video
    border_cells = [cell for cell in grid_cells if is_bordering(cell, cell_size, base_video)]
    for video in featured_videos:
        # assign_video(video, border_cells, assignments)
        assign_video(video, grid_cells, assignments)

    # Then assign groups
    connected_videos.sort(key=lambda v: v['group'])
    grouped_videos = [list(g) for _, g in groupby(connected_videos, key=lambda v: v['group'])]          
    for group in grouped_videos:
      assert(len(group) < 3)
      p1,p2 = group

      p2_spot = assign_video(p1, grid_cells, assignments)
      if p2_spot is not None:
        assignments[p2['key']] = p2_spot
      else: # failed
        assign_video(p2, grid_cells, assignments)



    # # Then, assign the connected videos
    # for video in connected_videos:
    #     if not assign_video(video, grid_cells, assignments):
    #         print(f"Warning: could not assign connected video {video}")

    # Finally, assign the remaining videos
    for video in other_videos:
        assign_video(video, grid_cells, assignments)



    # Return the final assignments

    return [(title_map[k], v) for k,v in assignments.items()]




# I’m using the following function to mask a video to a hexagon or circular shape, with 
# a border. I’d like to make the border dynamic. Specifically:
#   - Each reaction video should have its own bright saturated HSV color assigned to it
#   - The HSV color for the respective reaction video should be mixed with white, 
#     inversely proportional to the volume of the track at that timestamp. That is, 
#     when there is no volume, the border should be white, and when it is at its 
#     loudest, it should be the HSV color.

def create_masked_video(clip, width, height, color_func, border_thickness=10, as_circle=False):

    # Resize the clip
    clip = clip.resize((width, height))

    # Create new PIL images with the same size as the clip, fill with black color
    mask_img_large = Image.new('1', (width, height), 0)
    mask_img_small = Image.new('1', (width, height), 0)
    draw_large = ImageDraw.Draw(mask_img_large)
    draw_small = ImageDraw.Draw(mask_img_small)

    if as_circle:
        # Draw larger and smaller circles on the mask images
        draw_large.ellipse([(0, 0), (width, height)], fill=1)
        draw_small.ellipse([(border_thickness, border_thickness), ((width - border_thickness), (height - border_thickness))], fill=1)
    else:
        # Define hexagon vertices for larger hexagon
        vertices_large = [(0, height*0.25), (width*0.5, 0), (width, height*0.25),
                          (width, height*0.75), (width*0.5, height), (0, height*0.75)]

        # Calculate adjustments for smaller hexagon vertices
        x_adjust = border_thickness * np.sqrt(3) / 2  # trigonometric calculation
        y_adjust = border_thickness / 2

        # Define hexagon vertices for smaller hexagon
        vertices_small = [(x_adjust, height*0.25 + y_adjust), 
                          (width*0.5, y_adjust), 
                          (width - x_adjust, height*0.25 + y_adjust),
                          (width - x_adjust, height*0.75 - y_adjust), 
                          (width*0.5, height - y_adjust), 
                          (x_adjust, height*0.75 - y_adjust)]

        # Draw the larger and smaller hexagons on the mask images
        draw_large.polygon(vertices_large, fill=1)
        draw_small.polygon(vertices_small, fill=1)

    # Subtract smaller mask from larger mask to create border mask
    border_mask = ImageChops.subtract(mask_img_large, mask_img_small)
    border_mask_np = np.array(border_mask)
    mask_img_small_np = np.array(mask_img_small)

    def make_frame(t):
        img = np.ones((height, width, 3))

        # Convert the color from HSV to RGB, then scale from 0-1 to 0-255
        color_rgb = np.array(color_func(t)) * 255

        img[border_mask_np > 0] = color_rgb  # apply color to border

        return img

    # Define make_mask function to create mask for the border
    def make_mask(t):
        mask = np.zeros((height, width))
        mask[border_mask_np > 0] = 1
        return mask

    border_clip = VideoClip(make_frame, duration=clip.duration)

    # Apply mask to the border_clip
    border_mask_clip = VideoClip(make_mask, ismask=True, duration=clip.duration)
    border_clip = border_clip.set_mask(border_mask_clip)

    # Create video clip by applying the smaller mask to the original video clip
    clip = clip.set_mask(ImageClip(mask_img_small_np, ismask=True))

    # Overlay the video clip on the border clip
    final_clip = CompositeVideoClip([border_clip, clip])

    return final_clip






def create_color_func(hsv_color, clip):
    h, s, v = hsv_color
    audio_volume = get_audio_volume(clip)

    def color_func(t):
        # Get the volume at current time
        volume = audio_volume[int(t * 22000)]

        # If volume is zero, return white
        if volume == 0:
            return 1, 1, 1  # White in RGB

        # Calculate the interpolated V value based on audio volume
        v_modulated = 1 - volume * (1 - v)

        # Convert modulated HSV color back to RGB
        r, g, b = colorsys.hsv_to_rgb(h, s, v_modulated)
        return r, g, b

    return color_func


from utilities import conversion_audio_sample_rate
def get_audio_volume(clip, fps=None):
    if fps is None:
      fps = conversion_audio_sample_rate
      
    """Calculate the volume of the audio clip"""
    audio = clip.audio.to_soundarray(fps=fps)
    audio_volume = np.sqrt(np.mean(np.square(audio), axis=1))  # RMS amplitude
    audio_volume /= np.max(audio_volume)  # normalize to range [0, 1]
    return audio_volume



def generate_hsv_colors(n, s, v):
    """Generates n evenly distributed HSV colors with the same S and V"""
    return [(i/n, s, v) for i in range(n)]


def is_in_center(cell, width):
  x = cell[0]
  return x > width * .25 and x < width * .75

def is_in_right_half(cell, width):
  x = cell[0]
  return x > width * .5

def is_in_left_half(cell, width):
  x = cell[0]
  return x < width * .5

def is_bordering(cell, cell_size, bounds):
  return distance_from_region(cell, bounds) <= np.sqrt(3) * cell_size


def distance_from_region(point, bounds):
    """
    Calculates the Euclidean distance from a point to a rectangle.
    Parameters:
    - point: Tuple (x, y) for the point.
    - bounds: Tuple (x1, y1, x2, y2) for the upper left and lower right points of the rectangle.
    Returns: distance as a float.
    """
    x, y = point
    x1, y1, x2, y2 = bounds

    if x1 <= x and x <= x2 and y1 <= y and y <= y2: 
      return 0

    dx = min(abs(x1 - x), abs(x2 - x))
    dy = min(abs(y1 - y), abs(y2 - y))

    return (dx ** 2 + dy ** 2) ** 0.5


def distance(point, point2):
    """
    Calculates the Euclidean distance from a point to a rectangle.
    Parameters:
    - point: Tuple (x, y) for the point.
    - bounds: Tuple (x1, y1, x2, y2) for the upper left and lower right points of the rectangle.
    Returns: distance as a float.
    """
    x, y = point
    x2, y2 = point2

    dx = x2 - x 
    dy = y2 - y

    return (dx ** 2 + dy ** 2) ** 0.5




def overlap_percentage(smaller, larger):
    # Unpack rectangle coordinates
    smaller_left, smaller_top, smaller_right, smaller_bottom = smaller
    larger_left, larger_top, larger_right, larger_bottom = larger

    # Calculate the overlapping rectangle coordinates
    overlap_left = max(smaller_left, larger_left)
    overlap_top = max(smaller_top, larger_top)
    overlap_right = min(smaller_right, larger_right)
    overlap_bottom = min(smaller_bottom, larger_bottom)

    # Calculate the area of the overlapping rectangle
    overlap_width = max(0, overlap_right - overlap_left)
    overlap_height = max(0, overlap_bottom - overlap_top)
    overlap_area = overlap_width * overlap_height

    # Calculate the area of the smaller rectangle
    smaller_width = smaller_right - smaller_left
    smaller_height = smaller_bottom - smaller_top
    smaller_area = smaller_width * smaller_height

    # Calculate the overlap percentage
    overlap_percentage = (overlap_area / smaller_area) * 100 if smaller_area != 0 else 0

    return overlap_percentage




def match_audio_peak(base_audio_as_array, audio_as_array, factor=1):
    # Compute scale factor based on peak amplitude
    scale_factor = np.max(base_audio_as_array) / np.max(audio_as_array)

    scale_factor *= factor

    # Scale the target audio
    adjusted_audio_data = audio_as_array * scale_factor

    return adjusted_audio_data

