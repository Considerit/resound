import numpy as np
import math
from utilities import conversion_frame_rate, conversion_audio_sample_rate
from itertools import groupby




def create_layout_for_composition(song, base_video, width, height, reactions):
    total_videos = len(reactions)


    include_base_video = song['include_base_video']

    if include_base_video:
      base_size = int(width * .55)

      base_video = base_video.resize(base_size / base_video.w)
      base_width, base_height = base_video.size

      if base_video.audio.fps != conversion_audio_sample_rate:
          base_video = base_video.set_audio(base_video.audio.set_fps(conversion_audio_sample_rate))

      x,y = (base_width / 2, base_height / 2)

      base_video = base_video.set_position((x - base_width / 2, y - base_height / 2))

      # upper left point of rectangle, lower right point of rectangle
      bounds = (x - base_width / 2, y - base_height / 2, x + base_width / 2, y + base_height / 2)
      center = None
    else: 
      bounds = [0,0,width,0]
      center = [width / 2, height / 2]

    hex_grid, cell_size = generate_hexagonal_grid(width, height, total_videos, bounds, center)
    hex_grid = sorted(hex_grid, key=lambda cell: distance_from_region( cell, bounds ), reverse = False)
    cell_size = math.floor(cell_size)
    print(f"len(hex_grid) = {len(hex_grid)}  total_videos={total_videos} cell_size={cell_size}")

    # print(f"total_videos = {total_videos}, cell_size={cell_size}, cells_returned={len(hex_grid)}, grid={hex_grid}")

    positions = assign_hex_cells_to_videos(width, height, hex_grid, cell_size, bounds, reactions)


    return (base_video, positions, cell_size)



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


