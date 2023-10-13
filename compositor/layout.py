import numpy as np
import math
from utilities import conf, conversion_frame_rate, conversion_audio_sample_rate
from itertools import groupby



def create_layout_for_composition(base_video, width, height):

    total_videos = 0
    for name, reaction in conf.get('reactions').items():
      total_videos += len(reaction.get('reactors'))

    include_base_video = conf['include_base_video']

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

    assign_hex_cells_to_videos(width, height, hex_grid, cell_size, bounds)


    return (base_video, cell_size)



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
      # Adjust the vertical spacing: every second row is moved up by .25 of the hexagon's height      
      dx = 2 * a
      dy = math.ceil(.75 * 2 * a) - 1 #np.sqrt(3) * a


      


      # Calculate the number of hexagons that can fit in the width and height
      nx = int(np.ceil(width / dx))
      ny = int(np.ceil(height / dy))

      # Generate a dense grid of hexagons around the center
      for j in range(-ny, ny + 1):
        for i in range(-nx, nx + 1):
          x = center[0] + dx * (i + 0.5 * (j % 2))
          y = center[1] + dy * j
          
          # Add the hexagon to the list if it's fully inside the width x height area
          if x - a >= 0 and y - a >= 0 and x + a <= width and y + a <= height and distance_from_region((x,y), outside_bounds) > 0:
            coords.append((x, y))

      # Reduce the size of the hexagons and try again if we don't have enough
      if len(coords) < min_cells:
        prev_a = a
        a = math.ceil(a * 0.99)
        if prev_a == a: 
          a -= 1

    # Sort hexagons by distance from the center
    coords.sort(key=lambda p: (p[0] - center[0])**2 + (p[1] - center[1])**2)

    assert len(coords) >= min_cells, f"Only {len(coords)} cells could be placed"
    
    cell_size = math.ceil(2 * a)

    print(f"\tLayout: len(hex_grid) = {len(coords)}  target_cells={min_cells} cell_size={cell_size}")

    return coords, cell_size  # return 2 * a (diameter) to match with the circle representation


def assign_hex_cells_to_videos(width, height, grid_cells, cell_size, base_video):
    # Assumes: grid_cells are sorted
    assignments = {}

    # Define a helper function to assign a video to a cell
    def assign_video(video, cells, assignments, featured=False, in_group=False):

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
            if featured:
                # Check if cell is adjacent to another featured video
                for assigned_key in assignments.keys():
                    if featured_by_key.get(assigned_key, False) and distance(cell, assignments[assigned_key]) <= 1.1*cell_size:
                        score -= 1

            if score > best_score: 
              if in_group:

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


        assignments[video['key']] = best_spot
        return best_adjacent


    featured_videos = []
    connected_videos = []
    other_videos = []

    featured_by_key = {}
    all_reactions = list(conf.get('reactions').items())
    all_reactions.sort(key=lambda x: x[1].get('priority', 50), reverse=True)
    for name, reaction in all_reactions:
      reactors = reaction.get('reactors')
      if reaction.get('featured'):
        featured_videos.extend(reactors)
        for featured in reactors:
          featured_by_key[featured['key']] = True
      elif len(reactors) > 1:
        connected_videos.append(reactors)
      else:
        print(name, reaction.get('priority', 50))
        other_videos.extend(reactors)
        
    print(f"\tLayout: featured={len(featured_videos)}  grouped={len(connected_videos)*2}   singular={len(other_videos)}")

    # ...Assign the featured reactors
    featured_videos.sort(key=lambda x: 0 if x['orientation'] == 'center' else 1, reverse=True)
    for video in featured_videos:
        assign_video(video, grid_cells, assignments, featured=True)

    # ...Assign the paired reactors
    for reactors in connected_videos:
      assert(len(reactors) < 3)
      p1,p2 = reactors

      p2_spot = assign_video(p1, grid_cells, assignments, in_group=True)
      if p2_spot is not None:
        assignments[p2['key']] = p2_spot
      else: # failed
        assign_video(p2, grid_cells, assignments)

    # ...Assign the remaining videos
    other_videos.sort(key=lambda x: 0 if x['orientation'] == 'center' else 1, reverse=True)
    for video in other_videos:
        assign_video(video, grid_cells, assignments)


    # Save the grid assigments to the reactors
    for name, reaction in conf.get('reactions').items():
      reactors = reaction.get('reactors')

      reactor_assignments = [assignments[reactor['key']] for reactor in reactors]

      if reaction.get('swap_grid_positions', False):
        reactor_assignments.reverse()

      for i, reactor in enumerate(reactors): 
        reactor['grid_assignment'] = reactor_assignments[i]




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


