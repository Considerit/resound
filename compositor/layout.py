import numpy as np
import math
from utilities import conf, conversion_frame_rate, conversion_audio_sample_rate
from itertools import groupby



def create_layout_for_composition(base_video, width, height, shape="hexagon"):
    base_video_proportion = conf.get('base_video_proportion')
    total_videos = 0
    for name, reaction in conf.get('reactions').items():
      total_videos += len(reaction.get('reactors', []))

    include_base_video = conf['include_base_video']

    if include_base_video:
      base_size = int(width * base_video_proportion)

      base_video = base_video.resize(base_size / base_video.w)
      base_width, base_height = base_video.size

      if base_video.audio.fps != conversion_audio_sample_rate:
          base_video = base_video.set_audio(base_video.audio.set_fps(conversion_audio_sample_rate))



      base_video_placement = conf.get('base_video_placement')
      if base_video_placement == "left / top":
        x,y = (base_width / 2, base_height / 2)          # left / top
      elif base_video_placement == "left / bottom":
        x,y = (base_width / 2, height - base_height / 2) # left / bottom
      elif base_video_placement == 'center / bottom':
        x,y = (width / 2, height - base_height / 2)      # center / bottom

      print(f"SETTING POSITION TO {(x - base_width / 2, y - base_height / 2)}")
      base_video_centroid = (x - base_width / 2, y - base_height / 2)      

      # upper left point of rectangle, lower right point of rectangle
      bounds = (x - base_width / 2, y - base_height / 2, 
                x + base_width / 2, y + base_height / 2)

      center = None
      value_relative_to = [x, height - base_height]
    else: 
      bounds = [0,0,width,0]
      center = [width / 2, height / 2]
      value_relative_to = [width / 2, height / 2]
      base_video_centroid = None
      base_width = base_height = 0 


    hex_grid, cell_size = generate_grid(width, height, total_videos, bounds, center, shape=shape)

    # hex_grid = sorted(hex_grid, key=lambda cell: distance_from_region( cell, bounds ), reverse = False)
    hex_grid = sorted(hex_grid, key=lambda cell: distance( cell, value_relative_to ), reverse = False)

    assign_seats_to_reactors(  seats=hex_grid, 
                               grid_centroid=value_relative_to,
                               seat_size=cell_size,
                               base_video_width=base_width, 
                               base_video_height=base_height,
                               grid_size=(width, height))

    return (base_video, cell_size, base_video_centroid)




# I want to create a hexagonal grid that fits a space of size width x height. There needs to be at 
# least n cells, each of which is fully visible. Cells should have equal height and width. 
# Cell size should be maximized given the constraints. Please write a function generate_hexagonal_grid 
# that is given a width and a height and a minimum number of cells and which returns a list of cell 
# centroids for this hexagonal grid, and the cell size. The centroids should be sorted by 
# distance to the center of the grid, with the first ones the closest. 

def generate_grid(width, height, min_cells, outside_bounds=None, center=None, shape="hexagon"):

    # min_cells += 3
    min_cells = max(1, min_cells)

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
      if (shape == 'hexagon' or shape == 'circle'):
        dy = math.ceil(.75 * 2 * a) - 1 #np.sqrt(3) * a
      elif shape == 'diamond':
        dy = math.ceil(.5 * 2 * a) - 1 #np.sqrt(3) * a

      # Calculate the number of hexagons that can fit in the width and height
      nx = int(np.ceil(width / dx))
      ny = int(np.ceil(height / dy))

      # Generate a dense grid of hexagons around the center
      for j in range(-ny, ny + 1):
        for i in range(-nx, nx + 1):
          x = center[0] + dx * (i + 0.5 * (j % 2))
          y = center[1] + dy * j
          
          # Add the hexagon to the list if it's fully inside the width x height area
          fully_inside_grid = x - a >= 0 and y - a >= 0 and x + a <= width and y + a <= height
          not_covering_base = distance_from_region((x,y), outside_bounds) > a - (outside_bounds[3] - outside_bounds[1]) / 10
          if fully_inside_grid and not_covering_base:
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


    # Make sure that available grid positions are horizontally centered. This is particularly 
    # desirable for compositions with fewer reactions and with a video that is centered horizontally.
    # I'm not sure if it is desirable when the video isn't centered horizontally.
    min_x = 99999999
    max_x = 0
    for x,y in coords:
      if x - cell_size / 2 < min_x:
        min_x = x - cell_size / 2
      if x + cell_size / 2 > max_x:
        max_x = x + cell_size / 2
    right_padding = width - max_x
    left_padding = min_x
    x_adj = (right_padding - left_padding) / 2
    coords = [ (x+x_adj, y) for x,y in coords ]


    print(f"\tLayout: len(hex_grid) = {len(coords)}  target_cells={min_cells} cell_size={cell_size}")





    return coords, cell_size  # return 2 * a (diameter) to match with the circle representation











seat_preferences = {}
def get_seat_preferences(reaction, grid_centroid, grid_size, all_seats, base_video_width, base_video_height):
    global seat_preferences

    base_video_placement = conf.get('base_video_placement')
    if base_video_placement != "center / bottom":
      raise(f"Please update this function to support {base_video_placement}")

    key = f"{conf.get('song_key')}-{reaction.get('channel')}"
    if key not in seat_preferences:
        orientation = reaction.get('face_orientation')
        if orientation is None:
            print("ORIENTATION IS NONE!", reaction.get('channel'))
            orientation=("left", "down")
        horizontal_orientation, vertical_orientation = orientation


        vw = base_video_width
        vh = base_video_height
        w,h = grid_size

        ideal_seat = list(grid_centroid)

        if horizontal_orientation == 'left':
          ideal_seat[0] += vw / 2
        elif horizontal_orientation == 'right':
          ideal_seat[0] -= vw / 2

        if horizontal_orientation != 'center':
            if vertical_orientation == 'middle':
              ideal_seat[1] += vh / 2
            elif vertical_orientation == 'up':
              ideal_seat[1] += vh
        else:
            ideal_seat[1] = h - vh   # This is only correct for video placement = down center
            if vertical_orientation == 'middle':
              ideal_seat[1] -= 1
            elif vertical_orientation == 'down':
              ideal_seat[1] -= 2 

        # This is only correct for video placement = down center
        if horizontal_orientation == 'center':
          left = w / 2 - vw / 2
          right = w / 2 + vw / 2
          top = 0
          bottom = h - vh
        else:
          if horizontal_orientation == 'right':
            left = 0 
            right = w / 2 - vw / 2
          elif horizontal_orientation == 'left':
            left = w / 2 + vw / 2
            right = w

          if vertical_orientation == 'middle':
            top = h - vh
            bottom = h - vh / 4
          elif vertical_orientation == 'up':
            top = h - vh / 2
            bottom = h
          elif vertical_orientation == 'down':
            top = 0
            bottom = h - vh


        my_section = (left, top, right, bottom)

        seat_preferences[key] = [ideal_seat, my_section]

    return seat_preferences[key]



def seat_score(reaction, seat, grid_centroid, max_distance, ideal_seat, section_preference, base_size):

    closeness_score = 1 + (max_distance - distance(seat, ideal_seat)) / max_distance

    section_score = (max_distance - distance_from_region(seat, section_preference)) / max_distance

    return closeness_score * section_score * section_score



def construct_seating(seats, seat_size):
    seats_with_adjacents = []

    for seat in seats:

        surrounding = []

        for t in seats:
            if t == seat:
                continue

            dist = distance(seat, t)
            if dist < 1.25 * seat_size: 
                surrounding.append(t)

        same_row = [s for s in surrounding if seat[1] == s[1]]
        seats_with_adjacents.append( (seat, same_row, surrounding) )


    return seats_with_adjacents


def assign_seats_to_reactors(seats, grid_centroid, seat_size, base_video_width, base_video_height, grid_size):

    print(f"\tAssigning seats to reactors. Grid Centroid = {grid_centroid}")
    seats_with_adjacents = construct_seating(seats, seat_size)

    ###############################
    # get reaction seat preferences
    seat_preferences = {}
    total_reactors = 0
    for channel, reaction in conf.get('reactions').items():
        seat_preferences[channel] = get_seat_preferences(reaction, grid_centroid, grid_size, seats, base_video_width, base_video_height)
        total_reactors += len(reaction.get('reactors'))

    ########################
    # initialize seat scores
    seat_scores = {}
    max_distance = math.sqrt(  grid_size[0] ** 2 + grid_size[1] ** 2  )

    for seat, same_row, adjacents in seats_with_adjacents:
        seat_key = str(seat)
        seat_scores[seat_key] = {}
        for channel, reaction in conf.get('reactions').items():
            ideal_seat, my_section = seat_preferences[channel]
            seat_scores[seat_key][channel] = seat_score(reaction, seat, grid_centroid, max_distance, ideal_seat=ideal_seat, section_preference=my_section, base_size=(base_video_width, base_video_height))


    ##############
    # assign seats
    seating_by_reactor = {} # maps from reactor => seat
    seating_by_seat = {}    # maps from seat key => (reaction, reactor)


    def assign_seats(chosen_seats, chosen_channel):
        reaction = conf.get('reactions')[chosen_channel]
        reactors = reaction['reactors']

        if len(reactors) > len(chosen_seats):
            print("AGG! HAVE TO SPLIT GROUP TO DIFFERENT SEATS!")
            while( len(chosen_seats) < len(reactors) ):
                for seat in seats:
                    seat_key = str(seat)
                    if seat_key not in seating_by_seat and seat_key not in [str(t) for t in chosen_seats]:
                        chosen_seats.append(seat)
                        if len(chosen_seats) >= len(reactors):
                          break

        assert len(reactors) == len(chosen_seats), f"{len(reactors)}  {len(chosen_seats)}"

        # align seats and reactors by their x-position
        chosen_seats.sort( key=lambda seat: seat[0] )
        reactors.sort( key=lambda r: r['x'])

        if reaction.get('swap_grid_positions', False):
            chosen_seats.reverse()

        # make assignments
        for i, reactor in enumerate(reactors):
            chosen_seat = chosen_seats[i]
            seating_by_reactor[reactor['key']] = chosen_seat  
            seating_by_seat[str(chosen_seat)] = (reaction, reactor)
            reactor['grid_assignment'] = chosen_seat

            print(f"\t\tASSIGNED {chosen_seat} to {chosen_channel} / {i}. Priority={reaction.get('priority')} Target position={seat_preferences[chosen_channel]}")

        # remove choice from consideration
        removed = {}

        for chosen_seat in chosen_seats:
            del seat_scores[str(chosen_seat)]
            removed[str(chosen_seat)] = 1

        # remove reactor from remaining seat considerations
        for seat in list(seat_scores.keys()):
            del seat_scores[str(seat)][chosen_channel]

    i = 0
    while( len(seating_by_reactor.keys()) < total_reactors ):
        i += 1

        # find next assignment
        chosen_channel = None
        chosen_seats = []
        highest_score = 0
        highest_priority = 0


        if len(seat_scores.keys()) == 0:
          raise Exception(f"We've seated {len(seating_by_reactor.keys())} of {total_reactors} reactors, but don't seem to have any more available.")

        for seat, same_row, adjacents in seats_with_adjacents:

            seat_key = str(seat)
            if seat_key not in seat_scores:
                continue

            for channel, score in seat_scores[seat_key].items():
                reaction = conf.get('reactions')[channel]

                # Make sure groups are seated next to each other
                num_reactors = len(reaction.get('reactors'))                
                if num_reactors > 1:
                    if num_reactors <= 3:  # place side-by-side
                        surrounding_seats = same_row
                    else:                  # place in group
                        surrounding_seats = adjacents

                    group_seats = [seat]

                    for adjacent_seat in surrounding_seats:
                        if str(adjacent_seat) not in seating_by_seat:
                            group_seats.append(adjacent_seat)
                            score += seat_scores[str(adjacent_seat)][channel]
                            if len(group_seats) == num_reactors:
                                break

                    if len(group_seats) < num_reactors:
                        score *= .5 * num_reactors / len(group_seats)

                # Make sure featured reactions are spaced out
                if reaction.get('featured', False):
                    # Check if cell is adjacent to another featured video
                    for surrounding_seat in adjacents:
                        if str(surrounding_seat) in seating_by_seat:
                            score *= .1

                priority = reaction.get('priority')
                score *= (priority + 50) / 100

                if score > highest_score or ( score == highest_score and priority > highest_priority):

                    chosen_channel = channel
                    if num_reactors > 1:
                        chosen_seats = group_seats
                    else:
                        chosen_seats = [seat]
                    highest_score = score
                    highest_priority = priority

        if chosen_channel:
          assign_seats(chosen_seats, chosen_channel)

    return seating_by_reactor





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

    # Point is inside the rectangle
    if x1 <= x <= x2 and y1 <= y <= y2:
        return 0

    # Point is to the left of the rectangle
    elif x < x1:
        if y < y1:  # Top left corner
            return math.hypot(x1 - x, y1 - y)
        elif y > y2:  # Bottom left corner
            return math.hypot(x1 - x, y - y2)
        else:  # Directly to the left
            return x1 - x

    # Point is to the right of the rectangle
    elif x > x2:
        if y < y1:  # Top right corner
            return math.hypot(x - x2, y1 - y)
        elif y > y2:  # Bottom right corner
            return math.hypot(x - x2, y - y2)
        else:  # Directly to the right
            return x - x2

    # Point is directly above or below the rectangle
    else:
        if y < y1:  # Directly above
            return y1 - y
        else:  # Directly below
            return y - y2


def distance_to_corner(point, bounds):
    """
    Calculates the Euclidean distance from a point to a rectangle.
    Parameters:
    - point: Tuple (x, y) for the point.
    - bounds: Tuple (x1, y1, x2, y2) for the upper left and lower right points of the rectangle.
    Returns: distance as a float.
    """
    x, y = point
    x1, y1, x2, y2 = bounds

    if x1 <= x <= x2 and y1 <= y <= y2: 
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








import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_perturbed_hexagon(center, size, perturbations):
    """ Create a hexagon and perturb its vertices """
    points = []
    for i, angle in enumerate(range(0, 360, 60)):
        x = center[0] + (size + perturbations[i]) * np.cos(np.radians(angle))
        y = center[1] + (size + perturbations[i]) * np.sin(np.radians(angle))
        points.append((x, y))
    return points

def generate_hex_grid(rows, cols, hex_size):
    """ Generate a regular hexagonal grid """
    grid = []
    for row in range(rows):
        for col in range(cols):
            x = col * 1.5 * hex_size
            y = row * np.sqrt(3) * hex_size + (col % 2) * np.sqrt(3)/2 * hex_size
            grid.append((x, y, col, row))  # Include col and row for perturbation reference
    return grid

def generate_irregular_hex_layout(ax, aspect_ratio=16/9, hex_size=0.1, rows=10, cols=10):
    """ Generate an irregular hexagonal layout """
    np.random.seed(0)  # For reproducible results
    grid = generate_hex_grid(rows, cols, hex_size)
    perturbations = {}

    for x, y, col, row in grid:
        # Create unique perturbations for each vertex
        if (col, row) not in perturbations:
            perturbations[(col, row)] = [np.random.rand() * hex_size * 0.3 - hex_size * 0.15 for _ in range(6)]

        # Ensure shared edges have the same perturbation
        hex_perturbations = perturbations[(col, row)]
        # ... (same as previous script for shared edge perturbations)

        # Plot the perturbed hexagon
        plot_hexagon(ax, (x, y), hex_size, 'black', hex_perturbations)

def plot_hexagon(ax, center, size, edge_color, perturbations):
    """ Plot a single hexagon on the given axes """
    hexagon = patches.Polygon(create_perturbed_hexagon(center, size, perturbations), closed=True, fill=False, edgecolor=edge_color)
    ax.add_patch(hexagon)


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

def compute_centroids(vor):
    """Compute centroids of Voronoi cells."""
    centroids = []
    for region in vor.regions:
        if not -1 in region and len(region) > 0:
            polygon = [vor.vertices[i] for i in region]
            polygon = np.array(polygon)
            centroid = np.mean(polygon, axis=0)
            centroids.append(centroid)
    return np.array(centroids)

def generate_voronoi_diagram(width, height, num_points, num_large_cells, iterations=5):
    # Initial random points
    points = np.random.rand(num_points, 2) * [width, height]
    large_cell_points = np.random.rand(num_large_cells, 2) * [width * 0.2, height * 0.2] + [width * 0.4, height * 0.4]
    all_points = np.vstack((points, large_cell_points))

    for _ in range(iterations):
        vor = Voronoi(all_points)
        centroids = compute_centroids(vor)
        all_points = centroids  # Update points to centroids for next iteration

    return vor




if __name__ == "__main__":


  ### Experiment 1
  # # Set up plot
  # fig, ax = plt.subplots()
  # ax.set_aspect('equal')
  # ax.set_xlim(0, 16/9)
  # ax.set_ylim(0, 1)
  # ax.axis('off')

  # # Generate and plot layout
  # generate_irregular_hex_layout(ax)

  # plt.show()



  # ### Experiment 2
  # # Parameters
  # width, height = 16, 9
  # num_points = 95
  # num_large_cells = 5

  # # Generate Voronoi diagram with Lloyd's Algorithm
  # vor = generate_voronoi_diagram(width, height, num_points, num_large_cells)

  # # Plot
  # fig, ax = plt.subplots()
  # voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black', line_width=2)
  # ax.set_xlim([0, width])
  # ax.set_ylim([0, height])
  # ax.set_aspect('equal')
  # plt.show()



  # Experiment 3

  # Example circle data (centroids and radii)
  circle_centroids = np.array([[0.2, 0.5], [0.4, 0.7], [0.6, 0.3], [0.8, 0.5]])  # Replace with your data
  circle_radii = np.array([0.05, 0.07, 0.06, 0.08])  # Replace with your data

  # Create Voronoi diagram from circle centroids
  vor = Voronoi(circle_centroids)

  # Plotting
  fig, ax = plt.subplots()
  voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black', line_width=2)

  # Plot original circles for comparison
  for centroid, radius in zip(circle_centroids, circle_radii):
      circle = plt.Circle(centroid, radius, edgecolor='red', facecolor='none')
      ax.add_patch(circle)

  ax.set_xlim([0, 1])
  ax.set_ylim([0, 1])
  ax.set_aspect('equal')
  plt.show()




