import numpy as np
import math
from utilities import conf, conversion_frame_rate, conversion_audio_sample_rate
from itertools import groupby



def create_layout_for_composition(base_video, width, height):
    base_video_proportion = conf.get('base_video_proportion', .45)
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

      x,y = (base_width / 2, base_height / 2)          # left / top
      # x,y = (base_width / 2, height - base_height / 2) # left / bottom
      x,y = (width / 2, height - base_height / 2)      # center / bottom

      print(f"SETTING POSITION TO {(x - base_width / 2, y - base_height / 2)}")
      base_video_position = (x - base_width / 2, y - base_height / 2)      

      # upper left point of rectangle, lower right point of rectangle
      bounds = (x - base_width / 2, y - base_height / 2, 
                x + base_width / 2, y + base_height / 2)

      center = None
      value_relative_to = [x, height - base_height]
    else: 
      bounds = [0,0,width,0]
      center = [width / 2, height / 2]
      value_relative_to = [width / 2, height / 2]
      base_video_position = None
      base_width = base_height = 0 
    hex_grid, cell_size = generate_hexagonal_grid(width, height, total_videos, bounds, center)

    # hex_grid = sorted(hex_grid, key=lambda cell: distance_from_region( cell, bounds ), reverse = False)
    hex_grid = sorted(hex_grid, key=lambda cell: distance( cell, value_relative_to ), reverse = False)

    assign_seats_to_reactors(  seats=hex_grid, 
                               grid_centroid=value_relative_to,
                               seat_size=cell_size,
                               base_video_width=base_width, 
                               base_video_height=base_height,
                               grid_size=(width, height))

    return (base_video, cell_size, base_video_position)



# I want to create a hexagonal grid that fits a space of size width x height. There needs to be at 
# least n cells, each of which is fully visible. Cells should have equal height and width. 
# Cell size should be maximized given the constraints. Please write a function generate_hexagonal_grid 
# that is given a width and a height and a minimum number of cells and which returns a list of cell 
# centroids for this hexagonal grid, and the cell size. The centroids should be sorted by 
# distance to the center of the grid, with the first ones the closest. 

def generate_hexagonal_grid(width, height, min_cells, outside_bounds=None, center=None):

    min_cells += 3

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
          if x - a >= 0 and y - a >= 0 and x + a <= width and y + a <= height and distance_from_region((x,y), outside_bounds) > a - (outside_bounds[3] - outside_bounds[1]) / 10:
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



seat_preferences = {}
def get_seat_preferences(reaction, grid_centroid, all_seats, base_video_width, base_video_height):
    global seat_preferences

    key = f"{conf.get('song_key')}-{reaction.get('channel')}"
    if key not in seat_preferences:
        orientation = reaction.get('face_orientation')
        if orientation is None:
            print("ORIENTATION IS NONE!", reaction.get('channel'))
            orientation=("left", "down")
        horizontal_orientation, vertical_orientation = orientation

        my_value = list(grid_centroid)

        if horizontal_orientation == 'left':
          my_value[0] += base_video_width / 2
        elif horizontal_orientation == 'right':
          my_value[0] -= base_video_width / 2

        if horizontal_orientation != 'center':
            if vertical_orientation == 'middle':
              my_value[1] += base_video_height / 2
            elif vertical_orientation == 'up':
              my_value[1] += base_video_height

        seat_preferences[key] = my_value

    return seat_preferences[key]


def position_score(orientation, seat, grid_target, base_size):

    seat_x, seat_y = seat
    grid_center, grid_middle = grid_target
    (base_video_width, base_video_height) = base_size

    horizontal_orientation, vertical_orientation = orientation

    vscore = hscore = 1

    if (horizontal_orientation == 'center'):
      if grid_center - base_video_width / 4 <= seat_x and seat_x <= grid_center + base_video_width / 4:
        hscore = 1.7
      elif grid_center - base_video_width / 2 <= seat_x and seat_x <= grid_center + base_video_width / 2:
        hscore = 1.2
      else: 
        hscore = .9

    elif (horizontal_orientation == 'right'):
      if seat_x <= grid_center * .25 or seat_x <= grid_center:
        hscore = 1.7
      elif seat_x <= grid_center * .5:
        hscore = 1.2
      elif seat_x <= grid_center * .65:
        hscore = .9
      elif seat_x <= grid_center * .75:
        hscore = .7
      else: 
        hscore = .5

    else:  # left
      if seat_x >= grid_center * .75 or seat_x >= grid_center:
        hscore = 1.7
      elif seat_x >= grid_center * .5:
        hscore = 1.2
      elif seat_x >= grid_center * .35:
        hscore = .9
      elif seat_x >= grid_center * .25:
        hscore = .7
      else: 
        hscore = .5


    if (vertical_orientation == 'middle'):
      if grid_middle * .25 <= seat_y and seat_y <= grid_middle * .75:
        vscore = 1.1

    elif (vertical_orientation == 'down'):
      if seat_y <= grid_middle * .5:
        vscore = 1.1

    else:  # up
      if seat_y >= grid_middle * .5:
        vscore = 1.1

    return hscore, vscore


def seat_score(reaction, seat, grid_centroid, my_value, max_distance, base_size):

    closeness_score = 1 + (max_distance - distance(seat, my_value)) / max_distance

    horizontal_pos_score, vertical_pos_score = position_score(reaction.get('face_orientation'), seat, grid_centroid, base_size)   

    return horizontal_pos_score * vertical_pos_score * closeness_score



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
    reaction_preferences = {}
    total_reactors = 0
    for channel, reaction in conf.get('reactions').items():
        reaction_preferences[channel] = get_seat_preferences(reaction, grid_centroid, seats, base_video_width, base_video_height)
        total_reactors += len(reaction.get('reactors'))

    ########################
    # initialize seat scores
    seat_scores = {}
    max_distance = math.sqrt(  grid_size[0] ** 2 + grid_size[1] ** 2  )

    for seat, same_row, adjacents in seats_with_adjacents:
        seat_key = str(seat)
        seat_scores[seat_key] = {}
        for channel, reaction in conf.get('reactions').items():
            my_value = reaction_preferences[channel]
            seat_scores[seat_key][channel] = seat_score(reaction, seat, grid_centroid, my_value, max_distance, (base_video_width, base_video_height))


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

        assert( len(reactors) == len(chosen_seats), f"{len(reactors)}  {len(chosen_seats)}" )

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

            print(f"\t\tASSIGNED {chosen_seat} to {chosen_channel} / {i}. Target position={reaction_preferences[chosen_channel]}")

        # remove choice from consideration
        for chosen_seat in chosen_seats:
            del seat_scores[str(chosen_seat)]
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
                if score > highest_score or (score == highest_score and priority > highest_priority):

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

