from PIL import Image, ImageDraw, ImageChops
import numpy as np
from moviepy.editor import ImageClip, CompositeVideoClip
import math
from utilities import conf, conversion_frame_rate


def animateZoomPans(clip, show_viewport=False):
    zoom_pans = conf.get('zoom_pans', [])
    if len(zoom_pans) == 0:
        return clip

    duration = clip.duration
    fps = conversion_frame_rate

    # duration = 4 # clip.duration  # 13
    # fps = 8 # conversion_frame_rate  # 10

    if show_viewport:
        viewport_rectangle = create_viewport_rectangle(clip.size, 'red', (0, 0), 5, clip.duration)
        clip = CompositeVideoClip([clip, viewport_rectangle]).set_fps(fps).set_duration(duration)

    initializeZoomPanState(clip.size)

    events = createZoomPanEvents(zoom_pans, duration)

    animated_clip = clip.fl(lambda gf, t: zoom_and_pan(gf, t, events, clip.size))
    animated_clip = animated_clip.set_duration(duration).set_fps(fps)


    return animated_clip

def createZoomPanEvents(zoom_pans, duration):

    events = [ ZoomPanEvent(dfn) for dfn in zoom_pans ]

    for i,evt in enumerate(events): # fill in gaps where non-original state is to be maintained

        if not evt.ends_in_original_state():

            if i == len(events) - 1:
                ends = duration
            else:
                ends = events[i+1].start_time

            evt.end_time = ends

    # events = [
    #     ZoomPanEvent(0,  5, start_scale=1, end_scale=4, end_position='anchored'),        
    #     ZoomPanEvent(5,  10, end_scale=2, transition='ease_out'),        

    #     # ZoomPanEvent(0,  2, start_scale=2, end_scale=3, transition='ease_out'),        
    #     # ZoomPanEvent(2,  6, end_position=(1920/4, 1080/4), end_scale=3, transition='ease_out'),
    #     # ZoomPanEvent(6,  9, end_position=(1920, 1080), movement='arc'),
    #     # ZoomPanEvent(9, 12, end_position='original', end_scale=1, transition='ease_out') # Zoom out
    # ]

    return events

class ZoomPanEvent:

    def __init__(self, event):

        global zoomPanState

        self.start_time = event.get('start')
        self.duration = event.get('duration')
        self.end_time = event.get('end_time')

        self.start_scale = event.get('start_scale', zoomPanState["last_scale"]) 
        self.end_scale = event.get('end_scale', self.start_scale) 

        trans_func = event.get('transition', 'linear')

        self.transition = animation_transitions[trans_func]
        self.movement = event.get('movement', 'straight') 


        start_position = event.get('start_position', None) 
        end_position = event.get('end_position', None) 

        w,h = zoomPanState["size"]

        if start_position:
            if start_position == 'original':
                self.start_position = zoomPanState["original_position"] 
            else:
                self.start_position = [round(start_position[0] * zoomPanState['size'][0]), round(start_position[1] * zoomPanState['size'][1])]
        else:
            self.start_position = zoomPanState["last_position"]

        anchor = event.get('anchored', False)
        if end_position or anchor:
            if end_position=='original':
                self.end_position = zoomPanState["original_position"]
            elif anchor:
                x,y = zoomPanState["original_position"]
                s = self.end_scale

                if isinstance(anchor, bool):
                    base_video_placement = conf.get('base_video_placement')
                    anchor = {}
                    if 'bottom' in base_video_placement:
                        anchor['y'] = 'bottom'
                    elif 'top' in base_video_placement:
                        anchor['y'] = 'top'
                    if "left" in base_video_placement:
                        anchor['x'] = 'left'
                    elif 'right' in base_video_placement:
                        anchor['x'] = 'right'


                # print("BEFORE", [x, y], [w, h], s)
                y_anchor = anchor.get('y', None)
                x_anchor = anchor.get('x', None)

                if y_anchor == 'bottom':
                    y += h * (1 - 1 / s) / 2      #y = h * s / 2 # works
                elif y_anchor == 'top':
                    y -= h * (1 - 1 / s) / 2
                if x_anchor == 'left':
                    x -= w * (1 - 1 / s) / 2
                elif x_anchor == 'right':
                    x += w * (1 - 1 / s) / 2

                # print("AFTER", [x, y], [w, h], s)

                self.end_position = [ x, y ]
            else:
                self.end_position = [round(end_position[0] * zoomPanState['size'][0]), round(end_position[1] * zoomPanState['size'][1])]
        else: 
            self.end_position = self.start_position


        zoomPanState["last_position"] = self.end_position
        zoomPanState["last_scale"] = self.end_scale

        # adjust because our position actually starts at (0,0) not the centroid  
        self.start_position = [self.start_position[0] - zoomPanState["original_position"][0], self.start_position[1] - zoomPanState["original_position"][1]]
        self.end_position   = [self.end_position[0]   - zoomPanState["original_position"][0], self.end_position[1]   - zoomPanState["original_position"][1]]

        # print("FINAL:", self.end_position)
        self.arc_center = event.get('arc_center', ((self.start_position[0] + self.end_position[0]) / 2, (self.start_position[1] + self.end_position[1]) / 2))

    def ends_in_original_state(self):
        return self.end_scale == 1 and self.end_position[0] == zoomPanState['original_position'][0] and self.end_position[1] == zoomPanState['original_position'][1]


# Generalized zoom and pan function
def zoom_and_pan(get_frame, t, events, clip_size):


    for event in events:
        if event.start_time <= t < event.end_time:

            # Calculate normalized time (0 to 1) within the event
            normalized_time = (t - event.start_time) / (event.duration)

            # Apply the transition function to the normalized time
            interp = event.transition(normalized_time)
            
            # Calculate current scale and position
            scale    =  np.interp(interp, [0, 1], [event.start_scale,       event.end_scale])
            
            # Calculate position based on movement type
            if event.movement == 'arc':
                position = calculate_arc_position(event.start_position, event.end_position, event.arc_center, interp)
            else:
                position = (np.interp(interp, [0, 1], [event.start_position[0], event.end_position[0]]),
                            np.interp(interp, [0, 1], [event.start_position[1], event.end_position[1]]))

            # x   y
            # 0.5 0.65
            # 1.5 1.25
            # 2   1.5
            # 2.5 1.7
            # 3   1.75
            # 5   1.85
            # 7.5 2.03
            # 10  2.15
            # 15  2.29
            scale_func = lambda t: t ** (.4688 * np.log(event.end_scale) + 1.1097)  

            scale    =  np.interp(scale_func(normalized_time), [0, 1], [event.start_scale,       event.end_scale])

            old_position = position
            position = [position[0] * scale, position[1] * scale]


            # Apply zoom and pan
            frame = get_frame(t)
            pil_frame = Image.fromarray(frame)  # Convert to PIL Image


            # Maintain aspect ratio while rounding dimensions
            aspect_ratio = clip_size[0] / clip_size[1]
            new_width = round(scale * clip_size[0])
            new_height = round(new_width / aspect_ratio)

            new_size = (new_width, new_height)

            # print('position:', round(position[1]), 'old_position', round(old_position[1]), "scale=", scale, "area=", new_size[0] * new_size[1] / (clip_size[0] / clip_size[1]))
            resized_frame = pil_frame.resize(new_size, Image.LANCZOS)  # Resize the frame


            left = round(position[0] + (new_size[0] - clip_size[0]) / 2)
            top  = round(position[1] + (new_size[1] - clip_size[1]) / 2)

            left = max(0, left)
            top = max(0, top)

            right = left + clip_size[0]
            bottom = top + clip_size[1]

            right = min(new_size[0], right)
            bottom = min(new_size[1], bottom)


            left = right - clip_size[0]
            top = bottom - clip_size[1]


            # left = max(0, left)
            # top = max(0, top)
            # right = min(new_size[0], right)
            # bottom = min(new_size[1], bottom)

            # left = round(left)
            # top = round(top)
            # right = round(right)
            # bottom = round(bottom)

            assert( right - left == clip_size[0] and bottom - top == clip_size[0] and left >= 0 and top >= 0, f"BAD CROP! [{left},{top}] => [{right}, {bottom}]")

            cropped_frame = resized_frame.crop( ( left, top, right, bottom ) )
            
            return np.array(cropped_frame)  # Convert back to numpy array

    return get_frame(t)


animation_transitions = {
    "linear":      lambda t: t,
    "ease_in":     lambda t: t**2,
    "ease_out":    lambda t: 1 - (1 - t)**2,
    "ease_in_out": lambda t: 4*t**3 if t < 0.5 else 1 - (-2*t + 2)**3 / 2
}

zoomPanState = None
def initializeZoomPanState(output_size):
    global zoomPanState
    zoomPanState = {
        "original_position": [output_size[0] / 2, output_size[1] / 2],
        "last_position": [output_size[0] / 2, output_size[1] / 2],
        "last_scale": 1,
        "size": output_size
    }



def calculate_arc_position(start, end, center, interp):
    """Calculate position along an arc."""
    start_angle = math.atan2(start[1] - center[1], start[0] - center[0])
    end_angle = math.atan2(end[1] - center[1], end[0] - center[0])
    angle = np.interp(interp, [0, 1], [start_angle, end_angle])

    radius = math.sqrt((start[0] - center[0])**2 + (start[1] - center[1])**2)
    x = center[0] + radius * math.cos(angle)
    y = center[1] + radius * math.sin(angle)
    return x, y





def create_viewport_rectangle(size, color, position, border_width, duration, grid_spacing=100):
    """Create a rectangle around the viewport."""
    rect = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(rect)

    # Draw the border rectangle
    draw.rectangle((0, 0, size[0], size[1]), outline=color, width=border_width)

    # Draw the grid
    for x in range(grid_spacing, size[0], grid_spacing):
        draw.line([(x, 0), (x, size[1])], fill=color, width=1)
    for y in range(grid_spacing, size[1], grid_spacing):
        draw.line([(0, y), (size[0], y)], fill=color, width=1)

    # Convert PIL Image to numpy array
    rect_np = np.array(rect)

    return ImageClip(rect_np).set_position(position).set_duration(duration)

