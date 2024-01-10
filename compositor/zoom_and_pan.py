from PIL import Image, ImageDraw, ImageChops
import numpy as np
from moviepy.editor import ImageClip
import math




def animateClip(clip, events):

    animated_clip = clip.fl(lambda gf, t: zoom_and_pan(gf, t, events, clip.size))
    animated_clip = animated_clip.set_duration(clip.duration).set_fps(clip.fps)

    return animated_clip



class ZoomPanEvent:

    def __init__(self, start_time, end_time, start_scale=None, end_scale=None, start_position=None, end_position=None, transition='linear', movement='straight', arc_center=None):
        
        global zoomPanState

        self.start_time = start_time
        self.end_time = end_time

        if start_position:
            self.start_position = zoomPanState["original_position"] if start_position == 'original' else start_position
        else:
            self.start_position = zoomPanState["last_position"]

        if end_position:
            self.end_position = zoomPanState["original_position"] if end_position=='original' else end_position
        else: 
            self.end_position = self.start_position

        self.start_scale = start_scale if start_scale is not None else zoomPanState["last_scale"]
        self.end_scale   = end_scale   if end_scale   is not None else self.start_scale

        zoomPanState["last_position"] = self.end_position
        zoomPanState["last_scale"] = self.end_scale

        # adjust because our position actually starts at (0,0) not the centroid  
        self.start_position = [self.start_position[0] - zoomPanState["original_position"][0], self.start_position[1] - zoomPanState["original_position"][1]]
        self.end_position   = [self.end_position[0]   - zoomPanState["original_position"][0], self.end_position[1]   - zoomPanState["original_position"][1]]

        self.transition = animation_transitions[transition]
        self.movement = movement
        self.arc_center = arc_center if arc_center else ((self.start_position[0] + self.end_position[0]) / 2, (self.start_position[1] + self.end_position[1]) / 2)



# Generalized zoom and pan function
def zoom_and_pan(get_frame, t, events, clip_size):

    for event in events:
        if event.start_time <= t < event.end_time:

            # Calculate normalized time (0 to 1) within the event
            normalized_time = (t - event.start_time) / (event.end_time - event.start_time)

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


            # Apply zoom and pan
            frame = get_frame(t)
            pil_frame = Image.fromarray(frame)  # Convert to PIL Image
            new_size = (int(scale * clip_size[0]), 
                        int(scale * clip_size[1]))
            resized_frame = pil_frame.resize(new_size, Image.LANCZOS)  # Resize the frame

            scale_factor = (scale - 1) * (position[0] + clip_size[0] / 2) / clip_size[0]

            left   =  position[0] + clip_size[0] * scale_factor
            top    =  position[1] + clip_size[1] * scale_factor

            right  = left + clip_size[0]
            bottom =  top + clip_size[1]

            cropped_frame = resized_frame.crop( ( left, top, right, bottom ) )
            
            return np.array(cropped_frame)  # Convert back to numpy array

    return get_frame(t)


animation_transitions = {
    "linear":      lambda t: t,
    "ease_in":     lambda t: t**2,
    "ease_out":    lambda t: 1 - (1 - t)**2,
    "ease_in_out": lambda t: t**2 if t < 0.5 else 1 - (1 - t)**2
}

zoomPanState = None
def initializeZoomPanState(output_size):
    global zoomPanState
    zoomPanState = {
        "original_position": [output_size[0] / 2, output_size[1] / 2],
        "last_position": [output_size[0] / 2, output_size[1] / 2],
        "last_scale": 1
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





def create_viewport_rectangle(size, color, position, border_width, duration):
    """Create a rectangle around the viewport."""
    rect = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(rect)
    draw.rectangle((0, 0, size[0], size[1]), outline=color, width=border_width)
    
    # Convert PIL Image to numpy array
    rect_np = np.array(rect)
    
    return ImageClip(rect_np).set_position(position).set_duration(duration)

# viewport_rectangle = create_viewport_rectangle(output_size, 'red', (0, 0), 5, final_clip.duration)
# Composite the markers with the final clip
# final_clip = CompositeVideoClip([final_clip, viewport_rectangle]).set_fps(10).set_duration(13)
