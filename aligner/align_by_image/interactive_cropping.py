def get_ground_truth_bounding_box(reaction):
    reaction_video_path = reaction.get("video_path")
    hash_output_dir = os.path.join(conf.get("temp_directory"), "image_hashes")
    channel = reaction.get("channel")

    coordinates_path = os.path.join(hash_output_dir, f"coordinates_for_music_vid.json")
    if os.path.exists(coordinates_path):
        coordinates = read_object_from_file(coordinates_path)
    else:
        coordinates = {}

    if channel not in coordinates:
        crop_coordinates = select_crop_region(reaction_video_path)
        if crop_coordinates is None:
            print("crop coordinates failed")
            return
        coordinates[channel] = crop_coordinates
        save_object_to_file(coordinates_path, coordinates)

    return coordinates[channel]


###############################################################
# Interactive definition of embedded music video position in reaction video.

import cv2

# Global variables for drawing the bounding box
drawing = False
start_point = None
end_point = None
crop_coords = None


def draw_rectangle(event, x, y, flags, param):
    """
    Mouse callback function for drawing a rectangle.
    This function is used to handle mouse events for selecting the crop region.
    """
    global start_point, end_point, drawing, crop_coords

    # On left mouse button down, start drawing the rectangle
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)  # Ensure this is always the top-left corner
        end_point = (x, y)

    # While the mouse moves and the button is held down, update the rectangle
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)

    # On left mouse button release, finalize the rectangle
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        # Ensure top_left is always the top-left corner
        top_left_x = min(start_point[0], end_point[0])
        top_left_y = min(start_point[1], end_point[1])
        width = abs(start_point[0] - end_point[0])
        height = abs(start_point[1] - end_point[1])
        crop_coords = (top_left_x, top_left_y, width, height)
        print(f"Crop Coordinates: {crop_coords}")  # Add debug to track coordinates


def select_crop_region(video_path):
    """
    Function to select the cropping region interactively from the first frame of a video.
    :param video_path: Path to the video file.
    :return: Crop coordinates (top, left, width, height) or None if no region was selected.
    """
    global crop_coords, start_point, end_point

    # Reset the global variables
    crop_coords = None
    start_point = None
    end_point = None

    # Open the video and capture the first frame
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open the video.")
        return None

    # Get total frame count and calculate the middle frame index
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame_index = total_frames // 2  # Adjusted to the exact middle frame

    # Set the video position to the middle frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read the frame.")
        return None

    # Check the resolution of the frame
    height, width, _ = frame.shape
    print(f"Frame size: {width}x{height}")  # Debug frame size to ensure no resizing

    # Create a window and set the mouse callback for drawing a rectangle
    cv2.namedWindow("Select Region")
    cv2.setMouseCallback("Select Region", draw_rectangle)

    while True:
        # Display the frame, with the rectangle being drawn if applicable
        temp_frame = frame.copy()
        if start_point and end_point:
            cv2.rectangle(temp_frame, start_point, end_point, (0, 255, 0), 2)

        cv2.imshow("Select Region", temp_frame)

        # Wait for key press, finalize selection on 'Enter', cancel on 'ESC'
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key to finalize selection
            break
        elif key == 27:  # ESC key to exit without selection
            crop_coords = None
            break

    # Release the video and destroy the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

    print(
        f"Final Crop Coordinates: {crop_coords}"
    )  # Debug output of final crop coordinates

    return crop_coords
