# Resound: Creating Reaction Concerts
A script pipeline that processes reaction videos to a particular song so that all reactions can be put together into a single combined video (called a reaction concert).

See some Reaction Concerts at https://www.youtube.com/channel/UCL7KNnyjribwiNTnSIosOeQ

Background on the overall project motivation at https://traviskriplean.com/video-considerit-sajpp9

## Modules
The main parts are:

1) **Alignment** Identifying when each reactor first encounters each part of the song. This enables creating a stripped down version of the reaction that is aligned perfectly with the song.
2) **Facial recognition and gaze tracking** Identifying unique faces in the reaction video so we can crop to the reactors. Also hypothesizes about the dominant gaze, for use later in the compositor.
3) **Backchannel isolation** Identifying when a reactor is saying something (or hooting!), and isolating just that sound.
4) **Composing and audio mixing** Creating a hexagonal grid, placing the song video, assigning a hex grid to each reactor based on dominant gaze, mixing / mastering audio including stereo panning based on grid position, and outputing the final video.

