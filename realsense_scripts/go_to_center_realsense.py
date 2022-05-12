import pyrealsense2 as rs
import numpy as np
import cv2
from wheel_control import MotionController
import time


def turn_left(mc):
    mc.go_left_and_right(-80, 80)
    time.sleep(0.1)
    mc.stop()
    time.sleep(0.01)


def move_forward(mc):
    mc.go_left_and_right(80, 80)
    time.sleep(0.1)
    mc.stop()
    time.sleep(0.01)


class realsense_cam:
    def __init__(self):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
        self.align = rs.align(rs.stream.color)
        pipeline.start(config)
        self.pipeline = pipeline
        self.color_frame = None
        self.depth_frame = None

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        self.color_frame = frames.get_color_frame()
        self.depth_frame = frames.get_depth_frame()
        # self.color_frame = np.asanyarray(color_frame.get_data())
        # self.depth_frame = np.asanyarray(depth_frame.get_data())
        return np.asanyarray(self.color_frame.get_data()), np.asanyarray(self.depth_frame.get_data())

    def get_pixel_depth(self, x, y):
        color_intrin = self.color_frame.profile.as_video_stream_profile().intrinsics
        udist = self.depth_frame.get_distance(x, y)
        point1 = rs.rs2_deproject_pixel_to_point(color_intrin, [x, y], udist)
        return point1

    def close(self):
        self.pipeline.stop()


def main():
    cam = realsense_cam()
    mc = MotionController()
    while True:
        color_image, depth_image = cam.get_frames()
        # Draw a rectangle over the color_image
        cv2.rectangle(color_image, (300, 15), (340, 25), (0, 255, 0), 2)
        cv2.imshow("color", color_image)
        cv2.imshow("depth", depth_image)
        _, _, depth_value = cam.get_pixel_depth(320, 20)
        # Algorithm Start
        print(f"Depth: {depth_value}")
        if depth_value > 0.68:
            move_forward(mc)
        else:
            turn_left(mc)
        # Algorithm End
        keyCode = cv2.waitKey(1) & 0xFF
        if keyCode == 27 or keyCode == ord("q"):
            break
    cam.close()


if __name__ == "__main__":
    main()
