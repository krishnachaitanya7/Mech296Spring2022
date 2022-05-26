import pyrealsense2 as rs
import numpy as np
import cv2
from wheel_control import MotionController
from wheel_control import SolenoidController
import time
from time import sleep
import jetson.inference
import jetson.utils

# Constants
ROTATION_PWM = 40
MIDDLE_RANGE = np.arange(240, 300)
REACHED_BALL_Y = 300
REACHED_GOAL_Y = 150


def robot_go(mc, left_pwm, right_pwm):
    mc.go_left_and_right(left_pwm, right_pwm)
    time.sleep(0.2)
    mc.stop()


def detect_bottom_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])
    lower_blue = np.array([17, 150, 90])
    upper_blue = np.array([126, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    num_green_pixels = (np.sum(green_mask) / 255) / (green_mask.shape[0] * green_mask.shape[1]) * 100
    num_blue_pixels = (np.sum(blue_mask) / 255) / (blue_mask.shape[0] * blue_mask.shape[1]) * 100
    if num_green_pixels > num_blue_pixels:
        return "green"
    else:
        return "blue"


class realsense_cam:
    def __init__(self):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
        self.align = rs.align(rs.stream.color)
        pipeline.start(config)
        self.pipeline = pipeline
        self.net = jetson.inference.detectNet(
            argv=[
                "--model=/home/robotvision/PycharmProjects/Mech296Spring2022/networks/my_network/mb1_ssd.onnx",
                "--labels=/home/robotvision/PycharmProjects/Mech296Spring2022/networks/my_network/labels.txt",
                "--input-blob=input_0",
                "--output-cvg=scores",
                "--output-bbox=boxes",
                "--threshold=0.1",
            ]
        )
        self.colorizer = rs.colorizer()
        self.color_frame = None
        self.depth_frame = None

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        self.color_frame = frames.get_color_frame()
        self.depth_frame = frames.get_depth_frame()
        # self.color_frame = np.asanyarray(color_frame.get_data())
        # self.depth_frame = np.asanyarray(depth_frame.get_data())
        # colorize the depth frame to see the depth data
        depth_colormap = np.asanyarray(self.colorizer.colorize(self.depth_frame).get_data())
        # np.asanyarray(self.depth_frame.get_data())
        return np.asanyarray(self.color_frame.get_data()), depth_colormap

    def get_pixel_depth(self, x, y):
        color_intrin = self.color_frame.profile.as_video_stream_profile().intrinsics
        udist = self.depth_frame.get_distance(x, y)
        point1 = rs.rs2_deproject_pixel_to_point(color_intrin, [x, y], udist)
        return point1[2]

    def detect_objects(self, img):
        detections = self.net.Detect(jetson.utils.cudaFromNumpy(img))
        return detections

    def close(self):
        self.pipeline.stop()


cam = realsense_cam()
mc = MotionController()
solenoid_controller = SolenoidController()


def go_to_ball():
    while True:
        best_ball = None
        color_image, _ = cam.get_frames()
        detections = cam.detect_objects(color_image)
        # get the detection whith highest confidence of class 2
        all_goal_detections = [detection for detection in detections if detection.ClassID == 1]
        all_ball_detections = [detection for detection in detections if detection.ClassID == 2]
        if len(all_goal_detections) > 0:
            best_goal = sorted(all_goal_detections, key=lambda x: x.Confidence, reverse=True)[0]
        if len(all_ball_detections) > 0:
            best_ball = sorted(all_ball_detections, key=lambda x: x.Confidence, reverse=True)[0]
        if best_ball is not None:
            x1, y1, x2, y2 = best_ball.ROI
            centroid_x, centroid_y = int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))
            print(f"Centroid: {centroid_x}, {centroid_y}, Confidence: {best_ball.Confidence}")
            depth_val = cam.get_pixel_depth(centroid_x, centroid_y)
            print(f"Ball Depth: {depth_val}")
            if centroid_x in MIDDLE_RANGE:
                robot_go(mc, 25, 25)
            elif centroid_x < MIDDLE_RANGE[0]:
                robot_go(mc, 18, 25)
            elif centroid_x > MIDDLE_RANGE[-1]:
                robot_go(mc, 25, 18)
            if centroid_x in MIDDLE_RANGE and centroid_y > REACHED_BALL_Y:
                mc.stop()
                break
            cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        else:
            robot_go(mc, -20, 20)
            time.sleep(0.02)
        cv2.imshow("color_image", color_image)
        # cv2.imshow("depth_image", depth_image)
        keyCode = cv2.waitKey(1) & 0xFF
        if keyCode == 27 or keyCode == ord("q"):
            cv2.destroyAllWindows()
            break
    print("Got the ball, moving to goal")
    # go_to_goal("blue")


def go_to_goal(goal_color):
    while True:
        best_goal = None
        best_ball = None
        color_image, _ = cam.get_frames()
        detections = cam.detect_objects(color_image)
        # get the detection whith highest confidence of class 2
        all_goal_detections = [detection for detection in detections if detection.ClassID == 1]
        all_ball_detections = [detection for detection in detections if detection.ClassID == 2]
        if len(all_ball_detections) > 0:
            best_ball = sorted(all_ball_detections, key=lambda x: x.Confidence, reverse=True)[0]
        if len(all_goal_detections) > 0:
            best_goal = sorted(all_goal_detections, key=lambda x: x.Confidence, reverse=True)[0]

        if best_ball is not None:
            x1, y1, x2, y2 = best_ball.ROI
            centroid_x_ball, centroid_y_ball = int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))
            print(f"Ball Centroid: {centroid_x_ball}, {centroid_y_ball}, Confidence: {best_ball.Confidence}")
            cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            if best_goal is not None:
                x1, y1, x2, y2 = best_goal.ROI
                centroid_x, centroid_y = int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))
                print(f"Goal Centroid: {centroid_x}, {centroid_y}, Confidence: {best_goal.Confidence}")
                if centroid_x in MIDDLE_RANGE:
                    robot_go(mc, 30, 30)
                    # solenoid_controller.machine_gun()
                elif centroid_x < MIDDLE_RANGE[0]:
                    robot_go(mc, 20, 30)
                elif centroid_x > MIDDLE_RANGE[-1]:
                    robot_go(mc, 30, 20)
                if centroid_x in MIDDLE_RANGE and centroid_y > REACHED_GOAL_Y:
                    mc.stop()
                    print(f"stopping reached the goal. Centroid: {centroid_x}, {centroid_y}")
                    solenoid_controller.machine_gun()
                    break
                cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            else:
                robot_go(mc, 25, 10)
                time.sleep(0.02)
        else:
            print("Ball not detected")
            go_to_ball()
        cv2.imshow("color_image", color_image)
        # cv2.imshow("depth_image", depth_image)
        keyCode = cv2.waitKey(1) & 0xFF
        if keyCode == 27 or keyCode == ord("q"):
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()
    cam.close()


if __name__ == "__main__":
    go_to_goal("blue")
