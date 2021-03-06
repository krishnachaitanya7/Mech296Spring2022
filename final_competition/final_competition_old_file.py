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

REACHED_BALL_Y = 410
REACHED_GOAL_Y = 195
REACHED_WALL_Y = 350
BALL_COLOR = (0, 255, 0)
GOAL_COLOR = (0, 0, 255)
ROBOT_COLOR = (255, 0, 0)
PERSON_COLOR = (255, 255, 0)


def backup_robot(mc, left_pwm, right_pwm):
    mc.stop()
    time.sleep(0.5)
    mc.go_left_and_right(left_pwm, right_pwm)
    time.sleep(0.1)
    mc.stop()


def robot_go(mc, left_pwm, right_pwm):
    mc.go_left_and_right(left_pwm, right_pwm)
    time.sleep(0.1)
    mc.stop()


def turn_with_ball(mc, left_pwm, right_pwm):
    mc.go_left_and_right(left_pwm, right_pwm)
    time.sleep(0.15)
    mc.stop()


def kalashnikov(mc, solenoid_control, left_pwm, right_pwm):
    for _ in range(0, 3):
        mc.go_left_and_right(left_pwm, right_pwm)
        solenoid_control.fire()
        sleep(0.1)


def detect_bottom_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower_green = np.array([40, 40, 40])
    # upper_green = np.array([70, 255, 255])
    lower_blue = np.array([80, 10, 10])
    upper_blue = np.array([140, 269, 193])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # green_mask = cv2.inRange(hsv, lower_green, upper_green)
    # num_green_pixels = (np.sum(green_mask) / 255) / (green_mask.shape[0] * green_mask.shape[1]) * 100
    num_blue_pixels = (np.sum(blue_mask) / 255) / (blue_mask.shape[0] * blue_mask.shape[1]) * 100
    # cv2.imshow("green_mask", green_mask)
    # cv2.imshow("blue_mask", blue_mask)
    # keyCode = cv2.waitKey(1) & 0xFF
    # if keyCode == 27 or keyCode == ord("q"):
    #     cv2.destroyAllWindows()
    if num_blue_pixels > 0:
        return "blue"
    else:
        return "green"


class realsense_cam:
    def __init__(self):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
        self.align = rs.align(rs.stream.color)
        pipeline.start(config)
        self.pipeline = pipeline
        self.mobilenet = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
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
        # self.mobilenet = self.net
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

    def detect_wall(self, img):
        best_person = None
        detections = self.mobilenet.Detect(jetson.utils.cudaFromNumpy(img))
        all_person_detections = [detection for detection in detections if detection.ClassID == 1]
        if len(all_person_detections) > 0:
            best_person = max(all_person_detections, key=lambda x: x.Confidence)
        if best_person is not None:
            x1, y1, x2, y2 = best_person.ROI
            # centroid_x, centroid_y = (x1 + x2) / 2, (y1 + y2) / 2
            # draw the rectangle around the person
            # round the values to the nearest integer
            x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
            cv2.rectangle(img, (x1, y1), (x2, y2), PERSON_COLOR, 2)
            return y2 - y1
        else:
            return None

    def close(self):
        self.pipeline.stop()


cam = realsense_cam()
mc = MotionController()
solenoid_controller = SolenoidController()


def go_to_ball():
    while True:
        MIDDLE_RANGE = np.arange(240, 300)
        best_ball = None
        color_image, _ = cam.get_frames()
        detections = cam.detect_objects(color_image)
        person_height = cam.detect_wall(color_image)
        if person_height is not None:
            if person_height > REACHED_WALL_Y:
                backup_robot(mc, -20, -20)
                continue
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
                robot_go(mc, 30, 30)
            elif centroid_x < MIDDLE_RANGE[0]:
                robot_go(mc, 18, 25)
            elif centroid_x > MIDDLE_RANGE[-1]:
                robot_go(mc, 25, 18)
            if centroid_x in MIDDLE_RANGE and centroid_y > REACHED_BALL_Y:
                # mc.stop()
                break
            cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), BALL_COLOR, 2)
            if best_goal is not None:
                x1, y1, x2, y2 = best_goal.ROI
                cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), GOAL_COLOR, 2)
        else:
            robot_go(mc, -25, 20)
            # time.sleep(0.01)
        cv2.imshow("color_image", color_image)
        keyCode = cv2.waitKey(1) & 0xFF
        if keyCode == 27 or keyCode == ord("q"):
            cv2.destroyAllWindows()
            break
    print("Got the ball, moving to goal")
    # go_to_goal("blue")


def go_to_goal(goal_color):
    while True:
        MIDDLE_RANGE = np.arange(240, 300)
        best_goal = None
        best_ball = None
        color_image, _ = cam.get_frames()
        # detect wall
        person_height = cam.detect_wall(color_image)
        if person_height is not None:
            if person_height > REACHED_WALL_Y:
                backup_robot(mc, -20, -20)
                continue
        detections = cam.detect_objects(color_image)
        # print(f"Bottom Color: {detect_bottom_color(color_image)}")
        # get the detection whith highest confidence of class 2
        all_goal_detections = [detection for detection in detections if detection.ClassID == 1]
        all_ball_detections = [detection for detection in detections if detection.ClassID == 2]
        if len(all_ball_detections) > 0:
            best_ball = sorted(all_ball_detections, key=lambda x: x.Confidence, reverse=True)[0]
        if len(all_goal_detections) > 0:
            best_goal = sorted(all_goal_detections, key=lambda x: x.Confidence, reverse=True)[0]
            x1, y1, x2, y2 = best_goal.ROI
            # round all coordinates
            x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
            # generate mask to keep only the line (x1+2,y2+2) and (x2+2,y2+2) in the image
            mask = np.zeros(color_image.shape[:2], dtype="uint8")
            cv2.line(mask, (x1, y2 + 2), (x2, y2 + 2), 255, 2)
            # get the masked image
            masked_image = cv2.bitwise_and(color_image, color_image, mask=mask)
            # get the masked image's color
            detected_goal_color = detect_bottom_color(masked_image)
            print(f"Goal Color: {detected_goal_color}")
            if detected_goal_color != goal_color:
                best_goal = None

        if best_ball is not None:
            x1, y1, x2, y2 = best_ball.ROI
            centroid_x_ball, centroid_y_ball = int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))
            print(f"Ball Centroid: {centroid_x_ball}, {centroid_y_ball}, Confidence: {best_ball.Confidence}")
            cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), BALL_COLOR, 2)
            if best_goal is not None:
                x1, y1, x2, y2 = best_goal.ROI
                centroid_x_goal, centroid_y_goal = int(round((x1 + x2) / 2)), y2 - y1  # int(round((y1 + y2) / 2))
                goal_depth = cam.get_pixel_depth(centroid_x_goal, centroid_y_goal)
                print(f"Depth of Goal Centroid {goal_depth}")
                print(f"Goal Centroid: {centroid_x_goal}, {centroid_y_goal}, Confidence: {best_goal.Confidence}")
                # if goal_depth > 0.8:
                if centroid_x_goal in MIDDLE_RANGE:
                    # robot_go(mc, 20, 20)
                    # solenoid_controller.machine_gun()
                    kalashnikov(mc, solenoid_controller, 20, 20)
                elif centroid_x_goal < MIDDLE_RANGE[0]:
                    robot_go(mc, 18, 25)
                elif centroid_x_goal > MIDDLE_RANGE[-1]:
                    robot_go(mc, 25, 18)
                if centroid_x_goal in MIDDLE_RANGE and centroid_y_goal > REACHED_GOAL_Y:
                    mc.stop()
                    print(f"stopping reached the goal. Centroid: {centroid_x_goal}, {centroid_y_goal}")
                    solenoid_controller.machine_gun()
                    break
                cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), GOAL_COLOR, 2)
            else:
                turn_with_ball(mc, 10, 25)
                # time.sleep(0.02)
        else:
            print("Ball not detected")
            go_to_ball()

        # cv2.imshow("depth_image", depth_image)
        cv2.imshow("color_image", color_image)
        keyCode = cv2.waitKey(1) & 0xFF
        if keyCode == 27 or keyCode == ord("q"):
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()
    cam.close()


if __name__ == "__main__":
    go_to_goal("blue")
