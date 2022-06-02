from black import main
import pyrealsense2 as rs
import numpy as np
import cv2
from wheel_control import MotionController
from wheel_control import SolenoidController
import time
from time import sleep
import jetson.inference
import jetson.utils
import sys
import logging

rs.option.hue = 16
logging.basicConfig(stream=sys.stdout)
# set logger level info


# Constants
REACHED_BALL_Y = 410
REACHED_GOAL_HEIGHT = 250
SHOOTING_DISTANCE = 130
REACHED_PERSON_HEIGHT = 350
BALL_COLOR = (0, 255, 0)
GOAL_COLOR = (0, 0, 255)
ROBOT_COLOR = (255, 0, 0)
PERSON_COLOR = (255, 255, 0)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
SHOW_IMAGES = True
goal_color_assigned = "blue"
defence_position_reached = False


def robot_go(mc, left_pwm, right_pwm):
    mc.go_left_and_right(left_pwm, right_pwm)
    time.sleep(0.18)
    mc.stop()


def turn_for_searching(mc, left_pwm, right_pwm):
    mc.go_left_and_right(left_pwm, right_pwm)
    time.sleep(0.2)
    mc.stop()
    time.sleep(0.2)


def backup_robot(mc, left_pwm, right_pwm, right_or_left):
    mc.stop()
    time.sleep(0.1)
    mc.go_left_and_right(left_pwm, right_pwm)
    time.sleep(0.2)
    if right_or_left == "right":
        mc.go_left_and_right(-20, 20)
        time.sleep(0.4)
    else:
        mc.go_left_and_right(20, -20)
        time.sleep(0.4)
    mc.stop()


def turn_with_ball(mc, left_pwm, right_pwm):
    mc.go_left_and_right(left_pwm, right_pwm)
    time.sleep(0.2)
    mc.stop()
    time.sleep(0.2)


def detect_bottom_color(img):
    # TODO: Fix the goal colors
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower_green = np.array([40, 40, 40])
    # upper_green = np.array([70, 255, 255])
    lower_blue = np.array([80, 10, 10])
    upper_blue = np.array([140, 269, 193])
    # lower_blue = np.array([70, 70, 60])
    # upper_blue = np.array([160, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # green_mask = cv2.inRange(hsv, lower_green, upper_green)
    # num_green_pixels = (np.sum(green_mask) / 255) / (green_mask.shape[0] * green_mask.shape[1]) * 100
    num_blue_pixels = np.sum(blue_mask) / 255
    logger.info(f"Num of Blue Pixels: {num_blue_pixels}")
    # cv2.imshow("green_mask", green_mask)
    # cv2.imshow("blue_mask", blue_mask)
    # cv2.imshow("mask_image", img)
    # keyCode = cv2.waitKey(1) & 0xFF
    # if keyCode == 27 or keyCode == ord("q"):
    #     cv2.destroyAllWindows()
    if num_blue_pixels > 1:
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
        self.mobilenet = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.4)
        self.net = jetson.inference.detectNet(
            argv=[
                "--model=/home/robotvision/PycharmProjects/Mech296Spring2022/networks/my_network/mb1_ssd.onnx",
                "--labels=/home/robotvision/PycharmProjects/Mech296Spring2022/networks/my_network/labels.txt",
                "--input-blob=input_0",
                "--output-cvg=scores",
                "--output-bbox=boxes",
                "--threshold=0.2",
            ]
        )
        self.robot_net = jetson.inference.detectNet(
            argv=[
                "--model=/home/robotvision/PycharmProjects/Mech296Spring2022/networks/my_new_network/mb1_ssd_new.onnx",
                "--labels=/home/robotvision/PycharmProjects/Mech296Spring2022/networks/my_new_network/labels.txt",
                "--input-blob=input_0",
                "--output-cvg=scores",
                "--output-bbox=boxes",
                "--threshold=0.2",
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
        color_frame_numpy = np.asanyarray(self.color_frame.get_data())
        color_frame_jetson = jetson.utils.cudaFromNumpy(color_frame_numpy)
        return color_frame_numpy, color_frame_jetson, depth_colormap

    def get_pixel_depth(self, x, y):
        color_intrin = self.color_frame.profile.as_video_stream_profile().intrinsics
        udist = self.depth_frame.get_distance(x, y)
        point1 = rs.rs2_deproject_pixel_to_point(color_intrin, [x, y], udist)
        return point1[2]

    def detect_objects(self, img_jetson):
        detections = self.net.Detect(img_jetson)
        return detections

    def detect_wall(self, img_jetson):
        detections = self.mobilenet.Detect(img_jetson)
        return detections

    def detect_robot(self, img_jetson):
        detections = self.robot_net.Detect(img_jetson)
        return detections

    def close(self):
        self.pipeline.stop()


cam = realsense_cam()
mc = MotionController()
solenoid_controller = SolenoidController()


def kalashnikov(mc, solenoid_control, left_pwm, right_pwm):
    mc.go_left_and_right(left_pwm, right_pwm)
    sleep(0.1)
    solenoid_control.fire()
    # sleep(0.1)


# def go_to_goal(centroid_x_goal, goal_height):
#     MIDDLE_RANGE = np.arange(240, 300)
#     REACHED_GOAL_Y_HEIGHT = 350

#     if goal_height > REACHED_GOAL_Y_HEIGHT:
#         robot_go(mc, 20, -20)
#     else:
#         if centroid_x_goal in MIDDLE_RANGE:
#             robot_go(mc, 50, 50)
#         elif centroid_x_goal < MIDDLE_RANGE[0]:
#             robot_go(mc, 18, 25)
#         elif centroid_x_goal > MIDDLE_RANGE[-1]:
#             robot_go(mc, 25, 18)


def main_loop():
    best_goal = None
    our_goal = None
    opponent_goal = None
    best_ball = None
    best_person = None
    best_robot = None
    color_image_numpy, color_image_jetson, _ = cam.get_frames()
    # Ball and Goal detection
    detections = cam.detect_objects(color_image_jetson)
    # Person detection
    person_detections = cam.detect_wall(color_image_jetson)
    # Robot Detection
    robot_detections = cam.detect_robot(color_image_jetson)
    # Find the best goal
    all_goal_detections = [detection for detection in detections if detection.ClassID == 1]
    if len(all_goal_detections) > 0:
        color_image = np.copy(color_image_numpy)
        best_goal = max(all_goal_detections, key=lambda x: x.Confidence)
        # show the goal
        gx1, gy1, gx2, gy2 = best_goal.ROI
        cv2.rectangle(
            color_image_numpy, (int(round(gx1)), int(round(gy1))), (int(round(gx2)), int(round(gy2))), GOAL_COLOR, 2
        )
        # Detect the bottom color in Goal
        mask = np.zeros(color_image.shape[:2], dtype="uint8")
        # cv2.rectangle(mask, (int(round(gx1)), int(round(gy2))), (int(round(gx2)), int(round(gy2)) + 5), 255, 2)
        # cv2.rectangle(
        #     mask,
        #     (0, int(round(gy2))),
        #     (639, int(round(gy2)) + 10),
        #     255,
        #     2,
        # )
        mask[int(round(gy2)) : int(round(gy2)) + 30, int(round(gx1)) : int(round(gx2))] = 255
        # get the masked image
        masked_image = cv2.bitwise_and(color_image, color_image, mask=mask)
        # get the masked image's color
        goal_color = detect_bottom_color(masked_image)
        # goal_color = detect_bottom_color(color_image_numpy)
        if goal_color == goal_color_assigned:
            logger.info(f"Looking at right goal color: {goal_color}")
            # put goal color on top of the box
            cv2.putText(
                color_image_numpy,
                f"{goal_color}, H: {round(gy2-gy1, 2)}",
                (int(round((gx1 + gx2) / 2)), int(round((gy1 + gy2) / 2))),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                GOAL_COLOR,
                2,
            )
            our_goal = best_goal
            opponent_goal = None
        # Commenting this part due to bug in goal color detection.
        # Remove this comment and uncomment below snippet
        else:
            opponent_goal = best_goal
            our_goal = None
            logger.info(f"Not detecting the right color goal: {goal_color}")

    # find the best ball
    all_ball_detections = [detection for detection in detections if detection.ClassID == 2]
    if len(all_ball_detections) > 0:
        best_ball = max(all_ball_detections, key=lambda x: x.Confidence)
        bx1, by1, bx2, by2 = best_ball.ROI
        cv2.rectangle(
            color_image_numpy, (int(round(bx1)), int(round(by1))), (int(round(bx2)), int(round(by2))), BALL_COLOR, 2
        )
        cv2.putText(
            color_image_numpy,
            f"H: {round(by2-by1, 2)}",
            (int(round((bx1 + bx2) / 2)), int(round((by1 + by2) / 2))),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            BALL_COLOR,
            2,
        )
    # find the best person
    all_person_detections = [detection for detection in person_detections if detection.ClassID == 1]
    if len(all_person_detections) > 0:
        best_person = max(all_person_detections, key=lambda x: x.Confidence)
        px1, py1, px2, py2 = best_person.ROI
        cv2.rectangle(
            color_image_numpy,
            (int(round(px1)), int(round(py1))),
            (int(round(px2)), int(round(py2))),
            PERSON_COLOR,
            2,
        )
        cv2.putText(
            color_image_numpy,
            f"H: {round(py2-py1, 2)}",
            (int(round((px1 + px2) / 2)), int(round((py1 + py2) / 2))),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            PERSON_COLOR,
            2,
        )
    # Find best other robots to obliterate
    all_robot_detections = [detection for detection in robot_detections if detection.ClassID == 3]
    if len(all_robot_detections) > 0:
        best_robot = max(all_robot_detections, key=lambda x: x.Confidence)
        rx1, ry1, rx2, ry2 = best_robot.ROI
        cv2.rectangle(
            color_image_numpy,
            (int(round(rx1)), int(round(ry1))),
            (int(round(rx2)), int(round(ry2))),
            ROBOT_COLOR,
            2,
        )
        cv2.putText(
            color_image_numpy,
            f"h: {round(ry2-ry1, 2)}",
            (int(round((rx1 + rx2) / 2)), int(round((ry1 + ry2) / 2))),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            ROBOT_COLOR,
            2,
        )
    if SHOW_IMAGES:
        cv2.imshow("color", color_image_numpy)
        # cv2.imshow("depth", depth_image)
        cv2.waitKey(1)
    # The Control Loop
    return our_goal, opponent_goal, best_goal, best_ball, best_person, best_robot


def go_to_ball():
    logger.info("Ball not detected, Going to Ball")
    global defence_position_reached
    start_time = time.time()
    while True:
        MIDDLE_RANGE = np.arange(240, 290)
        our_goal, opponent_goal, best_goal, best_ball, best_person, best_robot = main_loop()
        if best_person is not None:
            px1, py1, px2, py2 = best_person.ROI
            person_height = py2 - py1
            person_x_centroid = (px1 + px2) / 2
            right_or_left = "right" if person_x_centroid > 320 else "left"
            if person_height > REACHED_PERSON_HEIGHT:
                backup_robot(mc, -20, -20, right_or_left)
                continue
        if best_ball is not None:
            start_time = time.time()
            defence_position_reached = False
            x1, y1, x2, y2 = best_ball.ROI
            centroid_x, centroid_y = int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))
            print(f"Centroid: {centroid_x}, {centroid_y}, Confidence: {best_ball.Confidence}")
            depth_val = cam.get_pixel_depth(centroid_x, centroid_y)
            print(f"Ball Depth: {depth_val}")
            if centroid_x in MIDDLE_RANGE:
                robot_go(mc, 25, 25)
            elif centroid_x < MIDDLE_RANGE[0]:
                robot_go(mc, 20, 25)
            elif centroid_x > MIDDLE_RANGE[-1]:
                robot_go(mc, 25, 20)
            if centroid_x in MIDDLE_RANGE and centroid_y > REACHED_BALL_Y:
                logger.critical("Reached The ball")
                return True
        else:
            turn_for_searching(mc, 20, -25)
            if time.time() - start_time > 5:
                logger.critical("Time limit exceeded")
                break
    return False


def go_to_goal():
    global defence_position_reached
    while True:
        # MIDDLE_RANGE = np.arange(240, 300)
        MIDDLE_RANGE = np.arange(220, 300)
        our_goal, opponent_goal, best_goal, best_ball, best_person, best_robot = main_loop()
        if best_person is not None:
            px1, py1, px2, py2 = best_person.ROI
            person_height = py2 - py1
            person_x_centroid = (px1 + px2) / 2
            right_or_left = "right" if person_x_centroid > 320 else "left"
            if person_height > REACHED_PERSON_HEIGHT:
                backup_robot(mc, -20, -20, right_or_left)
                continue
        if go_to_ball():
            logger.info("Going for goal")
            defence_position_reached = False
            our_goal, opponent_goal, best_goal, best_ball, best_person, best_robot = main_loop()
            x1, y1, x2, y2 = best_ball.ROI
            centroid_x_ball, centroid_y_ball = int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))
            if opponent_goal is not None:
                x1, y1, x2, y2 = opponent_goal.ROI
                centroid_x_goal, centroid_y_goal = int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))
                goal_height = y2 - y1
                if centroid_x_goal in MIDDLE_RANGE:
                    robot_go(mc, 25, 25)
                    # solenoid_controller.machine_gun()
                    # if goal_height > SHOOTING_DISTANCE:
                    #     logger.critical("Reached The Shooting Distance")
                    #     kalashnikov(mc, solenoid_controller, 20, 20)
                elif centroid_x_goal < MIDDLE_RANGE[0]:
                    logger.info("Goal on Left Side")
                    robot_go(mc, 18, 25)
                elif centroid_x_goal > MIDDLE_RANGE[-1]:
                    logger.info("Goal on Right Side")
                    robot_go(mc, 25, 18)
                if goal_height > REACHED_GOAL_HEIGHT:
                    mc.stop()
                    logger.info(f"Stopping. reached the goal. Centroid: {centroid_x_goal}, {centroid_y_goal}")
                    for _ in range(4):
                        logger.info("Shoot the fucking ball")
                        solenoid_controller.machine_gun()
                    # break
            else:
                turn_with_ball(mc, 25, -15)
        else:

            # reached_ball = go_to_ball()
            # if not reached_ball:
            while not defence_position_reached:
                # if not reached_ball:
                logger.info("No ball after 5 seconds, Defending the goal")
                # Means that 5 seconds elapsed, and I am not able to find the ball
                our_goal, opponent_goal, best_goal, best_ball, best_person, best_robot = main_loop()
                if best_ball is not None:
                    defence_position_reached = True
                    break
                if our_goal is not None:
                    x1, y1, x2, y2 = our_goal.ROI
                    centroid_x_goal, centroid_y_goal = int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))
                    goal_height = y2 - y1
                    if centroid_x_goal in MIDDLE_RANGE:
                        logger.info("Found Our Goal in the middle")
                        robot_go(mc, 30, 30)
                    elif centroid_x_goal < MIDDLE_RANGE[0]:
                        logger.info("Found Our Goal in the Left Side")
                        robot_go(mc, 18, 25)
                    elif centroid_x_goal > MIDDLE_RANGE[-1]:
                        logger.info("Found Our Goal in the Right Side")
                        robot_go(mc, 25, 18)
                    if goal_height > REACHED_GOAL_HEIGHT:
                        defence_position_reached = True
                        break
                else:
                    logger.critical("Goal not detected even after 5 seconds. Turning...")
                    turn_for_searching(mc, 20, -20)


if __name__ == "__main__":
    go_to_goal()
