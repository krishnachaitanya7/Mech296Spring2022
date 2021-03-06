import cv2
import pyrealsense2 as rs
import numpy as np
import jetson.inference
import jetson.utils
from wheel_control import MotionController


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

    def detect_objects(self, img):
        detections = self.net.Detect(jetson.utils.cudaFromNumpy(img))
        return detections


def main():
    cam = realsense_cam()
    while True:
        best_goal = None
        best_ball = None
        best_robot = None
        color_image, _ = cam.get_frames()
        detections = cam.detect_objects(color_image)
        # get the detection whith highest confidence of class 2
        all_goal_detections = [detection for detection in detections if detection.ClassID == 1]
        all_ball_detections = [detection for detection in detections if detection.ClassID == 2]
        all_robot_detections = [detection for detection in detections if detection.ClassID == 3]
        if len(all_goal_detections) > 0:
            best_goal = sorted(all_goal_detections, key=lambda x: x.Confidence, reverse=True)[0]
        if len(all_ball_detections) > 0:
            best_ball = sorted(all_ball_detections, key=lambda x: x.Confidence, reverse=True)[0]
        if len(all_robot_detections) > 0:
            best_robot = sorted(all_robot_detections, key=lambda x: x.Confidence, reverse=True)[0]
        if best_ball is not None:
            x1, y1, x2, y2 = best_ball.ROI
            cv2.rectangle(
                color_image, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), (0, 255, 0), 2
            )
            centroid_x, centroid_y = int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))
            print(f"Ball Centroid: {centroid_x}, {centroid_y}, Ball Confidence: {best_ball.Confidence}")
        if best_goal is not None:
            x1, y1, x2, y2 = best_goal.ROI
            cv2.rectangle(
                color_image, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), (255, 0, 0), 2
            )
            centroid_x, centroid_y = int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))
            print(f"Goal Centroid: {centroid_x}, {centroid_y}, Goal Confidence: {best_goal.Confidence}")
        if best_robot is not None:
            x1, y1, x2, y2 = best_robot.ROI
            cv2.rectangle(
                color_image, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), (0, 0, 255), 2
            )
            centroid_x, centroid_y = int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))
            print(f"Robot Centroid: {centroid_x}, {centroid_y}, Robot Confidence: {best_robot.Confidence}")
        cv2.imshow("color_image", color_image)
        keyCode = cv2.waitKey(1) & 0xFF
        if keyCode == 27 or keyCode == ord("q"):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()
