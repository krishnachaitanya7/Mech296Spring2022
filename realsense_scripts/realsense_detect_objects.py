import cv2
import pyrealsense2 as rs
import numpy as np
import jetson.inference
import jetson.utils


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
                "--threshold=0.3",
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
        color_image, _ = cam.get_frames()
        detections = cam.detect_objects(color_image)
        for detection in detections:
            if detection.ClassID != 1:
                x1, y1, x2, y2 = detection.ROI
                if detection.ClassID == 1:
                    detected_class = "Goal"
                elif detection.ClassID == 2:
                    detected_class = "Soccer Ball"
                print(f"Class: {detected_class}, Confidence: {detection.Confidence}")
                cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.imshow("color_image", color_image)
        keyCode = cv2.waitKey(1) & 0xFF
        if keyCode == 27 or keyCode == ord("q"):
            break


if __name__ == "__main__":
    main()
