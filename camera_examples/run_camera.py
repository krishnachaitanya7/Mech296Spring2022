import cv2
import jetson.inference
import jetson.utils
from gsp import gstreamer_pipeline as gsp


net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.1)


def main():
    window_title = "CSI Camera"

    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    cap = cv2.VideoCapture(gsp(flip_method=0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        try:
            while True:
                ret, img = cap.read()
                imgCuda = jetson.utils.cudaFromNumpy(img)
                # the reverse is done by: img = jetson.utils.cudaToNumpy(imgCuda)
                detections = net.Detect(imgCuda)
                for d in detections:
                    # print(d)
                    xt, yt, xb, yb = int(d.Left), int(d.Top), int(d.Right), int(d.Bottom)
                    className = net.GetClassDesc(d.ClassID)
                    if className == "person":
                        cv2.rectangle(img, (xt, yt), (xb, yb), (255, 0, 0), 2)
                        cv2.putText(img, className, (xt + 5, yt + 1), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 0), 2)

                cv2.imshow(window_title, img)
                keyCode = cv2.waitKey(10) & 0xFF
                # Stop the program on the ESC key or 'q'
                if keyCode == 27 or keyCode == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open CSI camera")


if __name__ == "__main__":
    main()
