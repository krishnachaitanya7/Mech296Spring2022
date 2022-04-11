import cv2


def main():
    window_title = "CSI Camera"
    cap = cv2.VideoCapture(gsp(flip_method=0), cv2.CAP_GSTREAMER)
    while 1:
        if cap.isOpened():
            ret, img = cap.read()
            cv2.imshow(window_title, img)
            keyCode = cv2.waitKey(10) & 0xFF
            if keyCode == 27 or keyCode == ord("q"):
                break
        else:
            print("Error: Unable to open CSI camera")


if __name__ == "__main__":
    main()
