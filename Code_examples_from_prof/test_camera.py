import sys
import cv2
from gsp import gstreamer_pipeline as gsp
def main():    
	window_title = "CSI Camera"
	# To flip the image, modify the flip_method parameter (0 and 2 are the most common)
	cap = cv2.VideoCapture(gsp(flip_method=0), cv2.CAP_GSTREAMER)
    while 1:
        if cap.isOpened():
            ret, img = cap.read()
            cv2.imshow("window_title", img)
            keyCode = cv2.waitKey(10) & 0xFF
            # Stop the program on the ESC key or 'q'
            if keyCode == 27 or keyCode == ord('q'):
                break	    
        else:
            print("Error: Unable to open CSI camera")


if __name__ == "__main__":
	main()
