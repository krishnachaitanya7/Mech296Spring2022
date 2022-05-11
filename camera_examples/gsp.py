
def gstreamer_pipeline(
	sensor_id=2,
	capture_width=1280,
	capture_height=720,
	display_width=640,
	display_height=480,
	framerate=20,
	flip_method=0,
):
	return (
		"nvarguscamerasrc sensor_id=%d !"
		"video/x-raw(memory:NVMM), "
		"width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
		"nvvidconv flip-method=%d ! "
		"video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
		"videoconvert ! "
		"video/x-raw, format=(string)BGR ! appsink drop=True"
		% (
			sensor_id,
			capture_width,
			capture_height,
			framerate,
			flip_method,
			display_width,
			display_height,
		)
	)
