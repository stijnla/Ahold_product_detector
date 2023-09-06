import cv2
cap = cv2.VideoCapture("../../../../rgb.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV v2.x used "CV_CAP_PROP_FPS"
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count/fps
print(duration)
new_duration = 10
new_fps = frame_count/new_duration
output_rgb_video = cv2.VideoWriter('rgb.mp4',cv2.VideoWriter_fourcc(*'mp4v'), new_fps, (1280,720))
print(new_fps)
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        output_rgb_video.write(frame)
    else:
        break
cap.release()