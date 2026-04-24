import cv2

cap = cv2.VideoCapture('sample_video.mp4')
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
out = cv2.VideoWriter('sample_video_240x480.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (480, 240))

while True:
    ret, frame = cap.read()
    if not ret: break
    out.write(cv2.resize(frame, (480, 240)))

cap.release()
out.release()
print("Downscaled.")
