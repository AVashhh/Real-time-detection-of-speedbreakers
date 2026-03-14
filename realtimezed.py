import cv2
import pyzed.sl as sl
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("./best_150.pt") 
# Initialize ZED camera
zed = sl.Camera()

init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720  # or HD1080 if GPU allows
init_params.camera_fps = 30

init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
init_params.coordinate_units = sl.UNIT.METER

status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    print("ZED Initialization Failed:", status)
    exit(1)

# Get camera resolution
camera_info = zed.get_camera_information()
resolution = camera_info.camera_configuration.resolution
print(f"Camera resolution: {resolution.width} x {resolution.height}")

# Create display window
cv2.namedWindow("YOLOv8x + ZED (Real-Time)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8x + ZED (Real-Time)", resolution.width, resolution.height)

# Runtime params
runtime = sl.RuntimeParameters()
image = sl.Mat()
depth = sl.Mat()

print("Starting real-time detection. Press 'q' to exit...")

while True:
    # Grab a frame from the live ZED camera
    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        # Retrieve image and depth
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        frame = image.get_data()

        # Convert BGRA to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Run YOLOv8 detection
        results = model(frame, conf=0.421, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())

                # Detect speed bumps (class 0)
                if cls == 0 and conf > 0.42:
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    # Clamp coordinates to frame bounds
                    h, w = frame.shape[:2]
                    cx = max(0, min(cx, w - 1))
                    cy = max(0, min(cy, h - 1))

                    # Get depth (distance)
                    err, dist = depth.get_value(cx, cy)
                    distance = dist if err == sl.ERROR_CODE.SUCCESS and dist > 0 else -1

                    # Draw results
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Speed bump {distance:.2f} m",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("YOLOv8x + ZED (Real-Time)", frame)

        # Exit if 'q' pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
zed.close()
cv2.destroyAllWindows()