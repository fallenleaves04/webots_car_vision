import numpy as np
import cv2
import math
import os
from controller import (Robot, Camera,
GPS, Keyboard, DistanceSensor,
Gyro, InertialUnit, Supervisor,
Display)
from vehicle import Driver
#from transformers import AutoImageProcessor, AutoModelForDepthEstimation
#from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
#import torch._dynamo
#torch._dynamo.config.suppress_errors = True
#import torch
#import torchvision.transforms as T

import visualise as vis
import camera_calibration as cc
import park_algo as palg
import stereo_yolo as sy

#from scipy.interpolate import splprep, splev
#from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import matplotlib._pylab_helpers
import time

# --------------------- Constants ---------------------
TIME_STEP = 64
NUM_DIST_SENSORS = 12
NUM_CAMERAS = 7
MAX_SPEED = 250.0
CAMERA_HEIGHT=2160
CAMERA_WIDTH=3840
global parking
parking = False
# PID parameters
KP = 0.25
KI = 0.006
KD = 2
KAW = 0.1

# Vehicle parameters
TRACK_FRONT = 1.628
TRACK_REAR = 1.628
WHEELBASE = 2.995
MAX_WHEEL_ANGLE = 0.5  # rad
CAR_WIDTH = 2.302
CAR_LENGTH = 4.85

# --------------------- Global Variables ---------------------

driver = Driver()

#robot = Robot()
#supervisor = Supervisor()
display = Display('display')
keyboard = Keyboard()
keyboard.enable(TIME_STEP)
gps = None
cameras = []
camera_names = []
cam_matrices = {}
images =[]
distance_sensors = []

speed = 0.0
steering_angle = 0.0
manual_steering = 0

previous_error = 0.0
integral = 0.0
homography_matrices = {}


# --------------------- Helper Functions ---------------------
def print_help():
    print("Samochód teraz jeździ.")
    print("Proszę użyć klawiszy UP/DOWN dla zwiększenia prędkości lub LEFT/RIGHT dla skrętu")

def set_speed(kmh):
    global speed
    speed = min(kmh, MAX_SPEED)
    driver.setCruisingSpeed(speed)
    print(f"Ustawiono prędkość {speed} km/h")

def set_steering_angle(wheel_angle):
    global steering_angle
    # Clamp steering angle to [-0.5, 0.5] radians (per vehicle constraints)
    wheel_angle = max(min(wheel_angle, MAX_WHEEL_ANGLE), -MAX_WHEEL_ANGLE)
    steering_angle = wheel_angle
    driver.setSteeringAngle(steering_angle)
    print(f"Skręcam {steering_angle} rad")

def change_manual_steering_angle(inc):
    global manual_steering
    new_manual_steering = manual_steering + inc
    if -25.0 <= new_manual_steering <= 25.0:
        manual_steering = new_manual_steering
        set_steering_angle(manual_steering * 0.02)


def check_keyboard():
    global parking
    key = keyboard.getKey()
    if key == Keyboard.UP:
        set_speed(speed + 0.5)
    elif key == Keyboard.DOWN:
        set_speed(speed - 0.5)
    elif key == Keyboard.RIGHT:
        change_manual_steering_angle(+2)
    elif key == Keyboard.LEFT:
        change_manual_steering_angle(-2)
    elif key == ord('P') or key == ord('p'):
        parking = not parking
        print("Rozpoczęto procedurę parkingu" if parking else "Ukończono procedurę parkingową")

#----------------------Sensor functions-----------------
def get_camera_image(camera):
    width = camera.getWidth()
    height = camera.getHeight()
    img = camera.getImage()
    if img is None:
        return None

    img_array = np.frombuffer(img, np.uint8).reshape((height, width, 4))[:, :, :3]
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    return img_array


# --------------------- Main Controller Loop ---------------------
def main():
    global processing
    # Initialize ultrasonic distance sensors
    sensor_names = [
        "distance sensor left front side", "distance sensor front left", "distance sensor front lefter",
        "distance sensor front righter", "distance sensor front right", "distance sensor right front side",
        "distance sensor left side", "distance sensor left", "distance sensor lefter",
        "distance sensor righter", "distance sensor right", "distance sensor right side"
    ]


    names_dists = {}
    dists = []
    for name in sensor_names:
        sensor = driver.getDevice(name)
        if sensor:
            sensor.enable(TIME_STEP)
            distance_sensors.append(sensor)

            print(f"Found sensor: {name}")
        else:
            print(f"Sensor not found: {name}")


    def process_distance_sensors(sen):
        l_dist = sen.getLookupTable()
        a_dist = (l_dist[0]-l_dist[3])/(l_dist[1]-l_dist[4])
        b_dist = l_dist[3]-l_dist[4]*a_dist
        value = sen.getValue()
        distance = a_dist*value+b_dist
        sigma = l_dist[2]
        noisy_distance = distance + np.random.normal(0, sigma)
        return noisy_distance
    # Initialize cameras
    camera_names = [
        "camera_front_bumper_wide","camera_rear",
        "camera_left_fender", "camera_right_fender",
        "camera_left_pillar",  "camera_right_pillar",
        "camera_front_top", "camera_front_top_add",
        "camera_helper"
    ]

    for name in camera_names:
        cam = driver.getDevice(name)
        if cam:
            cam.enable(TIME_STEP)
            cameras.append(cam)
            width = cam.getWidth()
            height = cam.getHeight()
            fov_rad = cam.getFov()
            K = sy.calculate_intrinsic_matrix(width, height, fov_rad)
            cam_matrices[name] = K
            print(f"Odnaleziono kamerę: {name}")


    cam_transfs = {}



    # initialize GPS
    gps = driver.getDevice("gps")
    if gps:
        gps.enable(TIME_STEP)
        print("GPS enabled")

    gyro = driver.getDevice("gyro")
    if gyro:
        gyro.enable(TIME_STEP)
        print("Gyro enabled")
    imu = driver.getDevice("inertial unit")
    if imu:
        imu.enable(TIME_STEP)
        print("IMU enabled")
    #print(cv2.__file__)

    print_help()



    #cv2.imshow("__",chessboard)
    # Main simulation loop.
    print("reading homographies...")
    right_H = np.load("right_homo.npy").astype(np.float32)
    left_H = np.load("left_homo.npy").astype(np.float32)
    front_H = np.load("front_homo.npy").astype(np.float32)
    right_fender_H = np.load("right_fender_homo.npy").astype(np.float32)
    left_fender_H = np.load("left_fender_homo.npy").astype(np.float32)
    rear_H = np.load("rear_homo.npy").astype(np.float32)

    homographies = []
    homographies.extend([front_H,right_H,right_fender_H,
    rear_H,left_fender_H,left_H])


    s = 2 # skala
    S = np.array([[1/s,0,0],[0,1/s,0],[0,0,1]]).astype(np.float32)
    homographies = [S @ H @ np.linalg.inv(S) for H in homographies]

    print("reading transformation matrices...")

    front_T = np.load("camera_front_bumper_wide_T_global.npy").astype(np.float32)
    left_T = np.load("camera_left_pillar_T_global.npy").astype(np.float32)
    right_T = np.load("camera_right_pillar_T_global.npy").astype(np.float32)
    left_fender_T = np.load("camera_left_fender_T_global.npy").astype(np.float32)
    right_fender_T = np.load("camera_right_fender_T_global.npy").astype(np.float32)
    rear_T = np.load("camera_rear_T_global.npy").astype(np.float32)
    front_top_T = np.load("camera_front_top_T_global.npy").astype(np.float32)

    first = True
    prev_x = 0
    prev_y = 0
    i = 0
    last_time = 0.0
    interval = 1024


    stream1 = cv2.cuda.Stream()
    stream2 = cv2.cuda.Stream()
    stream3 = cv2.cuda.Stream()
    stream4 = cv2.cuda.Stream()
    stream5 = cv2.cuda.Stream()
    stream6 = cv2.cuda.Stream()
    stream7 = cv2.cuda.Stream()
    stream8 = cv2.cuda.Stream()
    stream9 = cv2.cuda.Stream()
    stream10 = cv2.cuda.Stream()
    stream11 = cv2.cuda.Stream()
    stream12 = cv2.cuda.Stream()
    stream13 = cv2.cuda.Stream()
    streams = (stream1,stream2,stream3,stream4,stream5,
    stream6,stream7,stream8,stream9,stream10,stream11,stream12,stream13)


    car = cv2.imread("bmw.png", flags=cv2.IMREAD_COLOR)

    #model = YOLO("yolo11m-seg.pt")
    #
    #
    #



    yaw_init = 0.0
    first = True
    parker = None
    plotter = None

    SENSOR_INTERVAL = 0.15
    IMAGE_INTERVAL  = 0.15
    KEYBOARD_INTERVAL = 0.03
    last_sensor_time = driver.getTime()
    last_image_time  = driver.getTime()
    last_key_time = driver.getTime()
    win = None

    fig, ax = plt.subplots()
    T_center_to_camera = front_top_T
    name_right = "camera_front_top"
    name_left = "camera_front_top_add"

    # Obiekt stereo matcher
    stereo_left = cv2.StereoSGBM_create(minDisparity=0,
    numDisparities=32,
    blockSize=20,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=8)
    # matcher dla prawego obrazu - trzeba użyć createRightMatcher z ximgproc
    stereo_right = cv2.ximgproc.createRightMatcher(stereo_left)


    while driver.step() != -1:

        dists.clear()
        names_dists.clear()
        now = driver.getTime()
        if parking:
            """
            if now - last_sensor_time >= SENSOR_INTERVAL:
                last_sensor_time = now

                dists = [process_distance_sensors(s) for s in distance_sensors]
                names_dists = dict(zip(sensor_names,dists))
                plt.ion()
                fig.show()


                vis.draw_cones(ax,fig,dists)


                times = time.time()
                # inicjalizacja parkingu raz
                if first:
                    plotter = palg.LivePlotter()
                    parker = palg.Parking(driver,"left",times)
                    yaw_init = imu.getRollPitchYaw()[2]
                    first = False

                plotter.update(0)
                plotter.val = names_dists["distance sensor left front side"]

                #plotter.run()
                yaw = imu.getRollPitchYaw()[2]-yaw_init
                #print(yaw)
                parker.update_state(names_dists, yaw)
                if parker.state == "waiting_for_park":
                    odom_pose,spot = parker.update_state(names_dists, yaw)
                    parker.exec_path(odom_pose,spot,names_dists["distance sensor front left side"])

                #images = [get_camera_image(c) for c in cameras]
                #names_images = dict(zip(camera_names, images))
                #img_right = names_images["camera_front_top"]
                #img_left = names_images["camera_front_top_add"]
                #cv2.imwrite("jakis2.png",cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))

                time.sleep(0)
            """
            if now - last_sensor_time >= SENSOR_INTERVAL:
                images = [get_camera_image(c) for c in cameras]
                names_images = dict(zip(camera_names, images))
                #front = cv2.cvtColor(names_images["camera_front_bumper_wide"],cv2.COLOR_BGR2RGB)
                #helper = cv2.cvtColor(names_images["camera_helper"],cv2.COLOR_BGR2RGB)
                viss = vis.alt_collect_homo(names_images,homographies,car,streams)
                cv2.imwrite("img_vis1.jpg",viss)
                cv2.waitKey(1)
                time.sleep(0)
            """
            if now - last_image_time >= IMAGE_INTERVAL:
                last_image_time = now

                images = [get_camera_image(c) for c in cameras]
                names_images = dict(zip(camera_names, images))
                img_right = names_images[name_right]
                img_left = names_images[name_left]
                right_copy = img_right.copy()
                results = model(img_right,half=True,device = 0,classes = [2,5,7,10],conf=0.6,verbose=False,imgsz=(1280,960))
                if results and results[0].masks is not None:

                    K_right = cam_matrices[name_right]
                    K_left = cam_matrices[name_left]
                    f_left = K_left[0][0]
                    f_right = K_right[0][0]
                    # Zamień na grayscale

                    grayL = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
                    grayR = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)


                    #TUTAJ DALEJ ODFILTROWANE DISPARITY
                    # oblicz disparity z lewej i prawej kamery
                    disp_left = stereo_left.compute(grayL, grayR).astype(np.float32) / 16.0
                    disp_right = stereo_right.compute(grayR, grayL).astype(np.float32) / 16.0

                    # utwórz filtr WLS
                    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo_left)
                    wls_filter.setLambda(8000)
                    wls_filter.setSigmaColor(1.9)

                    # filtruj disparity
                    filtered_disp = wls_filter.filter(disp_left, grayL, None, disp_right)

                    disp_vis = cv2.normalize(filtered_disp, None, 0, 255, cv2.NORM_MINMAX)
                    disp_vis = np.nan_to_num(disp_vis, nan=0.0, posinf=0.0, neginf=0.0)
                    disp_vis = np.uint8(disp_vis)
                    cv2.namedWindow("Disparity WLS filtered",cv2.WINDOW_NORMAL)
                    cv2.imshow("Disparity WLS filtered", disp_vis)




                    #annotated_frame = results[0].plot()
                    masks = results[0].masks.data.cpu().numpy()  # shape: (num_detections, H, W)
                    orig_h, orig_w = img_right.shape[:2]

                    for i, mask in enumerate(masks):
                        # Resize do rozmiaru obrazu
                        mask_resized = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

                        # Kolor losowy
                        color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)

                        # Nałóż maskę
                        colored = np.zeros_like(right_copy, dtype=np.uint8)
                        for c in range(3):
                            colored[:, :, c] = color[c] * mask_resized

                        # Przezroczyste nałożenie
                        alpha = 0.6
                        right_copy = cv2.addWeighted(right_copy, 1.0, colored, alpha, 0)

                        filtered_disp_clean = np.nan_to_num(filtered_disp, nan=0.0, posinf=0.0, neginf=0.0)
                        disparity_masked = filtered_disp_clean * mask_resized

                        # Znajdź indeks punktu z największą disparity (czyli najmniejszą odległością)
                        # W masce disparity może być 0 tam gdzie brak danych, więc pomijamy
                        # Pobierz disparity tylko w masce i >0
                        valid_disparities = disparity_masked[(mask_resized > 0) & (disparity_masked > 0)]

                        if len(valid_disparities) == 0:
                            continue

                        mean_disp = valid_disparities.mean()
                        depth_m = f_right * 0.03 / mean_disp

            """
            """
                        if depth_m is not None:
                            #print(f"Obiekt {i}: Średnia disparity = {mean_disp:.2f}, odległość = {depth_m:.2f} m")
                            # Oblicz różnice abs między disparity a mean_disp
                            diffs = np.abs(disparity_masked - mean_disp)
                            diffs[mask_resized == 0] = np.inf  # poza maską ustawiamy na nieskończoność, aby je odrzucić

                            min_idx = np.unravel_index(np.argmin(diffs), diffs.shape)
                            v, u = min_idx

                            cv2.circle(right_copy, (u, v), 10, (0, 255, 0), 2)
            """
            """
                        #mask_resized = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                        #print(f"DLA OBIEKTU {i}")
                        #print("-----------------------------")



                        p1_3d, p2_3d = sy.points_from_mask_to_3D(mask_resized, filtered_disp, K_right, 0.03, T_center_to_camera)
                        if p1_3d is not None and p2_3d is not None:
                            #print(f"Punkt 1: {p1_3d}")
                            #print(f"Punkt 2: {p2_3d}")


                            p1_3d = np.append(p1_3d, 1.0)  # -> [X, Y, Z, 1]

                            p1_3d = p1_3d[:3]
                            p1_3d[2] = 0
                            p2_3d = np.append(p2_3d, 1.0)  # -> [X, Y, Z, 1]

                            p2_3d = p2_3d[:3]
                            p2_3d[2]=0


                            # Rzut na obraz
                            p1_3d_px = sy.project_points_world_to_image([p1_3d], T_center_to_camera, K)
                            p2_3d_px = sy.project_points_world_to_image([p2_3d], T_center_to_camera, K)
                            if len(p1_3d) > 0 and len(p2_3d) > 0:
                                # Rysuj na obrazie
                                u1,v1 = p1_3d_px[0]
                                u2,v2 = p2_3d_px[0]
                                color_tuple = tuple(int(c) for c in color)
                                cv2.circle(right_copy, (u1, v1), 6, color_tuple, -1)
                                cv2.putText(right_copy, f"PT1", (u1 + 5, v1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_tuple, 1)

                                cv2.circle(right_copy, (u2, v2), 6, color_tuple, -1)
                                cv2.putText(right_copy, f"PT2", (u2 + 5, v2 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_tuple, 1)

                                box_corners = sy.get_ground_box_corners(p1_3d, p2_3d)
                                sy.draw_ground_box_on_image(right_copy, box_corners, T_center_to_camera, K_right, color_tuple)

                            #print("-----------------------------")
                            #print("_____________________________")

                        ys, xs = np.where(mask_resized > 0)
                        if len(xs) == 0:
                            return
                        center_x = int(np.mean(xs))
                        center_y = int(np.mean(ys))

                        cv2.putText(right_copy, str(i), (center_x, center_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)


                    # Po pętli pokaż obraz
                    if depth_m is not None:
                        cv2.namedWindow("Maski z punktami najblizszymi",cv2.WINDOW_NORMAL)
                        cv2.imshow("Maski z punktami najblizszymi", right_copy)




                vis.collect_homo(names_images, homographies_scaled, car, streams)
                time.sleep(0)
                cv2.waitKey(1)
        else:
           #win.close()
           #app.quit()

           manager = matplotlib._pylab_helpers.Gcf.get_active()
           if manager:

                manager.canvas.manager.window.destroy()
                manager.canvas.stop_event_loop()

           cv2.destroyAllWindows()
        """
        if now - last_key_time >= KEYBOARD_INTERVAL:
            last_key_time = now
            check_keyboard()
            time.sleep(0)







if __name__ == "__main__":
    main()


#cv2.imwrite("lewa_kolumna_6.png",names_images["camera_left_pillar"])
#cv2.imwrite("lewy_blotnik_6.png",names_images["camera_left_fender"])
"""
T_center_to_camera = front_top_T
name = "camera_front_top"
results = model(names_images[name],half=True,device = 0,classes = [2,10,11,12,14,72],conf=0.6)

annotated_frame = results[0].plot()

K = cam_matrices[name]
#Ekstrakcja bounding boxów w formacie [x1, y1, x2, y2]
det_dim = results[0].boxes.xyxy.cpu().numpy()

for det in det_dim:
   x1, y1, x2, y2 = det[:4]
   w = x2 - x1
   h = y2 - y1
   x = x1
   y = y1

   bbox = (int(x), int(y), int(w), int(h))

   anchor_world, side = classify_object_position_and_anchor(bbox, K, T_center_to_camera,name)
   if anchor_world is None:
       continue

   box_world = create_3d_box(anchor_world, side)



   image_points = project_points_world_to_image(box_world, T_center_to_camera, K)

   # Rysowanie 3D boxa
   # Rysowanie dołu
   for i in range(4):
       safe_line(annotated_frame, image_points, i, (i + 1) % 4, (0, 255, 0), 2)

   # Rysowanie góry
   for i in range(4, 8):
       safe_line(annotated_frame, image_points, i, 4 + (i + 1) % 4, (0, 0, 255), 2)

   # Piony
   for i in range(4):
       safe_line(annotated_frame, image_points, i, i + 4, (255, 0, 0), 2)

cv2.namedWindow("yolo", cv2.WINDOW_NORMAL)
cv2.imshow("yolo", annotated_frame)
cv2.waitKey(1)


cc.solve_chess_size(names_images["camera_front_bumper_wide"],"camera_front_bumper_wide",(7,5))

"""
