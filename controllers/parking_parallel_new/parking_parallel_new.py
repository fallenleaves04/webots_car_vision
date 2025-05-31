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
from ultralytics import YOLO

#from scipy.interpolate import splprep, splev
#from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import matplotlib._pylab_helpers
import time
import threading, queue

# --------------------- Stałe ---------------------
TIME_STEP = 128
NUM_DIST_SENSORS = 12
NUM_CAMERAS = 8
MAX_SPEED = 250.0
CAMERA_HEIGHT=2160
CAMERA_WIDTH=3840
global parking

SENSOR_INTERVAL = 0.06
IMAGE_INTERVAL  = 0.2
KEYBOARD_INTERVAL = 0.04

# Parametry samochodu, charakterystyki z symulatora
TRACK_FRONT = 1.628
TRACK_REAR = 1.628
WHEELBASE = 2.995
MAX_WHEEL_ANGLE = 0.5  # rad
CAR_WIDTH = 2.302
CAR_LENGTH = 4.85

# --------------------- Zmienne globalne ---------------------

driver = Driver()

#robot = Robot()
#supervisor = Supervisor()
display = Display('display')
keyboard = Keyboard()
keyboard.enable(TIME_STEP)
gps = None
Driver.synchronization = False
cameras = []
print(driver.synchronization)
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
    print("Naciśnij klawisz P, aby rozpocząć poszukiwanie miejsca")
    print("Podczas parkowania, wciśnij Q aby szukać z prawej strony")
    print("albo E aby szukać miejsca z lewej strony")

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



#----------------------Sensor functions-----------------

camera_names = [
        "camera_front_bumper_wide","camera_rear",
        "camera_left_fender", "camera_right_fender",
        "camera_left_pillar",  "camera_right_pillar",
        "camera_front_top", "camera_front_top_add",
        "camera_helper"
    ]

def get_camera_image(camera):
    width = camera.getWidth()
    height = camera.getHeight()
    img = camera.getImage()
    if img is None:
        return None

    img_array = np.frombuffer(img, np.uint8).reshape((height, width, 4))[:, :, :3]
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    return img_array

sensor_names = [
        "distance sensor left front side", "distance sensor front left", "distance sensor front lefter",
        "distance sensor front righter", "distance sensor front right", "distance sensor right front side",
        "distance sensor left side", "distance sensor left", "distance sensor lefter",
        "distance sensor righter", "distance sensor right", "distance sensor right side"
    ]

def process_distance_sensors(sen):
    l_dist = sen.getLookupTable()
    a_dist = (l_dist[0]-l_dist[3])/(l_dist[1]-l_dist[4])
    b_dist = l_dist[3]-l_dist[4]*a_dist
    value = sen.getValue()
    distance = a_dist*value+b_dist
    sigma = l_dist[2]
    noisy_distance = distance + np.random.normal(0, sigma)
    return noisy_distance


# --------------------- Main Controller Loop ---------------------
def main():

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

    # Inicjalizuj kamery
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

    # GPS inicjalizacja
    gps = driver.getDevice("gps")
    if gps:
        gps.enable(TIME_STEP)
        print("GPS enabled")
    #Inicjalizacja IMU
    imu = driver.getDevice("inertial unit")
    if imu:
        imu.enable(TIME_STEP)
        print("IMU enabled")


    print_help()



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


    prev_x = 0
    prev_y = 0



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

    model = YOLO("yolo11m-seg.pt")
    #
    #
    #

    last_sensor_time = driver.getTime()
    last_image_time  = driver.getTime()
    last_key_time = driver.getTime()


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


    #----------------------------------------- Dalej wątki ---------------------------------------------------

    plt.ion()
    fig, ax_cones, ax_live = None, None, None
    parking = False
    parker = None
    def check_keyboard():
        nonlocal parker,parking, fig, ax_cones, ax_live
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
            if parking:
                # Utwórz okno i osie tylko raz

                fig, (ax_cones,ax_live) = plt.subplots(1, 2, figsize=(20,10))
                #odkomentować niżej jeżeli chce się tylko wszystkie czujniki
                #fig, ax_cones = plt.subplots(1, 1, figsize=(12,12))
                fig.suptitle("Parkowanie")

                print("Rozpoczęto parking")
            else:

                plt.close(fig)
                cv2.destroyAllWindows()
                print("Ukończono parking")
        if parker is not None:
            if parking and parker.state != "parking":
                if key in (ord('Y'),ord('y')):
                    parker.state = "parking"
                elif key in (ord('N'),ord('n')):
                    parker.state == "searching_start"
                    print("Rozpoczynam od nowa...")
                elif key in (ord('Q'),ord('q')):
                    print("Szukam z lewej strony...")
                    parker.side = "left"
                elif key in (ord('E'),ord('e')):
                    print("Szukam z prawej strony...")
                    parker.side = "right"

    first_call = True
    odom = 0.0
    spot = 0.0
    while driver.step() != -1:

        now = driver.getTime()

        if parking:
            # jeżeli z klawiatury włączono parkowanie, to:


            if now - last_sensor_time >= SENSOR_INTERVAL:
                # co SENSOR_INTERVAL (zadany na początku) wykonujemy:
                last_sensor_time = now
                if first_call:
                    plotter = palg.LivePlotter(ax_live)
                    parker = palg.Parking(driver, "left", now)
                    yaw_init = imu.getRollPitchYaw()[2]
                first_call = False
                # odczyt odległości
                dists = [process_distance_sensors(s) for s in distance_sensors]
                names = dict(zip(sensor_names, dists))

                # draw_cones na ax_cones
                vis.draw_cones(ax_cones, fig, dists)

                # live-plot na ax_live
                if (parker.side == "left"):
                    plotter.val = names["distance sensor left front side"]
                elif (parker.side == "right"):
                    plotter.val = names["distance sensor right front side"]
                plotter.update(0)

                # automaty parkowania
                yaw = imu.getRollPitchYaw()[2] - yaw_init
                parker.update_state(names, yaw)

                if parker.state == "parking":
                    odom, spot = parker.update_state(names, yaw)
                    if (parker.side == "left"):
                        parker.exec_path(odom, spot, names["distance sensor left front side"])
                    elif (parker.side == "right"):
                        parker.exec_path(odom, spot, names["distance sensor right front side"])
                #to niżej do wizualizacji - zakomentować razem z "plotter"
                #i "vis.draw_cones" jeżeli nie chce się wizualizacji
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            
            # co IMAGE_INTERVAL – przetwarzanie obrazów
            if now - last_image_time >= IMAGE_INTERVAL:
                last_image_time = now
                images = [get_camera_image(c) for c in cameras]
                names_images = dict(zip(camera_names, images))

                viss = vis.alt_collect_homo(names_images, homographies, car, streams)
                #cv2.imwrite("img3_vis.png",viss)

            cv2.waitKey(1)
            
        elif not parking:
            first_call = True
        if now - last_key_time >= KEYBOARD_INTERVAL:
            last_key_time = now
            check_keyboard()







if __name__ == "__main__":
    main()

#cv2.imwrite("lewa_kolumna_6.png",names_images["camera_left_pillar"])
#cv2.imwrite("lewy_blotnik_6.png",names_images["camera_left_fender"])

#SKOPIOWAĆ DO PĘTLI, JEŻELI CHCE SIĘ UŻYĆ STAREJ WERSJI Z BRYŁAMI YOLO
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

   anchor_world, side = sy.classify_object_position_and_anchor(bbox, K, T_center_to_camera,name)
   if anchor_world is None:
       continue

   box_world = sy.create_3d_box(anchor_world, side)



   image_points = sy.project_points_world_to_image(box_world, T_center_to_camera, K)

   # Rysowanie 3D boxa
   # Rysowanie dołu
   for i in range(4):
       sy.safe_line(annotated_frame, image_points, i, (i + 1) % 4, (0, 255, 0), 2)

   # Rysowanie góry
   for i in range(4, 8):
       sy.safe_line(annotated_frame, image_points, i, 4 + (i + 1) % 4, (0, 0, 255), 2)

   # Piony
   for i in range(4):
       sy.safe_line(annotated_frame, image_points, i, i + 4, (255, 0, 0), 2)

cv2.namedWindow("yolo", cv2.WINDOW_NORMAL)
cv2.imshow("yolo", annotated_frame)
cv2.waitKey(1)

"""
