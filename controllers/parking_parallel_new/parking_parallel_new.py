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
import torch
import torchvision.transforms as T

import visualise as vis
import camera_calibration as cc
import park_algo as palg
from scipy.interpolate import splprep, splev
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import matplotlib._pylab_helpers
import time


from ultralytics import YOLO

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
        "camera_front_top", "camera_front_top_add"
    ]
    def calculate_intrinsic_matrix(width, height, fov_rad):
        """
        Calculate the camera intrinsic matrix for given resolution and field of view.

        :param width: Image width in pixels
        :param height: Image height in pixels
        :param fov_rad: Horizontal field of view in radians
        :return: 3x3 intrinsic matrix
        """
        fx = (width / 2) / math.tan(fov_rad / 2)
        fy = fx  # Assuming square pixels (can be adjusted if needed)
        cx = width / 2
        cy = height / 2

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ])

        return K
    for name in camera_names:
        cam = driver.getDevice(name)
        if cam:
            cam.enable(TIME_STEP)
            cameras.append(cam)
            width = cam.getWidth()
            height = cam.getHeight()
            fov_rad = cam.getFov()
            K = calculate_intrinsic_matrix(width, height, fov_rad)
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
    homographies_scaled = [S @ H @ np.linalg.inv(S) for H in homographies]

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


    model = YOLO("yolo11m-seg.pt")
    #
    #
    #
    """
    Dwie poniższe funkcje są do zbudowania pozy kamery i szachownicy
    w postaci macierzy przekształcenia jednorodnego.
    """
    def build_homogeneous_transform(R, t):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        return T

    def build_pose_matrix(position, yaw_deg):

        yaw_rad = np.deg2rad(yaw_deg)
        R = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
            [np.sin(yaw_rad),  np.cos(yaw_rad), 0],
            [0, 0, 1]
        ])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = position
        return T


    """
    Poniższa funkcja jest dla przeliczenia punktu na płaszczyznie
    drogi zgodnie z położeniem kamery w układzie samochodu -
    przelicza kliknięty punkt w pikselach na punkt
    w metrach odnośnie samochodu.
    """
    def get_click_position(event, x, y, flags, param):
        global click_position,image

        T_center_to_camera,K = param
        if event == cv2.EVENT_LBUTTONDOWN:
            click_position = (x, y)
            print(f"Kliknięto na pozycji: {click_position}")

            # Rysowanie punktu na obrazie
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("kamerka", image)
            point_on_ground = pixel_to_world(x, y, K, T_center_to_camera)
            print("Punkt na ziemi w układzie BEV:", point_on_ground)

    """
    Funkcja pixel_to_world robi to, co wskazuje jej nazwa -
    przelicza piksel na obrazie kamery do współrzędnych globalnych
    na podstawie parametrów wewnętrznych i zewnętrznych kamery.
    Parametry zewnętrzne są określone macierzą T_center_to_camera -
    położenie kamery w układzie samochodu.
    """
    def pixel_to_world(u, v, K, T_center_to_camera):
        pixel = np.array([u, v, 1.0])  # Piksel w przestrzeni obrazu (homogeniczny)
        ray_camera = np.linalg.inv(K) @ pixel
        ray_camera = ray_camera / np.linalg.norm(ray_camera)  # Normalizowanie

        ray_world = T_center_to_camera[:3, :3] @ ray_camera
        camera_position = T_center_to_camera[:3, 3]

        if ray_world[2] == 0:
            return None  # Promień równoległy do ziemi, brak przecięcia

        t = -camera_position[2] / ray_world[2]
        point_on_ground = camera_position + t * ray_world

        return point_on_ground  # Punkt na ziemi (X, Y, 0)

    click_position = None
    global image

    """
    Przelicza punkty z układu globalnego 3D na obraz kamery 2D.
    Posługuje się macierzą kamery oraz jej połozeniem
    w układzie globalnym (samochodu).
    """
    def project_points_world_to_image(points_world, T_world_to_camera, K):
        projected_points = []

        # Inverse -> Camera <- World
        T_camera_to_world = np.linalg.inv(T_world_to_camera)

        for point in points_world:
            point_h = np.append(point, 1)  # homogeneous
            point_in_camera = T_camera_to_world @ point_h
            Xc, Yc, Zc = point_in_camera[:3]

            if Zc <= 0:
                continue  # behind camera

            # Project
            p_image = K @ np.array([Xc, Yc, Zc])
            u = p_image[0] / p_image[2]
            v = p_image[1] / p_image[2]
            projected_points.append((int(u), int(v)))

        return projected_points

    
    """
    Automatyczne utworzenie punktów podstawy górnej oraz dolnej
    bryły o danym punkcie zaczepienia i ustalonych wymiarach.
    Pierwsza próba wizualizacji brył na obiektach z YOLO.
    
    def create_3d_box(anchor_point, side,length=4.5, width=1.8, height=1.8):
        x, y, z = anchor_point

        dx = length
        dy = width

        if side == 'right':
            base = [
                [x, y, 0],
                [x, y+dy, 0],
                [x + dx, y+dy, 0],
                [x+dx, y, 0]
            ]
        elif side == 'left':
            base = [
                [x, y, 0],
                [x , y-dy, 0],
                [x + dx, y - dy, 0],
                [x +dx, y, 0]
            ]

        top = [[px, py, height] for (px, py, _) in base]
        return base + top  # 8 punktów


    def classify_object_position_and_anchor(bbox, K, T_center_to_camera,camera_name):
        x, y, w, h = bbox

        # Dolny lewy punkt (lewa strona)
        pt_left = pixel_to_world(x, y + h, K, T_center_to_camera)

        # Dolny prawy punkt (prawa strona)
        pt_right = pixel_to_world(x + w, y + h, K, T_center_to_camera)
        # Jeśli którykolwiek z punktów jest None – pomijamy
        if pt_left is None or pt_right is None:
            return None, None

        if pt_right[1] < 0:  # ujemne Y → prawa strona
            anchor = pt_right
            side = "right"
        else:
            anchor = pt_left
            side = "left"

        if camera_name == "camera_front_top" or camera_name == "camera_rear":
            anchor = anchor
            side = side
        elif camera_name == "camera_right_pillar":
            anchor = pt_right
            side = "right"
        elif camera_name == "camera_left_pillar":
            anchor = pt_left
            side = "left"
        elif camera_name == "camera_right_fender":
            anchor = pt_left
            side = "left"
        elif camera_name == "camera_left_fender":
            anchor = pt_right
            side = "right"


        return anchor, side
    
    Pomocnicza funkcja, która nie pozwala na rysowanie
    punktów za obrazem (dla pominięcia błędów).
    
    def safe_line(img, pts, i, j, color, thickness=2):
        if i < len(pts) and j < len(pts):
            cv2.line(img, pts[i], pts[j], color, thickness)

    """

    #car_res = cv2.resize(car, (width, height + 1000), interpolation=cv2.INTER_LINEAR)
    def disp_to_cam3D(u, v, d, K, B):
        if d <= 0:
            return None  # brak danych

        f = K[0, 0]
        cx = K[0, 2]
        cy = K[1, 2]

        Z = f * B / d
        X = (u - cx) * Z / f
        Y = (v - cy) * Z / f
        return np.array([X, Y, Z])  # w układzie kamery

    def vector_angle(v):
        # Kąt między wektorem v a osią Z pojazdu (0,0,1)
        # v = [x,y,z]
        z_axis = np.array([0,0,1])
        v_norm = v / np.linalg.norm(v)
        cos_theta = np.clip(np.dot(v_norm, z_axis), -1, 1)
        angle = np.arccos(cos_theta)  # w radianach
        return angle
    """
    def find_extreme_points(mask):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None, None
        left_x = np.min(xs)
        right_x = np.max(xs)
        # Wybieramy dowolne y dla lewej i prawej krawędzi spośród pikseli maski
        left_candidates = ys[xs == left_x]
        right_candidates = ys[xs == right_x]
        # Wybieramy środkowe y, żeby mieć reprezentatywny punkt (np. medianę)
        left_y = int(np.median(left_candidates))
        right_y = int(np.median(right_candidates))
        return (left_x, left_y), (right_x, right_y)
    """
    def points_from_mask_to_3D(mask_resized, filtered_disp, K, baseline, T_cam_to_car):
        """
        Given a segmentation mask and filtered disparity map, compute 3D coordinates of the left and right extreme points of the object.

        Args:
            mask_resized (np.ndarray): Binary mask of shape (H, W)
            filtered_disp (np.ndarray): Disparity map (float32) of shape (H, W)
            K (np.ndarray): Intrinsic matrix of the right (or left) camera
            baseline (float): Stereo baseline in meters (e.g., 0.03 m)
            T_cam_to_car (np.ndarray): 4x4 transformation matrix from camera to car frame

        Returns:
            tuple: Two 3D points (np.ndarray) in car coordinate frame or (None, None) if no points found
        """
       
        # Step 1: Find leftmost and rightmost mask points
        
        ys, xs = np.where(mask_resized > 0)
        if len(xs) == 0:
            return None, None

        left_x = np.min(xs)
        right_x = np.max(xs)

        left_y = int(np.median(ys[xs == left_x]))
        right_y = int(np.median(ys[xs == right_x]))
        
        
        def get_valid_disparity(disp_map, x, y, window=30):
            h, w = disp_map.shape
            x0 = max(0, x - window)
            x1 = min(w, x + window + 1)
            y0 = max(0, y - window)
            y1 = min(h, y + window + 1)
            patch = disp_map[y0:y1, x0:x1]
            valid = patch[patch > 0]
            if valid.size == 0:
                return None
            return np.median(valid)
        # Step 2: Get disparity at these points
        
        disp_left = get_valid_disparity(filtered_disp, left_x, left_y)
        disp_right = get_valid_disparity(filtered_disp, right_x, right_y)
        
        if disp_left is None or disp_right is None:
            return None, None
        if disp_left <= 0 or disp_right <= 0:
            return None, None

        # Step 3: Reproject using pinhole stereo model
        f = K[0, 0]
        cx = K[0, 2]
        cy = K[1, 2]

        # Point 1 (left)
        Z1 = f * baseline / disp_left
        X1 = (left_x - cx) * Z1 / f
        Y1 = (left_y - cy) * Z1 / f
        
        point_cam1 = np.array([X1, Y1, Z1, 1.0])

        # Point 2 (right)
        Z2 = f * baseline / disp_right
        X2 = (right_x - cx) * Z2 / f
        Y2 = (right_y - cy) * Z2 / f
        
        point_cam2 = np.array([X2, Y2, Z2, 1.0])

        # Step 4: Transform to car frame
        point_car1 = T_cam_to_car @ point_cam1
        point_car2 = T_cam_to_car @ point_cam2

        return point_car1[:3], point_car2[:3]

    

    def get_ground_box_corners(p1, p2):
        """
        Tworzy narożniki prostokąta na ziemi na podstawie dwóch punktów 3D.
        """
        p1 = np.array(p1)
        p2 = np.array(p2)

        x1, y1 = p1[:2]
        x2, y2 = p2[:2]

        return [
            [x1, y1, 0],
            [x1, y2, 0],
            [x2, y2, 0],
            [x2, y1, 0]
        ]
    def draw_ground_box_on_image(image, box_corners, T_car_to_cam, K, color):
        proj_pts = project_points_world_to_image(box_corners, T_car_to_cam, K)
        if len(proj_pts) == 4:
            for j in range(4):
                pt1 = proj_pts[j]
                pt2 = proj_pts[(j + 1) % 4]
                cv2.line(image, pt1, pt2, color, 2)

    
    
    yaw_init = 0.0
    first = True
    parker = None
    plotter = None

    SENSOR_INTERVAL = 0.05
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



                        p1_3d, p2_3d = points_from_mask_to_3D(mask_resized, filtered_disp, K_right, 0.03, T_center_to_camera)
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
                            p1_3d_px = project_points_world_to_image([p1_3d], T_center_to_camera, K)
                            p2_3d_px = project_points_world_to_image([p2_3d], T_center_to_camera, K)
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

                                box_corners = get_ground_box_corners(p1_3d, p2_3d)
                                draw_ground_box_on_image(right_copy, box_corners, T_center_to_camera, K_right, color_tuple)

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
