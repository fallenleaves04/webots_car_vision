import numpy as np
import cv2
from ultralytics import YOLO

"""
W tym pliku są pomocnicze funkcje, które zajmowały niepotrzebne miejsce
w pliku kontrolera. Odpowiadają za narysowanie brył i punktów pochodzących
ze stereowizji i z YOLO
"""

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
"""
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
