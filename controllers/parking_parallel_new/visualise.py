"""
plik z funkcjami pomocniczymi wizualizacyjnymi
"""
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import cv2 as cv
import camera_calibration as cc
from scipy.linalg import logm, expm, sqrtm

# Vehicle parameters
TRACK_FRONT = 1.628
TRACK_REAR = 1.628
WHEELBASE = 2.995
MAX_WHEEL_ANGLE = 0.5  # rad
CAR_WIDTH = 1.95
CAR_LENGTH = 4.85

global s
s = 2 # skala, jeżeli obrazy były wykonane przy innej rozdzielczości

def draw_cones(ax,fig,distances):
    """
    Do wizualizacji strefy detekcji czujników ultradzwiękowych.
    Jest wywoływana za każdym razem w pętli kontrolera - rysuje na pustej kanwie
    matplotlib'a i 
    """
    ax.clear()

    front_sensor_angles = [90,45,15,-15,-45,-90]
    rear_sensor_angles = [90,135,180,180,-135,-90]

    sensor_positions = np.array([
    [ 3.515873,  0.865199],  # front left side
    [ 3.588074,  0.81069 ],  # front left
    [ 3.799743,  0.375011],  # front lefter
    [ 3.799743, -0.375011],  # front righter
    [ 3.588074, -0.81069 ],  # front right
    [ 3.515873, -0.865199],  # front right side
    [-0.505871,  0.923198],  # left side
    [-0.845978,  0.798194],  # left
    [-0.929999,  0.32    ],  # lefter
    [-0.930001, -0.32    ],  # righter
    [-0.840982, -0.789534],  # right
    [-0.505875, -0.9232  ]   # right side
    ])

    sensor_angles = front_sensor_angles + rear_sensor_angles
    sensor_fovs_front = [45]*6
    sensor_fovs_rear = [45]*6
    sensor_fovs = sensor_fovs_front + sensor_fovs_rear
    # obrót do osi współrzędnych
    rot = np.array([[0,-1],[1,0]])
    sensor_positions_rot = sensor_positions @ rot.T
    sensor_angles_rot = np.array(sensor_angles) + 90


    for pos, angle_deg, fov_deg, dist in zip(sensor_positions_rot, sensor_angles_rot, sensor_fovs, distances):
        angle_rad = np.radians(angle_deg)
        fov_rad = np.radians(fov_deg)

        theta = np.linspace(angle_rad - fov_rad/2, angle_rad + fov_rad/2, 30)
        x = pos[0] + dist * np.cos(theta)
        y = pos[1] + dist * np.sin(theta)

        ax.fill(np.append(pos[0], x), np.append(pos[1], y), color='lightblue', alpha=0.3)
        ax.plot([pos[0], x[0]], [pos[1], y[0]], 'b:', alpha=0.5)
        ax.plot([pos[0], x[-1]], [pos[1], y[-1]], 'b:', alpha=0.5)
        ax.plot(pos[0], pos[1], 'ko')

    ax.add_patch(plt.Rectangle(
        (-CAR_WIDTH/2, -1), CAR_WIDTH, CAR_LENGTH,
        edgecolor='gray', facecolor='lightgray', alpha=0.8
    ))

    ax.set_xlim(-8, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    ax.set_title("Czujniki ultradźwiękowe")
    ax.grid(True)
    fig.canvas.draw()
    fig.canvas.flush_events()



def draw_distance(distances):
    #patrzymy z przodu, oś symetrii patrzy do przodu
    front_sensor_angles = [90,45,15,-15,-45,-90]
    #tutaj liczymy dodatnio przeciw wskazówek, ujemnie - ze wskazówkami
    rear_sensor_angles = [90,135,180,180,-135,-90]
    #rozmieszczenia czujników
    sensor_positions = np.array([
    [ 3.515873,  0.865199],  # front left side
    [ 3.588074,  0.81069 ],  # front left
    [ 3.799743,  0.375011],  # front lefter
    [ 3.799743, -0.375011],  # front righter
    [ 3.588074, -0.81069 ],  # front right
    [ 3.515873, -0.865199],  # front right side
    [-0.505871,  0.923198],  # left side
    [-0.845978,  0.798194],  # left
    [-0.929999,  0.32    ],  # lefter
    [-0.930001, -0.32    ],  # righter
    [-0.840982, -0.789534],  # right
    [-0.505875, -0.9232  ]   # right side
    ])
    sensor_fovs_front = [90,90,90,90,90,90]
    sensor_fovs_rear = [90,90,90,90,90,90]
    angles_deg = front_sensor_angles + rear_sensor_angles
    angles_rad = np.radians(angles_deg)
    endpoints = sensor_positions + np.stack(
        [distances * np.cos(angles_rad), distances * np.sin(angles_rad)],
        axis=-1
    )

    polygon_points = np.concatenate([endpoints[:6], endpoints[6:][::-1]])


    rot = np.array([[0, -1], [1, 0]])
    rotated_points = polygon_points @ rot.T
    rotated_sensor_positions = sensor_positions @ rot.T


    # zamiast dokładać closure point, pracujemy tylko na unikalnych
    pts = rotated_points  # shape (N,2), bez dodatkowego powtórzenia

    # przygotowujemy listę x,y
    x, y = pts[:,0], pts[:,1]

    # SPRÓBUJ splprep, a jak się nie uda, to fallback na moving average
    try:
        tck, u = splprep([x, y], s=0, per=True, k=3)
        unew = np.linspace(0, 1.0, 200)
        out = splev(unew, tck)
        xs, ys = out[0], out[1]
    except ValueError:
        # Fallback: prosta średnia ruchoma z oknem 5 punktów (cykliczne)
        xs = uniform_filter1d(x, size=5, mode='wrap')
        ys = uniform_filter1d(y, size=5, mode='wrap')


    # Rysowanie
    plt.clf()
    plt.fill(xs, ys, color='lightblue', alpha=0.5)
    plt.plot(rotated_sensor_positions[:, 0],
             rotated_sensor_positions[:, 1], 'ko')
    plt.plot(xs, ys, 'b--')

    # Twój prostokąt-auta, osie itd.
    plt.gca().add_patch(plt.Rectangle(
        (-CAR_WIDTH/2, -1), 2.302, 4.85,
        edgecolor='gray', facecolor='lightgray', alpha=0.8, zorder=0
    ))
    plt.grid(True)
    plt.xlim(-6, 6)
    plt.ylim(-10, 10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Strefa detekcji")
    plt.pause(0.001)


def collect_homo(names_images,homographies,car,streams):
    """
    DRUGI SPOSÓB
    """





    h, w = int(3600/s),int(3600/s)

    (stream1,stream2,stream3,stream4,
    stream5,stream6,stream7,stream8,
    stream9,stream10,stream11,stream12,stream13) = streams
    (front_H,right_H,right_fender_H,
    rear_H,left_fender_H,left_H)= homographies
    imgs = []
    # warp CUDA wszystkie kamery
    left = warp_with_cuda(names_images["camera_left_pillar"], left_H, "left homo", h, w,stream1)
    right = warp_with_cuda(names_images["camera_right_pillar"], right_H, "right homo", h, w,stream2)
    rear = warp_with_cuda(names_images["camera_rear"], rear_H, "rear homo", h, w,stream3)
    front = warp_with_cuda(names_images["camera_front_bumper_wide"], front_H, "front homo", h, w,stream4)
    # front_wind = warp_with_cuda(names_images["camera_front_top"], front_wind_H, "front wind homo", h, w)
    right_fender = warp_with_cuda(names_images["camera_right_fender"], right_fender_H, "right fender homo", h, w,stream5)
    left_fender = warp_with_cuda(names_images["camera_left_fender"], left_fender_H, "left fender homo", h, w,stream6)
    #imgs.extend([left,right,rear,front,

    S = np.array([[1/s,0,0],[0,1/s,0],[0,0,1]]).astype(np.float32)

    right_to_front_H = np.array([[ 1.0103254e+00,  8.9369901e-03,  2.4237607e+03],
 [-4.8033521e-03,  1.0200684e+00 , 1.8881282e+03],
 [-1.9079014e-06,  2.8290551e-06,  1.0000000e+00]]).astype(np.float32)
    right_to_front_H = S @ right_to_front_H @ np.linalg.inv(S)

    left_to_front_H = np.array([[ 1.0053335e+00 ,-6.1842590e-04, -2.4580974e+03],
 [-5.9892777e-03,  1.0021796e+00,  1.8966011e+03],
 [-2.1722203e-06, -4.8802093e-07,  1.0000000e+00]]).astype(np.float32)
    left_to_front_H = S @ left_to_front_H @ np.linalg.inv(S)



    left_to_rear_H = np.array([[ 9.7381920e-01, -2.3735706e-03, -2.2782476e+03],
 [ 4.3283560e-04,  9.7134042e-01, -1.7275361e+03],
 [-3.2049848e-06,  5.5537777e-07,  1.0000000e+00]]).astype(np.float32)

    right_to_rear_H = np.array([[ 9.8992777e-01,  1.6606528e-02,  4.6191958e+03],
 [-2.1194941e-03 , 9.9684310e-01, -1.7071634e+01],
 [-7.0434027e-07,  3.0691269e-06 , 1.0000000e+00]]).astype(np.float32)



    left_to_rear_H = S @ left_to_rear_H @ np.linalg.inv(S)
    right_to_rear_H = S @ right_to_rear_H @ np.linalg.inv(S)

    rear_to_front_H = np.array([[ 1.1127317e+00,  7.0112962e-03 ,-8.0307449e+01],
 [ 5.2037694e-02,  1.0801761e+00,  4.4567339e+03],
 [ 1.0615680e-05,  6.2511299e-07,  1.0000000e+00]]).astype(np.float32)
    rear_to_front_H = S @ rear_to_front_H @ np.linalg.inv(S)

    canvas_front = blend_warp_GPUONLY(front,right,right_to_front_H,stream7)
    canvas_front = blend_warp_GPUONLY(canvas_front,left,left_to_front_H,stream8)
    canvas_rear = blend_warp_GPUONLY(rear,left_fender,left_to_rear_H,stream9)
    canvas_rear = blend_warp_GPUONLY(canvas_rear,right_fender,right_to_rear_H,stream10)
    canvas = blend_warp_GPUONLY(canvas_front,canvas_rear,rear_to_front_H,stream11)

    canvas = canvas.download()

    cv.namedWindow("bev",cv.WINDOW_NORMAL)
    cv.imshow("bev",canvas)
    #cv.imwrite("img_vis.jpg",cv.cvtColor(canvas,cv.COLOR_BGR2RGB))
    """
    crop_scale = 0.5
    margin = 0
    ch,cw = canvas.shape[0], canvas.shape[1]
    crop_h = int(ch * crop_scale / 2)
    crop_w = int(cw * crop_scale / 2)

    center_h, center_w = ch // 2, cw // 2

    y1 = max(center_h - crop_h - margin, 0)
    y2 = min(center_h + crop_h + margin, h)
    x1 = max(center_w - crop_w - margin, 0)
    x2 = min(center_w + crop_w + margin, w)

    cropped = canvas[y1:y2, x1:x2]


    """
    #DZIALA TYLKO DLA FULLHD OBRAZOW, S=2
    cropped = canvas
    h, w = cropped.shape[:2]

    crop_top_px    = int(0.184 * h)
    crop_bottom_px = int(0.215 * h)
    crop_left_px = int(0.14 * w)
    crop_right_px = int(0.14 * w)

    y1 = crop_top_px
    y2 = h - crop_bottom_px
    x1 = crop_left_px
    x2 = w - crop_right_px
    cropped = cropped[y1:y2, x1:x2]

    scalex = 0.18 # np. 20% szerokości BEV
    scaley = 0.45
    bev_h, bev_w = cropped.shape[:2]
    new_w = int(bev_w * scalex)
    new_h = int(bev_h * scaley)
    car_resized = cv.resize(car, (new_w, new_h), interpolation=cv.INTER_AREA)

    # 3. Oblicz pozycję tak, by wstawić go na środku
    x_offset = (bev_w - new_w) // 2 + 15
    y_offset = (bev_h - new_h) // 2 - 15

    cropped[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = car_resized


    cv.namedWindow("bev",cv.WINDOW_NORMAL)
    cv.imshow("bev",cropped)
    cv.imwrite("img_vis.png",cropped)
    #H,_ = cc.chess_homography(canvas,names_images["camera_helper"],(10,6))

    return canvas

def test_homo(names_images,homographies,streams):
    h, w = int(3600/s),int(3600/s)

    (stream1,stream2,stream3,stream4,
    stream5,stream6,stream7,stream8,
    stream9,stream10,stream11,stream12,stream13) = streams
    (front_H,right_H,right_fender_H,
    rear_H,left_fender_H,left_H)= homographies
    imgs = []
    # warp CUDA wszystkie kamery
    left = warp_with_cuda(names_images["camera_left_pillar"], left_H, "left homo", h, w,stream1)
    right = warp_with_cuda(names_images["camera_right_pillar"], right_H, "right homo", h, w,stream2)
    rear = warp_with_cuda(names_images["camera_rear"], rear_H, "rear homo", h, w,stream3)
    front = warp_with_cuda(names_images["camera_front_bumper_wide"], front_H, "front homo", h, w,stream4)
    # front_wind = warp_with_cuda(names_images["camera_front_top"], front_wind_H, "front wind homo", h, w)
    right_fender = warp_with_cuda(names_images["camera_right_fender"], right_fender_H, "right fender homo", h, w,stream5)
    left_fender = warp_with_cuda(names_images["camera_left_fender"], left_fender_H, "left fender homo", h, w,stream6)

    H,_ = cc.chess_homography(left,front,(7,5))
    if H is not None:
        vis = warp_and_blend_gpu(front,left,H)
        cv.imwrite("img_vis.jpg",cv.cvtColor(vis,cv.COLOR_BGR2RGB))


def alt_collect_homo(names_images,homographies,car,streams):
    """
    TRZECI SPOSOB
    """
    h, w = int(3600/s),int(3600/s)

    (stream1,stream2,stream3,stream4,
    stream5,stream6,stream7,stream8,
    stream9,stream10,stream11,stream12,stream13) = streams
    (front_H,right_H,right_fender_H,
    rear_H,left_fender_H,left_H)= homographies
    imgs = []
    # warp CUDA wszystkie kamery
    left = warp_with_cuda(names_images["camera_left_pillar"], left_H, "left homo", h, w,stream1)
    right = warp_with_cuda(names_images["camera_right_pillar"], right_H, "right homo", h, w,stream2)
    rear = warp_with_cuda(names_images["camera_rear"], rear_H, "rear homo", h, w,stream3)
    front = warp_with_cuda(names_images["camera_front_bumper_wide"], front_H, "front homo", h, w,stream4)
    # front_wind = warp_with_cuda(names_images["camera_front_top"], front_wind_H, "front wind homo", h, w)
    right_fender = warp_with_cuda(names_images["camera_right_fender"], right_fender_H, "right fender homo", h, w,stream5)
    left_fender = warp_with_cuda(names_images["camera_left_fender"], left_fender_H, "left fender homo", h, w,stream6)

    #CZĘŚĆ KODU DO SPORZĄDZENIA HOMOGRAFII RZUTOWANIA NA KANWĘ

    canvas_size = (6000,6000)
    canvas_cpu = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    canvas = cv.cuda_GpuMat()
    canvas.upload(canvas_cpu,stream7)

    px = np.array([[0,0],[6000/s,0],[6000/s,6000/s],[0,6000/s]]).astype(np.float32)
    met = np.array([[10,10],[10,-10],[-10,-10],[-10,10]],dtype = np.float32) #

    H_px_to_m_bev,_ = cv.findHomography(met,px,cv.RANSAC,5.0)

    S = np.array([[1/s,0,0],[0,1/s,0],[0,0,1]]).astype(np.float32)
    H_left_to_bev= np.array([[5.78196165e-01, 1.36768422e-03 ,5.75901552e+02],
 [2.08805960e-03, 5.78244815e-01, 1.31412439e+03],
 [9.24389963e-07 ,7.19948103e-07, 1.00000000e+00]],dtype=np.float32)
    H_left_to_bev = S @ H_left_to_bev @ np.linalg.inv(S)

    H_right_to_bev = np.array([[ 5.74277218e-01, -8.39567193e-05,  3.34911956e+03],
 [ 3.93911249e-06,  5.73836613e-01,  1.31777314e+03],
 [ 1.31049336e-07, -1.42342160e-07,  1.00000000e+00]],dtype=np.float32)
    H_right_to_bev = S @ H_right_to_bev @ np.linalg.inv(S)

    H_left_fender_to_bev = np.array([[ 5.60635651e-01,  3.13768463e-02,  6.29376393e+02],
 [-1.14353803e-02,  6.03217079e-01, 2.80548010e+03],
 [-4.06125598e-06,  1.18455527e-05,  1.00000000e+00]],dtype=np.float32)
    H_corr_lf = np.array(
    [[ 9.95078761e-01, -5.70508469e-04,  5.23265217e+00],
     [-3.64430112e-03,  9.96849037e-01,  7.87846839e+00],
     [-1.04463326e-06,-2.72402103e-07,  1.00000000e+00]]).astype(np.float32)
    H_left_fender_to_bev = H_corr_lf @ H_left_fender_to_bev
    H_left_fender_to_bev = S @ H_left_fender_to_bev @ np.linalg.inv(S)

    H_right_fender_to_bev = np.array([[ 5.68654217e-01,  3.69548635e-02,  3.30027369e+03],
 [-5.18751953e-03,  6.04990446e-01 , 2.80305923e+03],
 [-2.20235231e-06,  1.11229040e-05 , 1.00000000e+00]],dtype=np.float32)
    H_corr_rf = np.array(
    [[ 9.86315188e-01, -3.03481721e-04,  2.78378345e+01],
     [-5.49422481e-03,  9.93028602e-01,  2.23023238e+01],
     [-1.63834578e-06, -8.44384115e-08 , 1.00000000e+00]]).astype(np.float32)
    H_right_fender_to_bev = H_corr_rf @ H_right_fender_to_bev
    H_right_fender_to_bev = S @ H_right_fender_to_bev @ np.linalg.inv(S)

    H_rear_to_bev = np.array([[5.72595558e-01, 2.77600165e-02 ,1.96983903e+03],
 [2.49365825e-04, 6.06721803e-01, 3.77306058e+03],
 [6.15294083e-08, 9.20599720e-06, 1.00000000e+00]],dtype=np.float32)
    H_corr_rear = np.array([[ 9.86009607e-01, -5.12444388e-05,  4.22106295e+01],
 [ 6.09014276e-05,  9.87099849e-01 , 4.70971490e+01],
 [ 1.79688786e-08, -1.22849686e-08,  1.00000000e+00]]).astype(np.float32)
    H_rear_to_bev = H_corr_rear @ H_rear_to_bev
    H_rear_to_bev = S @ H_rear_to_bev @ np.linalg.inv(S)

    H_front_to_bev = np.array([[5.78772185e-01, 3.67819798e-03, 1.95954679e+03],
 [2.99076766e-04, 5.83709774e-01, 2.15103801e+02],
 [1.77376103e-07 ,1.22892995e-06, 1.00000000e+00]],dtype = np.float32)
    H_front_to_bev = S @ H_front_to_bev @ np.linalg.inv(S)

    bev_left = blend_warp_GPUONLY(canvas,left,H_left_to_bev,stream7,canvas_size=(6000//s,6000//s))
    bev_right = blend_warp_GPUONLY(bev_left,right,H_right_to_bev,stream8,canvas_size=(6000//s,6000//s))
    bev_left_fender = blend_warp_GPUONLY(bev_right,left_fender,H_left_fender_to_bev,stream9,canvas_size=(6000//s,6000//s))
    bev_right_fender = blend_warp_GPUONLY(bev_left_fender,right_fender,H_right_fender_to_bev,stream10,canvas_size=(6000//s,6000//s))
    bev_rear = blend_warp_GPUONLY(bev_right_fender,rear,H_rear_to_bev,stream11,canvas_size=(6000//s,6000//s))
    bev_front = blend_warp_GPUONLY(bev_rear,front,H_front_to_bev,stream12,canvas_size=(6000//s,6000//s))


    bev = bev_front.download()


    # Granice w metrach (obszar 8x6 metrów -> prostokąt)
    meters = np.array([
        [ 5.6,  6],
        [ 5.6, -6],
        [-5.95, -6],
        [-5.95,  6]
    ], dtype=np.float32)

    # Dodaj trzecią współrzędną (homogeniczne)
    meters_hom = np.hstack([meters, np.ones((meters.shape[0], 1))])

    # Przekształć na piksele
    pxs = (H_px_to_m_bev @ meters_hom.T).T

    # Normalizuj (dziel przez ostatni element, jeśli nie równe 1)
    #pxs /= pxs[:, [2]]

    # Rzutuj na int (piksele)
    pxs_int = pxs[:, :2].astype(int)

    # Ustal minimalne i maksymalne współrzędne pikseli (x_min, x_max, y_min, y_max)
    x_min = np.min(pxs_int[:, 0])
    x_max = np.max(pxs_int[:, 0])
    y_min = np.min(pxs_int[:, 1])
    y_max = np.max(pxs_int[:, 1])

    # UWAGA: Sprawdzenie czy zakres jest w granicach obrazu
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, canvas_cpu.shape[1])
    y_max = min(y_max, canvas_cpu.shape[0])

    # Obcinanie obrazu (ROI -> Region of Interest)
    cropped = bev[y_min:y_max, x_min:x_max]
    # Pokaż wynik
    #cv.namedWindow("ROI Visualization", cv.WINDOW_NORMAL)
    #cv.imshow("ROI Visualization", bev_front)

    """
    # ======= PRZYKŁADOWE PUNKTY PROSTOKĄTA W METRACH =======
    rectangle_meters = np.array([
        [4, 4],   # prawy górny
        [4, -4],  # prawy dolny
        [2.4, -4], # lewy dolny
        [2.4, 4]   # lewy górny
    ], dtype=np.float32)

    pixels = np.array([
    [0, 0],  # lewy górny
    [cropped.shape[1], 0],  # prawy górny
    [cropped.shape[1], cropped.shape[0]],  # prawy dolny
    [0, cropped.shape[0]]  # lewy dolny
    ], dtype=np.int32)
    H_m_to_px, _ = cv.findHomography(meters, pixels)
    # Dodaj homogeniczną współrzędną
    rectangle_hom = np.hstack([rectangle_meters, np.ones((rectangle_meters.shape[0], 1))])

    # Przekształcenie do pikseli
    rectangle_pixels = (H_m_to_px @ rectangle_hom.T).T


    # Rzutuj na int do rysowania
    rectangle_pixels_int = rectangle_pixels[:, :2].astype(int)

    # Rysowanie prostokąta (żółty prostokąt)
    for i in range(4):
        pt1 = tuple(rectangle_pixels_int[i])
        pt2 = tuple(rectangle_pixels_int[(i + 1) % 4])
        cv.line(cropped, pt1, pt2, (0, 255, 255), 4)  # żółty, grubość 4

    # Dla lepszej wizualizacji narysuj też rogi
    for pt in rectangle_pixels_int:
        cv.circle(cropped, tuple(pt), 10, (0, 0, 255), -1)  # czerwone kropki
    """
    cv.namedWindow("bev",cv.WINDOW_NORMAL)
    cv.imshow("bev",cropped)
    return cropped



def chain_collect_homo(names_images,homographies,car,streams):

    h, w = int(3600/s),int(3600/s)

    (stream1,stream2,stream3,stream4,
    stream5,stream6,stream7,stream8,
    stream9,stream10,stream11,stream12,stream13) = streams
    (front_H,right_H,right_fender_H,
    rear_H,left_fender_H,left_H)= homographies
    imgs = []
    # warp CUDA wszystkie kamery
    left = warp_with_cuda(names_images["camera_left_pillar"], left_H, "left homo", h, w,stream1)
    right = warp_with_cuda(names_images["camera_right_pillar"], right_H, "right homo", h, w,stream2)
    rear = warp_with_cuda(names_images["camera_rear"], rear_H, "rear homo", h, w,stream3,show=True)
    front = warp_with_cuda(names_images["camera_front_bumper_wide"], front_H, "front homo", h, w,stream4)
    # front_wind = warp_with_cuda(names_images["camera_front_top"], front_wind_H, "front wind homo", h, w)
    right_fender = warp_with_cuda(names_images["camera_right_fender"], right_fender_H, "right fender homo", h, w,stream5)
    left_fender = warp_with_cuda(names_images["camera_left_fender"], left_fender_H, "left fender homo", h, w,stream6)

    #cv.imwrite("lewa_kolumna_6.jpg",left)
    #cv.imwrite("lewy_blotnik_6.jpg",left_fender)

    #H1 - lewa do frontalnej,
    #H2 - H1 do prawej
    #H3 - H2 do prawego błotnika
    #H4 - tylna zarówno do prawego, jak i lewego błotnika (dołączono )
    #H5 - lewy błotnik ostatni,
    S = np.array([[1/s,0,0],[0,1/s,0],[0,0,1]]).astype(np.float32)

    H1 = np.array([[      1.013,   0.0036699,       -2443],
 [   0.008471,     1.0221,      1880.2],
 [ 3.1404e-06,  5.2511e-06,           1]]).astype(np.float32)
    H1 = S @ H1 @ np.linalg.inv(S)
    H2 =np.array([[    0.99331,    0.024914 ,     4847.5],
 [ -0.0033566,      1.0098,        1888],
 [-1.2409e-06,  4.6596e-06,           1]]).astype(np.float32)
    H2 = S @ H2 @ np.linalg.inv(S)
    H3 = np.array([[    0.99215,    0.027157,      4769.2],
 [ -0.0035934 ,     1.0118 ,     4438.5],
 [-1.3227e-06 , 5.0418e-06,           1]]).astype(np.float32)
    H3 = S @ H3 @ np.linalg.inv(S)
    H4 = np.array([[     1.0145,    0.034837,      2429.5],
 [    0.01286,       1.038,      6119.2],
[  1.771e-06,  7.0018e-06 ,          1]]).astype(np.float32)
    H4 = S @ H4 @ np.linalg.inv(S)
    H5 = np.array([[1.0146343e+00, 1.3305261e-02, 6.5544266e+01],
 [1.1184645e-02, 1.0057985e+00, 4.4326597e+03],
 [1.5989363e-06 ,4.4307935e-06, 1.0000000e+00]]).astype(np.float32)
    H5 = S @ H5 @ np.linalg.inv(S)
    """
    H6 = np.array([[ 1.0074713e+00,  1.0635464e-03,  3.0394157e+01],
 [-2.9231559e-03,  1.0117992e+00 , 2.5832144e+03],
 [-7.9523011e-07,  3.8797717e-07,  1.0000000e+00]]).astype(np.float32)
    H6 = S @ H6 @ np.linalg.inv(S)
    """
    canvas = blend_warp_GPUONLY(front,left,H1,stream7)
    canvas = blend_warp_GPUONLY(canvas,right,H2,stream8)
    canvas = blend_warp_GPUONLY(canvas,right_fender,H3,stream9)
    canvas = blend_warp_GPUONLY(canvas,rear,H4,stream10)
    canvas = blend_warp_GPUONLY(canvas,left_fender,H5,stream11)

    canvas = canvas.download()
    #cv.namedWindow("bev",cv.WINDOW_NORMAL)
    #cv.imshow("bev",canvas)

    cropped = canvas
    h, w = cropped.shape[:2]

    crop_top_px    = int(0.1965 * h)
    crop_bottom_px = int(0.177 * h)
    crop_left_px = int(0.11 * w)
    crop_right_px = int(0.11 * w)

    y1 = crop_top_px
    y2 = h - crop_bottom_px
    x1 = crop_left_px
    x2 = w - crop_right_px
    cropped = cropped[y1:y2, x1:x2]

    scalex = 0.18 # np. 20% szerokości BEV
    scaley = 0.45
    bev_h, bev_w = cropped.shape[:2]
    new_w = int(bev_w * scalex)
    new_h = int(bev_h * scaley)
    car_resized = cv.resize(car, (new_w, new_h), interpolation=cv.INTER_AREA)

    # 3. Oblicz pozycję tak, by wstawić go na środku
    x_offset = (bev_w - new_w) // 2 + 5
    y_offset = (bev_h - new_h) // 2 - 19

    cropped[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = car_resized


    cv.namedWindow("bev",cv.WINDOW_NORMAL)
    cv.imshow("bev",cropped)



    return cropped


def warp_with_cuda(image, H, name,h,w,stream, gpu=True, show=False,first_time=True):
    """
    Funkcja pozwalająca na wyprostowanie obrazu za pomocą homografii H
    Można zarówno jak i wykorzystywać ją do gładkiego przetwarzania na GPU,
    jak i z pobieraniem z powrotem do CPU. Również show pozwala pokazać w
    oddzielnym oknie obraz.
    """


    if first_time:
        gpu_img = cv.cuda_GpuMat()
        gpu_img.upload(image,stream=stream)
    else:
        gpu_img = image
    H_corrected = np.array(np.zeros((3,3))).astype(np.float32)

    if name == "left fender homo":
        #pierwsza w kolumnie translacji - x, druga liczba - y;
        translation = np.array([
        [1, 0, 24],
        [0, 1, 0],
        [0, 0, 1]
        ], dtype=np.float32)
        H_corrected = translation @ H
        #print(H_corrected)
    else:
        H_corrected = H

    warped_gpu = cv.cuda.warpPerspective(gpu_img, H_corrected, (w, h),stream=stream)
    warped = cv.cuda.cvtColor(warped_gpu,cv.COLOR_BGR2RGB,stream=stream)

    if not gpu:
        warped = warped.download()
    else:
        warped = warped
    if show:
        if gpu:
            warp = warped.download()
        else:
            warp=warped
        cv.namedWindow(name, cv.WINDOW_NORMAL)
        cv.imshow(name, warp)
    #stream.waitForCompletion()
    return warped

def warp_and_blend_gpu(img1, img2, H, canvas_size=None, alpha=0.8):
    #canvas_size=(6000//s,6000//s)
    """
    Fast GPU‐only homography warp + simple blend:
      1) compute output canvas bounds & translation
      2) warp img1, img2 onto that canvas
      3) build binary masks on GPU via threshold
      4) do a weighted blend in the overlap, and bitwise OR outside

    img1, img2 : BGR uint8
    H          : float32 homography (3×3) mapping img2 → img1 frame
    canvas_size: (w,h) to force output size, or None to auto‐compute
    alpha      : blend weight for img2 in the overlap region
    """
    # upload
    g1 = cv.cuda_GpuMat(); g1.upload(img1)
    g2 = cv.cuda_GpuMat(); g2.upload(img2)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # compute translation & canvas size if needed
    if canvas_size is None:
        pts1 = np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2)
        pts2 = np.float32([[0,0],[w2,0],[w2,h2],[0,h2]]).reshape(-1,1,2)
        pts2t = cv.perspectiveTransform(pts2, H)
        all_pts = np.vstack([pts1, pts2t])
        x_min, y_min = np.int32(all_pts.min(axis=0).ravel() - 0.5)
        x_max, y_max = np.int32(all_pts.max(axis=0).ravel() + 0.5)
        trans = np.array([[1,0,-x_min],[0,1,-y_min],[0,0,1]],dtype=np.float32)
        out_w, out_h = x_max - x_min, y_max - y_min
    else:
        trans = np.eye(3, dtype=np.float32)
        out_w, out_h = canvas_size

    # warp both onto GPU canvas
    g1w = cv.cuda.warpPerspective(g1, trans,     (out_w, out_h))
    g2w = cv.cuda.warpPerspective(g2, trans @ H, (out_w, out_h))

    # build masks by thresholding grayscale on GPU
    gray1 = cv.cuda.cvtColor(g1w, cv.COLOR_BGR2GRAY)
    gray2 = cv.cuda.cvtColor(g2w, cv.COLOR_BGR2GRAY)
    _, m1w = cv.cuda.threshold(gray1, 1, 255, cv.THRESH_BINARY)
    _, m2w = cv.cuda.threshold(gray2, 1, 255, cv.THRESH_BINARY)

    # overlap region mask
    overlap = cv.cuda.bitwise_and(m1w, m2w)

    # simple blend in overlap
    blend = cv.cuda.addWeighted(g1w, 1-alpha, g2w, alpha, 0)

    # non‐overlap contributions
    inv2 = cv.cuda.bitwise_not(m2w)
    inv1 = cv.cuda.bitwise_not(m1w)
    part1 = cv.cuda.bitwise_and(g1w, cv.cuda.cvtColor(inv2, cv.COLOR_GRAY2BGR))
    part2 = cv.cuda.bitwise_and(g2w, cv.cuda.cvtColor(inv1, cv.COLOR_GRAY2BGR))
    partB = cv.cuda.bitwise_and(blend, cv.cuda.cvtColor(overlap, cv.COLOR_GRAY2BGR))

    # sum them up
    tmp = cv.cuda.add(part1, part2)
    out = cv.cuda.add(tmp, partB)

    return out.download()

def blend_warp_GPUONLY(g1, g2, H,stream, canvas_size=None, alpha=0.85):
    """
    Szybkie mieszanie i prostowanie dwóch obrazów na GPU.
    Wykorzystuje
    Fast GPU‐only homography warp + simple blend:
      1) compute output canvas bounds & translation
      2) warp img1, img2 onto that canvas
      3) build binary masks on GPU via threshold
      4) do a weighted blend in the overlap, and bitwise OR outside

    img1, img2 : BGR uint8
    H          : float32 homography (3×3) mapping img2 → img1 frame
    canvas_size: (w,h) to force output size, or None to auto‐compute
    alpha      : blend weight for img2 in the overlap region
    """
    # upload

    w1, h1 = g1.size()
    w2, h2 = g2.size()


    # compute translation & canvas size if needed
    if canvas_size is None:
        pts1 = np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2)
        pts2 = np.float32([[0,0],[w2,0],[w2,h2],[0,h2]]).reshape(-1,1,2)
        pts2t = cv.perspectiveTransform(pts2, H)
        all_pts = np.vstack([pts1, pts2t])
        x_min, y_min = np.int32(all_pts.min(axis=0).ravel() - 0.5)
        x_max, y_max = np.int32(all_pts.max(axis=0).ravel() + 0.5)
        trans = np.array([[1,0,-x_min],[0,1,-y_min],[0,0,1]],dtype=np.float32)
        out_w, out_h = x_max - x_min, y_max - y_min
    else:
        trans = np.eye(3, dtype=np.float32)
        out_w, out_h = canvas_size

    # warp both onto GPU canvas
    g1w = cv.cuda.warpPerspective(g1, trans,     (out_w, out_h),stream=stream)
    g2w = cv.cuda.warpPerspective(g2, trans @ H, (out_w, out_h),stream=stream)

    # build masks by thresholding grayscale on GPU
    gray1 = cv.cuda.cvtColor(g1w, cv.COLOR_BGR2GRAY,stream=stream)
    gray2 = cv.cuda.cvtColor(g2w, cv.COLOR_BGR2GRAY,stream=stream)
    _, m1w = cv.cuda.threshold(gray1, 1, 255, cv.THRESH_BINARY,stream=stream)
    _, m2w = cv.cuda.threshold(gray2, 1, 255, cv.THRESH_BINARY,stream=stream)

    # overlap region mask
    overlap = cv.cuda.bitwise_and(m1w, m2w,stream=stream)

    # simple blend in overlap
    blend = cv.cuda.addWeighted(g1w, 1-alpha, g2w, alpha, 0,stream=stream)

    # non‐overlap contributions
    inv2 = cv.cuda.bitwise_not(m2w,stream=stream)
    inv1 = cv.cuda.bitwise_not(m1w,stream=stream)
    g1w = cv.cuda.bitwise_and(g1w, cv.cuda.cvtColor(inv2, cv.COLOR_GRAY2BGR),stream=stream)
    g2w = cv.cuda.bitwise_and(g2w, cv.cuda.cvtColor(inv1, cv.COLOR_GRAY2BGR),stream=stream)
    blend = cv.cuda.bitwise_and(blend, cv.cuda.cvtColor(overlap, cv.COLOR_GRAY2BGR),stream=stream)

    # sum them up
    out = cv.cuda.add(g1w,g2w,stream=stream)
    out = cv.cuda.add(out, blend,stream=stream)
    #stream.waitForCompletion()
    return out

#DALSZE FRAGMENTY KODU DO SKOPIOWANIA W ALT_COLLECT_HOMO - TO SĄ DO WYZNACZENIA
#POŁOŻENIA SZACHOWNIC I ODPOWIADAJĄCYCH PUNKTÓW NA KANWIE

"""
def ch_points_calc(pattern_size,square_size,centerpoint):
    half_width = square_size*pattern_size[0]/2
    half_height = square_size*pattern_size[1]/2
    points = np.array(
    [[centerpoint[0]+half_height,centerpoint[1]+half_width],
    [centerpoint[0]+half_height,centerpoint[1]-half_width],
    [centerpoint[0]-half_height,centerpoint[1]-half_width],
    [centerpoint[0]-half_height,centerpoint[1]+half_width]],
    dtype=np.float32)
    return points

left_cp = np.array([-0.425+2.6,3.23],dtype=np.float32)
right_cp = np.array([-0.425+2.2,-3.51],dtype=np.float32)
left_fender_cp = np.array([-0.425-2.85,3.58],dtype=np.float32)
right_fender_cp = np.array([-0.425-2.85,-3.58],dtype=np.float32)
front_cp = np.array([-0.425+4.66,0],dtype=np.float32)
rear_cp = np.array([-0.425-5.25,0],dtype=np.float32)

objp_left = ch_points_calc((6,8),0.6,left_cp)
objp_right = ch_points_calc((5,6),0.6,right_cp)
objp_left_fender = ch_points_calc((8,7),0.5,left_fender_cp)
objp_right_fender = ch_points_calc((8,7),0.5,right_fender_cp)
objp_front = ch_points_calc((10,4),0.4,front_cp)
objp_rear = ch_points_calc((8,5),0.6,rear_cp)

#cor_left = cc.solve_chess_size(left,"left",(7,9),None)
#cor_right = cc.solve_chess_size(right,"right",(6,7),None)
#cor_left_fender = cc.solve_chess_size(left_fender,"left1",(9,8),None)
#cor_right_fender = cc.solve_chess_size(right_fender,"right1",(9,8),None)
#cor_front = cc.solve_chess_size(front,"front",(11,5),None)
#cor_rear = cc.solve_chess_size(rear,"rear",(11,5),None)

def apply_homography_to_points(points, H):
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])  # Dodanie współrzędnych jedności
    points_transformed = np.dot(H, points_homogeneous.T).T  # Mnożenie homografii
    points_transformed /= points_transformed[:, 2].reshape(-1, 1)  # Normalizacja przez Z (homogenizacja)
    return points_transformed[:, :2]  # Zwrócenie tylko współrzędnych x i y

# Zastosowanie homografii do punktów szachownic (lewa i prawa szachownica)
transformed_left_points = apply_homography_to_points(objp_left, H_px_to_m_bev)
transformed_right_points = apply_homography_to_points(objp_right, H_px_to_m_bev)

transformed_front_points = apply_homography_to_points(objp_front, H_px_to_m_bev)
transformed_rear_points = apply_homography_to_points(objp_rear, H_px_to_m_bev)

transformed_left_fender_points = apply_homography_to_points(objp_left_fender, H_px_to_m_bev)
transformed_right_fender_points = apply_homography_to_points(objp_right_fender, H_px_to_m_bev)
bev_left= np.eye(3,3).astype(np.float32)
bev_right = np.eye(3,3).astype(np.float32)
bev_left_fender = np.eye(3,3).astype(np.float32)
bev_right_fender = np.eye(3,3).astype(np.float32)




#rear = warp_and_blend_gpu(canvas,rear,H_rear_to_bev)
#cor_rear = cc.solve_chess_size(rear,"rear",(9,6),None)
"""
"""
if cor_left is not None:
    #H_left_met_to_px,_ = cv.findHomography(objp_left,cor_left,cv.RANSAC,3.0)
    #H_left_to_bev = H_px_to_m_bev @ H_left_met_to_px
    H_left_to_bev,_ = cv.findHomography(cor_left,transformed_left_points,cv.RANSAC,2.0)


    if H_left_to_bev is not None:
        bev_left = warp_and_blend_gpu(canvas,left,H_left_to_bev)
        # Rysowanie punktów na obrazie (kanwie)
        for point in transformed_left_points:
            cv.circle(bev_left, (int(point[0]), int(point[1])), 10, (0, 255, 0), -1)  # Zielone punkty dla lewej szachownicy

        for point in transformed_right_points:
            cv.circle(bev_left, (int(point[0]), int(point[1])), 10, (255, 0, 0), -1)
        print("h_left_to_bev")
        print(H_left_to_bev)
        print("------------------------------------")
        #cv.namedWindow("left_bev",cv.WINDOW_NORMAL)
        #cv.imshow("left_bev",bev_left)

if cor_right is not None and cor_left is not None:
    #H_right_met_to_px,_ = cv.findHomography(objp_right,cor_right,cv.RANSAC,3.0)
    #H_right_to_bev = H_px_to_m_bev @ H_right_met_to_px
    H_right_to_bev,_ = cv.findHomography(cor_right,transformed_right_points,cv.RANSAC,3.0)
    if H_right_to_bev is not None:
        bev_right = warp_and_blend_gpu(bev_left,right,H_right_to_bev)
        for point in transformed_left_points:
            cv.circle(bev_right, (int(point[0]), int(point[1])), 10, (0, 255, 0), -1)  # Zielone punkty dla lewej szachownicy

        for point in transformed_right_points:
            cv.circle(bev_right, (int(point[0]), int(point[1])), 10, (0, 0, 255), -1)
        print("H_right_to_bev")
        print(H_right_to_bev)
        print("------------------------------------")
        #print(bev_right.shape)
        #cv.namedWindow("right_bev",cv.WINDOW_NORMAL)
        #cv.imshow("right_bev",bev_right)
"""
"""
#
if cor_left_fender is not None:
    #H_left_met_to_px,_ = cv.findHomography(objp_left,cor_left,cv.RANSAC,3.0)
    #H_left_to_bev = H_px_to_m_bev @ H_left_met_to_px
    H_left_fender_to_bev,_ = cv.findHomography(cor_left_fender,transformed_left_fender_points,cv.RANSAC,3.0)
    if H_left_fender_to_bev is not None:
        bev_left_fender = warp_and_blend_gpu(canvas,left_fender,H_left_fender_to_bev)
        # Rysowanie punktów na obrazie (kanwie)
        for point in transformed_left_fender_points:
            cv.circle(bev_left_fender, (int(point[0]), int(point[1])), 10, (0, 255, 0), -1)  # Zielone punkty dla lewej szachownicy

        for point in transformed_right_fender_points:
            cv.circle(bev_left_fender, (int(point[0]), int(point[1])), 10, (0, 0, 255), -1)
        cv.namedWindow("left_bev",cv.WINDOW_NORMAL)
        cv.imshow("left_bev",bev_left_fender)

        print("H_left_fender_to_bev")
        print( H_left_fender_to_bev)
        print("------------------------------------")

#
if cor_right_fender is not None :
    #H_right_met_to_px,_ = cv.findHomography(objp_right,cor_right,cv.RANSAC,3.0)
    #H_right_to_bev = H_px_to_m_bev @ H_right_met_to_px
    H_right_fender_to_bev,_ = cv.findHomography(cor_right_fender,transformed_right_fender_points,cv.RANSAC,3.0)
    if H_right_fender_to_bev is not None:
        bev_right_fender = warp_and_blend_gpu(canvas,right_fender,H_right_fender_to_bev)
        for point in transformed_left_fender_points:
            cv.circle(bev_right_fender, (int(point[0]), int(point[1])), 10, (0, 255, 0), -1)  # Zielone punkty dla lewej szachownicy

        for point in transformed_right_fender_points:
            cv.circle(bev_right_fender, (int(point[0]), int(point[1])), 10, (0, 0, 255), -1)
        #print(bev_right.shape)
        cv.namedWindow("right_bev",cv.WINDOW_NORMAL)
        cv.imshow("right_bev",bev_right_fender)

        print("H_right_fender_to_bev")
        print(H_right_fender_to_bev)
        print("------------------------------------")
"""
"""
if cor_rear is not None:
    #H_left_met_to_px,_ = cv.findHomography(objp_left,cor_left,cv.RANSAC,3.0)
    #H_left_to_bev = H_px_to_m_bev @ H_left_met_to_px
    H_rear_to_bev,_ = cv.findHomography(cor_rear,transformed_rear_points,cv.RANSAC,3.0)
    if H_rear_to_bev is not None:
        bev_rear = warp_and_blend_gpu(canvas,rear,H_rear_to_bev)
        # Rysowanie punktów na obrazie (kanwie)
        for point in transformed_rear_points:
            cv.circle(bev_rear, (int(point[0]), int(point[1])), 10, (0, 255, 0), -1)  # Zielone punkty dla lewej szachownicy

        for point in transformed_front_points:
            cv.circle(bev_rear, (int(point[0]), int(point[1])), 10, (0, 0, 255), -1)
        print("git")
        print(H_rear_to_bev)
        cv.namedWindow("rear_bev",cv.WINDOW_NORMAL)
        cv.imshow("rear_bev",bev_rear)
"""
"""
if cor_front is not None:
    #H_left_met_to_px,_ = cv.findHomography(objp_left,cor_left,cv.RANSAC,3.0)
    #H_left_to_bev = H_px_to_m_bev @ H_left_met_to_px
    H_front_to_bev,_ = cv.findHomography(cor_front,transformed_front_points,cv.RANSAC,3.0)
    if H_front_to_bev is not None:
        bev_front = warp_and_blend_gpu(bev_rear,front,H_front_to_bev)
        # Rysowanie punktów na obrazie (kanwie)
        for point in transformed_front_points:
            cv.circle(bev_front, (int(point[0]), int(point[1])), 10, (0, 255, 0), -1)  # Zielone punkty dla lewej szachownicy

        #for point in transformed_right_points:
            #cv.circle(bev_front, (int(point[0]), int(point[1])), 10, (0, 0, 255), -1)
        print("jest git")
        print(H_front_to_bev)
        cv.namedWindow("front_bev",cv.WINDOW_NORMAL)
        cv.imshow("front_bev",bev_front)
"""
