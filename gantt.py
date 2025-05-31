import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Define project timeline
start_date = datetime(2025, 2, 20)
end_date = datetime(2025, 5, 31)

# Tasks and durations based on user input
tasks = [
    ("Środowisko Webots", datetime(2025, 2, 20), datetime(2025, 3, 10)),
    ("Widok z lotu ptaka", datetime(2025, 2, 20), datetime(2025, 5, 30)),
    ("CUDA (równolegle)", datetime(2025, 3, 20), datetime(2025, 5, 1)),
    ("Estymacja pozy i kalibracja kamer", datetime(2025, 4, 1), datetime(2025, 4, 30)),
    ("Detekcja YOLO", datetime(2025, 4, 26), datetime(2025, 5, 25)),
    ("Czujniki ultradźwiękowe (kondycjonowanie)", datetime(2025, 2, 20), datetime(2025, 3, 20)),
    ("Czujniki ultradźwiękowe (lokalizacja w przestrzeni)", datetime(2025, 5, 10), datetime(2025, 5, 31)),
    ("Nawigacja i algorytm planowania ścieżki", datetime(2025, 5, 10), datetime(2025, 5, 31)),
    ("Estymacja głębi (sieci neuronowe)", datetime(2025, 2, 20), datetime(2025, 3, 6)),
    ("Estymacja głębi (stereowizja)", datetime(2025, 4, 26), datetime(2025, 5, 25)),
    ("Wielowątkowość", datetime(2025, 4, 15), datetime(2025, 4, 22)),
    ("Wielowątkowość (kontynuacja)", datetime(2025, 5, 19), datetime(2025, 5, 31)),
    ("Sporządzenie sprawozdania projektowego",datetime(2025, 5, 8), datetime(2025, 5, 31))
]

# Plotting Gantt chart
fig, ax = plt.subplots(figsize=(10, 6))

for i, (task, start, finish) in enumerate(tasks):
    ax.barh(i, finish-start, left=start, height=0.5, align='center')

# Formatting the y-axis
ax.set_yticks(range(len(tasks)))
ax.set_yticklabels([task[0] for task in tasks])

# Formatting the x-axis (dates)
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.xticks(rotation=45)

# Labels and title
ax.set_xlabel('Data')
ax.set_title('Wykres Gantta dla realizacji projektu')

plt.tight_layout()
plt.grid(axis='x')
plt.show()



# Obiekt stereo matcher
stereo_left = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=32,
    blockSize=20,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=8
)
stereo_right = cv2.ximgproc.createRightMatcher(stereo_left)

# obliczenie mask i disparity
results = model(img_right, half=True, device=0,
                classes=[2,5,7,10], conf=0.6,
                verbose=False, imgsz=(1280,960))
if results and results[0].masks is not None:
    disp_left  = stereo_left.compute(grayL, grayR).astype(np.float32)/16.0
    disp_right = stereo_right.compute(grayR, grayL).astype(np.float32)/16.0

    wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo_left)
    wls.setLambda(8000);  wls.setSigmaColor(1.9)
    filtered_disp = wls.filter(disp_left, grayL, None, disp_right)

    for mask in results[0].masks.data.cpu().numpy():
        mask_u8 = cv2.resize(mask.astype(np.uint8),
                             (orig_w, orig_h),
                             interpolation=cv2.INTER_NEAREST)
        disp_masked = np.nan_to_num(filtered_disp) * mask_u8
        valid = disp_masked[(mask_u8>0)&(disp_masked>0)]
        if valid.size==0: continue

        p1_3d, p2_3d = points_from_mask_to_3D(
            mask_u8, filtered_disp,
            K_right, 0.03, T_center_to_camera
        )
        if p1_3d is None or p2_3d is None: continue

        # rzut 3D→2D
        p1_px = project_points_world_to_image([p1_3d], T_center_to_camera, K)
        p2_px = project_points_world_to_image([p2_3d], T_center_to_camera, K)
        # … dalsze operacje na obrazie …