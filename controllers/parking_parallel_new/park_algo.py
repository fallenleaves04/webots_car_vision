import math
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation

# Vehicle parameters
TRACK_FRONT = 1.628
TRACK_REAR = 1.628
WHEELBASE = 2.995
MAX_WHEEL_ANGLE = 0.5  # rad
CAR_WIDTH = 1.95
CAR_LENGTH = 4.85
MAX_SPEED = -3.0
def normalize_angle(angle: float) -> float:
    """
    Zamienia dowolny kąt w radianach na równoważny w przedziale [-π, π).
    """
    # opcja z pętlami
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle <= -math.pi:
        angle += 2 * math.pi
    return angle

class Parking:
    def __init__(self,driver,side,times,min_width=CAR_WIDTH*1.1,min_length=CAR_LENGTH*1.25,threshold = 1*CAR_WIDTH):
        self.min_width = min_width
        self.min_length = min_length
        self.threshold = threshold

        self.state = "searching_start"
        self.start_pose = None
        self.spots = []
        self.driver = driver

        self.prev_distance_front = 0.0
        self.prev_distance_rear = 0.0
        self.side = side
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.last_time = times


        self.Kp = 1.2
        self.Kd = 0.001
        self.Kl = 0.5
        self.prev_yaw_err = 0.0
    def update_odometry(self,yaw):
        # prędkość w m/s


        # prędkość [m/s]
        v = self.driver.getCurrentSpeed() / 3.6

        # integracja
        self.yaw = yaw
        dx = v * math.cos(self.yaw) * 0.05
        dy = v * math.sin(self.yaw) * 0.05
        self.x += dx
        self.y += dy

        # debug
        #print(f"v={v:.2f}m/s → dx={dx:.2f}, dy={dy:.2f} | x={self.x:.2f}, y={self.y:.2f}")
        return (self.x, self.y, self.yaw)

    def update_state(self, dists_names,yaw):
        if self.side == "left":
            distance_front = dists_names["distance sensor left front side"]
            distance_rear = dists_names["distance sensor left side"]
            delta_front = distance_front - self.prev_distance_front
            delta_rear = distance_rear - self.prev_distance_rear
            self.prev_distance_front = distance_front
            self.prev_distance_rear = distance_rear
        elif self.side == "right":
            distance_front = dists_names["distance sensor right front side"]
            distance_rear = dists_names["distance sensor right side"]
            delta_front = distance_front - self.prev_distance_front
            delta_rear = distance_rear - self.prev_distance_rear
            self.prev_distance_front = distance_front
            self.prev_distance_rear = distance_rear


        odom_pose = self.update_odometry(yaw)
        # projekcja punktu bocznego na mapę
        x, y, yaw = odom_pose
        #print(f"wsp. x: {x:.2f}, wsp. y: {y:.2f}, skret: {yaw:.2f}")
        if self.state == "searching_start":

            # czy odległość gwałtownie spadła?
            if delta_front > self.threshold:
                # zapamiętujemy początek miejsca
                self.start_pose = odom_pose
                self.state = "searching_progress"
                print("Kandydat na miejsce znaleziony.")
        elif self.state == "searching_progress":
            # czy odległość wzrosła z powrotem?
            if -delta_front > self.threshold:
                end_pose = odom_pose
                spot = self._make_spot(self.start_pose, end_pose)
                if spot is not None:
                    self.spots.append(spot)
                    print("Miejsce znalezione!!!")
                    print(spot)
                    self.state = "waiting_for_park"
                    return odom_pose,spot
                    #self.exec_path(odom_pose,spot,distance_front)
                elif spot is None:
                    print("Miejsce okazało się za małe.")

                    self.state = "searching_start"

                # resetujemy do kolejnej detekcji

    def exec_path(self, curr_pose, end_pose, lateral_dist):
        # curr_pose = (x, y, yaw), end_pose = (x_end,y_end,yaw_end)
        x, y, yaw = curr_pose
        x_e, y_e, yaw_e = end_pose

        # Kąt do mety
        target_yaw = math.atan2(y_e - y, x_e - x)
        yaw_err = normalize_angle(target_yaw - yaw)

        # odległość do mety
        dist_forward = math.hypot(x_e - x, y_e - y)


        steer = self.Kp*yaw_err + self.Kd*(yaw_err - self.prev_yaw_err)
        self.prev_yaw_err = yaw_err

        self.driver.setSteeringAngle(steer)
        # self.driver.setCruisingSpeed(min(dist_forward*0.5, MAX_SPEED))
        self.driver.setCruisingSpeed(MAX_SPEED)
        # korygować lateralnie, trzymając odległość od krawężnika:
        # jeśli lateral_dist != desired (np. 1.0m), deltalat = lateral_dist - desired
        deltalat = lateral_dist - (CAR_WIDTH/2 + 0.1)  # 20cm odległości od krawężnika
        steer += self.Kl * deltalat
        self.driver.setSteeringAngle(steer)
    def _make_spot(self, start, end):
        """
        Na podstawie dwóch poz. pojazdu tworzy kandydat na miejsce:
        start, end: (x, y, heading)
        Zwraca ((x1,y1),(x2,y2)) — współrzędne końców krawędzi równoległej do boku,
        lub None, jeśli rozmiar < min_length.
        """
        x0, y0, yaw0 = start
        x1, y1, yaw1 = end

        # wektor kierunku boku pojazdu (prostopadły do osi long.)
        # heading = kąt podłużnej osi pojazdu od osi X

        dx = math.cos(yaw0 + math.pi/2)
        dy = math.sin(yaw0 + math.pi/2)

        # punkty graniczne wzdłuż boku
        p_start = (x0 + dx * self.min_width/2, y0 + dy * self.min_width/2)
        p_end   = (x1 + dx * self.min_width/2, y1 + dy * self.min_width/2)

        # długość wzdłuż ruchu pojazdu
        length = math.hypot(x1 - x0, y1 - y0)
        if length < self.min_length:

            return None
        p_start
        return (p_start, p_end)

    def get_spots(self):
        """Zwraca listę wykrytych miejsc parkingowych."""
        return self.spots


class LivePlotter:
    def __init__(self, max_points=100):
        self.max_points = max_points
        self.data = []
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'r-')
        self.ax.set_xlim(0, self.max_points)
        self.ax.set_ylim(-10, 10)  # dopasuj do zakresu swojego czujnika
        self.ax.set_title("Wartości z czujnika na żywo")
        self.ax.set_xlabel("Pomiar")
        self.ax.set_ylabel("Wartość")
        self.val = 0.0
    def update(self, frame):
        # Tu pobierasz dane z czujnika — zastąp ten fragment swoim odczytem!
        sensor_value = self.val
        self.data.append(sensor_value)
        if len(self.data) > self.max_points:
            self.data.pop(0)
        xdata = list(range(len(self.data)))
        ydata = self.data
        self.line.set_data(xdata, ydata)
        self.ax.set_xlim(0, max(self.max_points, len(self.data)))
        # Opcjonalnie: dynamiczne skalowanie osi Y
        self.ax.set_ylim(min(ydata) - 1, max(ydata) + 1)
        return (self.line,)


    def run(self):
        ani = animation.FuncAnimation(self.fig, self.update,interval=100, blit=True)
        plt.show()
