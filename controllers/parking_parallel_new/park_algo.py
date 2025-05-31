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
class Parking:
    # Domyślne marginesy startu i końca manewru
    marg_start = 0.2
    marg_end   = 0.4


    def __init__(self, driver, side, times,
                 min_width=CAR_WIDTH*1.1,
                 min_length=CAR_LENGTH*1.25,
                 threshold=1*CAR_WIDTH):
        # Inicjalizacja parametrów parkingu
        self.min_width  = min_width       # minimalna szerokość miejsca
        self.min_length = min_length      # minimalna długość miejsca
        self.threshold  = threshold       # próg zmiany odległości

        self.state           = "searching_start"  # aktualny stan maszyny stanów
        self.start_pose      = None               # miejsce wykrycia początku luki
        self.spots           = []                 # wykryte miejsca parkingowe
        self.driver          = driver             # interfejs do samochodu w Webots
        self.spot = None
        # Pomocnicze zmienne do detekcji zmian odległości
        self.prev_distance_front = 6.0
        self.prev_distance_rear  = 6.0
        self.dist_start_far      = 6.0
        self.dist_start_cl       = 6.0
        self.dist_end_far        = 6.0
        self.dist_end_cl         = 6.0

        self.side      = side    # strona parkowania: "left" lub "right"
        self.x = 0.0             # os x w układzie globalnym
        self.y = 0.0             # os y w układzie globalnym
        self.yaw = 0.0           # orientacja pojazdu
        self.last_time = times   # czas ostatniej aktualizacji

        # Parametry regulatora PID dla sterowania
        self.Kp = 1.2
        self.Kd = 0.001
        self.Kl = 0.5
        self.prev_yaw_err = 0.0

    def update_odometry(self, yaw):
        """
        Aktualizuje pozycję (x,y) na podstawie prędkości i kąta yaw.
        """
        # Pobierz prędkość w km/h i przelicz na m/s
        v = self.driver.getCurrentSpeed() / 3.6

        # Integracja prostokątna z krokiem czasowym 0.05s
        self.yaw = yaw
        dx = v * math.cos(self.yaw) * 0.06
        dy = v * math.sin(self.yaw) * 0.06
        self.x += dx
        self.y += dy

        return (self.x, self.y, self.yaw)

    def update_state(self, dists_names, yaw):
        """
        Maszyna stanów wykrywająca luki parkingowe.
        """
        # Wybór odpowiednich czujników w zależności od strony
        if self.side == "left":
            distance_front = dists_names["distance sensor left front side"]
            distance_rear  = dists_names["distance sensor left side"]
        else:  # self.side == "right"
            distance_front = dists_names["distance sensor right front side"]
            distance_rear  = dists_names["distance sensor right side"]

        # Oblicz zmiany odległości
        delta_front = distance_front - self.prev_distance_front
        delta_rear  = distance_rear  - self.prev_distance_rear
        self.prev_distance_front = distance_front
        self.prev_distance_rear  = distance_rear

        # Aktualizuj pozycję z odometrii
        odom_pose = self.update_odometry(yaw)
        x, y, yaw = odom_pose
        #print(f"Odometria : {odom_pose}")
        # Stan: poszukiwanie początku luki
        if self.state == "searching_start":
            if delta_front > self.threshold:
                # Znaleziono gwałtowny wzrost odległości ⇒ początek luki
                self.start_pose      = (x - self.marg_start, y, yaw)
                self.dist_start_far  = distance_front
                self.dist_start_cl   = self.prev_distance_front
                self.state           = "searching_progress"
                print("Kandydat na miejsce znaleziony.")

        # Stan: poszukiwanie końca luki
        elif self.state == "searching_progress":

            if -delta_front > self.threshold:
                # Znaleziono gwałtowny spadek odległości ⇒ koniec luki
                end_pose = (x + self.marg_end, y, yaw)
                self.dist_end_far = distance_front
                self.dist_end_cl  = distance_front - delta_front

                # Utwórz obiekt miejsca parkingowego
                spot = self._make_spot(self.start_pose, end_pose)
                if spot:
                    self.state = "waiting_for_park"
                    self.spot = spot
                    print("Miejsce znalezione. Wciśnij Y, aby rozpocząć parkowanie. (NIE ZALECANE, NIE UMIE PARKOWAĆ)", spot)
                else:
                    print("Miejsce okazało się za małe.")
                    self.state = "searching_start"

        if self.spot is not None: return odom_pose, self.spot

    @staticmethod
    def normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2*math.pi
        while angle <= -math.pi:
            angle += 2*math.pi
        return angle

    def exec_path(self, curr_pose, end_pose, lateral_dist):
        """
        Generuje sterowanie pojazdem do osiągnięcia end_pose. Przykładowe, póki co nie działa.
        """
        x, y, yaw   = curr_pose
        x_e, y_e, _ = end_pose

        # Oblicz kąt do punktu docelowego i błąd yaw
        target_yaw = math.atan2(y_e - y, x_e - x)
        yaw_err = self.normalize_angle(target_yaw - yaw)

        # Odległość do celu
        dist_forward = math.hypot(x_e - x, y_e - y)

        # Regulator PD dla yaw
        steer = self.Kp * yaw_err + self.Kd * (yaw_err - self.prev_yaw_err)
        self.prev_yaw_err = yaw_err

        # Ustaw kąty sterowania i prędkość
        self.driver.setSteeringAngle(steer)
        self.driver.setCruisingSpeed(MAX_SPEED)

        # Dodatkowa korekcja boczna (trzymanie od krawężnika)
        deltalat = lateral_dist - (CAR_WIDTH/2 + 0.1)
        steer += self.Kl * deltalat
        self.driver.setSteeringAngle(steer)

    def _make_spot(self, start, end):
        """
        Tworzy opis miejsca parkingowego na podstawie pozycji start/end.
        """
        sp_len = end[0] - start[0]
        if sp_len < self.min_width:
            return None

        # Szerokość miejsca na podstawie różnicy odczytów sonaru
        sp_wid = ((self.dist_start_far - self.dist_start_cl) +
                  (self.dist_end_far   - self.dist_end_cl)) / 2

        # Środek luki parkingowej
        sp_cen_x = start[0] + sp_len / 2
        sp_cen_y = start[1] + (self.dist_start_cl + self.dist_start_far) / 2

        # Ustawienie w zależności od strony parkowania
        if self.side == "left":
            sen_pos = [3.515873,  0.865199,  90]
        elif self.side == "right":
            sen_pos = [3.515873, -0.865199, -90]

        # Punkt końcowy manewru
        x_end = sp_cen_x + sen_pos[0] - 1.425
        y_end = sp_cen_y + sen_pos[1]

        return [x_end, y_end, start[2]]

    def get_spots(self):
        """Zwraca listę wykrytych miejsc parkingowych."""
        return self.spots


class LivePlotter:
    def __init__(self, ax, max_points=100):
        # Inicjalizacja wykresu: ograniczenia i etykiety
        self.max_points = max_points
        self.data       = []
        self.ax         = ax
        self.line,     = self.ax.plot([], [], 'r-')
        self.ax.set_xlim(0, self.max_points)
        self.ax.set_ylim(-10, 10)
        self.ax.set_title("Wartości z czujnika na żywo")
        self.ax.set_xlabel("Pomiar")
        self.ax.set_ylabel("Wartość")
        self.val = 0.0

    def update(self, frame=None):
        """
        Aktualizacja danych na wykresie w czasie rzeczywistym.
        """
        sensor_value = self.val          # pobierz aktualną wartość czujnika
        self.data.append(sensor_value)
        if len(self.data) > self.max_points:
            self.data.pop(0)
        xdata = list(range(len(self.data)))
        ydata = self.data

        # Zaktualizuj dane linii i zakres osi
        self.line.set_data(xdata, ydata)
        self.ax.set_xlim(0, max(self.max_points, len(self.data)))
        self.ax.set_ylim(min(ydata) - 1, max(ydata) + 1)
        return (self.line,)


    #def run(self):
        #ani = animation.FuncAnimation(self.fig, self.update,interval=100, blit=True)
        #plt.show()
