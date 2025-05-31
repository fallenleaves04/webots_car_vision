
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
        print(f"Odometria : {odom_pose}")
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
                    print("Miejsce znalezione. Wciśnij Y, aby rozpocząć parkowanie.", spot)
                else:
                    print("Miejsce okazało się za małe.")
                    self.state = "searching_start"
       
        if self.spot is not None: return odom_pose, self.spot 
