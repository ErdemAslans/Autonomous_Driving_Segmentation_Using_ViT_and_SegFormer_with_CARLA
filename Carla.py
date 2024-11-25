import sys
import random
import time
import numpy as np
import threading
import cv2  # OpenCV kütüphanesi
import zlib  # Veri sıkıştırma için zlib
import logging  # Loglama için
from collections import deque
import queue  # Görselleştirme için kuyruk
from enum import Enum
import heapq

# CARLA yolunu ekleyin ve modülü içe aktarın
sys.path.append("C:/Users/Erdem/Downloads/CARLA_0.9.13/WindowsNoEditor/PythonAPI/carla/dist/carla-0.9.13-py3.7-win-amd64.egg")  # Yolunuza göre değiştirin
import carla

# Logging Ayarları
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("carla_simulation.log"),
                        logging.StreamHandler(sys.stdout)
                    ])

# CARLA istemcisi ve dünya ayarları
client = carla.Client('localhost', 2000)  # CARLA sunucusunun adresi ve portu
client.set_timeout(10.0)  # Zaman aşımı süresi
world = client.get_world()  # Dünya nesnesini alın

# Spawn noktaları ve blueprint kütüphanesi
spawn_points = world.get_map().get_spawn_points()
blueprint_library = world.get_blueprint_library()

# Global Değişkenler
lane_following = True
visualization_queue = queue.Queue()  # Görselleştirme kuyruğu

TARGET_SPEED = 20.0  # km/h
PID_KP = 1.0  # Kp'yi artırdık
PID_KI = 0.0  # Ki'yi sıfırladık
PID_KD = 0.3  # Kd'yi artırdık
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Global sensör listesi (sensör referanslarını korumak için)
sensors = []

def log_debug_info(debug_info):
    """Debug bilgilerini loglamak için basit bir fonksiyon."""
    logging.info("Debug Bilgileri:")
    for key, value in debug_info.items():
        if isinstance(value, np.ndarray):
            logging.info(f"{key}: {value.shape}")
        else:
            logging.info(f"{key}: {value}")

def calculate_brake_force(current_speed, distance, vehicle_mass=1500, road_condition="dry"):
    """
    Geliştirilmiş fren kuvveti hesaplama fonksiyonu.
    
    Args:
        current_speed (float): Araç hızı (km/h).
        distance (float): Engel ile araç arasındaki mesafe (m).
        vehicle_mass (float): Araç kütlesi (kg). Varsayılan: 1500 kg.
        road_condition (str): Yol koşulu ("dry", "wet", "icy"). Varsayılan: "dry".
        
    Returns:
        float: Fren kuvveti (0.0 - 1.0 arasında).
    """
    if distance <= 0:  # Negatif mesafe durumları
        return 1.0  # Acil fren

    # Hızı m/s'ye çevir
    speed_m_s = current_speed / 3.6

    # Frenleme katsayıları (yol koşuluna göre)
    road_conditions = {
        "dry": 0.8,  # Kuru yol (yüksek sürtünme)
        "wet": 0.6,  # Islak yol (orta sürtünme)
        "icy": 0.3   # Buzlu yol (düşük sürtünme)
    }
    road_friction = road_conditions.get(road_condition, 0.8)

    # Minimum güvenlik mesafesi (araç dinamiklerine dayalı)
    # D = v^2 / (2 * µ * g)  (fiziksel formül)
    g = 9.81  # Yerçekimi ivmesi (m/s^2)
    min_safe_distance = (speed_m_s**2) / (2 * road_friction * g)

    # Risk seviyesi belirleme
    critical_distance = min_safe_distance * 0.8  # Daha yüksek risk seviyesi
    warning_distance = min_safe_distance * 1.5  # Daha düşük risk seviyesi

    # Frenleme kuvveti hesaplama
    if distance <= critical_distance:
        # Acil durum frenleme
        brake_force = 1.0
    elif distance <= min_safe_distance:
        # Yüksek risk frenleme (lineer artış)
        brake_force = 0.5 + 0.5 * (min_safe_distance - distance) / min_safe_distance
    elif distance <= warning_distance:
        # Orta düzey frenleme
        brake_force = 0.3 + 0.2 * (warning_distance - distance) / warning_distance
    else:
        # Frenleme gerekmez
        brake_force = 0.0

    # Ağırlığa bağlı fren ayarı (daha ağır araçlarda fren kuvveti artırılır)
    weight_factor = min(vehicle_mass / 1500, 2.0)  # 1500 kg referans ağırlık
    brake_force *= weight_factor

    # Frenleme kuvvetini sınırlama
    brake_force = min(brake_force, 1.0)

    return brake_force

def setup_main_vehicle(world, spawn_points, vehicle_config):
    """
    Ana aracı oluşturur ve yapılandırır.
    
    Args:
        world (carla.World): CARLA dünyası.
        spawn_points (list): Spawn noktaları listesi.
        vehicle_config (dict): Araç fiziksel ve spawn ayarları.
    
    Returns:
        carla.Vehicle: Oluşturulan araç aktörü.
    """
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find(vehicle_config.get("vehicle_model", "vehicle.tesla.model3"))

    # Araç oluşturma denemeleri
    vehicle = None
    retries = vehicle_config.get("spawn_retries", 3)
    while retries > 0 and vehicle is None:
        try:
            spawn_point = random.choice(spawn_points)
            vehicle = world.spawn_actor(vehicle_bp, spawn_point)
            logging.info("Araç başarıyla oluşturuldu")
            break
        except Exception as e:
            logging.error(f"Araç oluşturma hatası: {e}")
            retries -= 1
            logging.info(f"Kalan deneme: {retries}")
            time.sleep(1)

    if vehicle is None:
        raise Exception("Araç oluşturulamadı!")

    # Fizik ayarları
    physics_control = vehicle.get_physics_control()
    physics_control.mass = vehicle_config.get("mass", 1500)  # kg
    physics_control.drag_coefficient = vehicle_config.get("drag_coefficient", 0.3)

    # Motor torku ayarları
    physics_control.torque_curve = vehicle_config.get("torque_curve", [
        carla.Vector2D(0, 400),
        carla.Vector2D(1500, 400),
        carla.Vector2D(3000, 300),
        carla.Vector2D(4500, 200)
    ])

    # Vites ayarları
    physics_control.gear_switch_time = vehicle_config.get("gear_switch_time", 0.1)
    physics_control.gear_ratio = vehicle_config.get("gear_ratio", [-1.0, 0.0, 3.5, 2.5, 1.5, 1.0])

    # Tekerlek sürtünme ayarları
    wheels = physics_control.wheels
    for wheel in wheels:
        wheel.tire_friction = vehicle_config.get("tire_friction", 3.5)
        wheel.damping_rate = vehicle_config.get("damping_rate", 1.5)
        wheel.max_steer_angle = vehicle_config.get("max_steer_angle", 70.0)

    # Physics control'ü uygula
    vehicle.apply_physics_control(physics_control)

    # Başlangıç hareketini engellememesi için el frenini devre dışı bırak
    vehicle.apply_control(carla.VehicleControl(
        throttle=0.0,
        steer=0.0,
        brake=0.0,
        hand_brake=vehicle_config.get("hand_brake", False),
        reverse=vehicle_config.get("reverse", False)
    ))

    return vehicle

class DrivingState(Enum):
    LANE_FOLLOWING = 1
    LANE_CHANGING = 2
    OBSTACLE_AVOIDANCE = 3
    INTERSECTION = 4
    TRAFFIC_LIGHT = 5

class HighLevelPlanner:
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.map = world.get_map()
        self.current_waypoints = []
        self.target_location = None
        self.closed_set = set()
        self.g_score = {}
        self.came_from = {}
        self.heap = []
    
    def set_target_location(self, target_location):
        self.target_location = target_location
    
    def a_star_search(self, start_waypoint, end_waypoint):
        """
        A* algoritmasını kullanarak başlangıç ve hedef arasında en kısa rotayı bulur.
        """
        open_set = []
        heapq.heappush(open_set, (0, start_waypoint))
        self.came_from = {}
        self.g_score = {start_waypoint: 0}
        self.closed_set = set()
        
        end_location = end_waypoint.transform.location

        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current == end_waypoint:
                # Yol oluşturuluyor
                route = []
                while current in self.came_from:
                    route.append(current)
                    current = self.came_from[current]
                return route[::-1]  # Ters çevir

            self.closed_set.add(current)
            
            for neighbor in current.next(2.0):  # Yakın waypointler
                if neighbor in self.closed_set:
                    continue
                
                tentative_g = self.g_score[current] + current.transform.location.distance(neighbor.transform.location)
                
                if neighbor not in self.g_score or tentative_g < self.g_score[neighbor]:
                    self.came_from[neighbor] = current
                    self.g_score[neighbor] = tentative_g
                    heuristic = neighbor.transform.location.distance(end_location)
                    f_score = tentative_g + heuristic
                    heapq.heappush(open_set, (f_score, neighbor))
        
        return []  # Rota bulunamadı

    def plan_route(self):
        if not self.target_location:
            logging.error("Hedef konum ayarlanmamış!")
            return

        start_waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True)
        end_waypoint = self.map.get_waypoint(self.target_location, project_to_road=True)
        self.current_waypoints = self.a_star_search(start_waypoint, end_waypoint)
        logging.info(f"{len(self.current_waypoints)} noktadan oluşan bir rota oluşturuldu.")

    def get_next_waypoint(self):
        """Bir sonraki waypoint'i al."""
        if not self.current_waypoints:
            return None
        return self.current_waypoints.pop(0)

class StateMachine:
    def __init__(self):
        self.state = DrivingState.LANE_FOLLOWING

    def transition(self, new_state):
        logging.info(f"Durum değişimi: {self.state.name} -> {new_state.name}")
        self.state = new_state

    def get_state(self):
        return self.state

class SensorFusion:
    def __init__(self):
        # Basit bir Kalman filtresi örneği (daha gelişmiş yöntemler için özelleştirilebilir)
        self.state = np.zeros(4)  # [x, y, dx, dy]
        self.P = np.eye(4) * 500.
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.R = np.eye(2) * 10.
        self.Q = np.eye(4)
    
    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, measurement):
        y = measurement - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state += K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
    
    def get_position(self):
        return self.state[:2]

class TrajectoryPlanner:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.safety_radius = 2.0  # Araç çevresinde güvenli mesafe
        self.num_points = 50  # Yörünge üzerindeki ara noktaların sayısı

    def plan_trajectory(self, current_position, target_position, obstacles):
        """
        Dinamik yörünge planlaması yapar.
        Args:
            current_position (tuple): (x, y) mevcut konum.
            target_position (tuple): (x, y) hedef konum.
            obstacles (list): Engellerin konumları [(x, y), ...].
        
        Returns:
            list: Planlanan yörünge noktaları.
        """
        trajectory = [current_position]
        for i in range(1, self.num_points):
            ratio = i / (self.num_points - 1)
            new_x = (1 - ratio) * current_position[0] + ratio * target_position[0]
            new_y = (1 - ratio) * current_position[1] + ratio * target_position[1]

            # Dinamik engel kontrolü
            new_x, new_y = self.avoid_obstacles((new_x, new_y), obstacles)

            trajectory.append((new_x, new_y))
        return trajectory

    def avoid_obstacles(self, point, obstacles):
        """
        Engellerden kaçınmak için bir noktayı düzeltir.
        Args:
            point (tuple): (x, y) noktasının koordinatları.
            obstacles (list): Engellerin konumları [(x, y), ...].
        
        Returns:
            tuple: Yeni (x, y) koordinatları.
        """
        x, y = point
        for obstacle in obstacles:
            obs_x, obs_y = obstacle
            distance = ((x - obs_x) ** 2 + (y - obs_y) ** 2) ** 0.5
            if distance < self.safety_radius:
                # Engel ile çakışmayı engellemek için yön değiştir
                angle = np.arctan2(y - obs_y, x - obs_x)
                x += np.cos(angle) * (self.safety_radius - distance)
                y += np.sin(angle) * (self.safety_radius - distance)
        return x, y

    def smooth_trajectory(self, trajectory):
        """
        Yörüngeyi yumuşatır.
        Args:
            trajectory (list): [(x, y), ...] yörünge noktaları.
        
        Returns:
            list: Yumuşatılmış yörünge.
        """
        smoothed_trajectory = [trajectory[0]]
        for i in range(1, len(trajectory) - 1):
            prev_point = trajectory[i - 1]
            current_point = trajectory[i]
            next_point = trajectory[i + 1]

            new_x = (prev_point[0] + current_point[0] + next_point[0]) / 3
            new_y = (prev_point[1] + current_point[1] + next_point[1]) / 3

            smoothed_trajectory.append((new_x, new_y))
        smoothed_trajectory.append(trajectory[-1])
        return smoothed_trajectory
        
class LaneDetector:
    def __init__(self):
        self.left_fit = deque(maxlen=10)  # Daha fazla geçmiş çerçeve
        self.right_fit = deque(maxlen=10)
        self.lane_width = None
        self.last_lane_center = None
        self.min_lane_width = 2.5  # metre cinsinden minimum genişlik
        self.max_lane_width = 4.5 # metre cinsinden maksimum genişlik
        self.alpha = 0.2  # Üstel hareketli ortalama için ağırlık
        self.last_valid_detection = None
        self.lost_counter = 0
        self.max_lost_frames = 5
        self.current_speed = 0.0
        self.last_steering_angle = 0.0  # Initialize last_steering_angle

    def perspective_transform(self, img):
        height, width = img.shape[:2]
        # Dinamik kaynak noktaları (görüntü boyutuna göre)
        src = np.float32([
            [width * 0.45, height * 0.65],  # Üst sol
            [width * 0.55, height * 0.65],  # Üst sağ
            [width * 0.9, height],          # Alt sağ
            [width * 0.1, height]           # Alt sol
        ])
        dst = np.float32([
            [width * 0.2, 0],               # Üst sol
            [width * 0.8, 0],               # Üst sağ
            [width * 0.8, height],          # Alt sağ
            [width * 0.2, height]           # Alt sol
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)  # Ters dönüşüm için
        warped = cv2.warpPerspective(img, M, (width, height))
        return warped, M, Minv

    def detect_lane_pixels(self, img):
        # Renk dönüşümleri ve eşikleme kombinasyonu
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        # Kenar tespiti
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # Adaptif eşikleme
        adaptive_thresh = cv2.adaptiveThreshold(
            l_channel, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 
            15, 
            8
        )

        # Sobel gradyan hesaplama
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Eşikleme kombinasyonu
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1

        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= 170) & (s_channel <= 255)] = 1

        # Tüm maskelerin kombinasyonu
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[
            (s_binary == 1) | 
            (sxbinary == 1) | 
            (adaptive_thresh > 0) |
            (edges > 0)
        ] = 1

        # Gürültü azaltma
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(combined_binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        return cleaned

    def fit_polynomial(self, binary_warped):
        # Histogram oluşturma
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        midpoint = len(histogram)//2

        # Sol ve sağ başlangıç noktaları
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Kayan pencere parametreleri
        nwindows = 9
        window_height = binary_warped.shape[0]//nwindows
        window_width = 100
        margin = window_width // 2

        # Etkin pikselleri bul
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Pencere pozisyonlarını takip et
        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            # Pencere sınırlarını belirle
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # İyi pikselleri tanımla
            good_left_inds = (
                (nonzeroy >= win_y_low) & 
                (nonzeroy < win_y_high) & 
                (nonzerox >= win_xleft_low) & 
                (nonzerox < win_xleft_high)
            ).nonzero()[0]
            
            good_right_inds = (
                (nonzeroy >= win_y_low) & 
                (nonzeroy < win_y_high) & 
                (nonzerox >= win_xright_low) & 
                (nonzerox < win_xright_high)
            ).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            minpix = 50
            # Sonraki pencere merkezini güncelle
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Dizileri birleştir
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Yeterli piksel bulunamadı
            logging.warning("Yeterli şerit pikseli bulunamadı!")
            return None, None

        # Şerit piksellerinin x ve y pozisyonlarını çıkar
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Polinom uydurma
        if len(leftx) > 300 and len(rightx) > 300:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            
            # Uydurmanın geçerliliğini kontrol et
            if self.validate_lane_detection(left_fit, right_fit):
                self.left_fit.append(left_fit)
                self.right_fit.append(right_fit)
                self.lost_counter = 0
                return left_fit, right_fit

        # Tespit başarısız oldu
        self.lost_counter += 1
        if self.lost_counter > self.max_lost_frames:
            # Son geçerli tespiti kullan
            if len(self.left_fit) > 0 and len(self.right_fit) > 0:
                return np.mean(self.left_fit, axis=0), np.mean(self.right_fit, axis=0)
        return None, None

    def validate_lane_detection(self, left_fit, right_fit, current_speed=0.0):
        """
        Şerit tespitinin geçerliliğini kontrol eder ve bir güven skoru hesaplar.

        Args:
            left_fit (numpy.ndarray): Sol şerit için polinom katsayıları.
            right_fit (numpy.ndarray): Sağ şerit için polinom katsayıları.
            current_speed (float): Araç hızı (km/h).

        Returns:
            bool: Tespit geçerli mi?
            float: Güven skoru (0.0 - 1.0 arası).
        """
        try:
            # 1. Test görüntüsü oluştur
            h = 720  # Görüntü yüksekliği
            plot_y = np.linspace(0, h - 1, h)

            # Şerit eğrilerini hesapla
            left_x = left_fit[0] * plot_y**2 + left_fit[1] * plot_y + left_fit[2]
            right_x = right_fit[0] * plot_y**2 + right_fit[1] * plot_y + right_fit[2]

            # Şerit genişliğini hesapla
            lane_width_pixels = np.mean(right_x - left_x)
            lane_width_meters = lane_width_pixels * (3.7 / 700)  # Pikselden metreye dönüşüm

            # 2. Şerit genişliği doğrulama
            min_lane_width = 2.5  # metre
            max_lane_width = 4.5  # metre
            if not (min_lane_width <= lane_width_meters <= max_lane_width):
                logging.warning(f"Geçersiz şerit genişliği: {lane_width_meters:.2f}m")
                return False, 0.5  # Orta güven skoru ile başarısız

            # 3. Şerit paralellik kontrolü
            left_slope = np.gradient(left_x)
            right_slope = np.gradient(right_x)
            slope_diff = np.abs(left_slope - right_slope)
            if np.mean(slope_diff) > 0.3:  # Paralellik farkı büyükse
                logging.warning(f"Şerit eğimleri paralel değil: Ortalama fark {np.mean(slope_diff):.2f}")
                return False, 0.6

            # 4. Eğrilik kontrolü
            left_curvature = ((1 + (2 * left_fit[0] * plot_y + left_fit[1])**2)**1.5) / np.abs(2 * left_fit[0])
            right_curvature = ((1 + (2 * right_fit[0] * plot_y + right_fit[1])**2)**1.5) / np.abs(2 * right_fit[0])
            curvature_diff = np.abs(np.mean(left_curvature) - np.mean(right_curvature))
            if curvature_diff > 2000:  # Eğrilik farkı çok büyükse
                logging.warning(f"Şerit eğrilik farkı fazla: {curvature_diff:.2f}")
                return False, 0.7

            # 5. Dinamik toleranslar (hız bazlı genişlik ve eğrilik toleransı)
            if current_speed > 30:
                max_lane_width += 0.5  # Hız arttıkça şerit genişliği toleransı
                curvature_diff += 1000  # Eğrilik farkı toleransı

            # 6. Güven skoru hesaplama
            width_score = 1.0 - min(max(abs(lane_width_meters - 3.7) / 1.0, 0.0), 1.0)
            slope_score = 1.0 - min(np.mean(slope_diff) / 0.3, 1.0)
            curvature_score = 1.0 - min(curvature_diff / 2000.0, 1.0)
            confidence = np.mean([width_score, slope_score, curvature_score])

            return confidence > 0.7, confidence

        except Exception as e:
            logging.error(f"Şerit doğrulama hatası: {e}")
            return False, 0.0


    def measure_lane_curvature(self, ploty, left_fit, right_fit):
        """Şerit eğriliğini gerçek dünya ölçeklerinde hesapla"""
        # Dönüşüm faktörleri
        ym_per_pix = 30/720  # metre/piksel y ekseni
        xm_per_pix = 3.7/700  # metre/piksel x ekseni

        # Şerit çizgilerini hesapla
        leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Gerçek dünya koordinatlarına dönüştür
        left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

        # Eğrilik hesaplama
        y_eval = np.max(ploty)
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        return left_curverad, right_curverad

    def calculate_vehicle_position(self, img_shape, left_fit, right_fit):
        """Aracın şeritte konumunu hesapla"""
        try:
            height, width = img_shape[0], img_shape[1]

            # Şerit çizgilerinin alt noktalarını hesapla
            left_bottom = left_fit[0]*height**2 + left_fit[1]*height + left_fit[2]
            right_bottom = right_fit[0]*height**2 + right_fit[1]*height + right_fit[2]

            # Şerit merkezi
            lane_center = (left_bottom + right_bottom) / 2
            image_center = width / 2

            # Şerit genişliği ve güvenilirlik kontrolü
            current_lane_width = right_bottom - left_bottom
            if current_lane_width <= self.min_lane_width * (700 / 3.7) or current_lane_width > self.max_lane_width * (700 / 3.7):
                if self.last_valid_detection is not None:
                    return self.last_valid_detection
                return 0.0, self.lane_width if self.lane_width else 3.5

            # Metre/piksel dönüşümü
            xm_per_pix = 3.7 / current_lane_width

            # Şerit genişliğini yumuşat
            if self.lane_width is None:
                self.lane_width = current_lane_width * xm_per_pix
            else:
                self.lane_width = 0.95 * self.lane_width + 0.05 * (current_lane_width * xm_per_pix)

            # Offset hesaplama ve yumuşatma
            offset = (image_center - lane_center) * xm_per_pix
            if self.last_valid_detection is not None:
                last_offset, _ = self.last_valid_detection
                offset = self.alpha * offset + (1 - self.alpha) * last_offset

            self.last_valid_detection = (offset, self.lane_width)
            return offset, self.lane_width

        except Exception as e:
            logging.error(f"Position calculation error: {e}")
            return 0.0, 3.5

    def get_steering_angle(self, img, debug=False, current_speed=0.0):
        """
        Direksiyon açısını hesaplayan temel algoritma.
        
        Args:
            img (numpy.ndarray): Kameradan alınan görüntü.
            debug (bool): Hata ayıklama modunu etkinleştir.
            current_speed (float): Aracın mevcut hızı.
    
        Returns:
            tuple: (Direksiyon açısı, debug bilgileri)
        """
        try:
            # 1. Görüntü ön işleme
            processed = self.preprocess_image(img)
            
            # 2. Şerit piksel tespiti ve perspektif dönüşüm
            binary = self.detect_lane_pixels(processed)
            warped, M, Minv = self.perspective_transform(binary)
    
            # 3. Şerit eğrisi tespiti
            left_fit, right_fit = self.fit_polynomial(warped)
            if left_fit is None or right_fit is None:
                return self.handle_detection_failure()  # Şerit tespiti başarısızsa varsayılan açıyı döndür
    
            # 4. Şerit eğriliği ve pozisyon hesaplama
            ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
            left_curverad, right_curverad = self.measure_lane_curvature(ploty, left_fit, right_fit)
            offset, lane_width = self.calculate_vehicle_position(img.shape, left_fit, right_fit)
    
            # 5. Dinamik Direksiyon Kontrolü
            curvature = (left_curverad + right_curverad) / 2  # Ortalama eğrilik
            base_angle = self.calculate_base_steering_angle(curvature)
    
            # 6. Hassas Direksiyon Ayarı (Offset Düzeltmesi)
            offset_correction = self.calculate_offset_correction(offset, current_speed)
            final_angle = self.combine_steering_inputs(base_angle, offset_correction)
    
            # 7. Direksiyon Açısını Yumuşatma
            smoothed_angle = self.apply_steering_smoothing(final_angle)
    
            # Debug bilgilerini hazırla
            if debug:
                debug_info = self.prepare_debug_info(
                    warped, binary, left_curverad, right_curverad,
                    offset, lane_width, smoothed_angle, img, Minv, 
                    left_fit, right_fit
                )
                return smoothed_angle, debug_info
    
            return smoothed_angle, None
    
        except Exception as e:
            logging.error(f"Direksiyon açısı hesaplama hatası: {e}")
            return self.last_steering_angle, None
    
    def preprocess_image(self, img):
        """Görüntü ön işleme"""
        # Gürültü azaltma
        denoised = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Kontrast iyileştirme
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def handle_detection_failure(self):
        """Şerit tespiti başarısız olduğunda"""
        if len(self.left_fit) > 0 and len(self.right_fit) > 0:
            # Son geçerli tespitlerin ortalamasını kullan
            return self.last_steering_angle, None
        return 0.0, None

    def calculate_base_steering_angle(self, curvature):
        """Eğriliğe göre temel direksiyon açısı"""
        if curvature == 0:
            return 0.0
            
        # Eğrilik bazlı açı
        base_angle = np.arctan(2.7 / curvature)  # 2.7m araç uzunluğu
        
        # [-1, 1] aralığına normalize et
        normalized_angle = base_angle / np.pi
        
        return np.clip(normalized_angle, -1.0, 1.0)

    def calculate_offset_correction(self, offset, current_speed):
        """Hıza bağlı offset düzeltmesi"""
        # Hıza bağlı düzeltme faktörü
        speed_factor = np.clip(current_speed / 30.0, 0.5, 1.5)
        
        # Offset bazlı düzeltme
        correction = 0.4 * offset / speed_factor
        
        return np.clip(correction, -0.5, 0.5)

    def combine_steering_inputs(self, base_angle, offset_correction):
        """Direksiyon girdilerini birleştir"""
        combined = base_angle + offset_correction
        return np.clip(combined, -1.0, 1.0)

    def apply_steering_smoothing(self, angle):
        """Direksiyon açısı yumuşatma"""
        # Üstel hareketli ortalama
        smoothed = self.alpha * angle + (1 - self.alpha) * self.last_steering_angle
        
        # Maksimum değişim hızı sınırlaması
        max_change = 0.1
        clamped = np.clip(
            smoothed, 
            self.last_steering_angle - max_change,
            self.last_steering_angle + max_change
        )
        
        self.last_steering_angle = clamped
        return clamped

    def prepare_debug_info(self, warped, binary, left_curverad, right_curverad,
                           offset, lane_width, steering_angle, img, Minv, 
                           left_fit, right_fit):
        """Debug bilgilerini hazırla"""
        # Görselleştirme
        visualization = self.visualize_detection(
            img, warped, Minv, left_fit, right_fit
        )
        
        return {
            'warped': warped,
            'binary': binary,
            'left_curverad': left_curverad,
            'right_curverad': right_curverad,
            'offset': offset,
            'lane_width': lane_width,
            'steering_angle': steering_angle,
            'visualization': visualization
        }

    def visualize_detection(self, img, warped, Minv, left_fit, right_fit):
        """Gelişmiş görselleştirme"""
        try:
            # Temel görselleştirme
            visualization = self.visualize_lanes(img, warped, Minv, left_fit, right_fit)
            
            # Merkez çizgisi
            height, width = img.shape[:2]
            cv2.line(visualization, 
                    (width//2, height),
                    (width//2, height-50),
                    (0, 255, 0), 2)
            
            # Tespit edilen şerit merkezi
            lane_center = int((left_fit[0]*height**2 + left_fit[1]*height + left_fit[2] +
                             right_fit[0]*height**2 + right_fit[1]*height + right_fit[2]) / 2)
            cv2.line(visualization,
                    (lane_center, height),
                    (lane_center, height-50),
                    (255, 0, 0), 2)
            
            return visualization
            
        except Exception as e:
            logging.error(f"Visualization error: {e}")
            return img

    def visualize_lanes(self, img, warped, Minv, left_fit, right_fit):
        """Temel şerit görselleştirme"""
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        out_img = np.copy(img)
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Şerit bölgesini doldur
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        cv2.polylines(color_warp, np.int_([pts_left]), False, (255, 0, 0), 15)
        cv2.polylines(color_warp, np.int_([pts_right]), False, (0, 0, 255), 15)
        
        # Perspektifi geri dönüştür
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
        return cv2.addWeighted(out_img, 1, newwarp, 0.3, 0)

class PIDController:
    def __init__(self, Kp, Ki, Kd, output_limits=(None, None)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0
        self.last_update_time = time.time()
        self.min_output, self.max_output = output_limits
        self.reset_threshold = 1.0  # Büyük hata değişimlerinde integratörü sıfırla

    def update(self, error, delta_time=None):
        # Zaman farkını hesapla
        current_time = time.time()
        if delta_time is None:
            delta_time = current_time - self.last_update_time
        self.last_update_time = current_time

        # Anti-windup için integral sınırlama
        if abs(error) > self.reset_threshold:
            self.integral = 0
        else:
            self.integral += error * delta_time
            if self.min_output is not None:
                self.integral = max(self.min_output, self.integral)
            if self.max_output is not None:
                self.integral = min(self.max_output, self.integral)

        # Türev hesaplama (düşük geçişli filtre ile)
        derivative = (error - self.prev_error) / delta_time
        self.prev_error = error

        # PID çıktısı
        output = (self.Kp * error + 
                 self.Ki * self.integral + 
                 self.Kd * derivative)

        # Çıktı sınırlama
        if self.min_output is not None:
            output = max(self.min_output, output)
        if self.max_output is not None:
            output = min(self.max_output, output)

        return output

    def set_parameters(self, Kp, Ki, Kd):
        """PID parametrelerini dinamik olarak güncelle"""
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        # Parametreler değiştiğinde integratörü sıfırla
        self.integral = 0

class LaneFollowingController:
    def __init__(self, high_level_planner, state_machine, sensor_fusion, trajectory_planner):
        self.lane_detector = LaneDetector()
        self.steering_history = deque(maxlen=10)
        self.pid = PIDController(Kp=PID_KP, Ki=PID_KI, Kd=PID_KD, output_limits=(-1.0, 1.0))
        self.current_speed = 0.0
        self.recovery_mode = False
        self.recovery_counter = 0
        self.max_recovery_frames = 30
        self.min_confidence_threshold = 0.6
        self.high_level_planner = high_level_planner
        self.state_machine = state_machine
        self.sensor_fusion = sensor_fusion
        self.trajectory_planner = trajectory_planner

    def update_pid_parameters(self):
        """Hıza ve duruma bağlı dinamik PID parametreleri"""
        current_state = self.state_machine.get_state()
        if current_state == DrivingState.OBSTACLE_AVOIDANCE:
            # Engel kaçınma modunda daha hassas kontrol
            self.pid.set_parameters(Kp=1.5, Ki=0.0, Kd=0.5)
        elif current_state == DrivingState.LANE_CHANGING:
            # Şerit değiştirme modunda
            self.pid.set_parameters(Kp=1.2, Ki=0.0, Kd=0.4)
        elif self.current_speed < 10:
            # Düşük hızda
            self.pid.set_parameters(Kp=1.2, Ki=0.0, Kd=0.4)
        elif 10 <= self.current_speed < 20:
            # Orta hızda
            self.pid.set_parameters(Kp=1.0, Ki=0.0, Kd=0.3)
        else:
            # Yüksek hızda
            self.pid.set_parameters(Kp=0.8, Ki=0.0, Kd=0.2)

    def update(self, image, debug=False, current_speed=0.0, target_location=None):
        """Ana güncelleme fonksiyonu"""
        self.current_speed = current_speed
        
        try:
            # Hedef konum ve rota planlama
            if target_location:
                self.high_level_planner.set_target_location(target_location)
                self.high_level_planner.plan_route()
            
            next_waypoint = self.high_level_planner.get_next_waypoint()
            if next_waypoint:
                target_position = (next_waypoint.transform.location.x, next_waypoint.transform.location.y)
            else:
                target_position = (self.high_level_planner.target_location.x, self.high_level_planner.target_location.y)
    
            # Şerit tespiti ve direksiyon açısı hesaplama
            steering_angle, debug_info = self.lane_detector.get_steering_angle(image, debug=debug, current_speed=current_speed)
            
            # Sensör füzyonu ile pozisyon güncelleme
            detected_position = self.sensor_fusion.get_position()
            self.sensor_fusion.predict()
            self.sensor_fusion.update(np.array(target_position))
    
            # Şerit tespiti güvenilirlik kontrolü
            if debug_info and 'lane_width' in debug_info:
                confidence = self.calculate_detection_confidence(debug_info)
                if confidence < self.min_confidence_threshold:
                    self.state_machine.transition(DrivingState.OBSTACLE_AVOIDANCE)
                    return self.handle_low_confidence()
    
            # Durum makinesi ve hareket kontrolü
            current_state = self.state_machine.get_state()
    
            if current_state == DrivingState.LANE_FOLLOWING:
                # Şerit takibi için PID kontrolü
                pass
            elif current_state == DrivingState.LANE_CHANGING:
                # Şerit değiştirme işlemi
                steering_angle = self.perform_lane_change()
            elif current_state == DrivingState.OBSTACLE_AVOIDANCE:
                # Engel kaçınma süreci
                lidar_data = self.sensor_fusion.get_position()  # LiDAR verisi alınır
                steering_angle, speed_factor = self.avoid_obstacle(lidar_data, self.current_speed)
                return {
                    'steering': steering_angle,
                    'speed_factor': speed_factor,
                    'debug_info': None
                }
    
            # Direksiyon açısını yumuşatma
            self.steering_history.append(steering_angle)
            smoothed_angle = self.calculate_smoothed_steering()
    
            # PID kontrolü ile direksiyon açısını ve hız faktörünü güncelle
            self.update_pid_parameters()
            final_steering = self.pid.update(smoothed_angle)
    
            # Hız kontrolü ve throttle hesaplama
            speed_factor = self.calculate_speed_factor(final_steering)
    
            # Risk analizi ve uyumlu kontrol
            if debug:
                self.log_debug_information(debug_info, final_steering, speed_factor)
    
            # Komutları döndür
            return {
                'steering': final_steering,
                'speed_factor': speed_factor,
                'debug_info': debug_info
            }
    
        except Exception as e:
            logging.error(f"Controller update error: {e}")
            return self.handle_error()

    def calculate_detection_confidence(self, debug_info):
        """Şerit tespiti güvenilirlik skoru"""
        confidence = 1.0
            
        # Şerit genişliği kontrolü
        lane_width = debug_info.get('lane_width', 0)
        if not (2.5 < lane_width < 4.5):
            confidence *= 0.5
            
        # Eğrilik kontrolü
        left_curve = debug_info.get('left_curverad', 0)
        right_curve = debug_info.get('right_curverad', 0)
        if abs(left_curve - right_curve) > 2000:
            confidence *= 0.7
            
        # Offset kontrolü
        offset = abs(debug_info.get('offset', 0))
        if offset > 1.0:
            confidence *= 0.8
            
        return confidence

    def handle_low_confidence(self):
        """Düşük güvenilirlik durumu kontrolü"""
        self.state_machine.transition(DrivingState.OBSTACLE_AVOIDANCE)
        self.recovery_mode = True
        self.recovery_counter += 1
            
        if self.recovery_counter > self.max_recovery_frames:
            return {
                'steering': 0.0,
                'speed_factor': 0.8,
                'debug_info': None
            }
        
        if len(self.steering_history) > 0:
            last_steering = self.steering_history[-1]
            return {
                'steering': last_steering * 0.8,  # Azaltılmış etki
                'speed_factor': 0.5,
                'debug_info': None
            }
            
        return {
            'steering': 0.0,
            'speed_factor': 0.5,
            'debug_info': None
        }

    def handle_error(self):
        """Hata durumu kontrolü"""
        return {
            'steering': 0.0,
            'speed_factor': 0.0,
            'debug_info': None
        }

    def calculate_smoothed_steering(self):
        """Yumuşatılmış direksiyon açısı hesaplama"""
        if len(self.steering_history) == 0:
            return 0.0
                
        # Medyan filtre ile aykırı değerleri temizle
        filtered_angles = np.array(self.steering_history)
        median = np.median(filtered_angles)
        mask = np.abs(filtered_angles - median) < 0.5
        valid_angles = filtered_angles[mask]
            
        if len(valid_angles) == 0:
            return median
            
        # Ağırlıklı ortalama (son değerlere daha fazla ağırlık ver)
        weights = np.exp(np.linspace(0, 1, len(valid_angles)))
        return np.average(valid_angles, weights=weights)

    def calculate_speed_factor(self, steering_angle):
        """Direksiyon açısına göre hız faktörü"""
        abs_angle = abs(steering_angle)
            
        if abs_angle > 0.5:
            return 0.5  # Keskin virajlarda
        elif abs_angle > 0.3:
            return 0.7  # Orta şiddetli virajlarda
        elif abs_angle > 0.1:
            return 0.85  # Hafif virajlarda
                
        return 1.0  # Düz yolda

    def perform_lane_change(self, direction="right"):
        """
        Şerit değiştirme işlemini gerçekleştir.

        Args:
            direction (str): "right" (sağ) veya "left" (sol) şerit değişikliği.

        Returns:
            float: Direksiyon açısı (-1.0 ile 1.0 arasında).
        """
        try:
            # 1. Şerit bilgilerini alın
            vehicle_transform = self.high_level_planner.vehicle.get_transform()
            current_position = np.array([vehicle_transform.location.x, vehicle_transform.location.y])
            yaw = np.radians(vehicle_transform.rotation.yaw)

            # 2. Şerit genişliği ve hedef pozisyon
            lane_width = 3.5  # Ortalama şerit genişliği (metre)
            offset = lane_width if direction == "right" else -lane_width
            target_position = current_position + np.array([np.cos(yaw) * offset, np.sin(yaw) * offset])

            # 3. Engelleri kontrol edin
            lidar_position = self.sensor_fusion.get_position()  # LiDAR pozisyonu
            safe_to_change = self.check_lane_clear(lidar_position, direction)
            if not safe_to_change:
                logging.warning(f"Şerit değiştirme başarısız! {direction} şerit dolu.")
                return 0.0  # Şerit değiştirme iptal

            # 4. Yörünge planlama
            trajectory = self.trajectory_planner.plan_trajectory(current_position, target_position, [])

            # 5. Direksiyon açısı hesaplama
            # Burada basitçe hedef noktaya doğru yönlendirme yapılmaktadır
            if len(trajectory) > 0:
                next_point = trajectory[0]
                dx = next_point[0] - current_position[0]
                dy = next_point[1] - current_position[1]
                angle_to_point = np.arctan2(dy, dx)
                angle_diff = angle_to_point - yaw
                steering_angle = np.clip(angle_diff, -1.0, 1.0)
                return steering_angle  # Bir sonraki kontrol döngüsüne aktar

            return 0.0

        except Exception as e:
            logging.error(f"Şerit değiştirme hatası: {e}")
            return  0.0

    def check_lane_clear(self, lidar_position, direction):
        """
        Hedef şeridin engellerden temiz olup olmadığını kontrol eder.

        Args:
            lidar_position (numpy.ndarray): LiDAR pozisyonu.
            direction (str): "right" veya "left".

        Returns:
            bool: Şerit temiz mi? True/False
        """
        lane_offset = 3.5 if direction == "right" else -3.5
        # Şerit merkezindeki engel kontrolü (basit bir mesafe kontrolü)
        distance = np.linalg.norm(lidar_position - np.array([lane_offset, 0]))
        if distance < 5.0:  # 5 metre içinde engel varsa
            return False
        return True

    def plan_lane_change_trajectory(self, start_position, target_position):
        """
        Şerit değişikliği için bir geçiş yörüngesi planlar.

        Args:
            start_position (tuple): (x, y) başlangıç konumu.
            target_position (tuple): (x, y) hedef konumu.

        Returns:
            list: Yörünge noktalarının listesi.
        """
        return self.trajectory_planner.plan_trajectory(start_position, target_position, [])

    def avoid_obstacle(self, lidar_data, vehicle_speed):
        """
        Geliştirilmiş engel kaçınma algoritması.
        Dinamik yörünge planlama ve hız ayarları içerir.
    
        Args:
            lidar_data (numpy.ndarray): LiDAR verisi (N, 4).
            vehicle_speed (float): Araç hızı (km/h).
    
        Returns:
            float: Direksiyon açısı (-1.0 ile 1.0 arasında).
            float: Hız faktörü (0.0 - 1.0 arasında).
        """
        try:
            # 1. LiDAR verisi analizi
            distances = np.sqrt(lidar_data[:, 0]**2 + lidar_data[:, 1]**2)
            angles = np.degrees(np.arctan2(lidar_data[:, 1], lidar_data[:, 0]))
    
            # Risk bölgeleri tanımı (hıza bağlı)
            critical_distance = 1.5 + (vehicle_speed / 20.0)  # Kritik mesafe
            danger_distance = 3.0 + (vehicle_speed / 15.0)  # Tehlike mesafesi
            safe_distance = 5.0 + (vehicle_speed / 10.0)  # Güvenli mesafe
    
            # Engel kategorileri
            critical_zone = distances < critical_distance
            danger_zone = (distances < danger_distance) & ~critical_zone
            safe_zone = (distances < safe_distance) & ~danger_zone & ~critical_zone
    
            # Engel analizine göre karar
            if np.any(critical_zone):
                logging.warning("Kritik mesafede engel tespit edildi! Acil kaçınma manevrası uygulanıyor.")
                return self.emergency_avoidance(lidar_data), 0.0
    
            if np.any(danger_zone):
                logging.info("Tehlike mesafesinde engel tespit edildi. Yavaşlama ve kaçınma başlatılıyor.")
                return self.plan_dynamic_avoidance(lidar_data), 0.5
    
            if np.any(safe_zone):
                logging.info("Güvenli mesafede engel tespit edildi. Proaktif kaçınma.")
                return self.plan_proactive_avoidance(lidar_data), 0.8
    
            # Engel yoksa normal sürüşe devam
            return 0.0, 1.0
    
        except Exception as e:
            logging.error(f"Engel kaçınma hatası: {e}")
            return 0.0, 0.0
    
    def emergency_avoidance(self, lidar_data):
        """
        Kritik durumlarda acil kaçınma manevrası.

        Args:
            lidar_data (numpy.ndarray): LiDAR verisi.

        Returns:
            float: Direksiyon açısı (-1.0 ile 1.0 arasında).
        """
        try:
            # En yakın engeli belirle
            distances = np.sqrt(lidar_data[:, 0]**2 + lidar_data[:, 1]**2)
            angles = np.degrees(np.arctan2(lidar_data[:, 1], lidar_data[:, 0]))
            closest_index = np.argmin(distances)

            # Engel sağda mı solda mı?
            if angles[closest_index] > 0:
                logging.info("Engel sağda. Sola kaçınıyor.")
                return -0.8  # Sola dön
            else:
                logging.info("Engel solda. Sağa kaçınıyor.")
                return 0.8  # Sağa dön

        except Exception as e:
            logging.error(f"Acil kaçınma hatası: {e}")
            return 0.0

    def plan_dynamic_avoidance(self, lidar_data):
        """
        Dinamik engel kaçınma yörüngesi planlama.

        Args:
            lidar_data (numpy.ndarray): LiDAR verisi.

        Returns:
            float: Direksiyon açısı (-1.0 ile 1.0 arasında).
        """
        try:
            # Dinamik kaçınma için en uygun yönü bul
            distances = np.sqrt(lidar_data[:, 0]**2 + lidar_data[:, 1]**2)
            angles = np.degrees(np.arctan2(lidar_data[:, 1], lidar_data[:, 0]))

            # Sol ve sağ bölgelere ayır
            left_distances = distances[angles < 0]
            right_distances = distances[angles > 0]

            # Hangi taraf daha güvenli?
            if np.mean(left_distances) > np.mean(right_distances):
                logging.info("Dinamik kaçınma: Sola yönleniyor.")
                return -0.5  # Sola kaçın
            else:
                logging.info("Dinamik kaçınma: Sağa yönleniyor.")
                return 0.5  # Sağa kaçın

        except Exception as e:
            logging.error(f"Dinamik kaçınma planlama hatası: {e}")
            return 0.0

    def plan_proactive_avoidance(self, lidar_data):
        """
        Güvenli mesafede proaktif kaçınma.

        Args:
            lidar_data (numpy.ndarray): LiDAR verisi.

        Returns:
            float: Direksiyon açısı (-1.0 ile 1.0 arasında).
        """
        try:
            # En geniş güvenli alanı belirle
            distances = np.sqrt(lidar_data[:, 0]**2 + lidar_data[:, 1]**2)
            angles = np.degrees(np.arctan2(lidar_data[:, 1], lidar_data[:, 0]))

            # Güvenli alan maskesi
            safe_zone = distances > 5.0

            # Güvenli alanın ortasına yönlen
            if np.any(safe_zone):
                safe_angles = angles[safe_zone]
                mean_angle = np.mean(safe_angles)
                steering = np.clip(mean_angle / 45.0, -1.0, 1.0)  # Direksiyon aralığına dönüştür
                logging.info(f"Proaktif kaçınma: Ortalama güvenli açı {mean_angle:.2f}, Direksiyon açısı: {steering:.2f}")
                return steering

            return 0.0

        except Exception as e:
            logging.error(f"Proaktif kaçınma hatası: {e}")
            return 0.0


class SensorManager:
    def __init__(self, vehicle, controller, high_level_planner):
        self.vehicle = vehicle
        self.controller = controller
        self.high_level_planner = high_level_planner
        self.blueprint_library = vehicle.get_world().get_blueprint_library()
        self.sensor_data_lock = threading.Lock()
        self.visualization_queue = visualization_queue  # Global kuyruğu kullanabilirsiniz
        self.sensors = []  # Sensör referanslarını saklamak için liste
        self.latest_camera_data = None
        self.latest_lidar_data = None
        self.setup_sensors()

    def setup_sensors(self):
        # Kamera Sensörü
        camera_bp = self.blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(CAMERA_WIDTH))
        camera_bp.set_attribute("image_size_y", str(CAMERA_HEIGHT))
        camera_bp.set_attribute("fov", "110")
        camera_bp.set_attribute("motion_blur_intensity", "0.5")
        camera_bp.set_attribute("lens_circle_falloff", "3.0")
        camera_bp.set_attribute("chromatic_aberration_intensity", "0.5")
        camera_spawn_point = carla.Transform(
            carla.Location(x=1.5, z=2.4),
            carla.Rotation(pitch=-15)
        )
        camera = self.vehicle.get_world().spawn_actor(camera_bp, camera_spawn_point, attach_to=self.vehicle)
        self.sensors.append(camera)
        camera.listen(self.camera_callback)

        # LiDAR Sensörü
        lidar_bp = self.blueprint_library.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", "16")         # Kanal sayısını 16'ya düşürdük
        lidar_bp.set_attribute("points_per_second", "250000")  # Nokta sayısını artırmak yerine varsayılanı koruduk
        lidar_bp.set_attribute("rotation_frequency", "20")
        lidar_bp.set_attribute("range", "85")            # Menzil artırıldı
        lidar_bp.set_attribute("upper_fov", "15")        # Üst görüş açısı genişletildi
        lidar_bp.set_attribute("lower_fov", "-35")       # Alt görüş açısı genişletildi
        lidar_bp.set_attribute("horizontal_fov", "135")  # Yatay görüş açısı genişletildi
        lidar_bp.set_attribute("noise_stddev", "0.005")  # Daha düşük gürültü
        lidar_bp.set_attribute("noise_seed", "1")
        lidar_bp.set_attribute("dropoff_general_rate", "0.05")  # Düşük kayıp oranı
        lidar_bp.set_attribute("dropoff_intensity_limit", "0.9") # Yüksek yoğunluk limiti
        lidar_bp.set_attribute("dropoff_zero_intensity", "0.2")  # Düşük sıfır yoğunluk oranı
        lidar_transform = carla.Transform(
            carla.Location(x=2.0, z=1.8),
            carla.Rotation(pitch=-5)
        )
        lidar = self.vehicle.get_world().spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
        self.sensors.append(lidar)
        lidar.listen(self.lidar_callback)

        # Collision Sensörü
        collision_bp = self.blueprint_library.find("sensor.other.collision")
        collision = self.vehicle.get_world().spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.sensors.append(collision)
        collision.listen(self.collision_callback)

        # Kamera içsel parametreleri
        fov = camera_bp.get_attribute("fov").as_float()
        focal_length = CAMERA_WIDTH / (2 * np.tan(np.radians(fov) / 2))
        self.intrinsic_matrix = np.array([
            [focal_length, 0, CAMERA_WIDTH / 2],
            [0, focal_length, CAMERA_HEIGHT / 2],
            [0, 0, 1]
        ])

        # Kamera ve LiDAR arasındaki dönüşüm matrisini hesaplayın
        camera_transform = camera.get_transform()
        lidar_transform = lidar.get_transform()
        camera_world_matrix = get_transformation_matrix(camera_transform)
        lidar_world_matrix = get_transformation_matrix(lidar_transform)
        self.lidar_to_camera_matrix = np.linalg.inv(camera_world_matrix) @ lidar_world_matrix

    def collision_callback(self, event):
        logging.critical("Çarpışma tespit edildi! Araç durduruluyor.")
        control_vehicle(self.vehicle, 0.0, 0.0, 0.0)  # Araç durduruluyor

    def camera_callback(self, image):
        try:
            # Aracın hızını alın
            vehicle_velocity = self.vehicle.get_velocity()
            speed_magnitude = np.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2) * 3.6  # km/h

            # Kamera verisini işleyin
            camera_image = np.array(image.raw_data).reshape((image.height, image.width, 4))[:, :, :3]

            if lane_following:
                # Aracın mevcut hızını kontrolcüyü güncellemek için iletin
                steering_angle, debug_info = self.controller.lane_detector.get_steering_angle(
                    camera_image, debug=True, current_speed=speed_magnitude
                )
                control_output = {
                    'steering': steering_angle,
                    'speed_factor': self.controller.calculate_speed_factor(steering_angle),
                    'debug_info': debug_info
                }
                if control_output['debug_info']:
                    control_vehicle(self.vehicle, control_output['steering'], control_output['speed_factor'])
                    log_debug_info(control_output['debug_info'])

                    # Görselleştirme görüntüsünü kuyruğa ekle
                    self.visualization_queue.put(control_output['debug_info']['visualization'])

            # Geliştirilmiş görüntü iyileştirme
            camera_image = cv2.GaussianBlur(camera_image, (5, 5), 0)
            camera_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2GRAY)
            _, camera_image = cv2.threshold(camera_image, 200, 255, cv2.THRESH_BINARY)

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            _, buffer = cv2.imencode('.jpg', camera_image, encode_param)
            jpg_as_text = buffer.tobytes()

            with self.sensor_data_lock:
                self.latest_camera_data = {
                    "camera": jpg_as_text,
                    "speed": speed_magnitude,
                    "timestamp": image.timestamp
                }
            self.send_combined_data()

        except Exception as e:
            logging.error(f"Kamera veri işleme hatası: {e}")

    def lidar_callback(self, point_cloud):
        try:
            # LiDAR verisi ön işleme
            lidar_data = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)

            # Zemin filtresi
            ground_mask = (lidar_data[:, 2] > -0.2) & (lidar_data[:, 2] < 2.0)
            intensity_mask = lidar_data[:, 3] > 0.1  # Yoğunluk bazlı filtreleme
            filtered_data = lidar_data[ground_mask & intensity_mask]

            # Mesafe filtresi
            distances = np.sqrt(filtered_data[:, 0]**2 + filtered_data[:, 1]**2)
            distance_mask = (distances > 0.5) & (distances < 80.0)
            filtered_data = filtered_data[distance_mask]

            # Yoğunluk bazlı filtreleme
            intensity_mask = filtered_data[:, 3] > 0.1
            filtered_data = filtered_data[intensity_mask]

            # Engel tespiti
            potential_obstacles = filtered_data[filtered_data[:, 2] > 0.3]
            min_obstacle_distance = float('inf')
            if len(potential_obstacles) > 0:
                min_obstacle_distance = np.min(np.sqrt(potential_obstacles[:, 0]**2 + potential_obstacles[:, 1]**2))
                if min_obstacle_distance < 3.0:  # 3 metre kritik mesafe
                    logging.warning("Kritik mesafede engel tespit edildi!")
                    # Aracın hızını doğrudan hesapla
                    vehicle_velocity = self.vehicle.get_velocity()
                    vehicle_speed = np.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2) * 3.6  # km/h
                    if vehicle_speed > 5.0:  # 5 km/h üzeri
                        # Acil durum sinyali gönder
                        control_vehicle(self.vehicle, 0.0, 0.5, min_obstacle_distance)  # Yavaşla

            # Veri sıkıştırma optimizasyonu
            compressed_lidar = zlib.compress(filtered_data.tobytes(), level=1)

            # Hız hesaplama
            vehicle_velocity = self.vehicle.get_velocity()
            speed_magnitude = np.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2) * 3.6  # km/h

            # Debug bilgisi
            if len(filtered_data) > 0:
                logging.info(f"LiDAR verisi şekli: {filtered_data.shape}")
                logging.info(f"İlk 5 LiDAR noktası: {filtered_data[:5]}")

            with self.sensor_data_lock:
                self.latest_lidar_data = {
                    "lidar": compressed_lidar,
                    "speed": speed_magnitude,
                    "timestamp": point_cloud.timestamp,
                    "obstacle_warning": min_obstacle_distance < 3.0
                }
            self.send_combined_data()

        except Exception as e:
            logging.error(f"LiDAR veri işleme hatası: {e}")

    def send_combined_data(self):
        global lane_following
        try:
            with self.sensor_data_lock:
                if self.latest_camera_data and self.latest_lidar_data:
                    time_diff = abs(self.latest_camera_data["timestamp"] - self.latest_lidar_data["timestamp"])

                    # Zaman senkronizasyonu kontrolü
                    if time_diff < 0.1:  # 100ms senkronizasyon toleransı
                        # Veriyi birleştir
                        combined_data = {
                            "camera": self.latest_camera_data["camera"],
                            "lidar": self.latest_lidar_data["lidar"],
                            "speed": (self.latest_camera_data["speed"] + self.latest_lidar_data["speed"]) / 2,
                            "timestamp": self.latest_camera_data["timestamp"],
                            "obstacle_warning": self.latest_lidar_data.get("obstacle_warning", False)
                        }

                        # Analiz işlemi
                        should_stop, distance, risk_level, lidar_data = analyze_data(combined_data)
                        command = "stop" if should_stop else "go"
                        logging.info(f"Komut: {command}")

                        if command == "stop":
                            control_vehicle(self.vehicle, 0.0, 0.0, distance)  # Araç durduruluyor
                            self.controller.state_machine.transition(DrivingState.OBSTACLE_AVOIDANCE)
                        elif command == "go":
                            # 'lane_following' flag'ini sadece araç durduysa yeniden etkinleştir
                            vehicle_velocity = self.vehicle.get_velocity()
                            current_speed = 3.6 * np.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2)
                            if current_speed < 1.0:
                                lane_following = True
                                initial_control = carla.VehicleControl(
                                    throttle=0.3,
                                    brake=0.0,
                                    steer=0.0
                                )
                                self.vehicle.apply_control(initial_control)
                                logging.info("Araç hareket etmeye başladı.")

                        # Görselleştirme
                        if self.latest_camera_data and self.latest_lidar_data:
                            # Grayscale görüntüyü yeniden oluştur
                            camera_image = np.frombuffer(zlib.decompress(self.latest_camera_data["camera"]),
                                                         dtype=np.uint8).reshape((CAMERA_HEIGHT, CAMERA_WIDTH))
                            # LiDAR verisini yeniden oluştur
                            lidar_data = np.frombuffer(zlib.decompress(self.latest_lidar_data["lidar"]),
                                                     dtype=np.float32).reshape(-1, 4)
                            
                            # LiDAR verisini görselleştir
                            visualized_image = self.controller.lane_detector.visualize_lidar_risks(camera_image, lidar_data, risk_level)
                            self.visualization_queue.put(visualized_image)
                            
                            # Gönderimden sonra tamponları temizle
                            self.latest_camera_data = None
                            self.latest_lidar_data = None
                        else:
                            logging.warning(f"Senkronizasyon dışı veriler: {time_diff:.3f}s fark")
        except Exception as e:
            logging.error(f"Veri birleştirme hatası: {e}")
            # Hata durumunda tamponları temizle
            self.latest_camera_data = None
            self.latest_lidar_data = None

    def visualize_lidar_risks(self, img, lidar_data, risk_level):
        """
        LiDAR verilerini kullanarak görüntü üzerinde risk alanlarını renklendirir.
        
        Args:
            img (numpy.ndarray): Görüntü (grayscale).
            lidar_data (numpy.ndarray): LiDAR verisi.
            risk_level (int): Risk seviyesi.
        
        Returns:
            numpy.ndarray: Güncellenmiş görüntü.
        """
        try:
            # Grayscale görüntüyü BGR'ye çevir
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            # Risk seviyelerine göre renk belirleme ve nokta büyüklüğü
            if risk_level == 3:
                color = (0, 0, 255)  # Kırmızı
                point_size = 4
            elif risk_level == 2:
                color = (0, 255, 255)  # Sarı
                point_size = 3
            elif risk_level == 1:
                color = (255, 0, 0)  # Mavi
                point_size = 2
            else:
                color = (0, 255, 0)  # Yeşil
                point_size = 1
            
            # LiDAR verilerini projekte et
            image_points = project_lidar_to_image(lidar_data, self.intrinsic_matrix, self.lidar_to_camera_matrix)
            
            # Vektörleştirilmiş OpenCV işlemi kullanarak daha hızlı çizim
            valid_indices = (
                (image_points[:, 0] >= 0) & (image_points[:, 0] < CAMERA_WIDTH) &
                (image_points[:, 1] >= 0) & (image_points[:, 1] < CAMERA_HEIGHT)
            )
            valid_points = image_points[valid_indices].astype(np.int32)
            
            # Noktaları çiz
            for point in valid_points:
                img_x, img_y = point
                cv2.circle(img_color, (img_x, img_y), point_size, color, -1)
            
            # Risk seviyesine göre ekranın üst kısmına uyarı ekleme
            if risk_level >= 1:
                warning_text = ""
                if risk_level == 3:
                    warning_text = "ACIL DURUM! DURDURULDU!"
                elif risk_level == 2:
                    warning_text = "YUKSEK RISK! YAVAŞLATILIYOR"
                elif risk_level == 1:
                    warning_text = "DIKKAT! HAZIR OL"
                
                if warning_text:
                    cv2.putText(img_color, warning_text, (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
            
            return img_color
        except Exception as e:
            logging.error(f"LiDAR görselleştirme hatası: {e}")
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

def get_transformation_matrix(transform):
    """Bir transformasyon objesinden 4x4 dönüşüm matrisi oluşturur."""
    # Euler açılarını radyana çevir
    yaw = np.radians(transform.rotation.yaw)
    pitch = np.radians(transform.rotation.pitch)
    roll = np.radians(transform.rotation.roll)
    
    # Rotasyon matrislerini oluştur
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1]
    ])
    
    R_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0,             1, 0],
        [-np.sin(pitch),0, np.cos(pitch)]
    ])
    
    R_roll = np.array([
        [1, 0,           0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    
    # Toplam rotasyon matrisini hesapla (Yaw * Pitch * Roll)
    R = R_yaw @ R_pitch @ R_roll
    
    # Çeviri vektörünü al
    translation = np.array([transform.location.x, transform.location.y, transform.location.z])
    
    # 4x4 dönüşüm matrisini oluştur
    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = translation
    
    return transformation

def project_lidar_to_image(lidar_points, intrinsic_matrix, extrinsic_matrix):
    """
    LiDAR noktalarını kamera görüntüsüne projekte eder.

    Args:
        lidar_points (numpy.ndarray): LiDAR verisi (N, 4).
        intrinsic_matrix (numpy.ndarray): Kamera içsel matrisi (3x3).
        extrinsic_matrix (numpy.ndarray): LiDAR'dan kameraya dönüşüm matrisi (4x4).

    Returns:
        numpy.ndarray: Görüntüdeki noktaların koordinatları (N, 2).
    """
    # Homojen koordinatlara çevir
    lidar_homogeneous = np.hstack((lidar_points[:, :3], np.ones((lidar_points.shape[0], 1))))
    
    # LiDAR'dan Kameraya dönüşüm
    camera_coords = extrinsic_matrix @ lidar_homogeneous.T  # (4, N)
    
    # Perspektif bölme
    camera_coords = camera_coords[:3, :] / camera_coords[2, :]
    
    # Kamera içsel matrisi ile projeksiyon
    image_points = intrinsic_matrix @ camera_coords  # (3, N)
    
    # Normalize et ve sadece 2D koordinatları al
    image_points = image_points[:2, :].T  # (N, 2)

    return image_points

def analyze_data(combined_data):
    """
    LiDAR verilerini analiz eder ve risk tespiti yapar.

    Args:
        combined_data (dict): Birleştirilmiş sensör verisi.

    Returns:
        tuple: (risk_detected (bool), distance (float), risk_level (int), lidar_data (numpy.ndarray))
    """
    try:
        lidar_data = np.frombuffer(zlib.decompress(combined_data['lidar']),
                                   dtype=np.float32).reshape(-1, 4)
        vehicle_speed = combined_data.get('speed', 0.0)

        # Veri kalite kontrolü
        if len(lidar_data) < 100:  # Çok az nokta
            logging.warning("Yetersiz LiDAR verisi!")
            return True, 0.0, 3, lidar_data  # Güvenli olmayabilir

        # Normal risk analizi
        risk_detected, distance, risk_level = enhanced_lidar_risk_analysis(lidar_data, vehicle_speed)

        # Geliştirilmiş risk mesajları
        risk_messages = {
            3: f"ACIL DURUM! Mesafe: {distance:.2f}m, Hız: {vehicle_speed:.1f}km/h - DURDURULDU!",
            2: f"Yüksek Risk! Mesafe: {distance:.2f}m, Hız: {vehicle_speed:.1f}km/h - YAVAŞLATILIYOR",
            1: f"Dikkat! Mesafe: {distance:.2f}m, Hız: {vehicle_speed:.1f}km/h - HAZIR OL",
            0: f"Güvenli Sürüş. Mesafe: {distance:.2f}m, Hız: {vehicle_speed:.1f}km/h"
        }

        logging.info(risk_messages.get(risk_level, "Durum değerlendirilemiyor"))

        # Geliştirilmiş durdurma kriterleri
        should_stop = any([
            risk_level >= 2,                              # Yüksek risk seviyesi
            vehicle_speed > 20 and distance < 2.0,        # Yüksek hızda yakın mesafe
        ])

        return should_stop, distance, risk_level, lidar_data

    except Exception as e:
        logging.error(f"Veri analiz hatası: {e}")
        return True, 0.0, 3, np.array([])  # Güvenli olmayabilir

def enhanced_lidar_risk_analysis(lidar_data, vehicle_speed=0.0):
    """
    Gelişmiş LiDAR risk analizi
    """

    # 1. Ön İşleme - Geliştirilmiş filtreleme
    ground_mask = (lidar_data[:, 2] > -0.2) & (lidar_data[:, 2] < 2.0)
    intensity_mask = lidar_data[:, 3] > 0.1  # Yoğunluk bazlı filtreleme
    filtered_data = lidar_data[ground_mask & intensity_mask]
    
    # 2. Geliştirilmiş Bölge Analizi
    distances = np.sqrt(filtered_data[:, 0]**2 + filtered_data[:, 1]**2)
    angles = np.degrees(np.arctan2(filtered_data[:, 1], filtered_data[:, 0]))
    
    # Dinamik risk bölgeleri (hıza bağlı)
    speed_factor = max(1.0, vehicle_speed / 30.0)  # 30 km/h referans hız
    critical_distance = 1.5 * speed_factor
    danger_distance = 3.0 * speed_factor
    warning_distance = 5.0 * speed_factor
    
    # Risk bölgeleri tanımı
    critical_zone = distances < critical_distance
    danger_zone = (distances < danger_distance) & ~critical_zone
    warning_zone = (distances < warning_distance) & ~danger_zone & ~critical_zone
    
    # Geliştirilmiş açı maskeleri
    front_center_mask = np.abs(angles) < 15
    front_sides_mask = (np.abs(angles) >= 15) & (np.abs(angles) < 45)
    rear_mask = np.abs(angles) > 150  # Arka bölge kontrolü
    
    # 3. Geliştirilmiş Risk Analizi
    critical_front = np.sum(critical_zone & front_center_mask)
    danger_front = np.sum(danger_zone & front_center_mask)
    warning_front = np.sum(warning_zone & front_center_mask)
    sides_points = np.sum(front_sides_mask & (distances < warning_distance))
    rear_points = np.sum(rear_mask & (distances < warning_distance))
    
    # Minimum mesafe hesaplama (bölgesel)
    min_distances = {
        'front': float('inf'),
        'sides': float('inf'),
        'rear': float('inf')
    }
    if len(distances[front_center_mask]) > 0:
        min_distances['front'] = np.min(distances[front_center_mask])
    if len(distances[front_sides_mask]) > 0:
        min_distances['sides'] = np.min(distances[front_sides_mask])
    if len(distances[rear_mask]) > 0:
        min_distances['rear'] = np.min(distances[rear_mask])
    
    # 4. Geliştirilmiş Risk Seviyesi Belirleme
    risk_level = 0
    speed_factor_m_s = vehicle_speed / 3.6
    dynamic_brake_distance = 2.0 + (speed_factor_m_s * 0.5) + (speed_factor_m_s ** 2 / 20)
    
    # Ön Riski Değerlendirme
    if min_distances['front'] < dynamic_brake_distance or (vehicle_speed > 25 and min_distances['front'] < dynamic_brake_distance):
        risk_level = max(risk_level, 3)
        logging.critical(f"ACIL DURUM! Ön Mesafe: {min_distances['front']:.2f}m, Yan Mesafe: {min_distances['sides']:.2f}m")
    
    # Yan Riski Değerlendirme
    if min_distances['sides'] < 1.5 and sides_points > 30:
        risk_level = max(risk_level, 2)
        logging.warning(f"YÜKSEK RİSK! Yan Mesafe: {min_distances['sides']:.2f}m")
    
    # Arka Riski Değerlendirme
    if min_distances['rear'] < 1.5 and rear_points > 20:
        risk_level = max(risk_level, 1)
        logging.info(f"DİKKAT! Arka Mesafe: {min_distances['rear']:.2f}m")
    
    return risk_level >= 2, min_distances['front'], risk_level

def control_vehicle(vehicle, steering_angle, speed_factor=1.0, distance=0.0):
    """
    Araç kontrol fonksiyonu.

    Args:
        vehicle (carla.Vehicle): Kontrol edilecek araç.
        steering_angle (float): Direksiyon açısı (-1.0 ile 1.0 arasında).
        speed_factor (float, optional): Hız faktörü. Varsayılan olarak 1.0.
        distance (float, optional): Engel ile araç arasındaki mesafe (m). Varsayılan olarak 0.0.
    """
    global lane_following, controller
    try:
        # Mevcut hız hesaplama
        velocity = vehicle.get_velocity()
        current_speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        logging.info(f"Current Speed: {current_speed:.2f} km/h")

        # Eğer araç duruyorsa ve "go" komutu geldiyse başlangıç hareketi
        if current_speed < 0.5 and lane_following:
            control = carla.VehicleControl(
                throttle=0.8,  # Başlangıç için daha yüksek gaz
                steer=0.0,
                brake=0.0,
                hand_brake=False,
                reverse=False
            )
            vehicle.apply_control(control)
            logging.info("Başlangıç hareketi uygulandı")
            return

        # Normal sürüş kontrolü için değerler
        target_speed = TARGET_SPEED * speed_factor
        speed_error = target_speed - current_speed
        logging.info(f"Target Speed: {target_speed:.2f} km/h, Speed Error: {speed_error:.2f}")

        # Hız PID kontrolü
        throttle = controller.pid.update(speed_error)
        throttle = np.clip(throttle, 0.0, 1.0)

        # Minimum gaz değeri belirleme
        if speed_error > 0 and throttle < 0.3:
            throttle = max(throttle, 0.3)  # Minimum gaz

        # Direksiyon kontrolü
        steering = np.clip(steering_angle, -1.0, 1.0)  # Daha geniş bir aralık

        # Fren kontrolü
        brake = 0.0
        if not lane_following:
            # Kalan mesafeye göre fren miktarını ayarla
            brake = calculate_brake_force(current_speed, distance)
            throttle = 0.0
            steering = 0.0  # Fren uygulanırken direksiyon açısını sıfırla
        elif speed_error < -5:  # Çok hızlıysa
            brake = min(-speed_error / 40.0, 1.0)  # Daha yumuşak frenleme
            throttle = 0.0
            steering = 0.0  # Fren uygulanırken direksiyon açısını sıfırla

        # Araç fiziği kontrolü (Opsiyonel: daha fazla hız kontrolü için)
        physics_control = vehicle.get_physics_control()
        if physics_control.mass < 1000:  # Araç çok hafifse
            throttle *= 1.2  # Gaz değerini artır

        # Debug bilgisi
        logging.info(f"Kontrol Değerleri:")
        logging.info(f"  Hız: {current_speed:.2f} km/h")
        logging.info(f"  Hedef: {target_speed:.2f} km/h")
        logging.info(f"  Gaz: {throttle:.2f}")
        logging.info(f"  Fren: {brake:.2f}")
        logging.info(f"  Direksiyon: {steering:.2f}")

        # Kontrol uygulama
        control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steering),
            brake=float(brake),
            hand_brake=False,
            reverse=False
        )

        vehicle.apply_control(control)

        # lane_following flag'ini güncelle
        if not lane_following and current_speed < 1.0:
            lane_following = True
            initial_control = carla.VehicleControl(
                throttle=0.3,
                brake=0.0,
                steer=0.0
            )
            vehicle.apply_control(initial_control)
            logging.info("Araç hareket etmeye başladı.")

    except Exception as e:
        logging.error(f"Araç kontrol hatası: {e}")
        # Hata durumunda güvenli duruş
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

def main():
    # Araç fiziksel ayarları
    vehicle_config = {
        "vehicle_model": "vehicle.tesla.model3",
        "spawn_retries": 5,
        "mass": 1600,  # kg
        "drag_coefficient": 0.32,
        "torque_curve": [
            carla.Vector2D(0, 450),
            carla.Vector2D(1500, 450),
            carla.Vector2D(3000, 350),
            carla.Vector2D(4500, 250)
        ],
        "gear_switch_time": 0.1,
        "gear_ratio": [-1.0, 0.0, 3.8, 2.6, 1.7, 1.0],
        "tire_friction": 3.8,
        "damping_rate": 1.6,
        "max_steer_angle": 75.0,
        "hand_brake": False,
        "reverse": False
    }

    # Ana araç kurulumu
    try:
        vehicle = setup_main_vehicle(world, spawn_points, vehicle_config)
        logging.info(f"Ana araç konumu: {vehicle.get_transform().location}")
        
        # Yüksek seviyeli planlayıcı ve TrajectoryPlanner'ı araç ile birlikte oluştur
        high_level_planner = HighLevelPlanner(world, vehicle)
        trajectory_planner = TrajectoryPlanner(vehicle)
        
        # Yörünge planlayıcıya aracı ata
        # (HighLevelPlanner zaten aracın referansını alıyor)
        
        # Durum makinesini oluştur
        state_machine = StateMachine()
        
        # Sensör Füzyonu oluştur
        sensor_fusion = SensorFusion()
        
        # Lane controller'ı başlat
        lane_controller = LaneFollowingController(high_level_planner, state_machine, sensor_fusion, trajectory_planner)
        
        # PID kontrolcülerini lane_controller içine taşıyın
        global controller
        controller = lane_controller

        # Kamerayı araca odakla
        spectator = world.get_spectator()
        spectator.set_transform(vehicle.get_transform())
    except Exception as e:
        logging.critical(f"Ana araç kurulum hatası: {e}")
        sys.exit(1)

    # Sensör kurulum ve veri işleme
    sensor_manager = SensorManager(vehicle, lane_controller, high_level_planner)

    # NPC araçları ekle
    vehicles = []
    for i in range(5):  # Daha fazla NPC araç eklemek için sayıyı artırabilirsiniz
        npc_vehicle_bp = random.choice(blueprint_library.filter("vehicle.*"))
        spawn_point = random.choice(spawn_points)
        npc_vehicle = world.try_spawn_actor(npc_vehicle_bp, spawn_point)
        if npc_vehicle:
            npc_vehicle.set_autopilot(True)
            vehicles.append(npc_vehicle)
    logging.info(f"{len(vehicles)} NPC araç simülasyona eklendi.")

    # Yayaları ekle
    walkers = []
    walker_controller_bp = blueprint_library.find("controller.ai.walker")
    for i in range(5):  # Daha fazla yaya eklemek için sayıyı artırabilirsiniz
        walker_bp = random.choice(blueprint_library.filter("walker.pedestrian.*"))
        spawn_point = random.choice(spawn_points)
        walker = world.try_spawn_actor(walker_bp, spawn_point)
        if walker:
            controller = world.try_spawn_actor(walker_controller_bp, carla.Transform(), walker)
            if controller:
                controller.start()
                controller.go_to_location(world.get_random_location_from_navigation())
                walkers.append((walker, controller))
    logging.info(f"{len(walkers)} yaya simülasyona eklendi.")

    # Görselleştirme Thread'i
    def visualization_thread():
        while True:
            try:
                visualization_image = visualization_queue.get(timeout=1)  # Timeout ekleyin
                cv2.imshow("Lane Detection and LiDAR Risk", visualization_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break  # 'q' tuşuna basıldığında çık
            except queue.Empty:
                continue  # Kuyruk boşsa devam et
            except Exception as e:
                logging.error(f"Görselleştirme thread hatası: {e}")
                break

    # Görselleştirme thread'ini başlat
    vis_thread = threading.Thread(target=visualization_thread, daemon=True)
    vis_thread.start()

    # Ana döngü
    try:
        while vis_thread.is_alive():
            # Kontrollü güncelleme için zaman uyumu
            time.sleep(0.1)  # Ana döngüyü hafifçe yavaşlat
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt yakalandı, simülasyon sonlandırılıyor.")
    finally:
        # Temizlik işlemleri
        for sensor in sensor_manager.sensors:
            sensor.stop()
            sensor.destroy()
        vehicle.destroy()
        for npc_vehicle in vehicles:
            npc_vehicle.destroy()
        for walker, controller in walkers:
            controller.stop()
            walker.destroy()
        cv2.destroyAllWindows()
        logging.info("Simülasyon sonlandırıldı ve tüm aktörler temizlendi.")

if __name__ == "__main__":
    main()
