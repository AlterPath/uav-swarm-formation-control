#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
import time
import threading
import os
import csv
import cv2
from datetime import datetime
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from std_msgs.msg import String, Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# --- Параметры (покрутила веса, чтобы строй держали) ---
K_ATT = 1.5           # Сделала притяжение сильнее, чтобы быстрее возвращались
K_REP_NEIGHBOR = 2.0  # Отталкивание от своих
K_REP_OBSTACLE = 5.0  # От препятствий жарим сильнее
K_FORMATION = 0.5     # Новая сила - пытаемся держаться на одной линии с соседями
D_INFLUENCE = 3.0     
D_SAFE = 1.2          
V_MAX = 2.0           
REACH_TOL = 0.7       
CONTROL_HZ = 20.0
CAM_FOV = 78.0        
OVERLAP = 0.2         

# Координаты препятствий (x, y, radius)
OBSTACLES = [
    (-15.0, -10.0, 2.5),
    (-30.0, -25.0, 3.0),
    (-45.0,  -8.0, 2.0),
    (-20.0, -40.0, 3.5),
    (-55.0, -30.0, 2.5),
]

class DroneNode:
    def __init__(self, drone_id, total_drones, ns):
        self.id = drone_id
        self.total = total_drones
        self.ns = ns.strip('/')
        
        rospy.init_node(f'drone_{self.id}_node')

        # Состояние и позиция
        self.state = State()
        self.pose = PoseStamped()
        self.got_pose = False
        
        # Соседи: {id: (x, y, z, time)}
        self.neighbors = {}
        self.n_lock = threading.Lock()
        
        # Параметры миссии (придут от лидера)
        self.field_l = 0
        self.field_w = 0
        self.alt = 0
        self.ready_to_go = threading.Event()
        
        # Для камеры
        self.bridge = CvBridge()
        self.last_frame = None
        self.im_lock = threading.Lock()

        # Настройка путей для логов (как в лабе просили)
        script_path = os.path.dirname(os.path.abspath(__file__))
        self.log_dir = os.path.join(script_path, "logs", self.ns)
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_file = open(os.path.join(self.log_dir, "mission.csv"), 'w')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(['time', 'event', 'x', 'y', 'z'])

        self._init_comms()
        
    def _init_comms(self):
        """Тут все подписки и сервисы"""
        p = f'/{self.ns}/mavros'
        rospy.Subscriber(f'{p}/state', State, self._cb_state)
        rospy.Subscriber(f'{p}/local_position/pose', PoseStamped, self._cb_pose)
        
        self.pub_vel = rospy.Publisher(f'{p}/setpoint_velocity/cmd_vel', TwistStamped, queue_size=10)
        self.pub_pos = rospy.Publisher(f'{p}/setpoint_position/local', PoseStamped, queue_size=10)
        
        self.srv_arm = rospy.ServiceProxy(f'{p}/cmd/arming', CommandBool)
        self.srv_mode = rospy.ServiceProxy(f'{p}/set_mode', SetMode)

        # Общий канал для роя
        self.pub_hb = rospy.Publisher('/swarm/heartbeat', Float32MultiArray, queue_size=10)
        rospy.Subscriber('/swarm/heartbeat', Float32MultiArray, self._cb_hb)
        
        self.pub_params = rospy.Publisher('/swarm/params', Float32MultiArray, queue_size=1, latch=True)
        rospy.Subscriber('/swarm/params', Float32MultiArray, self._cb_params)

        # Камера (по твоему топику)
        rospy.Subscriber(f'/iris{self.id}/down_camera_link/down_raw_image', Image, self._cb_img)

    def _cb_state(self, msg): self.state = msg
    def _cb_pose(self, msg): 
        self.pose = msg
        self.got_pose = True

    def _cb_img(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            with self.im_lock:
                self.last_frame = cv_img
        except: pass

    def _cb_hb(self, msg):
        if int(msg.data[0]) == self.id: return
        with self.n_lock:
            # Сохраняем x, y, z и время получения
            self.neighbors[int(msg.data[0])] = (msg.data[1], msg.data[2], msg.data[3], time.time())

    def _cb_params(self, msg):
        if not self.ready_to_go.is_set():
            self.field_l, self.field_w, self.alt = msg.data
            self.ready_to_go.set()

    def get_apf_vel(self, target):
        """Считаем итоговую скорость через потенциальные поля"""
        curr_x = self.pose.pose.position.x
        curr_y = self.pose.pose.position.y
        
        # 1. Сила притяжения к точке (Attraction)
        dx = target[0] - curr_x
        dy = target[1] - curr_y
        dist_t = math.hypot(dx, dy) + 0.1
        
        v_att_x = K_ATT * (dx / dist_t)
        v_att_y = K_ATT * (dy / dist_t)

        v_rep_x = 0
        v_rep_y = 0

        # 2. Отталкивание от соседей + удержание строя (чтобы не разлетались)
        with self.n_lock:
            for nid, (nx, ny, nz, t) in self.neighbors.items():
                if time.time() - t > 2.0: continue # Сосед пропал
                
                rx = curr_x - nx
                ry = curr_y - ny
                dist_n = math.hypot(rx, ry)
                
                if dist_n < D_INFLUENCE:
                    # Сила отталкивания
                    f = K_REP_NEIGHBOR * (1.0/max(0.5, dist_n) - 1.0/D_INFLUENCE)
                    v_rep_x += f * (rx / dist_n)
                    v_rep_y += f * (ry / dist_n)
                
                # Маленький лайфхак: если сосед сильно впереди/позади по Y, 
                # чуть-чуть подтягиваемся к нему (пружинка строя)
                v_rep_y += K_FORMATION * (ny - curr_y) * 0.1

        # 3. Отталкивание от препятствий
        for ox, oy, orad in OBSTACLES:
            rx = curr_x - ox
            ry = curr_y - oy
            dist_o = math.hypot(rx, ry) - orad
            
            if dist_o < 5.0: # Видим препятствие издалека
                f = K_REP_OBSTACLE * (1.0/max(0.1, dist_o) - 1.0/5.0)
                v_rep_x += f * (rx / (dist_o + orad))
                v_rep_y += f * (ry / (dist_o + orad))

        # Итог
        vx = v_att_x + v_rep_x
        vy = v_att_y + v_rep_y
        
        # Ограничение скорости
        v_norm = math.hypot(vx, vy)
        if v_norm > V_MAX:
            vx = (vx / v_norm) * V_MAX
            vy = (vy / v_norm) * V_MAX
            
        return vx, vy

    def takeoff(self):
        """Взлет и переход в OFFBOARD"""
        print(f"[{self.ns}] Взлетаю на {self.alt}...")
        # Сначала шлем точки, потом режим
        p = PoseStamped()
        p.pose.position.z = self.alt
        for _ in range(40):
            self.pub_pos.publish(p)
            rospy.sleep(0.05)
            
        self.srv_mode(custom_mode='OFFBOARD')
        self.srv_arm(True)
        rospy.sleep(2)

    def run_mission(self):
        # Ждем коннекта и параметров
        while not self.got_pose and not rospy.is_shutdown():
            rospy.sleep(0.1)
        
        if self.id == 0: # Лидер спрашивает размеры
            L = float(input("Длина поля: "))
            W = float(input("Ширина поля: "))
            H = float(input("Высота полета: "))
            self.pub_params.publish(Float32MultiArray(data=[L, W, H]))
        
        self.ready_to_go.wait()
        self.takeoff()
        
        # Строим змейку
        points = []
        fov_rad = math.radians(CAM_FOV)
        width_m = 2 * self.alt * math.tan(fov_rad/2)
        step_y = width_m * (1 - OVERLAP)
        
        # Каждый дрон берет свои линии
        my_lines = [i for i in range(10) if i % self.total == self.id]
        for line_idx in my_lines:
            y = line_idx * step_y
            if line_idx % 2 == 0:
                points.append((-self.field_l, y, self.alt))
                points.append((0, y, self.alt))
            else:
                points.append((0, y, self.alt))
                points.append((-self.field_l, y, self.alt))

        # Погнали по точкам
        rate = rospy.Rate(CONTROL_HZ)
        for pt in points:
            print(f"[{self.ns}] Иду к {pt}")
            while not rospy.is_shutdown():
                dist = math.hypot(pt[0]-self.pose.pose.position.x, pt[1]-self.pose.pose.position.y)
                if dist < REACH_TOL:
                    # Фоткаем
                    with self.im_lock:
                        if self.last_frame is not None:
                            cv2.imwrite(f"{self.log_dir}/img_{time.time()}.jpg", self.last_frame)
                    break
                
                # Скорость по APF
                vx, vy = self.get_apf_vel(pt)
                
                # Шлем в MAVROS
                cmd = TwistStamped()
                cmd.twist.linear.x = vx
                cmd.twist.linear.y = vy
                # Держим высоту
                cmd.twist.linear.z = 0.5 * (self.alt - self.pose.pose.position.z)
                self.pub_vel.publish(cmd)
                
                # Heartbeat (чтобы другие нас видели)
                self.pub_hb.publish(Float32MultiArray(data=[self.id, 
                    self.pose.pose.position.x, 
                    self.pose.pose.position.y, 
                    self.pose.pose.position.z]))
                
                rate.sleep()

        print(f"[{self.ns}] Все! Садимся.")
        self.srv_mode(custom_mode='AUTO.LAND')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--total', type=int, default=1)
    parser.add_argument('--ns', type=str, default='uav0')
    args = parser.parse_args()

    node = DroneNode(args.id, args.total, args.ns)
    try:
        node.run_mission()
    except rospy.ROSInterruptException:
        pass