

import argparse
import csv
import math
import os
import threading
import time
import cv2
import rospy
from datetime import datetime
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32MultiArray
from cv_bridge import CvBridge

K_ATT = 1.5
K_REP = 2.5
K_TAN = 1.2
K_LANE = 0.8
D_INFLUENCE = 4.0
D_SAFE = 1.5
V_MAX = 2.5
REACH_TOL = 0.8
POINT_TIMEOUT = 60.0
HEARTBEAT_HZ = 2.0
HEARTBEAT_DEAD = 3.0
HEARTBEAT_DEAD_STARTUP = 15.0
STARTUP_GRACE = 20.0
CONTROL_HZ = 20.0
CAM_FOV_DEG = 78.0
OVERLAP = 0.2
K_REP_OBS = 4.0
D_INF_OBS = 8.0
LOG_HZ = 5.0

OBSTACLES = []

class DroneNode:
    def __init__(self, drone_id, total, ns, log_dir=None):
        self.id = drone_id
        self.total = total
        self.ns = ns.strip('/')
        self.is_leader = (drone_id == 0)

        rospy.init_node('swarm_apf_lanes_' + self.ns, anonymous=False)

        self.state = State()
        self.pose = PoseStamped()
        self.have_pose = False

        self.neighbors = {}
        self.neighbors_lock = threading.Lock()
        self.seen_ids = {self.id}
        self.alive_ids = {self.id}
        self.alive_ids_locked = False

        self.startup_time = time.time()

        self.mission_started = False
        self.ready_set = set()
        self.ready_lock = threading.Lock()

        self.field_len = None
        self.field_wid = None
        self.alt = None
        self.params_received = threading.Event()

        self.my_lane_y = None
        self.alt_original = None

        self.path_length = 0.0
        self.prev_x = None
        self.prev_y = None
        self.mission_t0 = None
        self.mission_done = False
        self.target = (0.0, 0.0, 0.0)
        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_apf = dict(
            att_x=0., att_y=0.,
            rep_x=0., rep_y=0.,
            tan_x=0., tan_y=0.,
            lane_y=0.
        )

        pref = '/' + self.ns + '/mavros'
        rospy.Subscriber(pref + '/state', State, self.state_cb)
        rospy.Subscriber(pref + '/local_position/pose', PoseStamped, self.pose_cb)

        self.meas_vx = 0.0
        self.meas_vy = 0.0
        self.meas_vz = 0.0
        rospy.Subscriber(pref + '/local_position/velocity_local',
                         TwistStamped, self.vel_cb)
        self.sp_pub = rospy.Publisher(pref + '/setpoint_position/local',
                                       PoseStamped, queue_size=10)
        self.vel_pub = rospy.Publisher(pref + '/setpoint_velocity/cmd_vel',
                                       TwistStamped, queue_size=10)
        rospy.wait_for_service(pref + '/cmd/arming')
        rospy.wait_for_service(pref + '/set_mode')
        self.srv_arm = rospy.ServiceProxy(pref + '/cmd/arming', CommandBool)
        self.srv_mode = rospy.ServiceProxy(pref + '/set_mode', SetMode)

        self.hb_pub = rospy.Publisher('/swarm/heartbeat',
                                          Float32MultiArray, queue_size=10)
        self.ready_pub = rospy.Publisher('/swarm/ready', String, queue_size=10)
        self.params_pub = rospy.Publisher('/swarm/params',
                                          Float32MultiArray, queue_size=1, latch=True)
        rospy.Subscriber('/swarm/heartbeat', Float32MultiArray, self.hb_cb)
        rospy.Subscriber('/swarm/ready', String, self.ready_cb)
        rospy.Subscriber('/swarm/params', Float32MultiArray, self.params_cb)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = log_dir or script_dir
        os.makedirs(log_dir, exist_ok=True)
        self.img_dir = os.path.join(log_dir, 'raw_images', self.ns)
        os.makedirs(self.img_dir, exist_ok=True)

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')

        mpath = os.path.join(log_dir, f'metrics_apf_lanes_{self.ns}_{ts}.csv')
        self.mf = open(mpath, 'w', newline='')
        self.mw = csv.writer(self.mf)
        self.mw.writerow([
            'time_s', 'drone_id',
            'pos_x', 'pos_y', 'pos_z',
            'target_x', 'target_y', 'target_z',
            'formation_error_m',
            'lane_error_m',
            'nearest_neighbor_m',
            'velocity_mps',
            'path_length_m',
            'mission_time_s',
            'coverage_pct',
            'f_att_x', 'f_att_y',
            'f_rep_x', 'f_rep_y',
            'f_tan_x', 'f_tan_y',
            'f_lane_y',
            'event'
        ])
        self.mf.flush()

        ppath = os.path.join(log_dir, f'photos_apf_lanes_{self.ns}_{ts}.csv')
        self.pf = open(ppath, 'w', newline='')
        self.pw = csv.writer(self.pf)
        self.pw.writerow(['time_s', 'filename',
                           'pos_x', 'pos_y', 'pos_z', 'agl_m',
                           'yaw_rad', 'roll_rad', 'pitch_rad',
                           'vel_mps',
                           'ok', 'event'])
        self.pf.flush()
        self.photo_counter = 0

        self.bridge = CvBridge()
        self.latest_image = None
        self.image_lock = threading.Lock()
        rospy.Subscriber('/iris' + str(self.id) + '/down_camera_link/down_raw_image',
                         Image, self.image_cb)

        try:
            obs_path = os.path.join(script_dir, 'obstacles.txt')
            self.obstacles = []
            with open(obs_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    parts = line.split(',')
                    self.obstacles.append(
                        (float(parts[0]), float(parts[1]), float(parts[2])))
            rospy.loginfo('[%s] препятствий: %d', self.ns, len(self.obstacles))
        except Exception:
            self.obstacles = list(OBSTACLES)

        rospy.loginfo('[%s] инициализирован (APF+lanes), id=%d', self.ns, self.id)

    def state_cb(self, msg):
        self.state = msg

    def pose_cb(self, msg):
        px = msg.pose.position.x
        py = msg.pose.position.y
        if self.prev_x is not None:
            self.path_length += math.hypot(px - self.prev_x, py - self.prev_y)
        self.prev_x = px
        self.prev_y = py
        self.pose = msg
        self.have_pose = True

    def vel_cb(self, msg):
        self.meas_vx = msg.twist.linear.x
        self.meas_vy = msg.twist.linear.y
        self.meas_vz = msg.twist.linear.z

    def orient_from_pose(self):
        q = self.pose.pose.orientation

        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        sinp = max(- 1.0, min(1.0, sinp))
        pitch = math.asin(sinp)

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw, roll, pitch

    def image_cb(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logwarn_throttle(5.0, '[%s] cv_bridge: %s', self.ns, e)
            return
        with self.image_lock:
            self.latest_image = cv_img

    def hb_cb(self, msg):
        if len(msg.data) < 5: return
        nid = int(msg.data[0])
        if nid == self.id: return
        with self.neighbors_lock:
            self.neighbors[nid] = (
                msg.data[1], msg.data[2], msg.data[3], time.time())
            self.seen_ids.add(nid)

    def ready_cb(self, msg):
        with self.ready_lock:
            try:
                self.ready_set.add(int(msg.data))
            except ValueError:
                pass

    def params_cb(self, msg):
        if len(msg.data) >= 3 and not self.params_received.is_set():
            self.field_len = float(msg.data[0])
            self.field_wid = float(msg.data[1])
            self.alt = float(msg.data[2])
            self.alt_original = self.alt
            self.params_received.set()
            rospy.loginfo('[%s] params: L=%.1f W=%.1f H=%.1f',
                          self.ns, self.field_len, self.field_wid, self.alt)

    def publish_heartbeat(self):
        m = Float32MultiArray()
        m.data = [float(self.id),
                  self.pose.pose.position.x,
                  self.pose.pose.position.y,
                  self.pose.pose.position.z,
                  time.time()]
        self.hb_pub.publish(m)

    def start_heartbeat_thread(self):
        def loop():
            r = rospy.Rate(HEARTBEAT_HZ)
            while not rospy.is_shutdown():
                if self.have_pose: self.publish_heartbeat()
                if self.params_received.is_set(): self.update_alive()
                r.sleep()
        threading.Thread(target=loop, daemon=True).start()

    def update_alive(self):
        now = time.time()
        in_startup = (now - self.startup_time) < STARTUP_GRACE
        dead_timeout = HEARTBEAT_DEAD_STARTUP if in_startup else HEARTBEAT_DEAD
        alive = {self.id}
        with self.neighbors_lock:
            for nid, v in self.neighbors.items():
                t = v[-1]
                if now - t < dead_timeout: alive.add(nid)
            seen = set(self.seen_ids)
        if self.alive_ids_locked:
            alive = alive | self.alive_ids
        lost = (self.alive_ids & seen) - alive
        if lost and not in_startup and not self.alive_ids_locked:
            rospy.logwarn('[%s] потеряны дроны: %s', self.ns, lost)
            if self.mission_started and self.alt_original and len(alive) > 0:
                k = math.sqrt(self.total / max(1, len(alive)))
                self.alt = self.alt_original * k
                rospy.logwarn('[%s] новая высота: %.2f', self.ns, self.alt)
        self.alive_ids = alive

    def formation_error(self):
        p = self.pose.pose.position
        tx, ty, tz = self.target
        return math.sqrt((p.x - tx) ** 2 + (p.y - ty) ** 2 + (p.z - tz) ** 2)

    def lane_error(self):
        if self.my_lane_y is None: return 0.0
        return abs(self.pose.pose.position.y - self.my_lane_y)

    def nearest_neighbor(self):
        px = self.pose.pose.position.x
        py = self.pose.pose.position.y
        now = time.time()
        best = float('inf')
        with self.neighbors_lock:
            for nid, (nx, ny, nz, ts) in self.neighbors.items():
                if now - ts > HEARTBEAT_DEAD: continue
                d = math.hypot(px - nx, py - ny)
                if d < best: best = d
        return best if best < float('inf') else - 1.0

    def coverage_pct(self):
        if self.field_wid is None: return 0.0
        fov = math.radians(CAM_FOV_DEG)
        stripe = 2.0 * self.alt * math.tan(fov / 2.0) * (1.0 - OVERLAP)
        n_alive = max(1, len(self.alive_ids))
        return min(100.0, stripe * n_alive / self.field_wid * 100.0)

    def log_metrics(self, event=''):
        now = time.time()
        mission_s = (now - self.mission_t0) if self.mission_t0 else 0.0
        vel = math.hypot(self.last_vx, self.last_vy)
        p = self.pose.pose.position
        tx, ty, tz = self.target
        a = self.last_apf

        self.mw.writerow([
            round(now, 3), self.id,
            round(p.x, 3), round(p.y, 3), round(p.z, 3),
            round(tx, 3), round(ty, 3), round(tz, 3),
            round(self.formation_error(), 3),
            round(self.lane_error(), 3),
            round(self.nearest_neighbor(), 3),
            round(vel, 3),
            round(self.path_length, 3),
            round(mission_s, 3),
            round(self.coverage_pct(), 2),
            round(a['att_x'], 4), round(a['att_y'], 4),
            round(a['rep_x'], 4), round(a['rep_y'], 4),
            round(a['tan_x'], 4), round(a['tan_y'], 4),
            round(a['lane_y'], 4),
            event
        ])
        self.mf.flush()

    def start_metrics_thread(self):
        def loop():
            r = rospy.Rate(LOG_HZ)
            while not rospy.is_shutdown() and not self.mission_done:
                self.log_metrics()
                r.sleep()
        threading.Thread(target=loop, daemon=True).start()

    def apf_velocity(self, target):
        px = self.pose.pose.position.x
        py = self.pose.pose.position.y
        tx, ty = target[0], target[1]
        dx = tx - px
        dy = ty - py

        f_att_x = K_ATT * dx
        f_att_y = K_ATT * dy
        att_mag = math.hypot(f_att_x, f_att_y)
        if att_mag > V_MAX:
            f_att_x = f_att_x * V_MAX / att_mag
            f_att_y = f_att_y * V_MAX / att_mag

        f_rep_x = f_rep_y = 0.0
        f_tan_x = f_tan_y = 0.0

        f_lane_y = 0.0
        if self.my_lane_y is not None: f_lane_y = K_LANE * (self.my_lane_y - py)

        def add_repulse(obj_x, obj_y, radius, k_rep, d_inf, danger=False):
            nonlocal f_rep_x, f_rep_y, f_tan_x, f_tan_y
            rx = px - obj_x
            ry = py - obj_y
            rdist = max(1e-3, math.hypot(rx, ry) - radius)
            if rdist < d_inf:
                mag = k_rep * (1.0 / rdist - 1.0 / d_inf) / (rdist ** 2)
                if danger: mag *= 5.0
                nlen = math.hypot(rx, ry) + 1e-9
                f_rep_x += mag * rx / nlen
                f_rep_y += mag * ry / nlen
                perp_x = - ry / nlen
                perp_y = rx / nlen
                if perp_x * dx + perp_y * dy < 0: perp_x, perp_y = - perp_x, - perp_y
                k_t = K_TAN * (1.0 - rdist / d_inf)
                f_tan_x += k_t * perp_x
                f_tan_y += k_t * perp_y

        with self.neighbors_lock:
            items = list(self.neighbors.items())
        now = time.time()
        for nid, (nx, ny, nz, t) in items:
            if now - t > HEARTBEAT_DEAD: continue
            r_ij = math.hypot(px - nx, py - ny)
            add_repulse(nx, ny, 0.0, K_REP, D_INFLUENCE, danger=(r_ij < D_SAFE))

        near_obstacle = False
        for ox, oy, orad in self.obstacles:
            r_ij = math.hypot(px - ox, py - oy) - orad
            if r_ij < D_INF_OBS: near_obstacle = True
            add_repulse(ox, oy, orad, K_REP_OBS, D_INF_OBS, danger=(r_ij < 1.5))

        if near_obstacle:
            f_att_x *= 0.6
            f_att_y *= 0.6
            f_lane_y *= 0.1

        self.last_apf = dict(
            att_x=f_att_x, att_y=f_att_y,
            rep_x=f_rep_x, rep_y=f_rep_y,
            tan_x=f_tan_x, tan_y=f_tan_y,
            lane_y=f_lane_y
        )

        vx = f_att_x + f_rep_x + f_tan_x
        vy = f_att_y + f_rep_y + f_tan_y + f_lane_y
        v = math.hypot(vx, vy)
        if v > V_MAX: vx, vy = vx * V_MAX / v, vy * V_MAX / v
        return vx, vy

    def arm_and_offboard(self):
        sp = PoseStamped()
        sp.pose.position.x = self.pose.pose.position.x
        sp.pose.position.y = self.pose.pose.position.y
        sp.pose.position.z = self.alt
        r = rospy.Rate(CONTROL_HZ)
        for _ in range(50):
            sp.header.stamp = rospy.Time.now()
            self.sp_pub.publish(sp)
            r.sleep()
        last_req = rospy.Time.now()
        while not rospy.is_shutdown():
            dt = (rospy.Time.now() - last_req).to_sec()
            if self.state.mode != 'OFFBOARD' and dt > 2.0:
                try:
                    self.srv_mode(custom_mode='OFFBOARD')
                except rospy.ServiceException as e:
                    rospy.logwarn('set_mode: %s', e)
                last_req = rospy.Time.now()
            elif not self.state.armed and dt > 2.0:
                try:
                    self.srv_arm(True)
                except rospy.ServiceException as e:
                    rospy.logwarn('arm: %s', e)
                last_req = rospy.Time.now()
            if self.state.armed and self.state.mode == 'OFFBOARD':
                rospy.loginfo('[%s] OFFBOARD+ARMED', self.ns)
                return
            sp.header.stamp = rospy.Time.now()
            self.sp_pub.publish(sp)
            r.sleep()

    def land(self):
        try:
            self.srv_mode(custom_mode='AUTO.LAND')
            rospy.loginfo('[%s] посадка', self.ns)
        except rospy.ServiceException as e:
            rospy.logwarn('land: %s', e)

    def barrier_wait(self):
        rospy.loginfo('[%s] ожидание барьера...', self.ns)
        r = rospy.Rate(2.0)
        while not rospy.is_shutdown():
            self.ready_pub.publish(String(data=str(self.id)))
            with self.ready_lock:
                self.ready_set.add(self.id)
                if self.alive_ids.issubset(self.ready_set) and self.alive_ids.issuperset(self.ready_set):
                    self.alive_ids_locked = True
                    rospy.loginfo('[%s] барьер пройден, состав группы: %s',
                                  self.ns, sorted(self.alive_ids))
                    return
            r.sleep()

    def build_snake(self):
        fov = math.radians(CAM_FOV_DEG)
        stripe = max(1e-3, 2.0 * self.alt * math.tan(fov / 2.0) * (1.0 - OVERLAP))
        alive_sorted = sorted(self.alive_ids)
        n_alive = max(1, len(alive_sorted))
        total_stripes = max(1, int(math.ceil(self.field_wid / stripe)))
        my_rank = alive_sorted.index(self.id) if self.id in alive_sorted else 0
        my_stripes = [i for i in range(total_stripes) if i % n_alive == my_rank]

        def shift_out(x, y):
            for ox, oy, orad in self.obstacles:
                d = math.hypot(x - ox, y - oy)
                safe = orad + 1.5
                if d < safe:
                    nx = (x - ox) / (d + 1e-9); ny = (y - oy) / (d + 1e-9)
                    x = ox + nx * safe; y = oy + ny * safe
            return x, y

        pts = []
        for k, s in enumerate(my_stripes):
            y = stripe * (s + 0.5)
            xs = [0.0, - self.field_len] if k % 2 == 0 else[- self.field_len, 0.0]
            step = max(5.0, stripe)
            n = max(2, int(math.ceil(self.field_len / step)) + 1)
            for i in range(n):
                a = i / (n - 1)
                xp = xs[0] + a * (xs[1] - xs[0])
                sx, sy = shift_out(xp, y)
                pts.append((sx, sy, self.alt))
        return pts

    def goto_apf(self, target, event_on_reach='WAYPOINT'):
        self.target = target
        t0 = time.time()
        r = rospy.Rate(CONTROL_HZ)
        while not rospy.is_shutdown():
            dx = target[0] - self.pose.pose.position.x
            dy = target[1] - self.pose.pose.position.y
            dz = target[2] - self.pose.pose.position.z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist < REACH_TOL:
                self.log_metrics(event_on_reach)
                return True
            if time.time() - t0 > POINT_TIMEOUT:
                rospy.logwarn('[%s] TIMEOUT %s', self.ns, target)
                self.log_metrics('TIMEOUT')
                return False
            vx, vy = self.apf_velocity(target)
            vz = max(- V_MAX, min(V_MAX, 0.8 * dz))
            self.last_vx = vx
            self.last_vy = vy
            cmd = TwistStamped()
            cmd.header.stamp = rospy.Time.now()
            cmd.twist.linear.x = vx
            cmd.twist.linear.y = vy
            cmd.twist.linear.z = vz
            self.vel_pub.publish(cmd)
            r.sleep()
        return False

    def take_photo(self, target, reached):
        self.photo_counter += 1
        fname = self.ns + '_photo_' + str(self.photo_counter).zfill(5) + '.jpg'
        fpath = os.path.join(self.img_dir, fname)
        event = 'SHOT' if reached else 'MISSED'
        p = self.pose.pose.position
        yaw, roll, pitch = self.orient_from_pose()
        vel_mps = math.hypot(self.meas_vx, self.meas_vy)
        agl_m = p.z
        saved = False
        if reached:
            with self.image_lock:
                img = None if self.latest_image is None else self.latest_image.copy()
            if img is not None:
                try:
                    cv2.imwrite(fpath, img)
                    saved = True
                except Exception as e:
                    rospy.logwarn('[%s] imwrite: %s', self.ns, e)
            else:
                event = 'MISSED'
        self.pw.writerow([
            round(time.time(), 3), fname if saved else '',
            round(p.x, 3), round(p.y, 3), round(p.z, 3), round(agl_m, 3),
            round(yaw, 4), round(roll, 4), round(pitch, 4),
            round(vel_mps, 3),
            int(saved), event
        ])
        self.pf.flush()
        self.log_metrics('PHOTO_' + event)

    def run(self):
        rospy.loginfo('[%s] ждём FCU...', self.ns)
        r = rospy.Rate(5.0)
        while not rospy.is_shutdown() and not self.state.connected: r.sleep()
        while not rospy.is_shutdown() and not self.have_pose: r.sleep()

        self.start_heartbeat_thread()

        if self.is_leader:
            if getattr(self, 'cli_len', None) is not None:
                self.field_len = self.cli_len
                self.field_wid = self.cli_wid
                self.alt = self.cli_alt
                self.alt_original = self.alt
                rospy.loginfo('[%s] параметры из CLI: L=%.1f W=%.1f H=%.1f',
                              self.ns, self.field_len, self.field_wid, self.alt)
            else:
                self.field_len = float(input('Длина : '))
                self.field_wid = float(input('Ширина: '))
                self.alt = float(input('Высота: '))
            m = Float32MultiArray()
            m.data = [self.field_len, self.field_wid, self.alt]
            rospy.sleep(1.0)
            for _ in range(5):
                self.params_pub.publish(m)
                rospy.sleep(0.2)
            self.params_received.set()
        else:
            rospy.loginfo('[%s] ждём параметры...', self.ns)
            while not rospy.is_shutdown() and not self.params_received.is_set(): rospy.sleep(0.2)

        self.arm_and_offboard()
        rospy.loginfo('[%s] ожидание появления всех соседей в группе...', self.ns)
        wait_start = time.time()
        while not rospy.is_shutdown() and time.time() - wait_start < 15.0:
            with self.neighbors_lock:
                n_seen = len(self.neighbors) + 1
            if n_seen >= self.total: break
            rospy.sleep(0.2)
        rospy.loginfo('[%s] видимый состав до барьера: %s', self.ns, sorted(self.alive_ids))

        self.barrier_wait()


        fov = math.radians(CAM_FOV_DEG)
        stripe = 2.0 * self.alt * math.tan(fov / 2.0) * (1.0 - OVERLAP)
        alive_sorted = sorted(self.alive_ids)
        my_rank = alive_sorted.index(self.id) if self.id in alive_sorted else self.id
        start_y = stripe * (my_rank + 0.5)
        self.my_lane_y = start_y
        start = (0.0, start_y, self.alt)

        sp = PoseStamped()
        sp.pose.position.x = start[0]
        sp.pose.position.y = start[1]
        sp.pose.position.z = start[2]
        t0 = time.time()
        ctrl = rospy.Rate(CONTROL_HZ)
        while not rospy.is_shutdown():
            sp.header.stamp = rospy.Time.now()
            self.sp_pub.publish(sp)
            dz = self.alt - self.pose.pose.position.z
            dxy = math.hypot(start[0] - self.pose.pose.position.x,
                             start[1] - self.pose.pose.position.y)
            if abs(dz) < 0.4 and dxy < 1.0: break
            if time.time() - t0 > 90.0:
                rospy.logwarn('[%s] долго взлетает', self.ns)
                break
            ctrl.sleep()
        self.mission_started = True
        self.mission_t0 = time.time()
        self.start_metrics_thread()
        self.log_metrics('MISSION_START')

        route = self.build_snake()
        rospy.loginfo('[%s] маршрут: %d точек', self.ns, len(route))

        last_lane_y = start_y
        for i, p in enumerate(route):
            rospy.loginfo('[%s] -> %d/%d %s', self.ns, i + 1, len(route), p)

            if abs(p[1] - last_lane_y) > 0.5:
                self.my_lane_y = p[1]
                last_lane_y = p[1]
            self.target = p
            ok = self.goto_apf(p)
            self.take_photo(p, reached=ok)

        rospy.loginfo('[%s] маршрут завершён', self.ns)
        self.my_lane_y = start_y
        self.goto_apf((0.0, start_y, self.alt), event_on_reach='RTL')
        self.land()
        self.mission_done = True
        self.log_metrics('MISSION_END')

        rospy.sleep(3.0)
        self.mf.flush(); self.mf.close()
        self.pf.flush(); self.pf.close()
        rospy.loginfo('[%s] миссия завершена | path=%.1fм', self.ns, self.path_length)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--id', type=int, required=True)
    p.add_argument('--total', type=int, required=True)
    p.add_argument('--ns', type=str, default=None)
    p.add_argument('--field-len', type=float, default=None)
    p.add_argument('--field-wid', type=float, default=None)
    p.add_argument('--height', type=float, default=None)
    p.add_argument('--log-dir', type=str, default=None)
    args = p.parse_args()
    ns = args.ns if args.ns else 'uav' + str(args.id)
    node = DroneNode(args.id, args.total, ns, log_dir=args.log_dir)

    node.cli_len = args.field_len
    node.cli_wid = args.field_wid
    node.cli_alt = args.height

    try:
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr('Ошибка: %s', e)
    finally:
        try:
            node._mf.close()
            node._pf.close()
        except Exception:
            pass

if __name__ == '__main__': main()

