

import rospy
import math
import threading
import csv
import os
import time
import cv2
from datetime import datetime
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Empty
from cv_bridge import CvBridge, CvBridgeError

CAMERA_FOV_DEG = 78.0
MIN_OVERLAP = 0.2
CONTROL_HZ = 10.0
REACH_TOL = 0.25
LOG_HZ = 5.0
HEARTBEAT_HZ = 5.0
HEARTBEAT_DEAD = 3.0

N_DRONES = 3
CONNECT_TIMEOUT = 30.0
ARM_RETRIES = 10
ARM_RETRY_DELAY = 2.0

_ready_barrier = threading.Barrier(N_DRONES)

class SurveyDrone:
    def __init__(self, drone_id, height, dist_between,
                 area_length, area_width, step_y, log_dir=None):
        self.id = drone_id
        self.height = height
        self.dist_between = dist_between
        self.area_length = area_length
        self.area_width = area_width
        self.step_y = step_y
        self.mavros_ns = f'uav{drone_id}'
        self.camera_ns = f'iris{drone_id}'

        self.current_state = State()
        self.current_pose = PoseStamped()
        self.leader_pose = PoseStamped()
        self.latest_image = None
        self.bridge = CvBridge()

        self.prev_x = None
        self.prev_y = None
        self.path_length = 0.0
        self.mission_start = None
        self.mission_done = False

        self.neighbors = {}
        self.neighbors_lock = threading.Lock()
        self.land_cmd_received = False

        rospy.Subscriber('/swarm/land_cmd', Empty, self.land_cmd_cb)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = log_dir if log_dir else script_dir
        os.makedirs(out_dir, exist_ok=True)
        self.img_dir = os.path.join(out_dir, 'raw_images', self.mavros_ns)
        os.makedirs(self.img_dir, exist_ok=True)

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        metrics_path = os.path.join(
            out_dir, f'metrics_hardcode_{self.mavros_ns}_{ts}.csv')
        self.metrics_file = open(metrics_path, 'w', newline='')
        self.mw = csv.writer(self.metrics_file)
        self.mw.writerow([
            'time_s', 'drone_id',
            'pos_x', 'pos_y', 'pos_z',
            'target_x', 'target_y', 'target_z',
            'formation_error_m', 'dist_to_neighbor_m',
            'path_length_m', 'velocity_mps',
            'mission_time_s', 'coverage_pct', 'event'
        ])
        self.metrics_file.flush()

        photos_path = os.path.join(
            out_dir, f'photos_hardcode_{self.mavros_ns}_{ts}.csv')
        self.photo_file = open(photos_path, 'w', newline='')
        self.pw = csv.writer(self.photo_file)
        self.pw.writerow(['time_s', 'filename',
                           'pos_x', 'pos_y', 'pos_z', 'agl_m',
                           'yaw_rad', 'roll_rad', 'pitch_rad',
                           'vel_mps',
                           'ok', 'event'])
        self.photo_file.flush()
        self.photo_idx = 0

        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = height

        rospy.Subscriber(f'{self.mavros_ns}/mavros/state',
                         State, self.state_cb)
        rospy.Subscriber(f'{self.mavros_ns}/mavros/local_position/pose',
                         PoseStamped, self.pose_cb)

        self.meas_vx = 0.0
        self.meas_vy = 0.0
        self.meas_vz = 0.0
        rospy.Subscriber(f'{self.mavros_ns}/mavros/local_position/velocity_local',
                         TwistStamped, self.vel_cb)
        rospy.Subscriber(f'{self.camera_ns}/down_camera_link/down_raw_image',
                         Image, self.image_cb)

        self.pos_pub = rospy.Publisher(
            f'{self.mavros_ns}/mavros/setpoint_position/local',
            PoseStamped, queue_size=10)
        self.arm_srv = rospy.ServiceProxy(
            f'{self.mavros_ns}/mavros/cmd/arming', CommandBool)
        self.mode_srv = rospy.ServiceProxy(
            f'{self.mavros_ns}/mavros/set_mode', SetMode)

        self.hb_pub = rospy.Publisher(
            '/swarm/heartbeat', Float32MultiArray, queue_size=10)
        rospy.Subscriber('/swarm/heartbeat',
                         Float32MultiArray, self.hb_cb)

        if self.id > 0:
            rospy.Subscriber('/uav0/mavros/local_position/pose',
                             PoseStamped, self.leader_cb)

        rospy.loginfo(f'[hardcode/uav{self.id}] инициализирован')

    def state_cb(self, msg): self.current_state = msg

    def pose_cb(self, msg):
        px, py = msg.pose.position.x, msg.pose.position.y
        if self.prev_x is not None:
            self.path_length += math.hypot(px - self.prev_x, py - self.prev_y)
        self.prev_x, self.prev_y = px, py
        self.current_pose = msg

    def vel_cb(self, msg):
        self.meas_vx = msg.twist.linear.x
        self.meas_vy = msg.twist.linear.y
        self.meas_vz = msg.twist.linear.z

    def orient_from_pose(self):
        q = self.current_pose.pose.orientation
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
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError:
            pass

    def leader_cb(self, msg): self.leader_pose = msg

    def hb_cb(self, msg):
        if len(msg.data) < 4: return
        nid = int(msg.data[0])
        if nid == self.id: return
        with self.neighbors_lock:
            self.neighbors[nid] = (
                float(msg.data[1]), float(msg.data[2]),
                float(msg.data[3]), time.time())

    def publish_heartbeat(self):
        m = Float32MultiArray()
        m.data = [float(self.id),
                  self.current_pose.pose.position.x,
                  self.current_pose.pose.position.y,
                  self.current_pose.pose.position.z]
        self.hb_pub.publish(m)

    def start_heartbeat_thread(self):
        def loop():
            r = rospy.Rate(HEARTBEAT_HZ)
            while not rospy.is_shutdown():
                self.publish_heartbeat()
                r.sleep()
        threading.Thread(target=loop, daemon=True).start()

    def formation_error(self):
        p = self.current_pose.pose.position
        return math.sqrt((p.x - self.target_x) ** 2 +
                         (p.y - self.target_y) ** 2 +
                         (p.z - self.target_z) ** 2)

    def nearest_neighbor_dist(self):
        px, py = (self.current_pose.pose.position.x,
                  self.current_pose.pose.position.y)
        now = time.time()
        best = float('inf')
        with self.neighbors_lock:
            for _, (nx, ny, _, ts) in self.neighbors.items():
                if now - ts > HEARTBEAT_DEAD: continue
                d = math.hypot(px - nx, py - ny)
                if d < best: best = d
        return best if best < float('inf') else - 1.0

    def coverage_pct(self):
        fov_rad = math.radians(CAMERA_FOV_DEG)
        stripe = 2.0 * self.height * math.tan(fov_rad / 2.0)
        eff = stripe * (1.0 - MIN_OVERLAP)
        n_stripes = max(1, math.ceil(self.area_width / eff))
        x_done = abs(self.current_pose.pose.position.x)
        done = min(n_stripes, x_done / max(self.area_length, 1e-3) * n_stripes)
        return min(100.0, done / n_stripes * 100.0)

    def log_metrics(self, event=''):
        now = time.time()
        ms = (now - self.mission_start) if self.mission_start else 0.0
        p = self.current_pose.pose.position
        vel = self.path_length / ms if ms > 1.0 else 0.0
        self.mw.writerow([
            round(now, 3), self.id,
            round(p.x, 3), round(p.y, 3), round(p.z, 3),
            round(self.target_x, 3), round(self.target_y, 3), round(self.target_z, 3),
            round(self.formation_error(), 3),
            round(self.nearest_neighbor_dist(), 3),
            round(self.path_length, 3), round(vel, 3),
            round(ms, 3), round(self.coverage_pct(), 2), event
        ])
        self.metrics_file.flush()

    def start_metrics_thread(self):
        def loop():
            r = rospy.Rate(LOG_HZ)
            while not rospy.is_shutdown() and not self.mission_done:
                self.log_metrics()
                r.sleep()
        threading.Thread(target=loop, daemon=True).start()

    def set_offboard_and_arm(self):
        rate = rospy.Rate(CONTROL_HZ)
        sp = PoseStamped()
        sp.pose.position.x = self.current_pose.pose.position.x
        sp.pose.position.y = self.current_pose.pose.position.y
        sp.pose.position.z = self.height

        rospy.loginfo(f'[uav{self.id}] прогрев setpoint...')
        for _ in range(100):
            sp.header.stamp = rospy.Time.now()
            self.pos_pub.publish(sp)
            rate.sleep()

        for attempt in range(1, ARM_RETRIES + 1):
            rospy.loginfo(f'[uav{self.id}] OFFBOARD+ARM попытка {attempt} / {ARM_RETRIES}')
            try:
                self.mode_srv(custom_mode='OFFBOARD')
            except rospy.ServiceException as e:
                rospy.logwarn(f'[uav{self.id}] SetMode: {e}')

            time.sleep(0.5)

            try:
                resp = self.arm_srv(True)
                if resp.success:
                    rospy.loginfo(f'[uav{self.id}] ARMED OK')
                    return True
            except rospy.ServiceException as e:
                rospy.logwarn(f'[uav{self.id}] ARM: {e}')

            for _ in range(int(ARM_RETRY_DELAY * CONTROL_HZ)):
                sp.header.stamp = rospy.Time.now()
                self.pos_pub.publish(sp)
                rate.sleep()

        rospy.logerr(f'[uav{self.id}] не удалось ARM после {ARM_RETRIES} попыток')
        return False

    def land(self):
        rospy.loginfo(f'[uav{self.id}] посадка...')
        self.log_metrics('LAND_START')
        try:
            self.mode_srv(custom_mode='AUTO.LAND')
        except rospy.ServiceException as e:
            rospy.logwarn(f'[uav{self.id}] AUTO.LAND: {e}')
            return

        rate = rospy.Rate(2.0)
        deadline = time.time() + 60.0
        while not rospy.is_shutdown():
            z = self.current_pose.pose.position.z
            if z < 0.3: break
            if time.time() > deadline:
                rospy.logwarn(f'[uav{self.id}] таймаут посадки')
                break
            rate.sleep()

        try:
            self.arm_srv(False)
        except Exception:
            pass
        rospy.loginfo(f'[uav{self.id}] приземлился')
        self.log_metrics('LAND_DONE')

    def navigate_to(self, x, y, z, event_on_arrive=''):
        self.target_x, self.target_y, self.target_z = x, y, z
        target = PoseStamped()
        target.pose.position.x = x
        target.pose.position.y = y
        target.pose.position.z = z
        rate = rospy.Rate(CONTROL_HZ)
        while not rospy.is_shutdown():
            p = self.current_pose.pose.position
            if math.sqrt((p.x - x) ** 2 + (p.y - y) ** 2 + (p.z - z) ** 2) < REACH_TOL:
                if event_on_arrive: self.log_metrics(event_on_arrive)
                time.sleep(2.0)
                return True
            target.header.stamp = rospy.Time.now()
            self.pos_pub.publish(target)
            rate.sleep()
        return False

    def land_cmd_cb(self, msg):
        self.land_cmd_received = True

    def follow_leader(self):
        rate = rospy.Rate(CONTROL_HZ)
        target = PoseStamped()

        offset_y = self.id * self.dist_between
        while not rospy.is_shutdown() and not self.mission_done and not self.land_cmd_received:
            lp = self.leader_pose.pose.position
            self.target_x = lp.x
            self.target_y = lp.y + offset_y
            self.target_z = self.height
            target.header.stamp = rospy.Time.now()
            target.pose.position.x = self.target_x
            target.pose.position.y = self.target_y
            target.pose.position.z = self.target_z
            self.pos_pub.publish(target)
            rate.sleep()

    def take_photo(self, x, y, z):
        self.photo_idx += 1
        fname = f'uav{self.id}_photo_{self.photo_idx:05d}.jpg'
        fpath = os.path.join(self.img_dir, fname)
        ok = False
        if self.latest_image is not None:
            try:
                cv2.imwrite(fpath, self.latest_image)
                ok = True
            except Exception as e:
                rospy.logwarn(f'[uav{self.id}] imwrite: {e}')

        p = self.current_pose.pose.position
        yaw, roll, pitch = self.orient_from_pose()
        vel_mps = math.hypot(self.meas_vx, self.meas_vy)
        agl_m = p.z
        event = 'SHOT' if ok else 'MISSED'
        self.pw.writerow([round(time.time(), 3), fname if ok else '',
                           round(p.x, 3), round(p.y, 3), round(p.z, 3), round(agl_m, 3),
                           round(yaw, 4), round(roll, 4), round(pitch, 4),
                           round(vel_mps, 3),
                           int(ok), event])
        self.photo_file.flush()
        self.log_metrics('PHOTO')

    def run_mission(self):

        rospy.loginfo(f'[uav{self.id}] ожидание MAVROS...')
        deadline = time.time() + CONNECT_TIMEOUT
        while not rospy.is_shutdown() and not self.current_state.connected:
            if time.time() > deadline:
                rospy.logerr(f'[uav{self.id}] таймаут MAVROS')
                return
            time.sleep(0.1)

        deadline = time.time() + CONNECT_TIMEOUT
        while not rospy.is_shutdown() and self.latest_image is None:
            if time.time() > deadline:
                rospy.logwarn(f'[uav{self.id}] камера не ответила, продолжаем')
                break
            time.sleep(0.1)

        self.start_heartbeat_thread()

        rospy.loginfo(f'[uav{self.id}] барьер...')
        try:
            _ready_barrier.wait(timeout=CONNECT_TIMEOUT)
        except threading.BrokenBarrierError:
            rospy.logwarn(f'[uav{self.id}] барьер сломан, стартуем')

        if not self.set_offboard_and_arm():
            rospy.logerr(f'[uav{self.id}] не взлетел')
            self.metrics_file.close()
            self.photo_file.close()
            return

        time.sleep(self.id * 1.0)

        self.mission_start = time.time()
        self.start_metrics_thread()
        self.log_metrics('MISSION_START')

        if self.id == 0: self.run_leader()
        else:
            self.run_follower()

        self.mission_done = True
        self.log_metrics('MISSION_END')

        rospy.loginfo(
            f'[uav{self.id}] завершён | '
            f'path={self.path_length:.1f}м | '
            f'time={(time.time() - self.mission_start):.0f}с')
        self.metrics_file.close()
        self.photo_file.close()

    def run_leader(self):

        current_y = 0.0
        swarm_total_width = N_DRONES * self.dist_between

        while current_y < self.area_width:

            x_pos = 0.0
            while x_pos <= self.area_length:
                self.navigate_to(x_pos, current_y, self.height,
                                 event_on_arrive='WAYPOINT')
                self.take_photo(x_pos, current_y, self.height)
                x_pos += self.step_y

            current_y += swarm_total_width
            if current_y < self.area_width:

                self.navigate_to(self.area_length, current_y,
                                 self.height, event_on_arrive='TURN')

                x_pos = self.area_length
                while x_pos >= 0:
                    self.navigate_to(x_pos, current_y, self.height,
                                     event_on_arrive='WAYPOINT')
                    self.take_photo(x_pos, current_y, self.height)
                    x_pos -= self.step_y
                current_y += swarm_total_width

        self.navigate_to(0, 0, self.height, event_on_arrive='RTL')

        rospy.loginfo(f'[uav{self.id}] команда рою на посадку')
        land_pub = rospy.Publisher('/swarm/land_cmd', Empty, queue_size=1)
        rospy.sleep(0.5)
        for _ in range(10):
            land_pub.publish(Empty())
            rospy.sleep(0.1)
        self.land()

    def run_follower(self):
        last_y = [- 999.0]
        photo_th = threading.Thread(
            target=self.follower_photo_loop, args=(last_y,), daemon=True)
        photo_th.start()
        self.follow_leader()

        photo_th.join(timeout=2.0)
        self.land()

    def follower_photo_loop(self, last_y):
        while not rospy.is_shutdown() and not self.mission_done:
            cy = self.current_pose.pose.position.y
            cx = self.current_pose.pose.position.x
            if abs(cy - last_y[0]) >= self.step_y * 0.9:
                time.sleep(1.5)
                self.take_photo(cx, cy, self.height)
                last_y[0] = cy
            time.sleep(0.2)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=str, default=None)
    parser.add_argument('--height', type=float, default=None)
    parser.add_argument('--field-len', type=float, default=None)
    parser.add_argument('--field-wid', type=float, default=None)
    args, _ = parser.parse_known_args()

    rospy.init_node('swarm_hardcode_survey', anonymous=False)

    try:
        h = args.height if args.height is not None else float(input('Высота (м): '))
        l = args.field_len if args.field_len is not None else float(input('Длина поля (м): '))
        w = args.field_wid if args.field_wid is not None else float(input('Ширина поля (м): '))
    except Exception:
        rospy.logwarn('Ввод не удался, используем значения по умолчанию')
        h, l, w = 5.0, 60.0, 30.0

    fov_rad = math.radians(CAMERA_FOV_DEG)
    footprint = 2.0 * h * math.tan(fov_rad / 2.0)
    dist_between = footprint * (1.0 - MIN_OVERLAP)
    step_y = footprint * (1.0 - MIN_OVERLAP)

    rospy.loginfo(
        f'H={h}м L={l}м W={w}м | '
        f'footprint={footprint:.2f}м dist={dist_between:.2f}м step={step_y:.2f}м')

    drones = [SurveyDrone(i, h, dist_between, l, w, step_y, log_dir=args.log_dir)
               for i in range(N_DRONES)]
    threads = [threading.Thread(target=d.run_mission, name=f'drone_{d.id}')
               for d in drones]

    for t in threads: t.start()
    for t in threads: t.join()

if __name__ == '__main__': main()
