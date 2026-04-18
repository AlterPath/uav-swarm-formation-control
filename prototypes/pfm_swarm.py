#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
import threading
import csv
import cv2
import os
import time
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError

CAMERA_FOV_RAD = 2.0
MIN_OVERLAP = 0.2
HEARTBEAT_TIMEOUT = 3.0
READY_TIMEOUT = 10.0
SAFE_DIST_FACTOR = 0.7
MIN_DIST_STEP = 1.0
MAX_STEP = 0.8
DRONE_SAFE_R = 3.0
WP_TIMEOUT = 30.0


class SmartSwarmDrone:
    def __init__(self, drone_id, total_n, h_orig, area_l, area_w):
        self.id = drone_id
        self.total_n = total_n
        self.h_orig = h_orig
        self.area_length = area_l
        self.area_width = area_w

        self.current_h = h_orig
        self.active_drones = {i for i in range(total_n)}
        self.last_heartbeat = {i: time.time() for i in range(total_n)}
        self._current_safe_r = DRONE_SAFE_R

        self.k_att = 1.2
        self.k_rep_drone = 25.0
        self.k_rep_obs = 50.0
        self.k_tan = 20.0

        # чётные/нечётные обходят препятствие с разных сторон
        self.tan_sign = 1.0 if (drone_id % 2 == 0) else -1.0

        self.map_obstacles = [
            {'x': 15.0, 'y': 20.0, 'radius': 4.0},
            {'x': 35.0, 'y': 45.0, 'radius': 6.0}
        ]

        self.mavros_ns = f"uav{drone_id}"
        self.camera_ns = f"iris{drone_id}"
        self.current_state = State()
        self.current_pose = PoseStamped()
        self.swarm_poses = {}
        self.swarm_ready = {}
        self.latest_image = None
        self.bridge = CvBridge()

        self.save_path = os.path.abspath(f"survey_results/drone_{drone_id}")
        os.makedirs(self.save_path, exist_ok=True)

        log_path = os.path.join(self.save_path, "mission_log.csv")
        self.log_file = open(log_path, 'w', newline='')
        self.log_writer = csv.writer(self.log_file)
        self.log_writer.writerow(['photo', 'x', 'y', 'z', 'status'])

        rospy.Subscriber(f'{self.mavros_ns}/mavros/state', State, self.state_cb)
        rospy.Subscriber(f'{self.mavros_ns}/mavros/local_position/pose', PoseStamped, self.pose_cb)
        rospy.Subscriber(f'{self.camera_ns}/down_camera_link/down_raw_image', Image, self.image_cb)
        for i in range(total_n):
            if i != self.id:
                rospy.Subscriber(f'/uav{i}/mavros/local_position/pose', PoseStamped, self.other_pose_cb, callback_args=i)
                rospy.Subscriber(f'/uav{i}/swarm/ready', Bool, self.ready_cb, callback_args=i)

        self.pos_pub = rospy.Publisher(f'{self.mavros_ns}/mavros/setpoint_position/local', PoseStamped, queue_size=10)
        self.ready_pub = rospy.Publisher(f'/uav{self.id}/swarm/ready', Bool, queue_size=1, latch=True)
        self.arm_srv = rospy.ServiceProxy(f'{self.mavros_ns}/mavros/cmd/arming', CommandBool)
        self.mode_srv = rospy.ServiceProxy(f'{self.mavros_ns}/mavros/set_mode', SetMode)

    def state_cb(self, msg): self.current_state = msg
    def pose_cb(self, msg): self.current_pose = msg
    def ready_cb(self, msg, d_id): self.swarm_ready[d_id] = msg.data

    def image_cb(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logwarn_throttle(5, f"[UAV{self.id}] CvBridgeError: {e}")

    def other_pose_cb(self, msg, d_id):
        self.swarm_poses[d_id] = msg
        self.last_heartbeat[d_id] = time.time()

    def signal_ready(self):
        self.ready_pub.publish(Bool(data=True))
        deadline = time.time() + READY_TIMEOUT
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and time.time() < deadline:
            alive = self._get_alive_ids()
            if all(self.swarm_ready.get(i, False) for i in alive - {self.id}):
                return
            rate.sleep()
        rospy.logwarn(f"[UAV{self.id}] sync timeout, proceeding without full swarm")

    def _get_alive_ids(self):
        now = time.time()
        alive = {self.id}
        for d_id in range(self.total_n):
            if d_id != self.id and (now - self.last_heartbeat.get(d_id, 0) < HEARTBEAT_TIMEOUT):
                alive.add(d_id)
        return alive

    def update_swarm_params(self):
        self.active_drones = self._get_alive_ids()
        n_active = len(self.active_drones)
        self.current_h = min(self.h_orig * (self.total_n / n_active), self.h_orig * 2.5)
        dist_between = max(
            2 * self.current_h * math.tan(CAMERA_FOV_RAD / 2) * (1 - MIN_OVERLAP),
            MIN_DIST_STEP
        )
        sorted_active = sorted(list(self.active_drones))
        my_idx = sorted_active.index(self.id)
        return my_idx, dist_between

    def get_pfm_vector(self, tx, ty, tz):
        curr = self.current_pose.pose.position

        f_x = self.k_att * (tx - curr.x)
        f_y = self.k_att * (ty - curr.y)
        f_z = self.k_att * (tz - curr.z)

        for d_id, pose in self.swarm_poses.items():
            if d_id not in self.active_drones:
                continue
            other = pose.pose.position
            dx, dy, dz = curr.x - other.x, curr.y - other.y, curr.z - other.z
            dist = math.sqrt(dx**2 + dy**2 + dz**2)
            safe_r = max(self._current_safe_r, DRONE_SAFE_R)
            if 0.3 < dist < safe_r:
                rep = self.k_rep_drone * (1.0/dist - 1.0/safe_r) * (1.0/dist**2)
                f_x += (dx / dist) * rep
                f_y += (dy / dist) * rep
                f_z += (dz / dist) * rep

        for obs in self.map_obstacles:
            dx, dy = curr.x - obs['x'], curr.y - obs['y']
            dist_xy = max(math.sqrt(dx**2 + dy**2), 0.5)
            influence_r = obs['radius'] + 3.0
            if dist_xy < influence_r:
                rep = self.k_rep_obs * (1.0/dist_xy - 1.0/influence_r) * (1.0/dist_xy**2)
                f_x += (dx / dist_xy) * rep
                f_y += (dy / dist_xy) * rep
                tan_mag = self.k_tan * (1.0 / dist_xy)
                f_x += self.tan_sign * (-dy / dist_xy) * tan_mag
                f_y += self.tan_sign * (dx  / dist_xy) * tan_mag

        # ограничиваем длину вектора смещения — дрон не прыгает дальше MAX_STEP за тик
        step = math.sqrt(f_x**2 + f_y**2 + f_z**2)
        if step > MAX_STEP:
            scale = MAX_STEP / step
            f_x, f_y, f_z = f_x * scale, f_y * scale, f_z * scale

        return curr.x + f_x, curr.y + f_y, curr.z + f_z

    def navigate_smart(self, tx, ty, tz, arrival_thr=0.4):
        rate = rospy.Rate(20)
        deadline = time.time() + WP_TIMEOUT
        while not rospy.is_shutdown():
            curr = self.current_pose.pose.position
            if math.sqrt((curr.x-tx)**2 + (curr.y-ty)**2 + (curr.z-tz)**2) < arrival_thr:
                return True

            if time.time() > deadline:
                curr = self.current_pose.pose.position
                rospy.logwarn(f"[UAV{self.id}] WP timeout ({tx:.1f},{ty:.1f},{tz:.1f}), skipping")
                self.log_writer.writerow([
                    'skipped',
                    f"{curr.x:.3f}", f"{curr.y:.3f}", f"{curr.z:.3f}",
                    f"wp_timeout target=({tx:.2f},{ty:.2f},{tz:.2f})"
                ])
                self.log_file.flush()
                return False

            my_idx, dist_step = self.update_swarm_params()
            self._current_safe_r = dist_step * SAFE_DIST_FACTOR

            nx, ny, nz = self.get_pfm_vector(tx, ty, tz)
            msg = PoseStamped()
            msg.header.stamp = rospy.Time.now()
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = nx, ny, nz
            self.pos_pub.publish(msg)
            rate.sleep()
        return False

    def save_image(self, photo_idx):
        curr = self.current_pose.pose.position
        if self.latest_image is None:
            return photo_idx
        filename = f"img_{photo_idx:04d}.jpg"
        path = os.path.join(self.save_path, filename)
        try:
            if cv2.imwrite(path, self.latest_image):
                self.log_writer.writerow([
                    filename,
                    f"{curr.x:.3f}", f"{curr.y:.3f}", f"{curr.z:.3f}",
                    'ok'
                ])
                self.log_file.flush()
                photo_idx += 1
            else:
                rospy.logwarn(f"[UAV{self.id}] imwrite failed: {path}")
        except Exception as e:
            rospy.logerr(f"[UAV{self.id}] save error: {e}")
        return photo_idx

    def run_mission(self):
        rate_wait = rospy.Rate(10)
        while not rospy.is_shutdown() and \
              (not self.current_state.connected or self.latest_image is None):
            rate_wait.sleep()

        for _ in range(50):
            self.pos_pub.publish(self.current_pose)
            time.sleep(0.05)

        self.mode_srv(custom_mode="OFFBOARD")
        self.arm_srv(True)
        time.sleep(2.0)

        curr = self.current_pose.pose.position
        self.navigate_smart(curr.x, curr.y, self.h_orig)
        self.signal_ready()

        photo_idx = 0
        current_base_x = 0.0

        while current_base_x <= self.area_width and not rospy.is_shutdown():
            y_pos = 0.0
            while y_pos <= self.area_length and not rospy.is_shutdown():
                my_idx, dist_step = self.update_swarm_params()
                self.navigate_smart(current_base_x + my_idx * dist_step, y_pos, self.current_h)
                photo_idx = self.save_image(photo_idx)
                y_pos += max(self.current_h * 0.5, 0.5)

            current_base_x += len(self.active_drones) * dist_step
            if current_base_x <= self.area_width:
                my_idx, dist_step = self.update_swarm_params()
                self.navigate_smart(current_base_x + my_idx * dist_step, self.area_length, self.current_h)

        self.navigate_smart(0, 0, self.h_orig)
        self.mode_srv(custom_mode="AUTO.LAND")
        self.log_file.close()
        rospy.loginfo(f"[UAV{self.id}] mission complete, {photo_idx} photos saved")


def main():
    rospy.init_node('swarm', anonymous=True)
    try:
        h = float(input("Height (m): "))
        l = float(input("Length (m): "))
        w = float(input("Width (m): "))
        n = int(input("Drones: "))
    except (ValueError, EOFError):
        h, l, w, n = 5.0, 60.0, 40.0, 3
        rospy.logwarn("Using defaults: h=5, l=60, w=40, n=3")

    drones = [SmartSwarmDrone(i, n, h, l, w) for i in range(n)]
    threads = [threading.Thread(target=d.run_mission, daemon=True) for d in drones]
    for i, t in enumerate(threads):
        t.start()
        time.sleep(0.3)
    for t in threads:
        t.join()


if __name__ == '__main__':
    main()