"""NaVILA on the real Unitree Go2 through go2_sdk (ROS 2 Humble).

Same high/mid-level stack as the Isaac Sim runner (navila_go2.common): camera frames -> NaVILA server -> mid-level
action -> closed-loop velocity commands. The low level is the Go2's own sport-mode locomotion controller, which
go2_sdk drives from /cmd_vel (through twist_mux: the joystick, priority 10, always overrides this node, priority 5).

UNTESTED ON HARDWARE. It only logs commands unless --enable_motion is given. Keep the remote at hand.

  # terminal 1: NaVILA server (GPU machine)    ./navila_go2/scripts/start_server.sh
  # terminal 2: go2_sdk driver                 ros2 launch go2_robot_sdk robot.launch.py
  # terminal 3:
  source /opt/ros/humble/setup.bash
  python3 navila_go2/real/navila_go2_node.py --instruction "Walk to the door and stop." [--enable_motion]
  ros2 topic pub --once /navila/instruction std_msgs/String "data: 'Turn left and go to the kitchen.'"
"""

import argparse
import math
import os.path as osp
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "..", "..")))
from navila_go2.common.client import FrameHistory, NaVilaClient  # noqa: E402
from navila_go2.common.mid_level import STOP, MidLevelConfig, MidLevelController  # noqa: E402

import cv2  # noqa: E402  (after numpy; ROS python may shadow it otherwise)


def image_to_rgb(msg: Image) -> np.ndarray:
    """sensor_msgs/Image -> HxWx3 uint8 RGB without cv_bridge."""
    buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, : msg.width * 3]
    img = buf.reshape(msg.height, msg.width, 3)
    return img[:, :, ::-1].copy() if msg.encoding.lower().startswith("bgr") else img.copy()


def to_navila_view(rgb: np.ndarray, mode: str, size: int = 512) -> np.ndarray:
    """NaVILA resizes every frame to a square ('image_aspect_ratio: resize'); decide how the 16:9 Go2 frame gets there.

    crop:   center square (no distortion, ~45 deg FOV on the Go2 front camera)
    squash: whole frame squeezed to a square (73 deg FOV, distorted; what NaVILA would do on its own)
    """
    if mode == "crop":
        h, w = rgb.shape[:2]
        s = min(h, w)
        rgb = rgb[(h - s) // 2 : (h + s) // 2, (w - s) // 2 : (w + s) // 2]
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def yaw_from_quat(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class NavilaGo2Node(Node):
    def __init__(self, args):
        super().__init__("navila_go2")
        self.args = args
        self.client = NaVilaClient(args.server)
        self.get_logger().info(f"waiting for NaVILA server {args.server} ...")
        self.get_logger().info(f"NaVILA server: {self.client.wait_ready()}")
        self.ctrl = MidLevelController(MidLevelConfig(lin_speed=args.lin_speed, ang_speed=math.radians(args.ang_speed_deg)))
        self.history = FrameHistory()
        self.frame, self.frame_t = None, 0.0
        self.pose, self.pose_t = None, 0.0
        self.instruction = args.instruction
        self.mode = "idle"
        self.future = None
        self.n_actions = 0
        self.units_recorded = 0

        self.cmd_pub = self.create_publisher(Twist, args.cmd_topic, 10)
        self.create_subscription(Image, args.camera_topic, self.on_image, qos_profile_sensor_data)
        self.create_subscription(Odometry, args.odom_topic, self.on_odom, 10)
        self.create_subscription(String, "navila/instruction", self.on_instruction, 10)
        self.create_timer(1.0 / args.rate, self.tick)
        if not args.enable_motion:
            self.get_logger().warn("DRY RUN: velocity commands are logged, not published (use --enable_motion)")

    # ---------------- callbacks ----------------
    def on_image(self, msg):
        self.frame, self.frame_t = to_navila_view(image_to_rgb(msg), self.args.image_mode), time.time()

    def on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.pose, self.pose_t = (p.x, p.y, yaw_from_quat(q)), time.time()

    def on_instruction(self, msg):
        self.instruction = msg.data.strip()
        self.mode = "idle"  # restarts in tick()
        self.ctrl.active = False

    # ---------------- control loop ----------------
    def start_episode(self):
        self.history.reset()
        self.ctrl.reset_heading(self.pose[2])
        self.n_actions = 0
        self.get_logger().info(f"new instruction: {self.instruction!r}")
        self.query()

    def query(self):
        self.history.add(self.frame)
        self.future = self.client.act_async(self.instruction, self.history)
        self.mode = "thinking"

    def publish(self, vx, vy, wz):
        vx = max(-self.args.max_lin, min(self.args.max_lin, vx))
        wz = max(-self.args.max_ang, min(self.args.max_ang, wz))
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = float(vx), float(vy), float(wz)
        if self.args.enable_motion:
            self.cmd_pub.publish(msg)
        elif abs(vx) + abs(wz) > 0:
            self.get_logger().info(f"[dry run] cmd vx={vx:.2f} wz={wz:.2f}", throttle_duration_sec=1.0)

    def tick(self):
        now = time.time()
        stale = self.frame is None or self.pose is None or now - self.frame_t > 0.5 or now - self.pose_t > 0.5
        if stale:
            if self.mode in ("thinking", "executing"):
                self.get_logger().warn("camera/odom stale: holding still", throttle_duration_sec=2.0)
            self.publish(0.0, 0.0, 0.0)
            return
        if self.mode == "idle" and self.instruction:
            self.start_episode()

        if self.mode == "thinking" and self.future is not None and self.future.done():
            fut, self.future = self.future, None
            try:
                action, out = fut.result()
            except Exception as e:  # never keep moving on a failed decision
                self.get_logger().error(f"NaVILA request failed: {e}; stopping")
                self.mode, self.instruction = "done", None
                action = None
            if action is not None:
                self.n_actions += 1
                self.get_logger().info(f"#{self.n_actions:02d} {out['text']!r} -> {action} ({out['latency_s']:.2f}s)")
                if action.kind == STOP or self.n_actions >= self.args.max_actions:
                    self.mode, self.instruction = "done", None
                    self.get_logger().info("episode finished")
                else:
                    self.ctrl.start(action, self.pose)
                    self.units_recorded = 0
                    self.mode = "executing"

        cmd = (0.0, 0.0, 0.0)
        if self.mode == "executing":
            cmd = self.ctrl.step(self.pose, 1.0 / self.args.rate)
            while self.units_recorded < self.ctrl.units_done and self.ctrl.active:
                self.history.add(self.frame)  # one frame per 25 cm / 15 deg, as in NaVILA's training data
                self.units_recorded += 1
            if not self.ctrl.active:
                self.query()
        # publish every tick (also zeros): twist_mux drops the navigation input after 0.5 s of silence
        self.publish(*cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--camera_topic", default="camera/image_raw")
    parser.add_argument("--odom_topic", default="odom")
    parser.add_argument("--cmd_topic", default="cmd_vel")
    parser.add_argument("--image_mode", choices=["crop", "squash"], default="crop")
    parser.add_argument("--lin_speed", type=float, default=0.5)
    parser.add_argument("--ang_speed_deg", type=float, default=30.0)
    parser.add_argument("--max_lin", type=float, default=0.5, help="Hard clamp on |vx| (m/s).")
    parser.add_argument("--max_ang", type=float, default=0.8, help="Hard clamp on |wz| (rad/s).")
    parser.add_argument("--max_actions", type=int, default=60)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--enable_motion", action="store_true", help="Actually publish velocity commands.")
    args = parser.parse_args()

    rclpy.init()
    node = NavilaGo2Node(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish(0.0, 0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
