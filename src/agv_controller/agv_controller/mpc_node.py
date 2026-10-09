import math

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist

from .mpc import NonlinearMPC


class MPCNode(Node):

    def __init__(self):
        super().__init__('mpc_node')

        # =====================================================
        # MPC
        # =====================================================

        self.mpc = NonlinearMPC()

        # 当前机器人状态
        # [x, y, yaw]
        self.current_state = None

        # B-spline 参考轨迹
        # [[x, y, yaw], ...]
        self.reference_trajectory = None

        # MPC 当前使用的参考点索引
        self.current_index = 0

        # =====================================================
        # ROS Subscribers
        # =====================================================

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.path_sub = self.create_subscription(
            Path,
            '/bspline_path',
            self.path_callback,
            10
        )

        # =====================================================
        # ROS Publisher
        # =====================================================

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # =====================================================
        # MPC 定时器
        # 10 Hz
        # =====================================================

        self.timer = self.create_timer(
            0.1,
            self.mpc_callback
        )

        self.get_logger().info(
            'MPC Node 已启动，等待 /odom 和 /bspline_path...'
        )

    # =========================================================
    # 四元数 → yaw
    # =========================================================

    @staticmethod
    def quaternion_to_yaw(q):
        siny_cosp = 2.0 * (
            q.w * q.z +
            q.x * q.y
        )

        cosy_cosp = 1.0 - 2.0 * (
            q.y * q.y +
            q.z * q.z
        )

        return math.atan2(
            siny_cosp,
            cosy_cosp
        )

    # =========================================================
    # /odom
    # =========================================================

    def odom_callback(self, msg):

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        yaw = self.quaternion_to_yaw(
            msg.pose.pose.orientation
        )

        self.current_state = [
            x,
            y,
            yaw
        ]

    # =========================================================
    # /bspline_path
    # =========================================================

    def path_callback(self, msg):

        if len(msg.poses) == 0:
            self.reference_trajectory = None
            return

        trajectory = []

        for pose in msg.poses:

            x = pose.pose.position.x
            y = pose.pose.position.y

            yaw = self.quaternion_to_yaw(
                pose.pose.orientation
            )

            trajectory.append([
                x,
                y,
                yaw
            ])

        self.reference_trajectory = trajectory

        # 新路径从头开始寻找
        self.current_index = 0

        self.get_logger().info(
            f'收到 B-spline 轨迹，共 {len(trajectory)} 个点'
        )

    # =========================================================
    # MPC
    # =========================================================

    def mpc_callback(self):

        # 没有机器人状态
        if self.current_state is None:
            return

        # 没有参考轨迹
        if self.reference_trajectory is None:
            return

        # 参考轨迹太短
        if len(self.reference_trajectory) < 2:
            return

        try:

            # 转成 numpy
            import numpy as np

            current_state = np.array(
                self.current_state,
                dtype=float
            )

            reference = np.array(
                self.reference_trajectory,
                dtype=float
            )

            # =================================================
            # 求解 MPC
            # =================================================

            control, nearest_index = self.mpc.solve(
                current_state,
                reference,
                self.current_index
            )

            # 更新当前参考索引
            self.current_index = nearest_index

            v = float(control[0])
            omega = float(control[1])

            # =================================================
            # 发布 /cmd_vel
            # =================================================

            cmd = Twist()

            cmd.linear.x = v
            cmd.angular.z = omega

            self.cmd_vel_pub.publish(cmd)

        except Exception as e:

            self.get_logger().error(
                f'MPC 求解失败: {e}'
            )

            # 出现异常立即停车
            cmd = Twist()

            self.cmd_vel_pub.publish(cmd)


def main(args=None):

    rclpy.init(args=args)

    node = MPCNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        # 退出前停车
        cmd = Twist()
        node.cmd_vel_pub.publish(cmd)

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':
    main()