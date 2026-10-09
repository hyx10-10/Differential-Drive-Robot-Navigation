import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

from .bspline import BSplineSmoother


class BSplineNode(Node):

    def __init__(self):

        super().__init__('bspline_node')

        # ==================================================
        # 参数
        # ==================================================

        self.declare_parameter('degree', 3)
        self.declare_parameter('smoothing', 0.2)
        self.declare_parameter('sample_spacing', 0.10)

        degree = self.get_parameter(
            'degree'
        ).value

        smoothing = self.get_parameter(
            'smoothing'
        ).value

        sample_spacing = self.get_parameter(
            'sample_spacing'
        ).value

        # ==================================================
        # 创建 B-spline 平滑器
        # ==================================================

        self.smoother = BSplineSmoother(
            degree=degree,
            smoothing=smoothing,
            sample_spacing=sample_spacing
        )

        # ==================================================
        # 订阅 A* 路径
        # ==================================================

        self.path_sub = self.create_subscription(
            Path,
            '/astar_path',
            self.path_callback,
            10
        )

        # ==================================================
        # 发布 B-spline 路径
        # ==================================================

        self.bspline_pub = self.create_publisher(
            Path,
            '/bspline_path',
            10
        )

        self.get_logger().info(
            'B-spline节点启动'
        )

        self.get_logger().info(
            f'degree = {degree}'
        )

        self.get_logger().info(
            f'smoothing = {smoothing}'
        )

        self.get_logger().info(
            f'sample_spacing = {sample_spacing} m'
        )

        self.get_logger().info(
            '等待 /astar_path ...'
        )

    # ======================================================
    # A*路径回调
    # ======================================================

    def path_callback(self, msg):

        if len(msg.poses) < 2:

            self.get_logger().warn(
                'A*路径点太少，无法进行B-spline平滑'
            )

            return

        # ==================================================
        # ROS Path → numpy N×2
        # ==================================================

        waypoints = np.array(
            [
                [
                    pose.pose.position.x,
                    pose.pose.position.y
                ]
                for pose in msg.poses
            ],
            dtype=float
        )

        self.get_logger().info(
            f'收到A*路径: {len(waypoints)} 个点'
        )

        # ==================================================
        # B-spline
        # ==================================================

        trajectory = self.smoother.sample_trajectory(
            waypoints
        )

        if trajectory is None:

            self.get_logger().error(
                'B-spline轨迹生成失败！'
            )

            return

        # trajectory:
        #
        # [x, y, yaw]
        #

        self.get_logger().info(
            f'B-spline轨迹生成成功: '
            f'{len(trajectory)} 个点'
        )

        # ==================================================
        # numpy → ROS Path
        # ==================================================

        bspline_path = Path()

        bspline_path.header.frame_id = (
            msg.header.frame_id
            if msg.header.frame_id
            else 'map'
        )

        bspline_path.header.stamp = (
            self.get_clock().now().to_msg()
        )

        for x, y, yaw in trajectory:

            pose = PoseStamped()

            pose.header = bspline_path.header

            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0

            # yaw → quaternion
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0

            pose.pose.orientation.z = np.sin(
                yaw / 2.0
            )

            pose.pose.orientation.w = np.cos(
                yaw / 2.0
            )

            bspline_path.poses.append(
                pose
            )

        # ==================================================
        # 发布
        # ==================================================

        self.bspline_pub.publish(
            bspline_path
        )

        self.get_logger().info(
            'B-spline路径已发布到 /bspline_path'
        )


def main(args=None):

    rclpy.init(args=args)

    node = BSplineNode()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':
    main()