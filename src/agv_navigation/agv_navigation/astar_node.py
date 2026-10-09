import numpy as np

import rclpy
from rclpy.node import Node

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy
)

from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped

from .astar import AStarPlanner


class AStarNode(Node):

    def __init__(self):

        super().__init__('astar_node')

        # ==================================================
        # 状态
        # ==================================================

        self.map_msg = None
        self.obstacle_grid = None
        self.planner = None

        self.robot_x = None
        self.robot_y = None

        self.goal_x = None
        self.goal_y = None

        # ==================================================
        # 是否有新的目标点需要规划
        #
        # True  → 允许执行一次 A*
        # False → 不重复规划
        # ==================================================

        self.goal_dirty = False

        # ==================================================
        # 障碍物膨胀参数
        #
        # 地图分辨率：
        # 0.05 m / 栅格
        #
        # 膨胀半径：
        # 0.20 m
        #
        # 对应：
        # 0.20 / 0.05 = 4 个栅格
        # ==================================================

        self.inflation_radius = 4

        # ==================================================
        # 地图订阅
        #
        # map_server 使用：
        # RELIABLE + TRANSIENT_LOCAL
        # ==================================================

        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            map_qos
        )

        # ==================================================
        # 机器人里程计
        # ==================================================

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # ==================================================
        # RViz目标点
        # ==================================================

        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )

        # ==================================================
        # A*路径发布
        # ==================================================

        self.path_pub = self.create_publisher(
            Path,
            '/astar_path',
            10
        )

        self.get_logger().info(
            'A*节点启动'
        )

        self.get_logger().info(
            '等待地图 /map、机器人 /odom 和目标 /goal_pose ...'
        )

        self.get_logger().info(
            f'障碍物膨胀半径: '
            f'{self.inflation_radius} 个栅格 '
            f'({self.inflation_radius * 0.05:.2f} m)'
        )

    # ======================================================
    # 地图回调
    # ======================================================

    def map_callback(self, msg):

        self.map_msg = msg

        self.get_logger().info(
            f'收到地图: '
            f'{msg.info.width} x {msg.info.height}'
        )

        self.get_logger().info(
            f'resolution = {msg.info.resolution}'
        )

        self.get_logger().info(
            f'origin = '
            f'({msg.info.origin.position.x}, '
            f'{msg.info.origin.position.y})'
        )

        # ==================================================
        # ROS地图 → numpy
        # ==================================================

        grid = np.array(
            msg.data,
            dtype=np.int8
        ).reshape(
            msg.info.height,
            msg.info.width
        )

        # ==================================================
        # 地图转换
        #
        # 0       → 可通行
        # 100     → 障碍物
        # -1      → 未知
        #
        # >= 50 → 障碍物
        # < 50  → 可通行
        #
        # 未知区域也作为障碍物
        # ==================================================

        obstacle_grid = np.zeros_like(
            grid,
            dtype=np.uint8
        )

        obstacle_grid[grid >= 50] = 1
        obstacle_grid[grid < 0] = 1

        # ==================================================
        # 障碍物膨胀
        # ==================================================

        inflated_grid = self.inflate_obstacles(
            obstacle_grid,
            self.inflation_radius
        )

        self.obstacle_grid = inflated_grid

        # ==================================================
        # 创建 A* 规划器
        # ==================================================

        self.planner = AStarPlanner(
            self.obstacle_grid
        )

        self.get_logger().info(
            '障碍物膨胀完成，A*地图准备完成'
        )

        # ==================================================
        # 地图准备好以后，
        # 如果已经收到新的目标，则尝试规划
        # ==================================================

        self.try_plan()

    # ======================================================
    # 障碍物膨胀
    # ======================================================

    @staticmethod
    def inflate_obstacles(
        obstacle_grid,
        inflation_radius
    ):

        """
        对障碍物进行圆形膨胀。

        obstacle_grid:
            0 = 可通行
            1 = 障碍物

        inflation_radius:
            膨胀半径，单位：栅格
        """

        height, width = obstacle_grid.shape

        inflated_grid = obstacle_grid.copy()

        # ==================================================
        # 找到所有障碍物栅格
        # ==================================================

        obstacle_indices = np.argwhere(
            obstacle_grid == 1
        )

        # ==================================================
        # 对每一个障碍物进行膨胀
        # ==================================================

        for y, x in obstacle_indices:

            for dy in range(
                -inflation_radius,
                inflation_radius + 1
            ):

                for dx in range(
                    -inflation_radius,
                    inflation_radius + 1
                ):

                    # 圆形膨胀
                    if (
                        dx * dx + dy * dy
                        >
                        inflation_radius * inflation_radius
                    ):
                        continue

                    nx = x + dx
                    ny = y + dy

                    # 检查是否在地图范围内
                    if (
                        0 <= nx < width
                        and
                        0 <= ny < height
                    ):

                        inflated_grid[ny, nx] = 1

        return inflated_grid

    # ======================================================
    # 机器人位置
    # ======================================================

    def odom_callback(self, msg):

        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        # 第一次收到机器人位置时打印
        if not hasattr(self, 'odom_received'):

            self.odom_received = True

            self.get_logger().info(
                f'机器人当前位置: '
                f'({self.robot_x:.3f}, '
                f'{self.robot_y:.3f})'
            )

        # ==================================================
        # 注意：
        #
        # 这里不调用 try_plan()
        #
        # 防止 /odom 持续更新导致 A* 重复规划
        # ==================================================

    # ======================================================
    # RViz目标点
    # ======================================================

    def goal_callback(self, msg):

        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y

        # ==================================================
        # 标记收到新的目标
        # ==================================================

        self.goal_dirty = True

        self.get_logger().info(
            '========================================'
        )

        self.get_logger().info(
            f'收到新的目标点: '
            f'({self.goal_x:.3f}, '
            f'{self.goal_y:.3f})'
        )

        self.get_logger().info(
            '开始执行 A* 规划...'
        )

        # ==================================================
        # 新目标 → 执行一次规划
        # ==================================================

        self.try_plan()

    # ======================================================
    # 尝试规划
    # ======================================================

    def try_plan(self):

        # ==================================================
        # 没有新的目标
        # ==================================================

        if not self.goal_dirty:
            return

        # ==================================================
        # 地图还没有
        # ==================================================

        if self.map_msg is None:
            return

        # ==================================================
        # A*还没有创建
        # ==================================================

        if self.planner is None:
            return

        # ==================================================
        # 机器人位置还没有
        # ==================================================

        if self.robot_x is None:
            return

        # ==================================================
        # 目标还没有
        # ==================================================

        if self.goal_x is None:
            return

        # ==================================================
        # 世界坐标 → 栅格坐标
        # ==================================================

        start = self.world_to_map(
            self.robot_x,
            self.robot_y,
            self.map_msg
        )

        goal = self.world_to_map(
            self.goal_x,
            self.goal_y,
            self.map_msg
        )

        self.get_logger().info(
            f'起点世界坐标: '
            f'({self.robot_x:.3f}, '
            f'{self.robot_y:.3f})'
        )

        self.get_logger().info(
            f'起点栅格坐标: {start}'
        )

        self.get_logger().info(
            f'终点世界坐标: '
            f'({self.goal_x:.3f}, '
            f'{self.goal_y:.3f})'
        )

        self.get_logger().info(
            f'终点栅格坐标: {goal}'
        )

        # ==================================================
        # 检查起点、终点是否在地图范围内
        # ==================================================

        height, width = self.obstacle_grid.shape

        if not (
            0 <= start[0] < width
            and
            0 <= start[1] < height
        ):

            self.get_logger().error(
                'A*起点超出地图范围！'
            )

            return

        if not (
            0 <= goal[0] < width
            and
            0 <= goal[1] < height
        ):

            self.get_logger().error(
                'A*终点超出地图范围！'
            )

            return

        # ==================================================
        # 检查起点和终点
        # ==================================================

        if self.obstacle_grid[
            start[1],
            start[0]
        ] == 1:

            self.get_logger().error(
                'A*起点位于障碍物内！'
            )

            return

        if self.obstacle_grid[
            goal[1],
            goal[0]
        ] == 1:

            self.get_logger().error(
                'A*终点位于障碍物内！'
            )

            return

        # ==================================================
        # A*
        # ==================================================

        path = self.planner.plan(
            start,
            goal,
            simplify=False
        )

        if path is None:

            self.get_logger().error(
                'A*规划失败！'
            )

            return

        self.get_logger().info(
            f'A*规划成功！'
            f'路径点数量: {len(path)}'
        )

        # ==================================================
        # 栅格路径 → ROS Path
        # ==================================================

        ros_path = Path()

        ros_path.header.frame_id = (
            self.map_msg.header.frame_id
            if self.map_msg.header.frame_id
            else 'map'
        )

        ros_path.header.stamp = (
            self.get_clock().now().to_msg()
        )

        for mx, my in path:

            x, y = self.map_to_world(
                mx,
                my,
                self.map_msg
            )

            pose = PoseStamped()

            pose.header = ros_path.header

            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0

            pose.pose.orientation.w = 1.0

            ros_path.poses.append(
                pose
            )

        # ==================================================
        # 发布路径
        # ==================================================

        self.path_pub.publish(
            ros_path
        )

        self.get_logger().info(
            'A*路径已经发布到 /astar_path'
        )

        self.get_logger().info(
            '========================================'
        )

        # ==================================================
        # 本次目标已经规划完成
        #
        # 后面的 /odom 不会重新规划
        # ==================================================

        self.goal_dirty = False

    # ======================================================
    # 世界坐标 → 地图栅格
    # ======================================================

    @staticmethod
    def world_to_map(
        x,
        y,
        map_msg
    ):

        resolution = map_msg.info.resolution

        origin_x = (
            map_msg.info.origin.position.x
        )

        origin_y = (
            map_msg.info.origin.position.y
        )

        mx = int(
            (x - origin_x)
            / resolution
        )

        my = int(
            (y - origin_y)
            / resolution
        )

        return mx, my

    # ======================================================
    # 地图栅格 → 世界坐标
    # ======================================================

    @staticmethod
    def map_to_world(
        mx,
        my,
        map_msg
    ):

        resolution = map_msg.info.resolution

        origin_x = (
            map_msg.info.origin.position.x
        )

        origin_y = (
            map_msg.info.origin.position.y
        )

        x = (
            origin_x
            +
            (mx + 0.5)
            * resolution
        )

        y = (
            origin_y
            +
            (my + 0.5)
            * resolution
        )

        return x, y


def main(args=None):

    rclpy.init(args=args)

    node = AStarNode()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
