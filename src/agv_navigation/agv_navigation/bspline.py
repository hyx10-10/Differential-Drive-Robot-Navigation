import numpy as np
from scipy.interpolate import splprep, splev


class BSplineSmoother:
    """
    B样条路径平滑器

    功能：
    1. 对A*路径进行预处理
    2. B样条平滑
    3. 按空间距离均匀采样
    4. 计算轨迹航向角
    5. 保持起点终点
    """

    def __init__(
            self,
            degree: int = 3,
            smoothing: float = 0.2,
            sample_spacing: float = 0.10
    ):
        """
        参数
        ----
        degree:
            B样条阶数

        smoothing:
            平滑因子

            0.0:
                严格经过所有点

            较大:
                更平滑，但更容易切弯

            建议：
                0.05 ~ 0.5

        sample_spacing:
            轨迹点之间的空间距离
            单位：栅格
        """

        self.degree = degree
        self.smoothing = smoothing
        self.sample_spacing = sample_spacing

    # ============================================================
    # 删除重复点
    # ============================================================

    @staticmethod
    def remove_duplicate_points(
            waypoints: np.ndarray
    ) -> np.ndarray:

        if len(waypoints) <= 1:
            return waypoints

        cleaned = [waypoints[0]]

        for i in range(1, len(waypoints)):

            dist = np.linalg.norm(
                waypoints[i]
                -
                cleaned[-1]
            )

            if dist > 1e-6:

                cleaned.append(
                    waypoints[i]
                )

        return np.array(
            cleaned,
            dtype=float
        )

    # ============================================================
    # 弦长参数化
    # ============================================================

    @staticmethod
    def chord_length_parameterization(
            points: np.ndarray
    ) -> np.ndarray:
        """
        根据累计路径长度生成参数u
        """

        if len(points) < 2:

            return np.array([0.0])

        diff = np.diff(
            points,
            axis=0
        )

        distances = np.linalg.norm(
            diff,
            axis=1
        )

        cumulative = np.concatenate([
            [0.0],
            np.cumsum(distances)
        ])

        total_length = cumulative[-1]

        if total_length < 1e-8:

            return np.linspace(
                0.0,
                1.0,
                len(points)
            )

        u = cumulative / total_length

        return u

    # ============================================================
    # 根据空间距离均匀重采样
    # ============================================================

    def resample_by_distance(
            self,
            points: np.ndarray
    ) -> np.ndarray:
        """
        将轨迹点重新采样，
        使相邻点距离接近 sample_spacing
        """

        if len(points) < 2:
            return points

        diff = np.diff(
            points,
            axis=0
        )

        distances = np.linalg.norm(
            diff,
            axis=1
        )

        cumulative = np.concatenate([
            [0.0],
            np.cumsum(distances)
        ])

        total_length = cumulative[-1]

        if total_length < 1e-8:
            return points

        n_samples = max(
            int(
                total_length
                /
                self.sample_spacing
            )
            + 1,
            2
        )

        sample_distances = np.linspace(
            0.0,
            total_length,
            n_samples
        )

        x = np.interp(
            sample_distances,
            cumulative,
            points[:, 0]
        )

        y = np.interp(
            sample_distances,
            cumulative,
            points[:, 1]
        )

        return np.column_stack([
            x,
            y
        ])

    # ============================================================
    # 计算航向角
    # ============================================================

    @staticmethod
    def compute_yaw(
            points: np.ndarray
    ) -> np.ndarray:

        n = len(points)

        yaw = np.zeros(n)

        if n < 2:
            return yaw

        for i in range(n - 1):

            dx = (
                points[i + 1, 0]
                -
                points[i, 0]
            )

            dy = (
                points[i + 1, 1]
                -
                points[i, 1]
            )

            yaw[i] = np.arctan2(
                dy,
                dx
            )

        # 最后一个点使用倒数第二个方向
        yaw[-1] = yaw[-2]

        # unwrap防止 π 到 -π 突变
        yaw = np.unwrap(yaw)

        # 再归一化
        yaw = (
            yaw + np.pi
        ) % (
            2 * np.pi
        ) - np.pi

        return yaw

    # ============================================================
    # 主函数
    # ============================================================

    def sample_trajectory(
            self,
            waypoints: np.ndarray
    ) -> np.ndarray | None:
        """
        输入：

        waypoints:
            N×2

        输出：

        trajectory:
            M×3

            [
                x,
                y,
                yaw
            ]
        """

        # ========================================================
        # 1. 输入检查
        # ========================================================

        waypoints = np.asarray(
            waypoints,
            dtype=float
        )

        if (
                len(waypoints.shape) != 2
                or
                waypoints.shape[1] != 2
        ):

            raise ValueError(
                "waypoints 必须为 N×2 数组"
            )

        # 删除重复点
        waypoints = (
            self.remove_duplicate_points(
                waypoints
            )
        )

        n_points = len(waypoints)

        if n_points < 2:

            return None

        # ========================================================
        # 2. 如果点太少，直接线性重采样
        # ========================================================

        if n_points < 4:

            smooth_points = (
                self.resample_by_distance(
                    waypoints
                )
            )

            yaw = self.compute_yaw(
                smooth_points
            )

            return np.column_stack([
                smooth_points,
                yaw
            ])

        # ========================================================
        # 3. B样条阶数
        # ========================================================

        k = min(
            self.degree,
            n_points - 1
        )

        # ========================================================
        # 4. 参数化
        # ========================================================

        u = (
            self.chord_length_parameterization(
                waypoints
            )
        )

        # ========================================================
        # 5. B样条拟合
        # ========================================================

        try:

            tck, _ = splprep(

                [
                    waypoints[:, 0],
                    waypoints[:, 1]
                ],

                u=u,

                s=self.smoothing,

                k=k

            )

        except Exception as e:

            print(
                "B样条拟合失败:",
                e
            )

            return None

        # ========================================================
        # 6. 初步密集采样
        # ========================================================

        dense_n = max(
            n_points * 20,
            200
        )

        u_dense = np.linspace(
            0.0,
            1.0,
            dense_n
        )

        x_dense, y_dense = splev(
            u_dense,
            tck
        )

        dense_points = np.column_stack([
            x_dense,
            y_dense
        ])

        # ========================================================
        # 7. 强制保持起点终点
        # ========================================================

        dense_points[0] = waypoints[0]

        dense_points[-1] = waypoints[-1]

        # ========================================================
        # 8. 按空间距离均匀采样
        # ========================================================

        smooth_points = (
            self.resample_by_distance(
                dense_points
            )
        )

        # 再次确保首尾点
        smooth_points[0] = waypoints[0]

        smooth_points[-1] = waypoints[-1]

        # ========================================================
        # 9. 计算yaw
        # ========================================================

        yaw = self.compute_yaw(
            smooth_points
        )

        # ========================================================
        # 10. 输出
        # ========================================================

        trajectory = np.column_stack([
            smooth_points,
            yaw
        ])

        return trajectory