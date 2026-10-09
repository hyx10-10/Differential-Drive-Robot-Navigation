import numpy as np
from scipy.optimize import minimize


class NonlinearMPC:
    """
    差分轮AGV非线性模型预测控制器

    状态：
        [x, y, yaw]

    控制：
        [v, omega]

    非线性模型：

        x(k+1) =
            x(k)
            +
            v(k)*cos(yaw(k))*dt

        y(k+1) =
            y(k)
            +
            v(k)*sin(yaw(k))*dt

        yaw(k+1) =
            yaw(k)
            +
            omega(k)*dt
    """

    def __init__(
            self,
            dt=0.1,
            horizon=10,
            v_max=0.8,
            omega_max=1.5
    ):

        self.dt = dt
        self.N = horizon

        # ========================================================
        # 控制约束
        # ========================================================

        self.v_min = 0.0
        self.v_max = v_max

        self.omega_min = -omega_max
        self.omega_max = omega_max

        # ========================================================
        # 代价权重
        # ========================================================

        # 位置误差
        self.Q_pos = 20.0

        # 航向误差
        self.Q_yaw = 3.0

        # 控制输入
        self.R_v = 0.2
        self.R_omega = 0.1

        # 控制变化
        self.R_dv = 0.5
        self.R_domega = 0.2

        # 终端误差
        self.Q_terminal_pos = 50.0
        self.Q_terminal_yaw = 5.0

        # ========================================================
        # 上一时刻控制
        # ========================================================

        self.u_prev = np.array([
            0.0,
            0.0
        ])

        # ========================================================
        # 上一次优化结果
        # 用于Warm Start
        # ========================================================

        self.last_solution = None

    # ============================================================
    # 角度归一化
    # ============================================================

    @staticmethod
    def normalize_angle(angle):

        return (
            angle + np.pi
        ) % (
            2.0 * np.pi
        ) - np.pi

    # ============================================================
    # 非线性运动学模型
    # ============================================================

    def vehicle_model(
            self,
            state,
            control
    ):
        """
        单步预测
        """

        x, y, yaw = state

        v, omega = control

        x_next = (
            x
            +
            v
            * np.cos(yaw)
            * self.dt
        )

        y_next = (
            y
            +
            v
            * np.sin(yaw)
            * self.dt
        )

        yaw_next = (
            yaw
            +
            omega
            * self.dt
        )

        yaw_next = self.normalize_angle(
            yaw_next
        )

        return np.array([
            x_next,
            y_next,
            yaw_next
        ])

    # ============================================================
    # 预测整个控制时域
    # ============================================================

    def predict_trajectory(
            self,
            current_state,
            control_sequence
    ):
        """
        根据控制序列预测未来轨迹

        control_sequence shape:

            (N, 2)
        """

        states = np.zeros(
            (self.N + 1, 3)
        )

        states[0] = current_state

        state = current_state.copy()

        for k in range(self.N):

            control = control_sequence[k]

            state = self.vehicle_model(
                state,
                control
            )

            states[k + 1] = state

        return states

    # ============================================================
    # 找最近轨迹点
    # ============================================================

    @staticmethod
    def find_nearest_index(
            current_state,
            ref_traj,
            start_index=0
    ):

        position = current_state[:2]

        ref_positions = ref_traj[
            start_index:,
            :2
        ]

        distances = np.linalg.norm(
            ref_positions - position,
            axis=1
        )

        local_index = np.argmin(
            distances
        )

        return start_index + local_index

    # ============================================================
    # 获取MPC参考窗口
    # ============================================================

    def get_reference_window(
            self,
            ref_traj,
            nearest_index
    ):
        """
        从参考轨迹中提取未来N步参考点

        注意：

        由于B样条轨迹按空间距离采样，
        不能简单认为：

            index + 1

        就对应一个控制周期。

        因此使用lookahead step。
        """

        total_points = len(ref_traj)

        refs = np.zeros(
            (self.N + 1, 3)
        )

        # 每一个MPC预测步
        # 前进的参考点数量

        lookahead_step = 2

        for k in range(self.N + 1):

            idx = (
                nearest_index
                +
                k * lookahead_step
            )

            if idx >= total_points:

                idx = total_points - 1

            refs[k] = ref_traj[idx]

        return refs

    # ============================================================
    # 代价函数
    # ============================================================

    def cost_function(
            self,
            u_flat,
            current_state,
            ref_window
    ):
        """
        MPC代价函数
        """

        # 转换控制变量

        controls = u_flat.reshape(
            self.N,
            2
        )

        # 非线性预测

        predicted_states = (
            self.predict_trajectory(
                current_state,
                controls
            )
        )

        total_cost = 0.0

        # ========================================================
        # 轨迹跟踪代价
        # ========================================================

        for k in range(self.N):

            state = predicted_states[k + 1]

            ref = ref_window[k + 1]

            # ----------------------------------------------------
            # 位置误差
            # ----------------------------------------------------

            dx = state[0] - ref[0]

            dy = state[1] - ref[1]

            pos_error_sq = (
                dx ** 2
                +
                dy ** 2
            )

            total_cost += (
                self.Q_pos
                *
                pos_error_sq
            )

            # ----------------------------------------------------
            # 航向误差
            # ----------------------------------------------------

            yaw_error = (
                self.normalize_angle(
                    state[2]
                    -
                    ref[2]
                )
            )

            total_cost += (
                self.Q_yaw
                *
                yaw_error ** 2
            )

            # ----------------------------------------------------
            # 控制输入代价
            # ----------------------------------------------------

            v = controls[k, 0]

            omega = controls[k, 1]

            total_cost += (
                self.R_v
                *
                v ** 2
            )

            total_cost += (
                self.R_omega
                *
                omega ** 2
            )

            # ----------------------------------------------------
            # 控制变化率
            # ----------------------------------------------------

            if k == 0:

                dv = (
                    controls[k, 0]
                    -
                    self.u_prev[0]
                )

                domega = (
                    controls[k, 1]
                    -
                    self.u_prev[1]
                )

            else:

                dv = (
                    controls[k, 0]
                    -
                    controls[k - 1, 0]
                )

                domega = (
                    controls[k, 1]
                    -
                    controls[k - 1, 1]
                )

            total_cost += (
                self.R_dv
                *
                dv ** 2
            )

            total_cost += (
                self.R_domega
                *
                domega ** 2
            )

        # ========================================================
        # 终端状态代价
        # ========================================================

        terminal_state = predicted_states[-1]

        terminal_ref = ref_window[-1]

        dx = (
            terminal_state[0]
            -
            terminal_ref[0]
        )

        dy = (
            terminal_state[1]
            -
            terminal_ref[1]
        )

        terminal_pos_error = (
            dx ** 2
            +
            dy ** 2
        )

        total_cost += (
            self.Q_terminal_pos
            *
            terminal_pos_error
        )

        terminal_yaw_error = (
            self.normalize_angle(
                terminal_state[2]
                -
                terminal_ref[2]
            )
        )

        total_cost += (
            self.Q_terminal_yaw
            *
            terminal_yaw_error ** 2
        )

        return total_cost

    # ============================================================
    # 生成初始控制猜测
    # ============================================================

    def get_initial_guess(self):

        # --------------------------------------------------------
        # 如果之前没有解
        # --------------------------------------------------------

        if self.last_solution is None:

            controls = np.zeros(
                (self.N, 2)
            )

            controls[:, 0] = 0.3

            controls[:, 1] = 0.0

            return controls.flatten()

        # --------------------------------------------------------
        # Warm Start
        #
        # 丢弃第一个控制
        # 后面的控制整体前移
        # --------------------------------------------------------

        old_controls = (
            self.last_solution.reshape(
                self.N,
                2
            )
        )

        new_controls = np.zeros_like(
            old_controls
        )

        new_controls[:-1] = (
            old_controls[1:]
        )

        new_controls[-1] = (
            old_controls[-1]
        )

        return new_controls.flatten()

    # ============================================================
    # 求解MPC
    # ============================================================

    def solve(
            self,
            current_state,
            ref_traj,
            current_index=0
    ):
        """
        求解非线性MPC

        返回：

            control:
                [v, omega]

            nearest_index:
                最近参考轨迹索引
        """

        # ========================================================
        # 1. 寻找最近点
        # ========================================================

        nearest_index = (
            self.find_nearest_index(
                current_state,
                ref_traj,
                current_index
            )
        )

        # ========================================================
        # 2. 获取参考窗口
        # ========================================================

        ref_window = (
            self.get_reference_window(
                ref_traj,
                nearest_index
            )
        )

        # ========================================================
        # 3. 初始猜测
        # ========================================================

        u0 = self.get_initial_guess()

        # ========================================================
        # 4. 控制变量边界
        # ========================================================

        bounds = []

        for _ in range(self.N):

            # v
            bounds.append(
                (
                    self.v_min,
                    self.v_max
                )
            )

            # omega
            bounds.append(
                (
                    self.omega_min,
                    self.omega_max
                )
            )

        # ========================================================
        # 5. 优化
        # ========================================================

        result = minimize(

            fun=self.cost_function,

            x0=u0,

            args=(
                current_state,
                ref_window
            ),

            method="SLSQP",

            bounds=bounds,

            options={

                "maxiter": 50,

                "ftol": 1e-3,

                "disp": False

            }

        )

        # ========================================================
        # 6. 求解失败处理
        # ========================================================

        if not result.success:

            # 不直接报错停止
            # 使用上一控制量

            print(
                "NMPC求解失败:",
                result.message
            )

            fallback = np.array([
                0.0,
                0.0
            ])

            return (
                fallback,
                nearest_index
            )

        # ========================================================
        # 7. 保存解
        # ========================================================

        self.last_solution = (
            result.x.copy()
        )

        # ========================================================
        # 8. 输出第一个控制
        # ========================================================

        optimal_controls = (
            result.x.reshape(
                self.N,
                2
            )
        )

        control = (
            optimal_controls[0]
        )

        # ========================================================
        # 9. 终点附近减速
        # ========================================================

        goal = ref_traj[-1, :2]

        distance_to_goal = np.linalg.norm(

            current_state[:2]

            -

            goal

        )

        if distance_to_goal < 2.0:

            speed_scale = max(
                0.2,
                distance_to_goal / 2.0
            )

            control[0] *= speed_scale

        # ========================================================
        # 10. 保存当前控制
        # ========================================================

        self.u_prev = control.copy()

        return (
            control,
            nearest_index
        )