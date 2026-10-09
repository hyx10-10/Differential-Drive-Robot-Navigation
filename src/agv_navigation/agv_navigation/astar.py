import numpy as np
import heapq


class AStarPlanner:
    """
    A* 栅格路径规划器

    支持：
    - 8邻域搜索
    - 禁止斜向穿越障碍物角落
    - Octile Distance启发函数
    - 路径简化
    """

    def __init__(self, grid: np.ndarray):
        """
        参数
        ----
        grid : np.ndarray
            二维栅格地图
            0 = 可通行
            1 = 障碍物
        """

        if len(grid.shape) != 2:
            raise ValueError("grid 必须是二维数组")

        self.grid = grid
        self.h, self.w = grid.shape

        # 8邻域
        self.motions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),

            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1)
        ]

    # ============================================================
    # 边界检查
    # ============================================================

    def in_bounds(self, x: int, y: int) -> bool:

        return (
            0 <= x < self.w
            and
            0 <= y < self.h
        )

    # ============================================================
    # 是否障碍物
    # ============================================================

    def is_obstacle(self, x: int, y: int) -> bool:

        if not self.in_bounds(x, y):
            return True

        return self.grid[y, x] == 1

    # ============================================================
    # Octile Distance
    # ============================================================

    def heuristic(
            self,
            x: int,
            y: int,
            gx: int,
            gy: int
    ) -> float:
        """
        适用于8邻域的启发函数

        D = 1
        D2 = sqrt(2)

        h = D*(dx+dy)
            + (D2-2D)*min(dx,dy)
        """

        dx = abs(x - gx)
        dy = abs(y - gy)

        D = 1.0
        D2 = np.sqrt(2)

        return (
            D * (dx + dy)
            +
            (D2 - 2 * D)
            * min(dx, dy)
        )

    # ============================================================
    # 判断移动是否合法
    # ============================================================

    def is_valid_move(
            self,
            x: int,
            y: int,
            dx: int,
            dy: int
    ) -> bool:
        """
        判断从 (x,y) 沿 (dx,dy) 移动是否合法

        特别处理：
        对角移动时禁止从障碍物角落穿过去

        例如：

        ■ .
        . ●

        ●不能直接斜向移动到■对角另一侧
        """

        nx = x + dx
        ny = y + dy

        # 边界
        if not self.in_bounds(nx, ny):
            return False

        # 目标点障碍物
        if self.is_obstacle(nx, ny):
            return False

        # --------------------------------------------------------
        # 对角线移动
        # --------------------------------------------------------

        if abs(dx) == 1 and abs(dy) == 1:

            # 禁止穿越障碍物角落
            #
            # (x,y) -> (x+dx,y+dy)
            #
            # 同时检查：
            # (x+dx,y)
            # (x,y+dy)
            #

            if self.is_obstacle(x + dx, y):
                return False

            if self.is_obstacle(x, y + dy):
                return False

        return True

    # ============================================================
    # A*规划
    # ============================================================

    def plan(
            self,
            start: tuple[int, int],
            goal: tuple[int, int],
            simplify: bool = False
    ) -> list[tuple[int, int]] | None:
        """
        A*路径规划

        参数
        ----
        start:
            (x, y)

        goal:
            (x, y)

        simplify:
            是否进行路径简化

        返回
        ----
        path:
            [(x,y), ...]

        找不到路径返回 None
        """

        sx, sy = start
        gx, gy = goal

        # --------------------------------------------------------
        # 起点终点边界检查
        # --------------------------------------------------------

        if not self.in_bounds(sx, sy):
            print("A*: 起点越界")
            return None

        if not self.in_bounds(gx, gy):
            print("A*: 终点越界")
            return None

        # --------------------------------------------------------
        # 起点终点障碍物检查
        # --------------------------------------------------------

        if self.is_obstacle(sx, sy):
            print("A*: 起点位于障碍物内")
            return None

        if self.is_obstacle(gx, gy):
            print("A*: 终点位于障碍物内")
            return None

        # 起点终点相同
        if start == goal:
            return [start]

        # --------------------------------------------------------
        # Open List
        # --------------------------------------------------------

        open_heap = []

        # 元素：
        #
        # (f, g, x, y)
        #
        # 同时保存 g，
        # 方便判断过期节点

        start_h = self.heuristic(
            sx, sy, gx, gy
        )

        heapq.heappush(
            open_heap,
            (
                start_h,
                0.0,
                sx,
                sy
            )
        )

        # --------------------------------------------------------
        # g cost
        # --------------------------------------------------------

        g_cost = np.full(
            (self.h, self.w),
            np.inf,
            dtype=float
        )

        g_cost[sy, sx] = 0.0

        # --------------------------------------------------------
        # parent
        # --------------------------------------------------------

        parent = {}

        # ========================================================
        # A* 主循环
        # ========================================================

        while open_heap:

            f_curr, g_curr, x_curr, y_curr = (
                heapq.heappop(open_heap)
            )

            # ----------------------------------------------------
            # 跳过过期节点
            #
            # 一个节点可能被多次加入heap
            # 如果当前g比最新g大，说明已经过期
            # ----------------------------------------------------

            if g_curr > g_cost[y_curr, x_curr]:
                continue

            # ----------------------------------------------------
            # 到达终点
            # ----------------------------------------------------

            if x_curr == gx and y_curr == gy:

                path = self.reconstruct_path(
                    parent,
                    start,
                    goal
                )

                if simplify:
                    path = self.simplify_path(path)

                return path

            # ----------------------------------------------------
            # 遍历邻居
            # ----------------------------------------------------

            for dx, dy in self.motions:

                # 合法移动检查
                if not self.is_valid_move(
                        x_curr,
                        y_curr,
                        dx,
                        dy
                ):
                    continue

                nx = x_curr + dx
                ny = y_curr + dy

                # ------------------------------------------------
                # 移动代价
                # ------------------------------------------------

                if abs(dx) == 1 and abs(dy) == 1:

                    move_cost = np.sqrt(2)

                else:

                    move_cost = 1.0

                new_g = (
                    g_cost[y_curr, x_curr]
                    +
                    move_cost
                )

                # ------------------------------------------------
                # 找到更优路径
                # ------------------------------------------------

                if new_g < g_cost[ny, nx]:

                    g_cost[ny, nx] = new_g

                    h = self.heuristic(
                        nx,
                        ny,
                        gx,
                        gy
                    )

                    f = new_g + h

                    heapq.heappush(
                        open_heap,
                        (
                            f,
                            new_g,
                            nx,
                            ny
                        )
                    )

                    parent[(nx, ny)] = (
                        x_curr,
                        y_curr
                    )

        # 找不到路径
        print("A*: 找不到可行路径")

        return None

    # ============================================================
    # 回溯路径
    # ============================================================

    @staticmethod
    def reconstruct_path(
            parent,
            start,
            goal
    ) -> list[tuple[int, int]]:

        path = []

        current = goal

        path.append(current)

        while current != start:

            current = parent[current]

            path.append(current)

        path.reverse()

        return path

    # ============================================================
    # 路径简化
    # ============================================================

    @staticmethod
    def simplify_path(
            path: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """
        删除同方向的冗余点

        原路径：

        A - B - C - D

        如果A→B、B→C、C→D方向相同

        简化为：

        A -------- D

        注意：
        这里只删除共线点，
        不进行跨障碍物的直线连接，
        所以安全性不会降低。
        """

        if len(path) <= 2:
            return path.copy()

        simplified = [path[0]]

        prev_dx = None
        prev_dy = None

        for i in range(1, len(path)):

            x0, y0 = path[i - 1]
            x1, y1 = path[i]

            dx = x1 - x0
            dy = y1 - y0

            # 单位方向
            dx = int(np.sign(dx))
            dy = int(np.sign(dy))

            if prev_dx is None:

                prev_dx = dx
                prev_dy = dy

                continue

            # 方向发生变化
            if (
                    dx != prev_dx
                    or
                    dy != prev_dy
            ):

                simplified.append(
                    path[i - 1]
                )

                prev_dx = dx
                prev_dy = dy

        simplified.append(path[-1])

        return simplified
