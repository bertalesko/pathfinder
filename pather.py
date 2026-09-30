import os
import numpy as np
import heapq
import matplotlib.pyplot as plt
from PIL import Image
from collections import deque

walkable_color = np.array([89, 89, 89])

pos_start = [210, 120]

rune_values = {
    "Opulent": {"LootMult": 1.35, "Avoid": False},
    "Bond": {"LootMult": 1.25, "Avoid": False},
    "Power": {"LootMult": 1.30, "Avoid": False},
    "Time": {"LootMult": 1.18, "Avoid": False},
    "Death": {"LootMult": 1.15, "Avoid": False},
    "Rebirth": {"LootMult": 1.10, "Avoid": False},
    "Wisdom": {"LootMult": 0.95, "Avoid": True},
    "Oath": {"LootMult": 0.75, "Avoid": True},
    "Bait": {"LootMult": 1.00, "Avoid": True},
}


def check_rune(name: str):
    if name not in rune_values:
        return {"LootMult": 1, "Avoid": False}
    else:
        return rune_values[name]


class Monolith:
    def __init__(self, data):
        self.id = data["id"]
        self.grid_pos = data["grid_pos"]  # [x, y]
        self.num_slots = data["num_slots"]
        self.crown_runes = data["crown_runes"]
        self.crown_slots = data["crown_slots"]
        self.value_runes = data["rune_values"]
        self.value_summed = sum(self.value_runes.values())
        # print(self.value_summed)

    def __str__(self) -> str:
        return f"Monolith {self.id} at {self.grid_pos} slots {self.num_slots}"


class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.f_score = float('inf')
        self.g_score = float('inf')
        self.h_score = float('inf')
        self.parent_i = None  # parent row (y)
        self.parent_j = None  # parent col (x)
        self.walkable = False
        self.is_target = False

    def __str__(self) -> str:
        return f"Cell at ({self.x}, {self.y})"

    __repr__ = __str__


class Board:
    def __init__(self, imgname):
        # data = pop_data()
        pil = Image.open(imgname).convert("RGB")
        self.im = np.array(pil)  # HxWx3, uint8, 0-255
        self.rows, self.cols = self.im.shape[:2]  # H, W
        # self.cols = self.w          # width  (x)
        # self.rows = self.h          # height (y)
        self.cells = []
        self.populate_cells()

    def populate_cells(self):
        for r in range(self.rows):  # r = row = y
            self.cells.append([])
            for c in range(self.cols):  # c = col = x
                cell = Cell(c, r)
                if np.array_equal(self.im[r][c], walkable_color):
                    cell.walkable = True
                self.cells[r].append(cell)  # Cell(x=col, y=row)

        """for i in range(self.h):
            for j in range(self.w):                
                    self.cells[i][j].walkable = True
                else:
                    self.cells[i][j].walkable = False"""

    def __getitem__(self, index):
        return self.cells[index]  # board[y] -> row, board[y][x] -> Cell

    def is_valid(self, row, col):
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_walkable(self, row, col):
        cell = self.cells[row][col]
        return cell.walkable


def pop_data():
    data = {}
    monoliths = [
        {
            "id": 1,
            "grid_pos": [150, 133],
            'num_slots': 4, 'crown_runes': 1, 'crown_slots': [0],
            'rune_type': {0: 'Bond', 1: 'Bait', 2: 'Rebirth', 3: 'Oath', 4: 'Lightning', 5: 'Lightning', 6: 'Lightning',
                          7: 'Lightning', 8: 'Lightning', 9: 'Lightning'},
            'rune_values': {0: 1.25, 1: 1.0, 2: 1.1, 3: 0.75, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}
        },
        {
            "id": 2,
            "grid_pos": [77, 37],
            'num_slots': 4, 'crown_runes': 2, 'crown_slots': [2, 4],
            'rune_type': {0: 'Oath', 1: 'Opulent', 2: 'Bond', 3: 'Power', 4: 'Lightning', 5: 'Lightning',
                          6: 'Lightning', 7: 'Lightning', 8: 'Lightning', 9: 'Lightning'},
            'rune_values': {0: 0.75, 1: 1.35, 2: 1.25, 3: 1.3, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}

        },
        {
            "id": 3,
            "grid_pos": [26, 58],
            'num_slots': 5, 'crown_runes': 2, 'crown_slots': [1, 2],
            'rune_type': {0: 'Opulent', 1: 'Bond', 2: 'Bond', 3: 'Rebirth', 4: 'Bond', 5: 'Lightning', 6: 'Lightning',
                          7: 'Lightning', 8: 'Lightning', 9: 'Lightning'},
            'rune_values': {0: 1.35, 1: 1.25, 2: 1.25, 3: 1.1, 4: 1.25, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}

        },
        {
            "id": 4,
            "grid_pos": [227, 252],
            'num_slots': 3, 'crown_runes': 1, 'crown_slots': [1],
            'rune_type': {0: 'Bond', 1: 'Oath', 2: 'Time', 3: 'Lightning', 4: 'Lightning', 5: 'Lightning',
                          6: 'Lightning', 7: 'Lightning', 8: 'Lightning', 9: 'Lightning'},
            'rune_values': {0: 1.25, 1: 0.75, 2: 1.18, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}

        },
        {
            "id": 5,
            "grid_pos": [325, 75],
            'num_slots': 7, 'crown_runes': 2, 'crown_slots': [2, 4],
            'rune_type': {0: 'Time', 1: 'Opulent', 2: 'Rebirth', 3: 'Rebirth', 4: 'Death', 5: 'Rebirth', 6: 'Death',
                          7: 'Lightning', 8: 'Lightning', 9: 'Lightning'},
            'rune_values': {0: 1.18, 1: 1.35, 2: 1.1, 3: 1.1, 4: 1.15, 5: 1.1, 6: 1.15, 7: 1, 8: 1, 9: 1}

        },
        {
            "id": 6,
            "grid_pos": [240, 310],
            'num_slots': 5, 'crown_runes': 2, 'crown_slots': [1, 2],
            'rune_type': {0: 'Opulent', 1: 'Bait', 2: 'Wisdom', 3: 'Death', 4: 'Bait', 5: 'Lightning', 6: 'Lightning',
                          7: 'Lightning', 8: 'Lightning', 9: 'Lightning'},
            'rune_values': {0: 1.35, 1: 1.0, 2: 0.95, 3: 1.15, 4: 1.0, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}

        }]
    data["monoliths"] = monoliths
    return data


def populate_monos(data):
    return [Monolith(m) for m in data["monoliths"]]


def plotLineLow(x0, y0, x1, y1):
    ret = []
    dx = x1 - x0
    dy = y1 - y0
    yi = 1
    if dy < 0:
        yi = -1
        dy = -dy
    D = (2 * dy) - dx
    y = y0
    for x in range(x0, x1):
        ret.append([x, y])
        if D > 0:
            y = y + yi
            D = D + (2 * (dy - dx))
        else:
            D = D + 2 * dy
    return ret


def plotLineHigh(x0, y0, x1, y1):
    ret = []
    dx = x1 - x0
    dy = y1 - y0
    xi = 1
    if dx < 0:
        xi = -1
        dx = -dx
    D = (2 * dx) - dy
    x = x0
    for y in range(y0, y1):
        ret.append([x, y])
        if D > 0:
            x = x + xi
            D = D + (2 * (dx - dy))
        else:
            D = D + 2 * dx
    return ret


def plotLine(x0, y0, x1, y1):
    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            return plotLineLow(x1, y1, x0, y0)
        else:
            return plotLineLow(x0, y0, x1, y1)
    else:
        if y0 > y1:
            return plotLineHigh(x1, y1, x0, y0)
        else:
            return plotLineHigh(x0, y0, x1, y1)


def calculate_h_value(row, col, dest):
    return ((row - dest[0]) ** 2 + (col - dest[1]) ** 2) ** 0.5


def is_destination(row, col, dest):
    return row == dest[0] and col == dest[1]


# 8-connected movement: (drow, dcol, cost)
_NEIGHBOURS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, 2 ** 0.5), (-1, 1, 2 ** 0.5), (1, -1, 2 ** 0.5), (1, 1, 2 ** 0.5),
]


def a_star(board, start, goal):
    if not (board.is_valid(*start) and board.is_valid(*goal)):
        print("start or goal off-board")
        return None
    if not (board.is_walkable(*start) and board.is_walkable(*goal)):
        print("start or goal is blocked")
        return None

    came_from = {}
    g_score = {start: 0.0}
    f_score = {start: calculate_h_value(*start, goal)}

    open_heap = [(f_score[start], start)]
    open_set = {start}
    closed = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        open_set.discard(current)

        if current == goal:
            return came_from

        closed.add(current)
        crow, ccol = current

        for drow, dcol, cost in _NEIGHBOURS:
            nrow, ncol = crow + drow, ccol + dcol
            neighbour = (nrow, ncol)

            if not board.is_valid(nrow, ncol):
                continue
            if not board.is_walkable(nrow, ncol) or neighbour in closed:
                continue

            tentative_g = g_score[current] + cost
            if tentative_g < g_score.get(neighbour, float('inf')):
                came_from[neighbour] = current
                g_score[neighbour] = tentative_g
                f_score[neighbour] = tentative_g + calculate_h_value(nrow, ncol, goal)
                if neighbour not in open_set:
                    heapq.heappush(open_heap, (f_score[neighbour], neighbour))
                    open_set.add(neighbour)

    return None


def reconstruct_path(came_from, current):
    total_path = deque([current])
    while current in came_from:
        current = came_from[current]
        total_path.appendleft(current)
    return total_path


def decide_order_slots(monus):
    monus.sort(key=lambda x: x.num_slots, reverse=False)
    # for a in monus:
    #    print(a.num_slots)
    return monus


def decide_order_vs(monus):
    monus.sort(key=lambda x: (x.num_slots, x.value_summed), reverse=False)
    # for a in monus:
    #    print(a.num_slots)
    return monus


def decide_order_wgts(monus):
    monus.sort(key=lambda x: x.value_summed, reverse=True)
    # for a in monus:
    #    print(a.num_slots)
    return monus


def decide_order(order_type, monus):
    match order_type:
        case 'weights':
            print('sorting by weights')
            return decide_order_wgts(monus)
        case 'slots':
            print('sorting by slots')
            return decide_order_slots(monus)
        case 'values_and_slots':
            print('sorting by values and slots')
            return decide_order_vs(monus)
        case _:
            print(f'unknown type: {order_type}')
            return monus


def add_marker(tgt_id: int, pos_x: int, pos_y: int, pos_radius: int, arr: np.ndarray, board: Board):
    x, y = (pos_x, pos_y)  # round(m.grid_pos[0]), round(m.grid_pos[1])
    r = pos_radius
    # print(m)
    h, w = arr.shape[:2]
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    x0, x1 = max(0, x - r), min(w, x + r + 1)

    if tgt_id == 1:
        color = [0, 255, 0]
    elif tgt_id == 5:
        color = [0, 0, 255]
        board[y][x].is_target = True
        # target = (y, x)
    elif tgt_id == 666:
        color = [49, 146, 168]
    elif tgt_id == 777:
        color = [49, 146, 168]
    elif tgt_id == 999:
        color = [0, 0, 0]
    else:
        color = [255, 0, 0]
    arr[y0:y1, x0:x1] = color

    pass


def main():
    num_steps = 55
    max_charges = 22
    decisions = ["slots", "values_and_slots", "weights", "x"]
    for decision in decisions:
        data = pop_data()
        board = Board(r"bg.png")
        monos = populate_monos(data)

        num_charges = max_charges
        # monos = decide_order_slots(monos)
        # monos = decide_order_vs(monos)
        # decide_order_wgts(monos)
        monos = decide_order(decision, monos)

        target = None
        add_marker(666, pos_start[1], pos_start[0], 3, board.im, board)
        for i, m in enumerate(monos, start=1):
            print(f'{i}: {m}')
            add_marker(i, m.grid_pos[0], m.grid_pos[1], 3, board.im, board)

        current_point = (pos_start[0], pos_start[1])
        color = [255, 255, 155]
        for i, m in enumerate(monos, start=1):

            target = (m.grid_pos[1], m.grid_pos[0])
            add_marker(777, current_point[1], current_point[0], 1, board.im, board)

            # color = [randint(0, 255), randint(0, 255), randint(0, 255)]
            color[0] -= (round(255 / len(monos)))
            color[1] -= (round(255 / len(monos)))
            # color[2] -= (round(255/len(monos)))
            if target is not None:
                came_from = a_star(board, current_point, target)

                if came_from is not None:
                    path = reconstruct_path(came_from, target)
                    print(f"path found, {len(path)} steps")

                    # i = 1
                    for e, (pr, pc) in enumerate(path):
                        if e % num_steps == 0:
                            if num_charges > 0:
                                add_marker(999, pc, pr, 1, board.im, board)
                                num_charges -= 1

                        board.im[pr, pc] = color

                else:
                    print("no path found")

            current_point = target

        fig, ax = plt.subplots()
        ax.imshow(board.im)
        x, y = round(pos_start[1]), round(pos_start[0])
        ax.text(x, y, f's', dict(size=10))
        w_x = 15
        w_y = board.rows - 11
        label = (
            f"{"s":>2} : "
            f"x={x:>3}, y={y:>3}  ")
        # ax.text(w_x, w_y, f's : {x}, {y}', dict(size=10))
        ax.text(w_x, w_y, label, fontsize=10, family="monospace")
        for i, m in enumerate(monos, start=1):
            # w_x += 15
            w_y -= 17
            x, y = round(m.grid_pos[0]), round(m.grid_pos[1])
            ax.text(x, y, f'{i}', dict(size=10))
            label = (
                f"{i:>2} : "
                f"x={x:>3}, y={y:>3}  "
                f"s={m.num_slots:>2}  "
                f"w={m.value_summed:>5.1f}"
            )

            ax.text(w_x, w_y, label, fontsize=10, family="monospace")
            # ax.text(w_x, w_y, f'{i} : {x}, {y} s: {m.num_slots}, w: {m.value_summed}', dict(size=10))
        w_y -= 18
        label = (
            f"charges={max_charges:>2}  "
            f"distance={num_steps:>2}"
        )
        ax.text(w_x, w_y, label, fontsize=10, family="monospace")
        ax.set_xticks(np.arange(0, board.cols, 100))
        ax.set_yticks(np.arange(0, board.rows, 100))
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        ax.xaxis.set_label_position("top")
        ax.tick_params(
            top=True, labeltop=True,
            bottom=False, labelbottom=False,
            right=False, labelright=False,
            left=True, labelleft=True,
        )
        # plt.show()
        plt.savefig(f'{os.getcwd()}\\saved_order\\{decision}.png', transparent=None, dpi=600, format=None,
                    metadata=None, bbox_inches=None, pad_inches=0.1,
                    facecolor='auto', edgecolor='auto', backend=None)


if __name__ == "__main__":
    main()
