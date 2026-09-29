import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import deque
import heapq

walkable_color = np.array([89,89,89])

class Monolith:
    def __init__(self, data):
        self.id = data["id"]
        self.grid_pos = data["grid_pos"]   # [x, y]
        self.num_slots = data["num_slots"]

    def __str__(self) -> str:
        return f"Monolith {self.id} at {self.grid_pos} slots {self.num_slots}"


class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.f_score = float('inf')
        self.g_score = float('inf')
        self.h_score = float('inf')
        self.parent_i = None   # parent row (y)
        self.parent_j = None   # parent col (x)
        self.walkable = True
        self.is_target = False

    def __str__(self) -> str:
        return f"Cell at ({self.x}, {self.y})"

    __repr__ = __str__


class Board:
    def __init__(self, cols, rows):
        self.cols = cols          # width  (x)
        self.rows = rows          # height (y)
        self.cells = []
        for r in range(self.rows):            # r = row = y
            self.cells.append([])
            for c in range(self.cols):        # c = col = x
                self.cells[r].append(Cell(c, r))   # Cell(x=col, y=row)

    def __getitem__(self, index):
        return self.cells[index]              # board[y] -> row, board[y][x] -> Cell

    def is_valid(self, row, col):
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_walkable(self, row, col):
        return self.cells[row][col].walkable


def pop_data():
    data = {}
    #                                    x,   y
    monoliths = [{"id": 1, "grid_pos": [150, 133], "num_slots": 3 },
                 {"id": 2, "grid_pos": [77,  37],  "num_slots": 4 },
                 {"id": 3, "grid_pos": [26,  58],  "num_slots": 2 },
                 {"id": 4, "grid_pos": [227, 252], "num_slots": 7 },
                 {"id": 5, "grid_pos": [325, 75],  "num_slots": 5 },
                 {"id": 6, "grid_pos": [240, 310], "num_slots": 7 }]
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
    (-1,  0, 1.0), (1,  0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
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

def decide_order(monuss):
    monuss.sort(key=lambda x: x["num_slots"], reverse=True)
    for a in monuss:
        print(monuss[a]["num_slots"])
    return monuss


def main():

    data = pop_data()
    pil = Image.open(r"bg.png").convert("RGB")
    im = np.array(pil)                      # HxWx3, uint8, 0-255
    h, w = im.shape[:2]
    print(f'{h} x {w}')

    board = Board(w, h)
    monos = populate_monos(data)

    for i in range(h):
        for j in range(w):

            if np.array_equal(im[i][j], walkable_color):
                board[i][j].walkable = True

            else:
                board[i][j].walkable = False

    target = None
    for i, m in enumerate(monos, start=1):
        x, y = round(m.grid_pos[0]), round(m.grid_pos[1])
        r = 3
        print(m)

        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)

        if i == 1:
            color = [0, 255, 0]
        elif i == 5:
            color = [0, 0, 255]
            board[y][x].is_target = True
            target = (y, x)
        else:
            color = [255, 0, 0]

        im[y0:y1, x0:x1] = color


    if target is not None:
        start_x, start_y = monos[0].grid_pos
        start = (start_y, start_x)
        came_from = a_star(board, start, target)
        if came_from:
            path = reconstruct_path(came_from, target)
            print(f"path found, {len(path)} steps")

            for (pr, pc) in path:
                im[pr, pc] = [255, 255, 0]
        else:
            print("no path found")


    fig, ax = plt.subplots()
    ax.imshow(im)
    for m in monos:
        x, y = round(m.grid_pos[0]), round(m.grid_pos[1])
        ax.text(x, y, f'{x}, {y}', dict(size=15))

    ax.set_xticks(np.arange(0, w, 100))
    ax.set_yticks(np.arange(0, h, 100))
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    plt.show()


if __name__ == "__main__":
    main()