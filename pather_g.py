import os
import numpy as np
import heapq
import matplotlib.pyplot as plt
from PIL import Image
from collections import deque
from itertools import permutations

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


def rune_mult(name):
    return rune_values.get(name, {}).get("LootMult", 1.0)


def rune_avoid(name):
    return rune_values.get(name, {}).get("Avoid", True)


class Monolith:
    def __init__(self, data):
        self.id = data["id"]
        self.grid_pos = data["grid_pos"]  # [x, y]
        self.num_slots = data["num_slots"]
        self.runes = data["runes"]  # list of rune NAMES in this stone
        self.crown_slots = data["crown_slots"]
        self.crown_runes = len(self.crown_slots)

    def local_mult(self):
        m = 1.0
        for name in self.runes:
            m *= rune_mult(name)
        return m

    @property
    def value_summed(self):
        return self.local_mult()

    def crown_contributions(self):
        out = []
        for s in self.crown_slots:
            if 0 <= s < len(self.runes):
                name = self.runes[s]
                if not rune_avoid(name):
                    out.append((name, rune_mult(name)))
        return out

    def crown_value(self):
        m = 1.0
        for _, mult in self.crown_contributions():
            m *= mult
        return m

    def waves(self):
        return max(1, self.num_slots - 1)

    def __str__(self) -> str:
        return f"Monolith {self.id} at {self.grid_pos} slots {self.num_slots}"

    __repr__ = __str__


class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.walkable = False
        self.is_target = False

    def __str__(self) -> str:
        return f"Cell at ({self.x}, {self.y})"

    __repr__ = __str__


class Board:
    def __init__(self, imgname):
        pil = Image.open(imgname).convert("RGB")
        self.im = np.array(pil)  # HxWx3, uint8
        self.rows, self.cols = self.im.shape[:2]  # H, W
        self.walk = np.all(self.im == walkable_color, axis=-1)
        self.cells = []
        self.populate_cells()

    def populate_cells(self):
        for r in range(self.rows):
            row = [Cell(c, r) for c in range(self.cols)]
            for c in range(self.cols):
                row[c].walkable = bool(self.walk[r, c])
            self.cells.append(row)

    def __getitem__(self, index):
        return self.cells[index]

    def is_valid(self, row, col):
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_walkable(self, row, col):
        return bool(self.walk[row, col])


def pop_data():
    data = {}
    monoliths = [
        {"id": 1, "grid_pos": [150, 133], "num_slots": 4, "crown_slots": [0],
         "runes": ["Bond", "Bait", "Rebirth", "Oath"]},
        {"id": 2, "grid_pos": [77, 37], "num_slots": 4, "crown_slots": [2],
         "runes": ["Oath", "Opulent", "Bond", "Power"]},
        {"id": 3, "grid_pos": [26, 58], "num_slots": 5, "crown_slots": [1, 2],
         "runes": ["Opulent", "Bond", "Bond", "Rebirth", "Bond"]},
        {"id": 4, "grid_pos": [227, 252], "num_slots": 3, "crown_slots": [1],
         "runes": ["Bond", "Oath", "Time"]},
        {"id": 5, "grid_pos": [325, 75], "num_slots": 7, "crown_slots": [2, 4],
         "runes": ["Time", "Opulent", "Rebirth", "Rebirth", "Death", "Rebirth", "Death"]},
        {"id": 6, "grid_pos": [240, 310], "num_slots": 5, "crown_slots": [1, 2],
         "runes": ["Opulent", "Bait", "Wisdom", "Death", "Bait"]},
    ]
    data["monoliths"] = monoliths
    return data


def populate_monos(data):
    return [Monolith(m) for m in data["monoliths"]]


def calculate_h_value(row, col, dest):
    return ((row - dest[0]) ** 2 + (col - dest[1]) ** 2) ** 0.5


# 8-connected movement: (drow, dcol, cost)
_NEIGHBOURS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, 2 ** 0.5), (-1, 1, 2 ** 0.5), (1, -1, 2 ** 0.5), (1, 1, 2 ** 0.5),
]


def a_star(board, start, goal, return_cost=False):
    if not (board.is_valid(*start) and board.is_valid(*goal)):
        return (None, float('inf')) if return_cost else None
    if not (board.is_walkable(*start) and board.is_walkable(*goal)):
        return (None, float('inf')) if return_cost else None

    came_from = {}
    g_score = {start: 0.0}
    open_heap = [(calculate_h_value(*start, goal), start)]
    open_set = {start}
    closed = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        open_set.discard(current)

        if current == goal:
            return (came_from, g_score[goal]) if return_cost else came_from

        closed.add(current)
        crow, ccol = current

        for drow, dcol, cost in _NEIGHBOURS:
            nrow, ncol = crow + drow, ccol + dcol
            neighbour = (nrow, ncol)
            if not board.is_valid(nrow, ncol):
                continue

            if drow != 0 and dcol != 0:
                if not (board.is_walkable(crow, ncol) and board.is_walkable(nrow, ccol)):
                    continue
            if not board.is_walkable(nrow, ncol) or neighbour in closed:
                continue

            tentative_g = g_score[current] + cost
            if tentative_g < g_score.get(neighbour, float('inf')):
                came_from[neighbour] = current
                g_score[neighbour] = tentative_g
                f = tentative_g + calculate_h_value(nrow, ncol, goal)
                if neighbour not in open_set:
                    heapq.heappush(open_heap, (f, neighbour))
                    open_set.add(neighbour)

    return (None, float('inf')) if return_cost else None


def reconstruct_path(came_from, current):
    total_path = deque([current])
    while current in came_from:
        current = came_from[current]
        total_path.appendleft(current)
    return total_path


def node_points(monos):
    pts = [tuple(pos_start)]  # already (row, col)
    for m in monos:
        pts.append((m.grid_pos[1], m.grid_pos[0]))  # x,y -> row,col
    return pts


def build_distance_matrix(board, monos):
    pts = node_points(monos)
    n = len(pts)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            _, cost = a_star(board, pts[i], pts[j], return_cost=True)
            dist[i][j] = dist[j][i] = cost
    return dist, pts


def order_indices(order, monos):
    """map an ordered list of monoliths to their column indices (1-based, 0=start).

    Keyed on monolith id() so it stays correct even after monos is reordered
    in place; the distance matrix columns follow the ORIGINAL monos order used
    to build it, which is what `base_monos` must preserve."""
    lookup = {id(m): k + 1 for k, m in enumerate(monos)}
    return [lookup[id(m)] for m in order]


def route_charges(order, monos, dist, num_steps):
    idxs = [0] + order_indices(order, monos)
    total = sum(dist[idxs[k]][idxs[k + 1]] for k in range(len(idxs) - 1))
    if total == float('inf'):
        return float('inf')
    return total / num_steps



def score_chain(order):
    """Walk the chain start->end. Each monolith's payout is its own loot
    multiplier times all DISTINCT crown runes already proliferating, times its
    wave count. It then adds its own (non-Avoid) crown runes to the stack for
    everything downstream. The endpoint collects the full multiplied stack.

    Stacking is MULTIPLICATIVE, matching the LootMult data shape. Same rune
    names do not stack. Avoid runes are filtered in crown_contributions, so an
    Oath in a crown slot boosts nothing downstream."""
    active = {}  # rune_name -> LootMult
    total = 0.0
    for m in order:
        boost = 1.0
        for mult in active.values():
            boost *= mult
        total += m.local_mult() * m.waves() * boost
        for name, mult in m.crown_contributions():
            if name not in active:  # 'same runes do not stack'
                active[name] = mult
    return total


def pick_endpoint(monos):
    return max(monos, key=lambda m: (m.num_slots, m.value_summed))


def best_order(monos, dist, num_steps, max_charges, base_monos=None, fix_endpoint=False):
    """Brute-force permutations (n small). Prefer the best-scoring chain that
    fits the charge budget; if none fits, return the best-scoring chain overall
    (budget a soft target, tie-broken by fewer charges).

    base_monos: the list the distance matrix columns were built from; charges
    are measured against it. Defaults to `monos`.
    fix_endpoint: if True, force the highest-slot stone last (old behaviour);
    default False lets the search choose the endpoint that actually scores best."""
    base = base_monos if base_monos is not None else monos

    if fix_endpoint:
        endpoint = pick_endpoint(monos)
        rest = [m for m in monos if m is not endpoint]
        candidates = (list(p) + [endpoint] for p in permutations(rest))
    else:
        candidates = (list(p) for p in permutations(monos))

    best_in_budget, best_in_budget_val = None, float('-inf')
    best_any, best_any_key = None, None  # key = (value, -charges)

    for order in candidates:
        val = score_chain(order)
        ch = route_charges(order, base, dist, num_steps)
        if ch <= max_charges and val > best_in_budget_val:
            best_in_budget, best_in_budget_val = order, val
        key = (val, -ch)
        if best_any_key is None or key > best_any_key:
            best_any, best_any_key = order, key

    if best_in_budget is not None:
        return best_in_budget
    print(f"  (no order fits {max_charges} charges; best-value order needs "
          f"{route_charges(best_any, base, dist, num_steps):.1f})")
    return best_any


def decide_order_slots(monus):
    monus.sort(key=lambda x: x.num_slots)
    return monus


def decide_order_vs(monus):
    monus.sort(key=lambda x: (x.num_slots, x.value_summed))
    return monus


def decide_order_wgts(monus):
    # ascending so highest value lands LAST (proliferation collector)
    monus.sort(key=lambda x: x.value_summed)
    return monus


def decide_order_crown(monus):
    # crown value ascending -> strongest proliferators earlier is handled by
    # score, so here we just surface crown strength as a simple baseline
    monus.sort(key=lambda x: x.crown_value())
    return monus


def decide_order(order_type, monus, dist=None, num_steps=55, max_charges=22,
                 base_monos=None):
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
        case 'crown':
            print('sorting by crown value')
            return decide_order_crown(monus)
        case 'optimal':
            print('brute-force proliferation-aware optimum')
            return best_order(monus, dist, num_steps, max_charges, base_monos=base_monos)
        case _:
            print(f'unknown type: {order_type}')
            return monus


def add_marker(tgt_id, pos_x, pos_y, pos_radius, arr, board):
    x, y = pos_x, pos_y
    r = pos_radius
    h, w = arr.shape[:2]
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    x0, x1 = max(0, x - r), min(w, x + r + 1)

    if tgt_id == 1:
        color = [0, 255, 0]
    elif tgt_id == 5:
        color = [0, 0, 255]
        board[y][x].is_target = True
    elif tgt_id in (666, 777):
        color = [49, 146, 168]
    elif tgt_id == 999:
        color = [0, 0, 0]
    else:
        color = [255, 0, 0]
    arr[y0:y1, x0:x1] = color


def main():
    num_steps = 55
    max_charges = 22
    decisions = ["slots", "values_and_slots", "weights", "crown", "optimal"]

    for decision in decisions:
        data = pop_data()
        board = Board(r"bg.png")

        # distance matrix once per board, keyed to base_monos' column order
        base_monos = populate_monos(data)  # stable reference for the matrix
        dist, _ = build_distance_matrix(board, base_monos)

        num_charges = max_charges
        # sort a copy so base_monos (matrix columns) stays put
        monos = decide_order(decision, list(base_monos), dist=dist,
                             num_steps=num_steps, max_charges=max_charges,
                             base_monos=base_monos)

        # report the chain's modelled reward + charge cost (charges vs base_monos)
        chain_val = score_chain(monos)
        chain_ch = route_charges(monos, base_monos, dist, num_steps)
        flag = "" if chain_ch <= max_charges else "  OVER"
        print(f"[{decision}] score={chain_val:.2f}  charges={chain_ch:.1f}/{max_charges}{flag}")

        add_marker(666, pos_start[1], pos_start[0], 3, board.im, board)
        for i, m in enumerate(monos, start=1):
            print(f'{i}: {m}')
            add_marker(i, m.grid_pos[0], m.grid_pos[1], 3, board.im, board)

        current_point = (pos_start[0], pos_start[1])
        color = [255, 255, 155]
        for i, m in enumerate(monos, start=1):
            target = (m.grid_pos[1], m.grid_pos[0])
            add_marker(777, current_point[1], current_point[0], 1, board.im, board)

            color[0] -= round(255 / len(monos))
            color[1] -= round(255 / len(monos))

            came_from = a_star(board, current_point, target)
            if came_from is not None:
                path = reconstruct_path(came_from, target)
                print(f"path found, {len(path)} steps")
                for e, (pr, pc) in enumerate(path):
                    if e % num_steps == 0 and num_charges > 0:
                        add_marker(999, pc, pr, 1, board.im, board)
                        num_charges -= 1
                    board.im[pr, pc] = color
            else:
                print("no path found")

            current_point = target

        fig, ax = plt.subplots()
        ax.imshow(board.im)
        x, y = round(pos_start[1]), round(pos_start[0])
        ax.text(x, y, 's', dict(size=10))
        w_x = 15
        w_y = board.rows - 11
        ax.text(w_x, w_y, f'{"s":>2} : x={x:>3}, y={y:>3}  ',
                fontsize=10, family="monospace")
        for i, m in enumerate(monos, start=1):
            w_y -= 17
            x, y = round(m.grid_pos[0]), round(m.grid_pos[1])
            ax.text(x, y, f'{i}', dict(size=10))
            label = (f"{i:>2} : x={x:>3}, y={y:>3}  "
                     f"s={m.num_slots:>2}  w={m.value_summed:>5.1f}  "
                     f"cr={m.crown_value():>4.1f}")
            ax.text(w_x, w_y, label, fontsize=10, family="monospace")
        w_y -= 18
        ax.text(w_x, w_y,
                f"charges={max_charges:>2}  distance={num_steps:>2}  "
                f"score={chain_val:>7.1f}",
                fontsize=10, family="monospace")
        ax.set_xticks(np.arange(0, board.cols, 100))
        ax.set_yticks(np.arange(0, board.rows, 100))
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        ax.xaxis.set_label_position("top")
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False,
                       right=False, labelright=False, left=True, labelleft=True)

        os.makedirs(os.path.join(os.getcwd(), "saved_order"), exist_ok=True)
        plt.savefig(os.path.join(os.getcwd(), "saved_order", f"{decision}.png"),
                    dpi=600, bbox_inches=None, pad_inches=0.1,
                    facecolor='auto', edgecolor='auto')
        plt.close(fig)


if __name__ == "__main__":
    main()
