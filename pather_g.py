import io
import os
import shutil
import struct
import zlib

import numpy as np
import heapq
import json
import matplotlib.pyplot as plt
from PIL import Image
from collections import deque
from itertools import permutations

walkable_color = np.array([89,89,89])

default_pos_start = [210, 120]

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
default_monos = [
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


def _encode_png_ga(mask, gray: int = 96, bg: int = 255) -> bytes:
    print("encoding png")
    rows, width = mask.shape
    ga = np.zeros((rows, width, 2), dtype=np.uint8)
    ga[..., 0] = np.where(mask, np.uint8(gray), np.uint8(bg))  # grey channel (walkable / background)
    ga[..., 1] = 255                                           # alpha channel (fully opaque)
    # Build the zlib stream: each scanline prefixed with filter byte 0 (none).
    stream = bytearray()
    flat = ga.reshape(rows, width * 2)
    for y in range(rows):
        stream.append(0)
        stream.extend(flat[y].tobytes())

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    sig = b"\x89PNG\r\n\x1a\n"
    #print(sig.hex())
    ihdr = struct.pack(">IIBBBBB", width, rows, 8, 4, 0, 0, 0)  # colortype 4 = GA
    idat = zlib.compress(bytes(stream), 6)
    print(f"png size: {len(idat)}")
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")

def load_npy(path):
    print(f"loading npy {path}")
    cells = np.load(file=path)
    print("cells loaded")
    if cells.ndim != 2:
        raise ValueError(f"{os.path.basename(path)} is {cells.ndim}D; expected a 2D grid")
    cells = cells.astype(np.uint8, copy=False)
    print("cells converted")
    rows, width = cells.shape
    bpr = width // 2  # 2 cells per byte (width = bpr*2)
    # Re-pack to the game's nibble layout: even x = low nibble, odd x = high.
    packed = np.zeros((rows, bpr), dtype=np.uint8)
    lo = cells[:, 0:bpr * 2:2]  # even columns
    hi = cells[:, 1:bpr * 2:2]  # odd columns
    packed |= (lo & 0x0F)
    packed |= (hi & 0x0F) << 4
    png = _encode_png_ga(cells != 0, gray=int(walkable_color[0]))

    # print(entry["png"])
    return png

class Board:
    def __init__(self, imgname):
        print("loading map")
        print(imgname)
        pil = Image.open(io.BytesIO(load_npy(imgname))).convert("RGB")

        self.im = np.array(pil)  # HxWx3, uint8
        self.rows, self.cols = self.im.shape[:2]  # H, W
        self.walk = np.all(self.im == walkable_color, axis=-1)
        self.cells = []
        self.pos_start = [0,0]
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


def pop_data(use_default = True, data_file=''):

    data = {}
    if use_default:
        data["monoliths"] = default_monos
        return data
    else:
        with open(data_file, 'r') as f:
            return json.load(f)



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


def node_points(monos, pos_start):
    pts = [tuple(pos_start)]  # already (row, col)
    for m in monos:
        pts.append((m.grid_pos[1], m.grid_pos[0]))  # x,y -> row,col
    return pts


def build_distance_matrix(board, monos):
    pts = node_points(monos, board.pos_start)
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


# brute force is only affordable for small n; above this we use a heuristic
_EXACT_MAX = 9


def _order_key(order, base, dist, num_steps, max_charges):
    """Total order over candidate chains, higher is better.

    (feasible?, value, -charges): a chain within the charge budget always beats
    one that is not; ties break on higher score, then on fewer charges. This
    matches the original preference (best-scoring in-budget chain, else the
    best-scoring chain overall)."""
    val = score_chain(order)
    ch = route_charges(order, base, dist, num_steps)
    return (1 if ch <= max_charges else 0, val, -ch)


def _local_search(order, base, dist, num_steps, max_charges, fixed_tail=0):
    """Improve an ordering with 2-opt segment reversals and single-node moves.
    The last `fixed_tail` elements are pinned (used to keep a fixed endpoint)."""
    order = list(order)
    best_key = _order_key(order, base, dist, num_steps, max_charges)
    n = len(order) - fixed_tail
    improved = True
    while improved:
        improved = False
        # 2-opt: reverse every sub-segment within the movable prefix
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                k = _order_key(cand, base, dist, num_steps, max_charges)
                if k > best_key:
                    order, best_key, improved = cand, k, True
        # relocation: move a single stone to another slot
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                cand = order[:i] + order[i + 1:]
                cand.insert(j, order[i])
                k = _order_key(cand, base, dist, num_steps, max_charges)
                if k > best_key:
                    order, best_key, improved = cand, k, True
    return order, best_key


def _greedy_order(monos, base, dist, num_steps, max_charges):
    """Build a chain by repeatedly appending the stone that most improves the
    running key. A proliferation-aware seed for local search."""
    remaining = list(monos)
    order = []
    while remaining:
        best_m = max(remaining, key=lambda m: _order_key(
            order + [m], base, dist, num_steps, max_charges))
        order.append(best_m)
        remaining.remove(best_m)
    return order


def best_order(monos, dist, num_steps, max_charges, base_monos=None, fix_endpoint=False):
    """Find a high-value monolith chain that fits the charge budget.

    Small inputs (n <= _EXACT_MAX) are solved exactly by brute force. Larger
    inputs use greedy seeds refined by 2-opt / relocation local search, which
    finishes in well under a second even for the full board (14!, the exact
    search, would never return).

    Prefer the best-scoring chain within the charge budget; if none fits, return
    the best-scoring chain overall (budget a soft target, tie-broken by fewer
    charges). base_monos is the list the distance matrix columns were built from;
    charges are measured against it. fix_endpoint pins the highest-slot stone
    last (old behaviour); default lets the search choose the endpoint."""
    base = base_monos if base_monos is not None else monos

    if fix_endpoint:
        endpoint = pick_endpoint(monos)
        movable = [m for m in monos if m is not endpoint]
    else:
        endpoint, movable = None, list(monos)

    def finalize(order):
        return order + [endpoint] if endpoint is not None else order

    keyf = lambda order: _order_key(order, base, dist, num_steps, max_charges)

    if len(movable) <= _EXACT_MAX:
        best = max((finalize(list(p)) for p in permutations(movable)), key=keyf)
    else:
        print(f"  ({len(monos)} stones: heuristic search instead of {len(movable)}! brute force)")
        # seed with a few sensible orderings, refine each, keep the best
        seeds = [
            list(movable),
            sorted(movable, key=lambda m: m.crown_value()),
            sorted(movable, key=lambda m: (m.num_slots, m.value_summed)),
            sorted(movable, key=lambda m: m.value_summed),
            _greedy_order(movable, base, dist, num_steps, max_charges),
        ]
        fixed_tail = 1 if endpoint is not None else 0
        best, best_key = None, None
        for seed in seeds:
            refined, key = _local_search(
                finalize(seed), base, dist, num_steps, max_charges, fixed_tail)
            if best_key is None or key > best_key:
                best, best_key = refined, key

    ch = route_charges(best, base, dist, num_steps)
    if ch > max_charges:
        print(f"  (no order fits {max_charges} charges; best-value order needs {ch:.1f})")
    return best


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
            print('optimal')
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


def main(default_data=False, data_dir=''):
    decisions = ["slots", "values_and_slots", "weights", "crown", "optimal"]

    # Loaded data has priority: whenever data_dir holds the required files we use
    # them, even if default_data was requested. Fall back to the built-in
    # defaults only when there is nothing to load.
    have_loaded = bool(data_dir) and \
        os.path.isfile(os.path.join(data_dir, "monoliths.json")) and \
        os.path.isfile(os.path.join(data_dir, "walkable.npy"))

    if have_loaded:
        use_default = False
    elif default_data:
        use_default = True
    else:
        print(f"no loaded data found in {data_dir!r}; falling back to default data")
        use_default = True

    if use_default:
        map_file = 'bg.png'
        data_file = ''
    else:
        data_file = os.path.join(data_dir, "monoliths.json")
        map_file = os.path.join(data_dir, "walkable.npy")

    results = []  # one entry per decision, for the end-of-run summary
    for decision in decisions:
        print(f"--- {decision} ---")
        data = pop_data(use_default, data_file=data_file)
        board = Board(map_file)
        board.pos_start = (data["detonator"]["grid_pos"][1], data["detonator"]["grid_pos"][0])
        num_steps = data["charges_dist"]
        max_charges = data["detonator"]["total_charges"]
        expl_radius = data["explo_radius"]
        #pos_start =
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
        fits = chain_ch <= max_charges
        flag = "" if fits else "  OVER"
        # efficiency = reward per charge spent (higher is better)
        efficiency = chain_val / chain_ch if chain_ch not in (0, float('inf')) else 0.0
        print(f"[{decision}] score={chain_val:.2f}  charges={chain_ch:.1f}/{max_charges}"
              f"  eff={efficiency:.2f}{flag}")
        results.append({
            "decision": decision, "score": chain_val, "charges": chain_ch,
            "fits": fits, "efficiency": efficiency,
        })

        add_marker(666, board.pos_start[1], board.pos_start[0], 3, board.im, board)
        for i, m in enumerate(monos, start=1):
            add_marker(i, m.grid_pos[0], m.grid_pos[1], 3, board.im, board)

        current_point = (board.pos_start[0], board.pos_start[1])
        color = [255, 255, 155]
        for i, m in enumerate(monos, start=1):
            target = (m.grid_pos[1], m.grid_pos[0])
            add_marker(777, current_point[1], current_point[0], 1, board.im, board)

            color[0] -= round(255 / len(monos))
            color[1] -= round(255 / len(monos))

            came_from = a_star(board, current_point, target)
            if came_from is not None:
                path = reconstruct_path(came_from, target)
                #print(f"path found, {len(path)} steps")
                for e, (pr, pc) in enumerate(path):
                    if e % num_steps == 0 and num_charges > 0:
                        add_marker(999, pc, pr, 1, board.im, board)
                        num_charges -= 1
                    board.im[pr, pc] = color
            else:
                print("no path found")

            current_point = target

        # image on the left, monolith list in its own panel on the right
        fig, (ax, ax_txt) = plt.subplots(
            1, 2, figsize=(12, 9),
            gridspec_kw={"width_ratios": [3, 1]})
        ax.imshow(board.im)
        x, y = round(board.pos_start[1]), round(board.pos_start[0])
        ax.text(x, y, 's', dict(size=7, color="red"))

        # build the label list; each monolith also gets its index drawn on the map
        lines = [f'{"s":>2} : x={x:>3}, y={y:>3}']
        for i, m in enumerate(monos, start=1):
            mx, my = round(m.grid_pos[0]), round(m.grid_pos[1])
            ax.text(mx, my, f'{i}', dict(size=7, color="red"))
            lines.append(f"{i:>2} : x={mx:>3}, y={my:>3}  "
                         f"s={m.num_slots:>2}  w={m.value_summed:>5.1f}  "
                         f"cr={m.crown_value():>4.1f}")
        lines.append("")
        lines.append(f"charges={max_charges:>2}  distance={num_steps:>2}  "
                     f"score={chain_val:>7.1f}")

        # render the whole list as one monospace text block so lines can't overlap
        ax_txt.axis("off")
        ax_txt.text(0.0, 1.0, "\n".join(lines), va="top", ha="left",
                    family="monospace", fontsize=9, transform=ax_txt.transAxes)

        ax.set_xticks(np.arange(0, board.cols, 200))
        ax.set_yticks(np.arange(0, board.rows, 100))
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        ax.xaxis.set_label_position("top")
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False,
                       right=False, labelright=False, left=True, labelleft=True)
        fig.tight_layout()

        os.makedirs(os.path.join(os.getcwd(), "saved_order"), exist_ok=True)
        plt.savefig(os.path.join(os.getcwd(), "saved_order", f"{decision}.png"),
                    dpi=600, bbox_inches=None, pad_inches=0.1,
                    facecolor='auto', edgecolor='auto')
        plt.close(fig)

    _summarize_and_pick(results)


def _summarize_and_pick(results):
    """Print a summary of every decision and pick the most efficient one.

    Efficiency is score per charge (reward per detonation charge). Orders that
    fit the charge budget are preferred; among those (or, if none fit, among all)
    the highest efficiency wins. The winning plot is copied to
    saved_order/BEST_<decision>.png."""
    if not results:
        return

    # most efficient, preferring budget-feasible orders
    best = max(results, key=lambda r: (r["fits"], r["efficiency"]))

    print("\n=== results (sorted by efficiency) ===")
    print(f"{'decision':<18}{'score':>9}{'charges':>10}{'eff':>8}  budget")
    for r in sorted(results, key=lambda r: r["efficiency"], reverse=True):
        mark = "  <-- BEST" if r is best else ""
        budget = "ok" if r["fits"] else "OVER"
        print(f"{r['decision']:<18}{r['score']:>9.2f}{r['charges']:>10.1f}"
              f"{r['efficiency']:>8.2f}  {budget}{mark}")

    note = "" if best["fits"] else "  (none fit the charge budget; picked best efficiency)"
    print(f"\nMost efficient: {best['decision']}  "
          f"(score={best['score']:.2f}, charges={best['charges']:.1f}, "
          f"eff={best['efficiency']:.2f}){note}")

    src = os.path.join(os.getcwd(), "saved_order", f"{best['decision']}.png")
    dst = os.path.join(os.getcwd(), "saved_order", f"BEST_{best['decision']}.png")
    if os.path.isfile(src):
        shutil.copyfile(src, dst)
        print(f"Saved winning plot to {dst}")


if __name__ == "__main__":
    main()
