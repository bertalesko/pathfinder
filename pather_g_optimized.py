import os
import shutil
import math

import numpy as np
import heapq
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
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


class Board:
    """Direct NumPy-backed board; no PNG conversion or Cell-object grid."""
    def __init__(self, imgname):
        cells = np.load(file=imgname)
        if cells.ndim != 2:
            raise ValueError(f"{os.path.basename(imgname)} is {cells.ndim}D; expected a 2D grid")
        self.walk = cells.astype(bool, copy=False)
        self.rows, self.cols = self.walk.shape
        self.pos_start = (0, 0)
        self._background = np.full((self.rows, self.cols, 3), 255, dtype=np.uint8)
        self._background[self.walk] = walkable_color

    def image_copy(self):
        return self._background.copy()

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
    closed = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
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
                # Heapq has no decrease-key operation.  Push the improved
                # priority and let the closed-set check discard stale entries.
                heapq.heappush(open_heap, (f, neighbour))

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


def build_distance_and_path_cache(board, monos):
    """Run each unordered point pair through A* once, retaining costs and paths."""
    pts = node_points(monos, board.pos_start)
    n = len(pts)
    dist = [[0.0] * n for _ in range(n)]
    paths = {}
    for i in range(n):
        for j in range(i + 1, n):
            came_from, cost = a_star(board, pts[i], pts[j], return_cost=True)
            dist[i][j] = dist[j][i] = cost
            paths[(i, j)] = None if came_from is None else list(reconstruct_path(came_from, pts[j]))
    return dist, paths, pts


def cached_path(paths, start_index, end_index):
    if start_index == end_index:
        return []
    if start_index < end_index:
        return paths[(start_index, end_index)]
    path = paths[(end_index, start_index)]
    return None if path is None else list(reversed(path))


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


def add_marker(tgt_id, pos_x, pos_y, pos_radius, arr):
    x, y, r = pos_x, pos_y, pos_radius
    h, w = arr.shape[:2]
    y0, y1 = max(0, y-r), min(h, y+r+1)
    x0, x1 = max(0, x-r), min(w, x+r+1)
    if tgt_id == 1:
        color = [0, 255, 0]
    elif tgt_id == 5:
        color = [0, 0, 255]
    elif tgt_id in (666, 777):
        color = [49, 146, 168]
    elif tgt_id == 999:
        color = [0, 0, 0]
    elif tgt_id == 888:
        color = [255, 165, 0]
    elif tgt_id == 555:
        color = [255, 0, 0]
    elif tgt_id == 444:
        color = [0, 0, 0]
    else:
        color = [255, 0, 0]
    arr[y0:y1, x0:x1] = color


def _step_cost(a, b):
    return math.sqrt(2) if a[0] != b[0] and a[1] != b[1] else 1.0


def build_charge_candidates(board, monos, explosion_radius):
    """Find meaningful, walkable charge sites and what each blast detonates.

    A site is considered only when it lies in at least one remnant's blast
    disk.  Sites with the same hit set and nearest-first detonation order are
    equivalent for scoring, so one central representative is retained.  This
    keeps the route graph small; it is a practical route-cost approximation,
    not an exhaustive optimization over every pixel in a blast disk.
    """
    radius = int(math.ceil(explosion_radius))
    radius_sq = explosion_radius * explosion_radius
    seen = set()
    groups = {}

    for monolith in monos:
        cx, cy = monolith.grid_pos
        for row in range(max(0, cy - radius), min(board.rows, cy + radius + 1)):
            for col in range(max(0, cx - radius), min(board.cols, cx + radius + 1)):
                point = (row, col)
                if point in seen or not board.is_walkable(row, col):
                    continue
                seen.add(point)
                hits = [
                    index for index, other in enumerate(monos)
                    if (col - other.grid_pos[0]) ** 2 + (row - other.grid_pos[1]) ** 2 <= radius_sq
                ]
                if not hits:
                    continue
                order = tuple(sorted(
                    hits,
                    key=lambda index: (
                        (col - monos[index].grid_pos[0]) ** 2 +
                        (row - monos[index].grid_pos[1]) ** 2,
                        monos[index].id,
                    ),
                ))
                mask = sum(1 << index for index in hits)
                groups.setdefault((mask, order), []).append(point)

    candidates = []
    for (mask, order), points in groups.items():
        hit_centres = [
            (monos[index].grid_pos[1], monos[index].grid_pos[0])
            for index in order
        ]
        # A centre-of-hit-set representative is a stable, useful placement for
        # both single and overlapping blast areas.
        point = min(
            points,
            key=lambda candidate: sum(
                (candidate[0] - centre[0]) ** 2 + (candidate[1] - centre[1]) ** 2
                for centre in hit_centres
            ),
        )
        candidates.append({"point": point, "sites": set(points), "mask": mask, "order": order})
    return candidates


def build_charge_route_cache(board, candidates):
    """Cache shortest walkable paths between the start and blast sites."""
    points = [board.pos_start] + [candidate["point"] for candidate in candidates]
    count = len(points)
    distances = [[0.0] * count for _ in range(count)]
    paths = {}
    for i in range(count):
        for j in range(i + 1, count):
            came_from, cost = a_star(board, points[i], points[j], return_cost=True)
            distances[i][j] = distances[j][i] = cost
            paths[(i, j)] = None if came_from is None else list(reconstruct_path(came_from, points[j]))
    return distances, paths


def advance_charge_sites(plan, candidates, route_paths):
    """Move each blast forward to the edge of its valid blast area.

    The optimiser uses a central representative to keep its candidate graph
    small.  Once it has chosen a blast sequence, follow the next route and use
    the last point that still has the identical hit set and explosion order.
    This keeps the same detonation result while avoiding needless placement on
    top of a remnant.
    """
    action_sites = [candidates[index]["point"] for index in plan["actions"]]
    for action_number, candidate_index in enumerate(plan["actions"][:-1]):
        next_candidate_index = plan["actions"][action_number + 1]
        path = cached_path(route_paths, candidate_index + 1, next_candidate_index + 1)
        if path is None:
            continue
        valid_sites = candidates[candidate_index]["sites"]
        for point in reversed(path):
            if point in valid_sites:
                action_sites[action_number] = point
                break
    return action_sites


def build_selected_route(board, action_sites, charge_distance):
    """Build exact paths and integer charge usage for chosen blast sites."""
    paths = []
    charges = 0
    current = board.pos_start
    for target in action_sites:
        came_from, cost = a_star(board, current, target, return_cost=True)
        if came_from is None:
            return None, math.inf
        paths.append(list(reconstruct_path(came_from, target)))
        charges += _edge_charges(cost, charge_distance)
        current = target
    return paths, charges


def _edge_charges(distance, charge_distance):
    """Charges needed to reach and place the next explosive site."""
    if math.isinf(distance):
        return math.inf
    return max(1, math.ceil((distance - 1e-9) / charge_distance))  #it's distance - 0.0000000001, to fix the alignment


def score_charge(candidate, detonated_mask, active_runes, monos):
    """Score one blast in its nearest-remnant-first explosion order."""
    active = set(active_runes)
    gained = 0.0
    hit_order = []
    for index in candidate["order"]:
        bit = 1 << index
        if detonated_mask & bit:
            continue
        boost = math.prod(rune_mult(name) for name in active)
        monolith = monos[index]
        gained += monolith.local_mult() * monolith.waves() * boost
        for name, _ in monolith.crown_contributions():
            active.add(name)
        hit_order.append(index)
    return gained, tuple(hit_order), frozenset(active)


def best_charge_plan(monos, candidates, distances, charge_distance, max_charges):
    """Find the highest-value feasible sequence of blast placements.

    The search can stop after any blast, so it naturally chooses a profitable
    subset of remnants.  A Pareto frontier per route/scoring state avoids
    retaining plans that are both lower-value and more expensive.
    """
    best = {"score": 0.0, "charges": 0, "actions": (), "hit_orders": ()}
    frontier = {}
    stack = [(0, 0, frozenset(), 0, 0.0, (), ())]

    while stack:
        last_node, detonated, active, spent, score, actions, hit_orders = stack.pop()
        state_key = (last_node, detonated, active)
        state_frontier = frontier.setdefault(state_key, [])
        if any(old_score >= score and old_spent <= spent for old_score, old_spent in state_frontier):
            continue
        state_frontier[:] = [
            (old_score, old_spent) for old_score, old_spent in state_frontier
            if not (score >= old_score and spent <= old_spent)
        ]
        state_frontier.append((score, spent))

        if (score, -spent, -len(actions)) > (best["score"], -best["charges"], -len(best["actions"])):
            best = {"score": score, "charges": spent, "actions": actions, "hit_orders": hit_orders}

        for candidate_index, candidate in enumerate(candidates):
            new_mask = candidate["mask"] & ~detonated
            if not new_mask:
                continue
            edge = _edge_charges(distances[last_node][candidate_index + 1], charge_distance)
            next_spent = spent + edge
            if next_spent > max_charges:
                continue
            gained, hit_order, next_active = score_charge(candidate, detonated, active, monos)
            stack.append((
                candidate_index + 1,
                detonated | new_mask,
                next_active,
                next_spent,
                score + gained,
                actions + (candidate_index,),
                hit_orders + (hit_order,),
            ))
    return best


def draw_charge_segment(image, path, charge_distance):
    """Draw evenly spaced orange charges and return their grid positions."""
    if not path:
        return []
    distances = [0.0]
    for previous, current in zip(path, path[1:]):
        distances.append(distances[-1] + _step_cost(previous, current))

    total_distance = distances[-1]
    charge_count = _edge_charges(total_distance, charge_distance)
    charge_points = []
    path_index = 0
    for charge_number in range(1, charge_count + 1):
        target_distance = total_distance * charge_number / charge_count
        while path_index < len(path) - 1 and distances[path_index] < target_distance:
            path_index += 1
        # The final charge must be the selected blast site.  Earlier markers
        # are snapped to the closer path cell for a visually even chain.
        if charge_number == charge_count:
            point = path[-1]
        else:
            previous_index = max(0, path_index - 1)
            if abs(distances[previous_index] - target_distance) <= abs(distances[path_index] - target_distance):
                point = path[previous_index]
            else:
                point = path[path_index]
        add_marker(888, point[1], point[0], 1, image)
        charge_points.append(point)
    return charge_points


def main(default_data=False, data_dir=''):


    if not default_data:
        data_file = os.path.join(data_dir, "monoliths.json")
        map_file = os.path.join(data_dir, "walkable.npy")
    else:
        print("should use default data")
        return
    missing = [path for path in (data_file, map_file) if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError("Selected data folder is missing: " + ", ".join(os.path.basename(path) for path in missing))

    data = pop_data(False, data_file=data_file)
    board = Board(map_file)
    board.pos_start = (data["detonator"]["grid_pos"][1], data["detonator"]["grid_pos"][0])
    charge_distance = data["charges_dist"]
    max_charges = data["detonator"]["total_charges"]
    explosion_radius = data["explo_radius"]
    monos = populate_monos(data)

    candidates = build_charge_candidates(board, monos, explosion_radius)
    if not candidates:
        raise ValueError("No walkable blast locations were found for the loaded remnants.")
    distances, route_paths = build_charge_route_cache(board, candidates)
    plan = best_charge_plan(monos, candidates, distances, charge_distance, max_charges)
    action_sites = advance_charge_sites(plan, candidates, route_paths)
    selected_paths, selected_charges = build_selected_route(board, action_sites, charge_distance)
    if selected_charges > max_charges:
        # Keep the planner's central sites if advancing a blast would require
        # an additional bridge charge through local terrain.
        action_sites = [candidates[index]["point"] for index in plan["actions"]]
        selected_paths, selected_charges = build_selected_route(board, action_sites, charge_distance)
    if selected_paths is None:
        raise RuntimeError("Planner selected an unreachable blast location.")
    plan["charges"] = selected_charges

    image = board.image_copy()
    add_marker(666, board.pos_start[1], board.pos_start[0], 3, image)
    execution_order = {}
    for hit_order in plan["hit_orders"]:
        for monolith_index in hit_order:
            execution_order[monolith_index] = len(execution_order) + 1

    placed_charges = []
    for action_number, path in enumerate(selected_paths, start=1):
        color = [255, max(0, 255 - action_number * 20), 155]
        for row, col in path:
            row0, row1 = max(0, row - 1), min(board.rows, row + 2)
            col0, col1 = max(0, col - 1), min(board.cols, col + 2)
            image[row0:row1, col0:col1] = color
        placed_charges.extend(draw_charge_segment(image, path, charge_distance))

    # Plot remnants after the path, then put charge dots on top if a blast
    # location overlaps a remnant.
    for monolith_index, monolith in enumerate(monos):
        marker_type = 555 if monolith_index in execution_order else 444
        add_marker(marker_type, monolith.grid_pos[0], monolith.grid_pos[1], 3, image)
    placed_charges = list(dict.fromkeys(placed_charges))
    for row, col in placed_charges:
        add_marker(888, col, row, 1, image)

    fig, (ax, ax_txt) = plt.subplots(1, 2, figsize=(12, 9),
                                     gridspec_kw={"width_ratios": [3, 1]})
    ax.imshow(image)
    for row, col in placed_charges:
        ax.add_patch(Circle((col, row), explosion_radius, fill=False,
                            edgecolor="orange", linewidth=0.8, alpha=0.8))
    start_x, start_y = board.pos_start[1], board.pos_start[0]
    ax.text(start_x, start_y, "s", dict(size=7, color="red"))
    lines = [f"start : x={start_x:>4}, y={start_y:>4}"]
    exploded_details = []
    for action_number, (site, hit_order) in enumerate(zip(action_sites, plan["hit_orders"]), start=1):
        row, col = site
        hit_ids = ", ".join(str(monos[index].id) for index in hit_order)
        lines.append(f"charge {action_number:>2}: x={col:>4}, y={row:>4}  hits [{hit_ids}]")
        for monolith_index in hit_order:
            exploded_details.append((execution_order[monolith_index], action_number, monolith_index))
    lines.extend(("", f"charges={plan['charges']:>2}/{max_charges:>2}  "
                   f"radius={explosion_radius:>4.0f}  score={plan['score']:>7.1f}"))
    lines.append("")
    lines.append("exploded remnants:")
    for order, action_number, monolith_index in sorted(exploded_details):
        monolith = monos[monolith_index]
        x, y = monolith.grid_pos
        lines.append(f"{order:>2}: id={monolith.id:>2}  c={action_number:>2}  "
                     f"x={x:>4}, y={y:>4}  s={monolith.num_slots:>2}  "
                     f"w={monolith.value_summed:>5.1f}  cr={monolith.crown_value():>4.1f}")
    for monolith_index, monolith in enumerate(monos):
        if monolith_index in execution_order:
            label, colour = str(execution_order[monolith_index]), "red"
        else:
            label, colour = str(monolith.id), "black"
        ax.text(monolith.grid_pos[0], monolith.grid_pos[1], label, dict(size=7, color=colour))

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

    output_dir = os.path.join(os.getcwd(), "saved_order")
    os.makedirs(output_dir, exist_ok=True)
    output = os.path.join(output_dir, "optimal.png")
    plt.savefig(output, dpi=600, bbox_inches=None, pad_inches=0.1,
                facecolor='auto', edgecolor='auto')
    plt.close(fig)

    result = {
        "decision": "optimal", "score": plan["score"], "charges": plan["charges"],
        "fits": plan["charges"] <= max_charges,
        "efficiency": plan["score"] / plan["charges"] if plan["charges"] else 0.0,
    }
    print(f"[optimal] score={plan['score']:.2f}  charges={plan['charges']}/{max_charges}  "
          f"eff={result['efficiency']:.2f}")
    _summarize_and_pick([result])


def _summarize_and_pick(results):
    if not results:
        return

    # The expedition objective is feasibility, then total reward; efficiency
    # is informational and fewer charges only break equal-score ties.
    best = max(results, key=lambda r: (r["fits"], r["score"], -r["charges"]))

    print("\n=== results (sorted by score) ===")
    print(f"{'decision':<18}{'score':>9}{'charges':>10}{'eff':>8}  budget")
    for r in sorted(results, key=lambda r: (r["fits"], r["score"], -r["charges"]), reverse=True):
        mark = "  <-- BEST" if r is best else ""
        budget = "ok" if r["fits"] else "OVER"
        print(f"{r['decision']:<18}{r['score']:>9.2f}{r['charges']:>10.1f}"
              f"{r['efficiency']:>8.2f}  {budget}{mark}")

    note = "" if best["fits"] else "  (no feasible route)"
    print(f"\nBest feasible route: {best['decision']}  "
          f"(score={best['score']:.2f}, charges={best['charges']:.1f}, "
          f"eff={best['efficiency']:.2f}){note}")

    src = os.path.join(os.getcwd(), "saved_order", f"{best['decision']}.png")
    dst = os.path.join(os.getcwd(), "saved_order", f"BEST_{best['decision']}.png")
    if os.path.isfile(src):
        shutil.copyfile(src, dst)
        print(f"Saved winning plot to {dst}")


if __name__ == "__main__":
    main(data_dir=os.path.join(os.path.dirname(__file__), "data"))


