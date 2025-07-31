// DStarLite.cpp

#include "DStarLite.h"

double DStarLite::INF = std::numeric_limits<double>::infinity();

DStarLite::DStarLite(int width_, int height_, int start_x_, int start_y_,
                     const std::vector<std::pair<int,int>>& goals,
                     pybind11::array_t<double> cost_map,
                     const std::vector<std::pair<int,int>>& known_obstacles)
    : width(width_), height(height_), start_x(start_x_), start_y(start_y_), km(0.0) 
{
    // Validate that the cost_map dimensions match the grid
    auto buf = cost_map.request();
    if (buf.ndim != 2 || buf.shape[0] != height || buf.shape[1] != width) {
        throw std::runtime_error("Cost map dimensions do not match given width and height");
    }
    // Store cost map reference and pointer to its data
    cost_array = cost_map;
    cost_data = static_cast<const double*>(buf.ptr);

    // Initialize obstacle grid
    obstacle_grid.assign(width * height, false);
    for (const auto& cell : known_obstacles) {
        int ox = cell.first;
        int oy = cell.second;
        if (inBounds(ox, oy)) {
            obstacle_grid[oy * width + ox] = true;
        }
        // Out-of-bounds obstacle coordinates are ignored
    }

    // Initialize g and rhs for all cells
    g.assign(width * height, INF);
    rhs.assign(width * height, INF);

    // Process goal positions
    goal_indices.clear();
    for (const auto& goal : goals) {
        int gx = goal.first;
        int gy = goal.second;
        if (!inBounds(gx, gy)) {
            throw std::runtime_error("Goal coordinates out of grid bounds");
        }
        int goal_idx = gy * width + gx;
        goal_indices.insert(goal_idx);
    }

    // Set initial conditions for goals: rhs = 0 and push into open list
    U_key.assign(width * height, Key(INF, INF));
    while (!U.empty()) U.pop();  // ensure queue is empty
    for (int goal_idx : goal_indices) {
        rhs[goal_idx] = 0.0;
        Key k = calculateKey(goal_idx);
        pushQueue(goal_idx, k);
    }
}

DStarLite::Key DStarLite::calculateKey(int index) const {
    // Calculate priority key for a given state index
    double g_rhs = (g[index] < rhs[index] ? g[index] : rhs[index]);
    int sx = start_x, sy = start_y;
    int x = index % width, y = index / width;
    double h = heuristic(sx, sy, x, y);
    return Key(g_rhs + h + km, g_rhs);
}

void DStarLite::pushQueue(int index, Key key) {
    // Insert state into the priority queue with given key
    U.push({ key, index });
    U_key[index] = key;
}

int DStarLite::popQueue() {
    // Remove and return the state with the smallest key (skip stale entries)
    while (!U.empty()) {
        PQItem item = U.top();
        if (U_key[item.index] != item.key) {
            // Outdated entry (state has a new key), ignore it
            U.pop();
            continue;
        }
        // Valid top entry found – remove it from queue and mark as no longer in queue
        U.pop();
        U_key[item.index] = Key(INF, INF);
        return item.index;
    }
    return -1;  // queue is empty
}

DStarLite::Key DStarLite::topKey() {
    // Peek at the smallest key in the priority queue (after removing any stale entries)
    while (!U.empty()) {
        PQItem item = U.top();
        if (U_key[item.index] != item.key) {
            // Remove stale entries at the top
            U.pop();
            continue;
        }
        return item.key;
    }
    return Key(INF, INF);
}

std::vector<int> DStarLite::neighbors(int index) const {
    // Generate all valid 8-way neighbors (not blocked by obstacles)
    std::vector<int> result;
    result.reserve(8);
    int x = index % width;
    int y = index / width;
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            if (dx == 0 && dy == 0) continue;
            int nx = x + dx;
            int ny = y + dy;
            if (!inBounds(nx, ny)) continue;
            int neighbor_idx = ny * width + nx;
            if (obstacle_grid[neighbor_idx]) continue;
            result.push_back(neighbor_idx);
        }
    }
    return result;
}

double DStarLite::cost(int u_index, int v_index) const {
    // Compute travel cost from cell u to cell v (returns INF if impassable)
    if (u_index < 0 || u_index >= width * height || v_index < 0 || v_index >= width * height) {
        return INF;
    }
    if (obstacle_grid[u_index] || obstacle_grid[v_index]) {
        return INF;
    }
    // Straight moves cost 1.0, diagonal moves cost SQRT2
    int ux = u_index % width, uy = u_index / width;
    int vx = v_index % width, vy = v_index / width;
    double base_cost = (ux != vx && uy != vy) ? SQRT2 : 1.0;
    // Add terrain cost for entering the neighbor cell v
    double terrain_cost = cost_data[v_index];
    return base_cost + terrain_cost;
}

void DStarLite::updateVertex(int index) {
    if (goal_indices.find(index) == goal_indices.end()) {
        // For non-goal states, update rhs value based on successors
        std::vector<int> nbrs = succ(index);
        double min_rhs = INF;
        for (int nb : nbrs) {
            double c = cost(index, nb);
            double new_val = c + g[nb];
            if (new_val < min_rhs) {
                min_rhs = new_val;
            }
        }
        rhs[index] = min_rhs;
    }
    // Remove any existing queue entry for this state (mark as stale)
    if (U_key[index].first != INF || U_key[index].second != INF) {
        U_key[index] = Key(INF, INF);
    }
    // If the state is inconsistent (g != rhs), push it into the queue with updated key
    if (g[index] != rhs[index]) {
        Key k = calculateKey(index);
        pushQueue(index, k);
    }
}

void DStarLite::computeShortestPath() {
    // Process states until the start state is locally consistent or no better path exists
    int start_idx = start_y * width + start_x;
    while (true) {
        Key top = topKey();
        Key start_key = calculateKey(start_idx);
        // If the smallest key is not less than start's key and start is consistent, we are done
        bool betterExists = (top.first < start_key.first) ||
                            (std::fabs(top.first - start_key.first) < 1e-9 && top.second < start_key.second);
        if (!betterExists && rhs[start_idx] == g[start_idx]) {
            break;
        }
        // Otherwise, pop the state u with the smallest key
        int u = popQueue();
        if (u == -1) break;  // queue empty, no path found
        if (g[u] > rhs[u]) {
            // State u has improved (or is a goal with rhs set to 0)
            g[u] = rhs[u];
            for (int pred_idx : pred(u)) {
                updateVertex(pred_idx);
            }
        } else {
            // State u has gotten worse (an edge cost increased or became obstacle)
            double old_g = g[u];
            g[u] = INF;
            // Update u and all its predecessors
            std::vector<int> all_preds = pred(u);
            all_preds.push_back(u);
            for (int p : all_preds) {
                if (rhs[p] == cost(p, u) + old_g) {
                    // If this neighbor was depending on u for its best path, recalc its rhs
                    if (goal_indices.find(p) == goal_indices.end()) {
                        double min_rhs = INF;
                        for (int nb : succ(p)) {
                            double c = cost(p, nb);
                            double new_val = c + g[nb];
                            if (new_val < min_rhs) {
                                min_rhs = new_val;
                            }
                        }
                        rhs[p] = min_rhs;
                    }
                }
                updateVertex(p);
            }
        }
    }
}

void DStarLite::setObstacle(int x, int y) {
    if (!inBounds(x, y)) return;
    int idx = y * width + x;
    if (!obstacle_grid[idx]) {
        // Add a new obstacle and update affected cells
        obstacle_grid[idx] = true;
        std::vector<int> affected = pred(idx);
        affected.push_back(idx);
        for (int p : affected) {
            updateVertex(p);
        }
    }
}

void DStarLite::updateStart(int new_start_x, int new_start_y) {
    if (new_start_x == start_x && new_start_y == start_y) return;
    // Increase the key modifier by the heuristic distance the start moved
    double dist = heuristic(start_x, start_y, new_start_x, new_start_y);
    km += dist;
    // Update start coordinates
    start_x = new_start_x;
    start_y = new_start_y;
}

std::vector<std::pair<int,int>> DStarLite::getShortestPath() {
    std::vector<std::pair<int,int>> path;
    int start_idx = start_y * width + start_x;
    if (g[start_idx] == INF) {
        // No known path to any goal
        return path;
    }
    // Reconstruct the path from start to a goal by following the lowest-cost successor
    int current_idx = start_idx;
    path.emplace_back(start_x, start_y);
    while (goal_indices.find(current_idx) == goal_indices.end()) {
        std::vector<int> nbrs = succ(current_idx);
        if (nbrs.empty()) {
            // Dead end (should not happen if g[start] is finite)
            path.clear();
            return path;
        }
        double best_cost = INF;
        int best_idx = -1;
        for (int nb : nbrs) {
            double val = cost(current_idx, nb) + g[nb];
            if (val < best_cost) {
                best_cost = val;
                best_idx = nb;
            }
        }
        if (best_idx == -1) {
            // No path found (should not happen if algorithm is correct)
            path.clear();
            return path;
        }
        // Append the chosen neighbor to the path
        int bx = best_idx % width;
        int by = best_idx / width;
        if (!path.empty() && bx == path.back().first && by == path.back().second) {
            // Detected a cycle (unexpected in positive-cost graphs)
            path.clear();
            return path;
        }
        path.emplace_back(bx, by);
        current_idx = best_idx;
    }
    return path;
}

// Pybind11 module definition for Python integration
PYBIND11_MODULE(dstar_planner, m) {
    pybind11::class_<DStarLite>(m, "DStarLite")
        .def(pybind11::init<int,int,int,int,
                            const std::vector<std::pair<int,int>>&,
                            pybind11::array_t<double>,
                            const std::vector<std::pair<int,int>>&>(),
             pybind11::arg("width"), pybind11::arg("height"),
             pybind11::arg("start_x"), pybind11::arg("start_y"),
             pybind11::arg("goals"),
             pybind11::arg("cost_map"),
             pybind11::arg("known_obstacles") = std::vector<std::pair<int,int>>())
        .def("computeShortestPath", &DStarLite::computeShortestPath)
        .def("setObstacle", &DStarLite::setObstacle)
        .def("updateStart", &DStarLite::updateStart)
        .def("getShortestPath", &DStarLite::getShortestPath);
}
