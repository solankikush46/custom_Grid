// DStarLite.cpp

#include "DStarLite.h"
#include <stdexcept>
#include <algorithm>

double DStarLite::INF = std::numeric_limits<double>::infinity();

/////////////////////////////////////////
// Public Wrappers for (x,y) interface //
/////////////////////////////////////////

void DStarLite::updateVertex(int x, int y) {
    int idx = y * width + x;
    updateVertex(idx);  // call private version
}

std::vector<std::pair<int,int>> DStarLite::neighbors(int x, int y) {
    int idx = y * width + x;
    auto nbrs_idx = succ(idx);
    std::vector<std::pair<int,int>> out;
    out.reserve(nbrs_idx.size());
    for (int n : nbrs_idx) {
        out.emplace_back(n % width, n / width);
    }
    return out;
}

/////////////////////////////////////////
//      Constructor implementations    //
/////////////////////////////////////////

DStarLite::DStarLite(int width_, int height_, int start_x_,
                     int start_y_, const std::vector<std::pair<int,int>>& goals,
                     pybind11::array_t<double> cost_map,
                     const std::vector<std::pair<int,int>>& known_obstacles)
    : width(width_), height(height_),
      start_x(start_x_), start_y(start_y_), km(0.0)
{
    auto buf = cost_map.request();
    if (buf.ndim != 2 || buf.shape[0] != height || buf.shape[1] != width)
        throw std::runtime_error("Cost map dimensions mismatch");

    cost_array = cost_map;
    cost_data  = static_cast<const double*>(buf.ptr);

    obstacle_grid.assign(width * height, false);
    for (auto const &cell : known_obstacles) {
        int ox = cell.first, oy = cell.second;
        if (inBounds(ox, oy))
            obstacle_grid[oy * width + ox] = true;
    }

    g.assign(width * height, INF);
    rhs.assign(width * height, INF);

    goal_indices.clear();
    for (auto const &gp : goals) {
        int gx = gp.first, gy = gp.second;
        if (!inBounds(gx, gy))
            throw std::runtime_error("Goal out of bounds");
        goal_indices.insert(gy * width + gx);
    }

    U_key.assign(width * height, Key(INF, INF));
    while (!U.empty()) U.pop();
    for (int gi : goal_indices) {
        rhs[gi] = 0.0;
        pushQueue(gi, calculateKey(gi));
    }
}

DStarLite::DStarLite(int width_, int height_, int start_x_,
                     int start_y_, const std::pair<int,int>& goal,
                     pybind11::array_t<double> cost_map,
                     const std::vector<std::pair<int,int>>& known_obstacles)
    : DStarLite(width_, height_, start_x_, start_y_,
                std::vector<std::pair<int,int>>{goal},
                cost_map, known_obstacles)
{}

/////////////////////////////////////////
//   Core D* Lite algorithm methods    //
/////////////////////////////////////////

void DStarLite::computeShortestPath() {
    int start_idx = start_y * width + start_x;
    while (true) {
        Key top = topKey();
        Key sk  = calculateKey(start_idx);
        bool need = (top.first < sk.first) ||
                    (std::fabs(top.first - sk.first) < 1e-9 && top.second < sk.second);
        if (!need && rhs[start_idx] == g[start_idx]) break;

        int u = popQueue();
        if (u < 0) break;

        if (g[u] > rhs[u]) {
            g[u] = rhs[u];
            for (int p : pred(u)) updateVertex(p);
        } else {
            double oldg = g[u];
            g[u] = INF;
            auto preds = pred(u);
            preds.push_back(u);
            for (int p : preds) {
                if (rhs[p] == cost(p, u) + oldg) {
                    double m = INF;
                    for (int s : succ(p))
                        m = std::min(m, cost(p, s) + g[s]);
                    if (!goal_indices.count(p)) rhs[p] = m;
                }
                updateVertex(p);
            }
        }
    }
}

std::vector<std::pair<int,int>> DStarLite::getShortestPath() {
    std::vector<std::pair<int,int>> path;
    int start_idx = start_y * width + start_x;
    if (g[start_idx] == INF) return path;

    int cur = start_idx;
    path.emplace_back(start_x, start_y);
    while (!goal_indices.count(cur)) {
        auto nbrs = succ(cur);
        if (nbrs.empty()) return {};
        double best = INF;
        int besti = -1;
        for (int n : nbrs) {
            double v = cost(cur, n) + g[n];
            if (v < best) { best = v; besti = n; }
        }
        if (besti < 0) return {};
        int bx = besti % width, by = besti / width;
        if (bx == path.back().first && by == path.back().second)
            return {};  // cycle detected
        path.emplace_back(bx, by);
        cur = besti;
    }
    return path;
}

/////////////////////////////////////////
//        Dynamic updates             //
/////////////////////////////////////////

void DStarLite::setObstacle(int x, int y) {
    if (!inBounds(x, y)) return;
    int idx = y * width + x;
    if (!obstacle_grid[idx]) {
        obstacle_grid[idx] = true;
        auto affected = pred(idx);
        affected.push_back(idx);
        for (int p : affected) updateVertex(p);
    }
}

void DStarLite::updateStart(int new_start_x, int new_start_y) {
    if (new_start_x == start_x && new_start_y == start_y) return;
    double d = heuristic(start_x, start_y, new_start_x, new_start_y);
    km += d;
    start_x = new_start_x;
    start_y = new_start_y;
}

// ─────────────────────────────────────────────────────────────────
// Private flat-index updateVertex definition that was missing
// ─────────────────────────────────────────────────────────────────
void DStarLite::updateVertex(int index) {
    if (!goal_indices.count(index)) {
        auto nbrs = succ(index);
        double best = INF;
        for (int n : nbrs) {
            double c = cost(index, n) + g[n];
            best = std::min(best, c);
        }
        rhs[index] = best;
    }
    // remove old queue entry
    U_key[index] = Key(INF, INF);
    // re-insert if inconsistent
    if (g[index] != rhs[index]) {
        pushQueue(index, calculateKey(index));
    }
}

/////////////////////////////////////////
//        Helper implementations       //
/////////////////////////////////////////

bool DStarLite::ComparePQ::operator()(PQItem const &a, PQItem const &b) const {
    if (a.key.first > b.key.first + 1e-9) return true;
    if (a.key.first < b.key.first - 1e-9) return false;
    if (a.key.second > b.key.second + 1e-9) return true;
    if (a.key.second < b.key.second - 1e-9) return false;
    return a.index > b.index;
}

DStarLite::Key DStarLite::calculateKey(int idx) const {
    double gr = std::min(g[idx], rhs[idx]);
    int x = idx % width, y = idx / width;
    double h = heuristic(start_x, start_y, x, y);
    return Key(gr + h + km, gr);
}

void DStarLite::pushQueue(int idx, Key k) {
    U.push({k, idx});
    U_key[idx] = k;
}

int DStarLite::popQueue() {
    while (!U.empty()) {
        auto it = U.top();
        if (U_key[it.index] != it.key) { U.pop(); continue; }
        U.pop();
        U_key[it.index] = Key(INF, INF);
        return it.index;
    }
    return -1;
}

DStarLite::Key DStarLite::topKey() {
    while (!U.empty()) {
        auto it = U.top();
        if (U_key[it.index] != it.key) { U.pop(); continue; }
        return it.key;
    }
    return Key(INF, INF);
}

bool DStarLite::inBounds(int x, int y) const {
    return x >= 0 && x < width && y >= 0 && y < height;
}

std::vector<int> DStarLite::neighbors(int idx) const {
    std::vector<int> out; out.reserve(8);
    int x = idx % width, y = idx / width;
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            if (!dx && !dy) continue;
            int nx = x + dx, ny = y + dy;
            if (!inBounds(nx, ny)) continue;
            int ni = ny * width + nx;
            if (!obstacle_grid[ni]) out.push_back(ni);
        }
    }
    return out;
}

std::vector<int> DStarLite::pred(int idx) const { return neighbors(idx); }
std::vector<int> DStarLite::succ(int idx) const { return neighbors(idx); }

double DStarLite::cost(int u, int v) const {
    if (u < 0 || u >= width*height || v < 0 || v >= width*height) return INF;
    if (obstacle_grid[u] || obstacle_grid[v]) return INF;
    int ux = u % width, uy = u / width;
    int vx = v % width, vy = v / width;
    double base = (ux != vx && uy != vy) ? SQRT2 : 1.0;
    return base + cost_data[v];
}

double DStarLite::heuristic(int x1, int y1, int x2, int y2) const {
    double dx = std::abs(x1 - x2), dy = std::abs(y1 - y2);
    return dx > dy ? dx : dy;
}

/////////////////////////////////////////
//    Pybind11 module registration     //
/////////////////////////////////////////

PYBIND11_MODULE(DStarLite, m) {
    pybind11::class_<DStarLite>(m, "DStarLite")
        .def(pybind11::init<
             int,int,int,int,
             const std::vector<std::pair<int,int>>&,
             pybind11::array_t<double>,
             const std::vector<std::pair<int,int>>&>(),
             pybind11::arg("width"),
             pybind11::arg("height"),
             pybind11::arg("start_x"),
             pybind11::arg("start_y"),
             pybind11::arg("goals"),
             pybind11::arg("cost_map"),
             pybind11::arg("known_obstacles") = std::vector<std::pair<int,int>>())
        .def("computeShortestPath", &DStarLite::computeShortestPath)
        .def("setObstacle",          &DStarLite::setObstacle)
        .def("updateStart",          &DStarLite::updateStart)
        .def("getShortestPath",      &DStarLite::getShortestPath)
        .def("updateVertex",
             static_cast<void (DStarLite::*)(int,int)>(&DStarLite::updateVertex),
             pybind11::arg("x"), pybind11::arg("y"))
        .def("neighbors",
             static_cast<std::vector<std::pair<int,int>> (DStarLite::*)(int,int)>(&DStarLite::neighbors),
             pybind11::arg("x"), pybind11::arg("y"));
}
