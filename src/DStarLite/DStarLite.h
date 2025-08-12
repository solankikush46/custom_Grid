// DStarLite.h

#ifndef DSTARBOT_LITE_H
#define DSTARBOT_LITE_H

#include <vector>
#include <utility>
#include <queue>
#include <unordered_set>
#include <limits>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

class DStarLite {
public:
    // Primary constructors
    DStarLite(int width,
              int height,
              int start_x,
              int start_y,
              const std::vector<std::pair<int,int>>& goals,
              pybind11::array_t<double> cost_map,
              const std::vector<std::pair<int,int>>& known_obstacles = {});

    DStarLite(int width,
              int height,
              int start_x,
              int start_y,
              const std::pair<int,int>& goal,
              pybind11::array_t<double> cost_map,
              const std::vector<std::pair<int,int>>& known_obstacles = {});

    // Core D* Lite API
    void computeShortestPath();
    void setObstacle(int x, int y);
    void updateStart(int new_start_x, int new_start_y);
    std::vector<std::pair<int,int>> getShortestPath();

    // Python‐friendly overloads
    void updateVertex(int x, int y);
    std::vector<std::pair<int,int>> neighbors(int x, int y);

private:
    int width, height;
    int start_x, start_y;

    std::unordered_set<int> goal_indices;

    pybind11::array_t<double> cost_array;
    const double* cost_data;

    std::vector<bool> obstacle_grid;

    std::vector<double> g;
    std::vector<double> rhs;

    typedef std::pair<double,double> Key;
    struct PQItem { Key key; int index; };
    struct ComparePQ { bool operator()(PQItem const &a, PQItem const &b) const; };

    std::priority_queue<PQItem, std::vector<PQItem>, ComparePQ> U;
    std::vector<Key> U_key;
    double km;

    static double INF;
    static constexpr double SQRT2 = 1.4142135623730951;

    // Internal helpers
    Key calculateKey(int idx) const;
    void pushQueue(int idx, Key k);
    int  popQueue();
    Key  topKey();
    bool inBounds(int x, int y) const;
    std::vector<int> neighbors(int idx) const;
    double cost(int u, int v) const;
    std::vector<int> pred(int idx) const;
    std::vector<int> succ(int idx) const;
    void updateVertex(int idx);
    double heuristic(int x1, int y1, int x2, int y2) const;
};

#endif // DSTARBOT_LITE_H
