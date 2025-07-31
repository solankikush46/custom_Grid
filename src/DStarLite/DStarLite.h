// DStarLite.h

#ifndef DSTARBOT_LITE_H
#define DSTARBOT_LITE_H

#include <vector>
#include <utility>
#include <queue>
#include <unordered_map>
#include <limits>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

class DStarLite {
public:
    // Constructor for multiple goals
    DStarLite(int width, int height, int start_x, int start_y,
              const std::vector<std::pair<int,int>>& goals,
              pybind11::array_t<double> cost_map,
              const std::vector<std::pair<int,int>>& known_obstacles = {});

    // Constructor for a single goal (delegates to the multiple-goal constructor)
    DStarLite(int width, int height, int start_x, int start_y,
              const std::pair<int,int>& goal,
              pybind11::array_t<double> cost_map,
              const std::vector<std::pair<int,int>>& known_obstacles = {})
        : DStarLite(width, height, start_x, start_y,
                    std::vector<std::pair<int,int>>{goal},
                    cost_map, known_obstacles) {}

    // Compute or repair the shortest path tree until the start is consistent
    void computeShortestPath();

    // Mark a cell as a new obstacle and update affected states
    void setObstacle(int x, int y);

    // Update the start position after the agent moves (adjusts the heuristic modifier)
    void updateStart(int new_start_x, int new_start_y);

    // Reconstruct and return the current shortest path from start to the nearest goal
    std::vector<std::pair<int,int>> getShortestPath();

private:
    // Grid parameters
    int width, height;
    int start_x, start_y;
    std::unordered_set<int> goal_indices;       // set of goal cell indices

    // Cost grid data (reference to a NumPy array from Python)
    pybind11::array_t<double> cost_array;       // keep Python array alive
    const double* cost_data;                    // pointer to cost grid values

    // Obstacles grid (true for obstacle cells)
    std::vector<bool> obstacle_grid;

    // Heuristic function (Chebyshev distance for 8-directional movement)
    inline double heuristic(int x1, int y1, int x2, int y2) const {
        double dx = std::abs(x1 - x2);
        double dy = std::abs(y1 - y2);
        return dx > dy ? dx : dy;
    }

    // Cost-to-come values and one-step lookahead values for each cell
    std::vector<double> g;
    std::vector<double> rhs;

    // Priority queue (min-heap) for states to update and an array for their current keys
    typedef std::pair<double,double> Key;
    struct PQItem {
        Key key;
        int index;
    };
    struct ComparePQ {
        bool operator()(const PQItem& a, const PQItem& b) const {
            // Compare two queue items lexicographically by key (min-heap)
            if (a.key.first > b.key.first + 1e-9) return true;
            if (a.key.first < b.key.first - 1e-9) return false;
            if (a.key.second > b.key.second + 1e-9) return true;
            if (a.key.second < b.key.second - 1e-9) return false;
            // If keys are effectively equal, tie-break by index for determinism
            return a.index > b.index;
        }
    };
    std::priority_queue<PQItem, std::vector<PQItem>, ComparePQ> U;
    std::vector<Key> U_key;  // latest key for each state (INF,INF if not in queue)
    double km;               // key modifier (increased when the start moves)

    // Helper methods for internal use
    Key calculateKey(int index) const;
    void pushQueue(int index, Key key);
    int popQueue();
    Key topKey();
    bool inBounds(int x, int y) const { return x >= 0 && x < width && y >= 0 && y < height; }
    std::vector<int> neighbors(int index) const;
    double cost(int u_index, int v_index) const;
    inline std::vector<int> pred(int index) const { return neighbors(index); }
    inline std::vector<int> succ(int index) const { return neighbors(index); }
    void updateVertex(int index);

    // Constants for infinity and diagonal movement cost
    static double INF;
    static constexpr double SQRT2 = 1.4142135623730951;
};

#endif // DSTARBOT_LITE_H
