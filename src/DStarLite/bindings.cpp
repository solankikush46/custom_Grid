#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "DStarLite.h"

namespace py = pybind11;

using XY     = std::pair<int,int>;
using XYList = std::vector<XY>;

// Match your header's array flags; we'll accept C-contiguous and forcecast (pybind11 bitmask)
using CostArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

PYBIND11_MODULE(dstar_lite, m) {
    m.doc() = "pybind11 bindings for D* Lite";

    py::class_<DStarLite>(m, "DStarLite")
        // Overload 1: single goal (pair<int,int>)
        .def(py::init<int,int,int,int,const XY&, CostArray, const XYList&>(),
             py::arg("W"), py::arg("H"),
             py::arg("start_x"), py::arg("start_y"),
             py::arg("goal_xy"),
             py::arg("cost_map"),
             py::arg("static_obs"))

        // Overload 2: multiple goals (vector<pair<int,int>>)
        .def(py::init<int,int,int,int,const XYList&, CostArray, const XYList&>(),
             py::arg("W"), py::arg("H"),
             py::arg("start_x"), py::arg("start_y"),
             py::arg("goals_xy"),
             py::arg("cost_map"),
             py::arg("static_obs"))

        // Methods — match non-const/const exactly as in your header
        .def("computeShortestPath",
             static_cast<void (DStarLite::*)()>(&DStarLite::computeShortestPath))

        .def("getShortestPath",
             static_cast<XYList (DStarLite::*)()>(&DStarLite::getShortestPath))

        .def("updateVertex",
             static_cast<void (DStarLite::*)(int,int)>(&DStarLite::updateVertex),
             py::arg("x"), py::arg("y"))

        .def("neighbors",
             static_cast<XYList (DStarLite::*)(int,int)>(&DStarLite::neighbors),
             py::arg("x"), py::arg("y"))

        .def("updateStart",
             static_cast<void (DStarLite::*)(int,int)>(&DStarLite::updateStart),
             py::arg("x"), py::arg("y"));
}
