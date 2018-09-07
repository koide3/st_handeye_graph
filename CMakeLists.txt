cmake_minimum_required(VERSION 2.8.3)
project(st_handeye)

add_compile_options(-std=c++11)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(G2O REQUIRED)
find_package(VISP REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED filesystem program_options)

include_directories(
	include
        ${G2O_INCLUDE_DIRS}
        ${VISP_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
	${Boost_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
)

link_directories(${G2O_LIBRARY_DIRS})

###########
## Build ##
###########

add_library(st_handeye_graph
  src/st_handeye_graph.cpp
)
target_link_libraries(st_handeye_graph
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${G2O_TYPES_DATA}
  ${G2O_CORE_LIBRARY}
  ${G2O_STUFF_LIBRARY}
  ${G2O_SOLVER_PCG}
  ${G2O_SOLVER_CSPARSE}
  ${G2O_SOLVER_CHOLMOD}
  ${G2O_TYPES_SLAM3D}
  ${G2O_TYPES_SLAM3D_ADDONS}
)

add_library(st_handeye_graph_xray
  src/st_handeye_graph_xray.cpp
)
target_link_libraries(st_handeye_graph_xray
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${G2O_TYPES_DATA}
  ${G2O_CORE_LIBRARY}
  ${G2O_STUFF_LIBRARY}
  ${G2O_SOLVER_PCG}
  ${G2O_SOLVER_CSPARSE}
  ${G2O_SOLVER_CHOLMOD}
  ${G2O_TYPES_SLAM3D}
  ${G2O_TYPES_SLAM3D_ADDONS}
)

add_library(st_handeye_visp
  src/st_handeye_visp.cpp
)
target_link_libraries(st_handeye_visp
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${VISP_LIBRARIES}
)

# handeye_simulation
add_executable(handeye_simulation
  src/handeye_simulation.cpp
)
target_link_libraries(handeye_simulation
  st_handeye_graph
  st_handeye_visp
)
add_dependencies(handeye_simulation
  st_handeye_graph
  st_handeye_visp
)

# handeye_simulation_xray
add_executable(handeye_simulation_xray
  src/handeye_simulation_xray.cpp
)
target_link_libraries(handeye_simulation_xray
    st_handeye_graph_xray
)
add_dependencies(handeye_simulation_xray
  st_handeye_graph_xray
)

# calibrate
add_executable(calibrate
  src/calibrate.cpp
)
target_link_libraries(calibrate
    st_handeye_graph
    st_handeye_visp
)
add_dependencies(calibrate
  st_handeye_graph
  st_handeye_visp
)
