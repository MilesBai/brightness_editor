cmake_minimum_required(VERSION 3.12)

project(whiten LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED) # Find OpenCV package

include_directories(${OpenCV_INCLUDE_DIRS}
)

add_executable(curve_brightness_editor curve_brightness_editor.cpp)
target_link_libraries(curve_brightness_editor ${OpenCV_LIBS})