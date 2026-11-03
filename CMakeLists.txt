cmake_minimum_required(VERSION 4.1)

project(ML_DEMO)

add_executable(ML_DEMO src/main.cpp
    src/math/algorithm/algo.cpp
    src/math/container/tensor.cpp
    src/nn/linear.cpp
    src/nn/network.cpp
)