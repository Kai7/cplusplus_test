cmake_minimum_required(VERSION 3.18.4)
project(cv)

set(CMAKE_CXX_INIT "-fPIC -Werror=all -fdiagnostics-color=always -std=gnu++17")
set(CMAKE_CXX_INIT "${CMAKE_CXX_INIT} -ffunction-sections -fdata-sections")
# set(CMAKE_CXX_INIT "${CMAKE_CXX_INIT} -Wl,-gc-sections")

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_INIT})

set(CV_INCLUDES ${CMAKE_SOURCE_DIR}/include/)
message("CV_INCLUDES: ${CV_INCLUDES}")

include_directories(${CV_INCLUDES})
add_subdirectory(modules)

install(DIRECTORY ${CV_INCLUDES} DESTINATION include/cv)
