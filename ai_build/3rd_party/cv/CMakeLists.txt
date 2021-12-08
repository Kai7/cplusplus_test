cmake_minimum_required(VERSION 3.18.4)
project(cv)

set(CMAKE_CXX_INIT "-fsigned-char -fPIC -Werror=all -fdiagnostics-color=always -std=gnu++17")

set(CV_INCLUDES ${CMAKE_SOURCE_DIR}/include/)
message("CV_INCLUDES: ${CV_INCLUDES}")

include_directories(${CV_INCLUDES})
add_subdirectory(modules)

install(DIRECTORY ${CV_INCLUDES} DESTINATION include/cv)
