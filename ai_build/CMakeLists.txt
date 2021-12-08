cmake_minimum_required(VERSION 3.18.4)
project(ai)

set(CMAKE_CXX_INIT "-fPIC -Werror=all -fdiagnostics-color=always -std=gnu++17")
set(CMAKE_CXX_INIT "${CMAKE_CXX_INIT} -ffunction-sections -fdata-sections -Wl,-gc-sections")

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_INIT})

set(CV_INCLUDES ${CV_SDK_ROOT}/include)
set(CV_LIBS ${CV_SDK_ROOT}/lib/libcv_core.a)

set(TEST_INCLUDES ${CMAKE_SOURCE_DIR}/include/)
message("TEST_INCLUDES: ${TEST_INCLUDES}")

include_directories(${TEST_INCLUDES} ${CV_INCLUDES})
add_subdirectory(modules)

install(DIRECTORY ${CV_INCLUDES} DESTINATION include/ai)
