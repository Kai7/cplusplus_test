cmake_minimum_required(VERSION 3.18.4)
project(ai)

# Find toolchain cmake file and toolchain folder
if("${CMAKE_TOOLCHAIN_FILE}" STREQUAL "")
  message(FATAL_ERROR "CMAKE_TOOLCHAIN_FILE is not set. Aborting.")
endif()

set(CMAKE_CXX_INIT "-fPIC -Werror=all -fdiagnostics-color=always -std=gnu++17")
set(CMAKE_CXX_INIT "${CMAKE_CXX_INIT} -ffunction-sections -fdata-sections -Wl,-gc-sections")
set(CMAKE_CXX_INIT "${CMAKE_CXX_INIT} -fvisibility=hidden")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_INIT} -s")

set(OPENCV_INCLUDES ${OPENCV_ROOT}/include)
set(OPENCV_LIBS ${OPENCV_ROOT}/lib/libopencv_core.a)

message("==================================================")
message("[Summary]")
message("C   compiler       ${CMAKE_C_COMPILER}")
message("CXX compiler       ${CMAKE_CXX_COMPILER}")
message("C Flags            ${CMAKE_C_FLAGS}")
message("CXX Flags          ${CMAKE_CXX_FLAGS}")
message("Install dir        ${CMAKE_INSTALL_PREFIX}")
#message("SHRINK_OPENCV_SIZE ${SHRINK_OPENCV_SIZE}")
message("==================================================")

set(AI_INCLUDES ${CMAKE_SOURCE_DIR}/include/)

include_directories(${AI_INCLUDES} ${OPENCV_INCLUDES})
add_subdirectory(modules)

install(DIRECTORY ${AI_INCLUDES} DESTINATION include/ai)
