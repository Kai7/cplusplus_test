set(SAMPLE_DEPS ai_core ${OPENCV_LIBS_EXT})

message("SAMPLE_DEPS : ${SAMPLE_DEPS}")

project(sample_imgwarp)
add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp)
target_link_libraries(${PROJECT_NAME} ${SAMPLE_DEPS})
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
