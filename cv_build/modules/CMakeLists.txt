
set(CORE_SRCS core.cpp imgwarp.cpp)

set(CORE_DEPS ${OPENCV_LIBS})

project(ai_core)
add_library(${PROJECT_NAME} SHARED ${CORE_SRCS})
target_link_libraries(${PROJECT_NAME} ${CORE_DEPS})
install(TARGETS ${PROJECT_NAME} DESTINATION lib)

project(ai_core-static)
add_library(${PROJECT_NAME} STATIC ${CORE_SRCS})
target_link_libraries(${PROJECT_NAME} ${CORE_DEPS})

install(TARGETS ${PROJECT_NAME} DESTINATION lib)
