
set(CORE_SRCS core.cpp)

set(CORE_DEPS ${CV_LIBS})

project(ai_core)
# add_library(${PROJECT_NAME} STATIC ${CORE_SRCS})
add_library(${PROJECT_NAME} SHARED ${CORE_SRCS})
target_link_libraries(${PROJECT_NAME} ${CORE_DEPS})
install(TARGETS ${PROJECT_NAME} DESTINATION lib)
