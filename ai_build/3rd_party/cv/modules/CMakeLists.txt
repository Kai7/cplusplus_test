add_compile_options(-fvisibility=hidden)

set(CORE_SRCS core.cpp)

project(cv_core)
add_library(${PROJECT_NAME} STATIC ${CORE_SRCS})
# add_library(${PROJECT_NAME} SHARED ${CORE_SRCS})
install(TARGETS ${PROJECT_NAME} DESTINATION lib)
