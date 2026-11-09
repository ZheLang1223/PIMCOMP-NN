cmake_minimum_required(VERSION 3.6)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -DNDEBUG")

project(PIMCOMP-NN)

#find_package(ZLIB REQUIRED)

file(GLOB_RECURSE SRC "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
file(GLOB_RECURSE BACKEND "${CMAKE_CURRENT_SOURCE_DIR}/backend/*.cpp")
file(GLOB_RECURSE BACKEND_H "${CMAKE_CURRENT_SOURCE_DIR}/backend/*.h")
file(GLOB_RECURSE ZSTR_H "${CMAKE_CURRENT_SOURCE_DIR}/backend/zstr/*.hpp")
file(GLOB_RECURSE JSON_H "${CMAKE_CURRENT_SOURCE_DIR}/backend/json/*.h")
file(GLOB_RECURSE EVALUATION "${CMAKE_CURRENT_SOURCE_DIR}/evaluation/*.cpp")
file(GLOB_RECURSE EVALUATION_H "${CMAKE_CURRENT_SOURCE_DIR}/evaluation/*.h")
list(APPEND SRC "configure.h")
list(APPEND SRC ${BACKEND} ${BACKEND_H})
list(APPEND SRC ${EVALUATION} ${EVALUATION_H})
list(APPEND SRC ${JSON_H} ${ZSTR_H})

include_directories(backend/zlib)
add_subdirectory(backend/zlib)
add_executable(${PROJECT_NAME}  ${SRC})
add_dependencies(${PROJECT_NAME} zlibstatic)
#target_link_libraries(${PROJECT_NAME} PRIVATE ZLIB::ZLIB)
target_link_libraries(${PROJECT_NAME} zlibstatic)