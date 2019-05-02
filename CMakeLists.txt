cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(voronoi)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -march=native -fno-omit-frame-pointer")

execute_process(COMMAND jemalloc-config --libs
  OUTPUT_VARIABLE JEMALLOC_LIBS
  OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND jemalloc-config --libdir
  OUTPUT_VARIABLE JEMALLOC_LIBDIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REPLACE " " ";" JEMALLOC_LIBS ${JEMALLOC_LIBS})
message(STATUS "jemalloc-libs: ${JEMALLOC_LIBS}, jemalloc-libdir: ${JEMALLOC_LIBDIR}")

set(SRC_DIR "src")
add_executable(main "${SRC_DIR}/main.cpp")
target_include_directories(main PUBLIC .)
target_link_libraries(main pthread)
target_link_options(main PUBLIC -L${JEMALLOC_LIBDIR} -Wl,-rpath,${JEMALLOC_LIBDIR} -ljemalloc "${JEMALLOC_LIBS}")
