CMAKE_MINIMUM_REQUIRED(VERSION 3.3.13)

set(CMAKE_MACOSX_RPATH 1)

project(3d_parser)


add_executable(main parse_main.cpp parse_main.h)
target_link_libraries(main lib3ds)
