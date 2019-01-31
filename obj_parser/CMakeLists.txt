project(object_parser)

add_executable(obj_file_parser obj_file_parser.cpp)
target_link_libraries(obj_file_parser tinyobjloader)



