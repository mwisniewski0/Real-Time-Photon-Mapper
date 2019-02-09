project(object_parser)

add_executable(obj_file_parser ./obj_file_parser.cpp ../common/geometry.cpp ../common/bvh.cpp ../common/scene.cpp)
target_link_libraries(obj_file_parser tinyobjloader)
target_compile_options(obj_file_parser PRIVATE "-D NO_CUDA")



