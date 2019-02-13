project(object_parser)

add_executable(test_obj_file_parser ./test.cpp ./obj_file_parser.cpp ../common/geometry.cpp ../common/bvh.cpp ../common/scene.cpp ../common/ply.cpp ../common/streamWriter.cpp)


target_link_libraries(test_obj_file_parser tinyobjloader)
target_compile_options(test_obj_file_parser PRIVATE "-D NO_CUDA -O3")




