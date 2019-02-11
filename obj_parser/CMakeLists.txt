project(object_parser)

add_executable(obj_file_parser ./obj_file_parser.cpp ../common/geometry.cpp ../common/bvh.cpp ../common/scene.cpp ./photon_builder.cpp ../common/ply.cpp ../common/streamWriter.cpp)
add_executable(photon_builder ./photon_builder.cpp ../common/geometry.cpp)


target_link_libraries(obj_file_parser tinyobjloader)
target_compile_options(obj_file_parser PRIVATE "-D NO_CUDA -O3")

target_compile_options(photon_builder PRIVATE "-D NO_CUDA -O3")



