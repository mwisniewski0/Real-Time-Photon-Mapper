# Common

Common contains all non-CUDA dependent code used by other parts of Photon.

* **[bvh.cpp](./bvh.cpp)** - Defines a BVH structure for accelerated, O(log(n)), triangle searches.
* **[cutil_math.h](./cutil_math.h)** - Based on NVidia's CUDA samples math_helpers.h header. Defines CUDA vector operations as well as defines the CUDA vectors for non-CUDA development environments.
* **[geometry.cpp](./geometry.cpp)** - Defines the geometry used by both stage 1 and stage 2.
* **[obj_file_parser.cpp](./obj_file_parser.cpp)** - Functions for loading `.obj` and `.mtl` files into Photon, and converting them into a valid scene.
* **[ply.cpp](./ply.cpp)** - Functions for loading `.ply` files into Photon as a list of triangles.
* **[scene.cpp](./scene.cpp)** - Defines a structure to hold all elements of a scene to be rendered
* **[streamWriter.cpp](./streamWriter.cpp)** - Functionality for reading and writing various Photon classes to and from streams.
