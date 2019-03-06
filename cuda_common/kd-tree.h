#ifndef __KD_TREE
#define __KD_TREE

// This file defines the KD-tree structure for fast k-nearest-points lookup.

#include <memory>
#include <vector>

#include "../common/cutil_math.h"
#include "../common/photon.h"

/*
 * Node in the tree data structure
 */
struct Node{
    Photon* photon;
    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;
    uint dimension;
};

// Prints the provided Node and its children recursively. Used for debugging.
void printTree(Node& head);

// Sorts the provided Photons into a KD-tree order. Such list can be used in nearestNeighbor() and
// gatherPhotons()
void sortPhotons(std::vector<Photon>& photonList);

// Finds num nearest photons to "point" using a photon map sorted using sortPhotons(). The indices
// of the closest photons are written to the "closest" parameter.
__device__ void nearestNeighbor(Photon* photonMap, uint len, float3 point, uint* closest, uint num);

// Finds the density of photons around the given point using a photon map sorted using sortPhotons()
__device__ float3 gatherPhotons(float3 point, Photon* map, uint len);

#endif
