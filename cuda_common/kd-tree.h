#ifndef __KD_TREE
#define __KD_TREE

#include <memory>
#include <vector>

#include "../common/cutil_math.h"
#include "../common/photon.h"

/*
 * Node in tree data structure
 */
struct Node{
    Photon* photon;
    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;
    uint dimension;
};

void printTree(Node& head);

std::unique_ptr<Node> buildTree(Photon* photons, int len, uint dimension);

void treeToArray(Photon* photon, Node& node, uint idx, uint max);

void sortPhotons(std::vector<Photon>& photonList);

__device__ void nearestNeighbor(Photon* photonMap, uint len, float3 point, uint* closest, uint num);

__device__ float3 gatherPhotons(float3 point, Photon* map, uint len);

#endif
