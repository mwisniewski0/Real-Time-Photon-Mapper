#ifndef __KD_TREE
#define __KD_TREE

#include <memory>
#include <vector>

#include "../common/cutil_math.h"
#include "../common/photon.h"
#include "cudaHelpers.h"

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

#define DIMENSION 3

//Returns the splitting plane on level i of the tree
inline __host__ __device__ uint GET_PLANE(uint i) {
	uint count = 0;
	while (i >>= 1) ++count;
	return count%DIMENSION;
}

__device__ __host__ inline void nearestNeighbor(const GPUVector<Photon>& photonMap, float3 point, uint* closest, uint num)
{
	//These are the only important things to store to make the search iterative
	struct StackNode {
		uint photonIdx;
		bool searchedFirst;
	};

	StackNode stack[64];
	int stackIdx = 0;
	stack[stackIdx] = { 0,false };
	float minDists[550];
	for (uint i = 0; i < num; ++i) {
		minDists[i] = 1e20;
		closest[i] = 0;
	}

	while (stackIdx >= 0) {
		uint currIdx = stack[stackIdx].photonIdx;
		const float3& currPos = photonMap[currIdx].pos;
		if (!stack[stackIdx].searchedFirst) {
			float3 diff = { currPos.x - point.x, currPos.y - point.y, currPos.z - point.z };
			float distSqr = diff.x*diff.x + diff.y*diff.y + diff.z*diff.z;
			if (distSqr < minDists[num - 1]) {
				uint i = num - 1;
				while (i && distSqr < minDists[i - 1]) {
					minDists[i] = minDists[i - 1];
					closest[i] = closest[i - 1];
					--i;
				}
				minDists[i] = distSqr;
				closest[i] = currIdx;
			}
		}
		if (2 * currIdx + 1 >= photonMap.size) { //leaf node
			--stackIdx; // done with node
			continue;
		}
		else {
			uint plane = GET_PLANE(currIdx);
			float pPlane, cPlane;
			if (plane == 0) {
				pPlane = point.x;
				cPlane = currPos.x;
			}
			else if (plane == 1) {
				pPlane = point.y;
				cPlane = currPos.y;
			}
			else {
				pPlane = point.z;
				cPlane = currPos.z;
			}
		        
			if (pPlane <= cPlane) { // search left then right
	
				if (stack[stackIdx].searchedFirst) {
					if (2 * currIdx + 2 < photonMap.size && pPlane + sqrt(minDists[num - 1]) > cPlane)
						stack[stackIdx] = { 2 * currIdx + 2, false };
					else
					{
						--stackIdx;
					}
					//continue;

				}
				else {
					stack[stackIdx].searchedFirst = true;
					if (pPlane - sqrtf(minDists[num - 1]) <= cPlane)
						stack[++stackIdx] = { 2 * currIdx + 1, false };
					//continue;

				}
			}
			else {
				if (stack[stackIdx].searchedFirst) {
					if (pPlane - sqrt(minDists[num - 1]) <= cPlane)
						stack[stackIdx] = { 2 * currIdx + 1, false };
					else
						--stackIdx;
					//continue;
				}
				else {
					stack[stackIdx].searchedFirst = true;
					if (2 * currIdx + 2 < photonMap.size && pPlane + sqrt(minDists[num - 1]) > cPlane)
						stack[++stackIdx] = { 2 * currIdx + 2, false };
					//continue;
				}
			}
		}
	}
}

__device__ inline float3 gatherPhotons(float3 point, const GPUVector<Photon>& map) {
	uint nearest[1];
	nearestNeighbor(map, point, nearest, 1);

	float3 total = make_float3(0, 0, 0);
	for (int i = 0; i < 1; ++i){  
	    float dist = length(point - map[nearest[i]].pos);
	    //printf("%f %f %f\n", point.x, point.y, point.z);
	    if(dist < 0.01)
		total += map[nearest[i]].power;
	    //total += map[i].power;
	}
	float dist = 1;// length(point - nearest[0]);
	return total / dist / dist + make_float3(0.2f);
}


#endif
