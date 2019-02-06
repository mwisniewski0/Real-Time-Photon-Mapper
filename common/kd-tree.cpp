#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#include <climits>

#include "photon.h"
#include "kd-tree.h"

#define DIMENSION 3

//Returns the splitting plane on level i of the tree
inline uint GET_PLANE(uint i){
    uint count = 0;
    while (i>>=1) ++count;
    return count%DIMENSION;
}

//Sort an array of photons in the x, y, and z directions respectively
struct{
    bool operator()(const Photon& a, const Photon& b){
	return a.pos.x < b.pos.x;
    }
} sortX;
struct{
    bool operator()(const Photon& a, const Photon& b){
	return a.pos.y < b.pos.y;
    }
} sortY;
struct{
    bool operator()(const Photon& a, const Photon& b){
	return a.pos.z < b.pos.z;
    }
} sortZ;

//Recursively go through tree and print the position of each photon
void printTree(Node& head){
    std::cout << head.photon->pos.x << " " << head.photon->pos.y << " " << head.photon->pos.z << std::endl;
    if (head.left){
	std::cout << "l ";
	printTree(*(head.left));
    }
    if(head.right){
	std::cout << "r ";
	printTree(*(head.right));
    }
}

//This is O(nlog^2n). It can be done in O(nlogn) but since this is done in preprocesing stage, I don't
//really care about going fast and this way is easier to code.
std::unique_ptr<Node> buildTree(Photon* photons, int len, uint dimension){
    auto node = std::make_unique<Node>();
    if (dimension == 0)
	std::sort(photons, photons + len, sortX);
    else if(dimension == 1)
	std::sort(photons, photons + len, sortY);
    else
	std::sort(photons, photons + len, sortZ);
    if (len == 1){
	node->photon = &(photons[0]);
	node->dimension = dimension;
	node->right = nullptr;
	node->left = nullptr;
	return node;
    }
    if (len == 2){
	node->photon = &(photons[1]);
	node->left = std::make_unique<Node>();
	node->left->photon = &(photons[0]);
	node->right=nullptr;
	return node;
    }
    uint mid = len/2;
    node->photon = &(photons[mid]);
    node->left = buildTree(photons, mid, (dimension+1)%DIMENSION);
    node->right = buildTree(photons+mid+1, len-mid-1, (dimension+1)%DIMENSION);
    return node;
}

//Copy kd-tree as structure of nodes with pointers into a continguous chunk of memory
//Left child is 2*i+1 and right is 2*i + 2
void treeToArray(Photon* photon, Node& node, uint idx, uint max){
    Photon* p= node.photon;
    photon[idx] = *(p);
    if(node.left)
	treeToArray(photon, *(node.left), 2*idx+1,max);
    else if (2*idx+1 < max){
	photon[2*idx+1].pos = {1e20,1e20,1e20};
    }
    if(node.right)
	treeToArray(photon, *(node.right), 2*idx+2,max);
    else if (2*idx+2 < max){
	photon[2*idx+2].pos = {1e20,1e20,1e20};
    }
}

//Take a list of photons, turn it into a kd-tree and copy it back to the original vector
//Note that in order to make it balance, the vector may end up a little longer than it started
void sortPhotons(std::vector<Photon>& photonList){
    std::vector<Photon> temp(1<<((int)std::ceil(log2(photonList.size()))));
    std::unique_ptr<Node> photonMap = buildTree(photonList.data(), photonList.size(), 0);
    treeToArray(temp.data(), *photonMap, 0, temp.size());
    photonList = std::move(temp);
}

//These are the only important things to store to make the search iterative
struct StackNode{
    uint photonIdx;
    bool searchedFirst;
};

//Use c-array here so we don't have to deal with std::vector in CUDA
__device__ void nearestNeighbor(Photon* photonMap, uint len, float3 point, uint* closest, uint num){
    StackNode stack[64];
    int stackIdx = 0;
    stack[stackIdx] = {0,false};
    float minDists[num];
    for (uint i = 0; i<num; ++i){
	minDists[i] = 1e20;
	closest[i] = 0;
    }
    while(stackIdx>=0){
	uint currIdx = stack[stackIdx].photonIdx;
	float3& currPos = photonMap[currIdx].pos;
	if(!stack[stackIdx].searchedFirst){
	    float3 diff = {currPos.x - point.x, currPos.y - point.y, currPos.z - point.z};
	    float distSqr = diff.x*diff.x + diff.y*diff.y + diff.z*diff.z;
	    if (distSqr < minDists[num-1]){
		uint i = num-1;
		while (i && distSqr < minDists[i-1]){
		    minDists[i] = minDists[i-1];
		    closest[i] = closest[i-1];
		    --i;
		}
		minDists[i] = distSqr;
		closest[i] = currIdx;
	    }
	}
	if (2*currIdx + 1 >= len){ //leaf node
	    --stackIdx; // done with node
	}
	else{
	    uint plane = GET_PLANE(currIdx);
	    float pPlane, cPlane;
	    if (plane == 0){
		pPlane = point.x;
		cPlane = currPos.x;
	    }
	    else if (plane == 1){
		pPlane = point.y;
		cPlane = currPos.y;
	    }
	    else{
		pPlane = point.z;
		cPlane = currPos.z;
	    }
	    if (pPlane <= cPlane){ // search left then right
		if(stack[stackIdx].searchedFirst){
		    if (2*currIdx+2<len && pPlane + sqrt(minDists[num-1]) > cPlane)
			stack[stackIdx] = {2*currIdx+2, false};
		    else
			--stackIdx;
		}
		else{
		    stack[stackIdx].searchedFirst = true;
		    if(pPlane - sqrtf(minDists[num-1]) <= cPlane)
			stack[++stackIdx] = {2*currIdx + 1, false};
		}
	    }
	    else{
		if(stack[stackIdx].searchedFirst){
		    if (pPlane - sqrt(minDists[num-1]) <= cPlane)
			stack[stackIdx] = {2*currIdx + 1, false};
		    else
			--stackIdx;
		}
		else{
		    stack[stackIdx].searchedFirst = true;
		    if(2*currIdx + 2 < len && pPlane + sqrt(minDists[num-1]) > cPlane)
			stack[++stackIdx] = {2*currIdx + 2, false};
		}
	    }
	}
    }
}

