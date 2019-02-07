#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#include <climits>

#include "../common/photon.h"
#include "../cuda_common/kd-tree.h"


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
void printTree(Node& head) {
	std::cout << head.photon->pos.x << " " << head.photon->pos.y << " " << head.photon->pos.z << std::endl;
	if (head.left) {
		std::cout << "l ";
		printTree(*(head.left));
	}
	if (head.right) {
		std::cout << "r ";
		printTree(*(head.right));
	}
}

//This is O(nlog^2n). It can be done in O(nlogn) but since this is done in preprocesing stage, I don't
//really care about going fast and this way is easier to code.
std::unique_ptr<Node> buildTree(Photon* photons, int len, uint dimension) {
	auto node = std::make_unique<Node>();
	if (dimension == 0)
		std::sort(photons, photons + len, sortX);
	else if (dimension == 1)
		std::sort(photons, photons + len, sortY);
	else
		std::sort(photons, photons + len, sortZ);
	if (len == 1) {
		node->photon = &(photons[0]);
		node->dimension = dimension;
		node->right = nullptr;
		node->left = nullptr;
		return node;
	}
	if (len == 2) {
		node->photon = &(photons[1]);
		node->left = std::make_unique<Node>();
		node->left->photon = &(photons[0]);
		node->right = nullptr;
		return node;
	}
	uint mid = len / 2;
	node->photon = &(photons[mid]);
	node->left = buildTree(photons, mid, (dimension + 1) % DIMENSION);
	node->right = buildTree(photons + mid + 1, len - mid - 1, (dimension + 1) % DIMENSION);
	return node;
}

//Copy kd-tree as structure of nodes with pointers into a continguous chunk of memory
//Left child is 2*i+1 and right is 2*i + 2
void treeToArray(Photon* photon, Node& node, uint idx, uint max) {
	Photon* p = node.photon;
	photon[idx] = *(p);
	if (node.left)
		treeToArray(photon, *(node.left), 2 * idx + 1, max);
	else if (2 * idx + 1 < max) {
		photon[2 * idx + 1].pos = { 1e20f,1e20f,1e20f };
	}
	if (node.right)
		treeToArray(photon, *(node.right), 2 * idx + 2, max);
	else if (2 * idx + 2 < max) {
		photon[2 * idx + 2].pos = { 1e20f,1e20f,1e20f };
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
