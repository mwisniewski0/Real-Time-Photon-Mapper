#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "../common/kd-tree.h"

std::default_random_engine gen;

void createRandom(std::vector<Photon>& list){
    for (uint i = 0; i<list.size(); ++i){
	list[i].pos.x = (float)gen()/gen.max();
	list[i].pos.y = (float)gen()/gen.max();
	list[i].pos.z = (float)gen()/gen.max();
    }
}

float findDist(float3 a, float3 b){
    float3 diff = {a.x-b.x, a.y-b.y, a.z-b.z};
    return sqrt(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
}

void testSort(){
    std::vector<Photon> list(100);
    std::vector<uint> nearest(list.size());
    float3 point;
    for (uint i= 0; i < 10; ++i){ 
	createRandom(list);
	sortPhotons(list);
	point.x = gen()/gen.max();
	point.y = gen()/gen.max();
	point.z = gen()/gen.max();
	nearestNeighbor(list.data(), list.size(), point, nearest.data(), nearest.size());
	
	for(uint j = 0; j<nearest.size()-1; ++j){
	    if(findDist(list[nearest[i]].pos,point) > findDist(list[nearest[i+1]].pos,point)){
		std::cout << "Test " << i << " failed" << std::endl;
	        exit(1);
	    }
	}
    }
}

void outputData(){;
    uint k = 512;
    std::cout << "Num Points" << "|" << "Sort Time" << "|" << "Sort Scaled" <<
	"|" << "Search Time" << "|" << "Search Scaled" << std::endl;
    bool printDetails = false;
    uint numSamples = 10;
    for (int i = 5; i <= 10; ++i){
	uint len = (1<<(2*i));
	double avgSortTime = 0;
	double avgSortScale = 0;
	double avgSearchTime = 0;
	double avgSearchScale = 0;
	for(uint j = 0; j<numSamples; ++j){
	    std::vector<Photon> list(len);
	    createRandom(list);
	    auto t0 = std::chrono::system_clock::now();
	    sortPhotons(list);
	    auto t1 = std::chrono::system_clock::now();
	    std::chrono::duration<double> diff = t1-t0;
	    double secs = diff.count();
	    avgSortTime+=secs;
	    double scale = (double)((t1-t0).count())/len/(2*i)/(2*i);
	    avgSortScale += scale;
	    if(printDetails){
		std::cout << std::setw(10) << len << "|" <<
		    std::setw(9) << secs*1000 << "|" << std::setw(11) << scale << "|";
	    }
	    std::vector<uint> nearest(k);
	    float3 point;
	    point.x = (float)gen()/gen.max();
	    point.y = (float)gen()/gen.max();
	    point.z = (float)gen()/gen.max();
	    t0 = std::chrono::system_clock::now();
	    nearestNeighbor(list.data(), list.size(), point, nearest.data(), nearest.size());
	    t1 = std::chrono::system_clock::now();
	    diff = t1-t0;
	    secs = diff.count();
	    avgSearchTime += secs;
	    scale = (double)((t1-t0).count())/(2*i);
	    avgSearchScale += scale;
	    if(printDetails){
		std::cout << std::setw(11) << secs*1000*1000 << "|"  << std::setw(13) <<
		    	scale << std::endl;
	    }
	}
	std::cout << "\e[1m" << std::setw(10) <<  len << "|" <<
	    std::setw(9) << avgSortTime/numSamples*1000 << "|" <<
	    "\e[91m" << std::setw(11) << avgSortScale/numSamples  << "\e[30m|" <<
	    std::setw(11) << avgSearchTime/numSamples*1000*1000 << "|" <<
	    "\e[91m" << std::setw(13) << avgSearchScale/numSamples << "\e[30m\e[0m" << std::endl;
    }
    std::cout << std::endl << "Sort time in milliseconds" << std::endl <<
	"Search time in microseconds" << std::endl;
}

int main(){
    testSort();
    outputData();

    /* std::vector<Photon> list(10);
    createRandom(list);
    sortPhotons(list);
    for(Photon p: list){
	std::cout << p.pos.x << " " << p.pos.y << " " << p.pos.z << std::endl;
	}*/
    return 0;
}
