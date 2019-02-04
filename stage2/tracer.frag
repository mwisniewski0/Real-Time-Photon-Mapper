#version 430 core
out vec3 color;
in vec2 screenPos;

uniform vec3 cameraTopLeft;
uniform vec3 cameraTopRight;
uniform vec3 cameraBottomLeft;
uniform vec3 cameraBottomRight;
uniform vec3 cameraEyePos;


const float EPSILON = 0.0001;
const float AIR_REFRACTIVE_INDEX = 1;

struct BVHNode {
	vec3 boundingBoxMinCoords;
	int leftOrCount;
	vec3 boundingBoxMaxCoords;
	int rightOrOffset;
};

bool isBVHNodeLeaf(BVHNode node) {
	return 0 != (node.leftOrCount & (1 << 31));
}

int getBVHNodeCount(BVHNode node) {
	return node.leftOrCount & 0x7fffffff;
}

struct Material {
    vec4 color;
    vec4 specularReflectivity;
	float refractiveIndex;
    int type; // 0 diffuse, 1 specular, 2 refractive

	int padding1;
	int padding2;
};

struct Triangle {
    vec4 a;
    vec4 b;
    vec4 c;
    Material material;
};

struct Sphere {
	vec4 center;
	float radius;
	Material material;
};

// Currently, spheres are only planned to be used for the demo, and will be removed in further
// iterations.
Sphere[3] spheres = Sphere[3](
	Sphere(vec4(-0.5,0.3,0,0), 0.2, Material(vec4(0), vec4(0.77, 0.83, 0.81, 0), 2.5, 1, 0,0)),
	Sphere(vec4(0,-0.3,0,0), 0.2, Material(vec4(0), vec4(0.77, 0.83, 0.81, 0), 2.5, 2, 0,0)),
	Sphere(vec4(0.5,0.3,0,0), 0.2, Material(vec4(0), vec4(0.77, 0.83, 0.81, 0), 2.5, 1, 0,0))
);

struct PointLightSource {
    vec4 position;
    vec4 intensity;  // vec4(r,g,b,_)
};

readonly layout(std430, binding = 2) buffer geometryLayout
{
    Triangle triangles[];
};

readonly layout(std430, binding = 3) buffer lightingLayout
{
    PointLightSource lightSources[];
};

readonly layout(std430, binding = 4) buffer bvhLayout
{
    BVHNode triangleBvhNodes[];
};

vec3 v3(vec4 v) {
	return v.xyz;
}

struct Ray {
    vec3 origin;
    vec3 dir;  // Unit dir vector
};

struct RayHit {
    vec3 pointOfHit;
    vec3 normal;  // Unit vector
    Material material;
};

float lengthSquared(vec3 v) {
    return dot(v, v);
}

vec3 pointAtDistance(Ray ray, float d) {
    return ray.origin + ray.dir * d;
}

Ray rayFromPoints(vec3 start, vec3 end) {
    return Ray(start, normalize(end-start));
}

bool rayIntersectsBVHNode(Ray ray, BVHNode node) {
	// Based on https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
	float t1 = (node.boundingBoxMinCoords.x - ray.origin.x)/ray.dir.x;
	float t2 = (node.boundingBoxMaxCoords.x - ray.origin.x)/ray.dir.x;
	float t3 = (node.boundingBoxMinCoords.y - ray.origin.y)/ray.dir.y;
	float t4 = (node.boundingBoxMaxCoords.y - ray.origin.y)/ray.dir.y;
	float t5 = (node.boundingBoxMinCoords.z - ray.origin.z)/ray.dir.z;
	float t6 = (node.boundingBoxMaxCoords.z - ray.origin.z)/ray.dir.z;

	float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
	float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

	return tmax >= 0 && tmin <= tmax;
}

vec3 getNormalAwayFromRay(Ray ray, Triangle t) {
    vec3 normal = cross(v3(t.b - t.a), v3(t.c - t.a));

    // Avoiding branching - this will flip the sign of the normal if the normal forms an obtuse
    // angle with the direction of the ray
    return normalize(normal*dot(normal, -ray.dir));
}

// Moller–Trumbore intersection algorithm
// Source: https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
bool intersectRayAndTriangle(Ray ray, Triangle t, out RayHit hit)
{
    vec3 edge1, edge2, h, s, q;
    float a,f,u,v;
    edge1 = v3(t.b - t.a);
    edge2 = v3(t.c - t.a);
    h = cross(ray.dir, edge2);
    a = dot(edge1, h);
    if (a > -EPSILON && a < EPSILON) {
        // This ray is parallel to this triangle.
        return false;
      }

    f = 1.0/a;
    s = ray.origin - v3(t.a);
    u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) {
        return false;
      }

    q = cross(s, edge1);
    v = f * dot(ray.dir, q);
    if (v < 0.0 || u + v > 1.0) {
        return false;
    }

    // At this stage we can compute d to find out where the intersection point is on the line.
    float d = f * dot(edge2, q);
    if (d > EPSILON) {
        // ray intersection
        hit.pointOfHit = pointAtDistance(ray, d);
        hit.normal = getNormalAwayFromRay(ray, t);
        hit.material = t.material;
        return true;
    }
    else {
        // This means that there is a line intersection but not a ray intersection.
        return false;
    }
}

bool intersectRayAndSphere(Ray ray, Sphere s, out RayHit hit)
{
    vec3 co = ray.origin - v3(s.center);
    float b = 2 * dot(co, ray.dir);
    float c = dot(co, co) - s.radius*s.radius;
    float delta = b*b - 4*c; // Since a is 1 (unitDirection dot unitDirection)

    if (delta < 0) return false;

    float sqrtDelta = sqrt(delta);
    float negT = (-b - sqrtDelta) / 2;
    float posT = (-b + sqrtDelta) / 2;

    if (negT <= 0 && posT <= 0) {
        // The sphere is behind the ray origin
        return false;
    }

    float dFromRayStart;
    bool collidedInside = false;
    if(negT <= 0 && posT > 0) {
        // We hit the sphere from the inside
        dFromRayStart = posT;
        collidedInside = true;
    } else {
        // Take the closer point of intersection
        dFromRayStart = negT;
    }
	hit.pointOfHit = pointAtDistance(ray, dFromRayStart);
	hit.normal = (hit.pointOfHit - v3(s.center)) * (1.0 / s.radius);
	hit.material = s.material;
	return true;
}

struct Camera {
    vec3 screenTopLeft;
    vec3 screenTopRight;
    vec3 screenBottomLeft;
    vec3 screenBottomRight;
    vec3 eyePos;
};

Ray getCameraRay(Camera cam, vec2 screenPos) {
    vec3 pointOnScreen = (cam.screenBottomRight - cam.screenBottomLeft) * screenPos.x +
                         (cam.screenTopLeft - cam.screenBottomLeft) * screenPos.y +
                         cam.screenBottomLeft;
    return rayFromPoints(cam.eyePos, pointOnScreen);
}

//bool castRay(Ray ray, out RayHit result) {
//	int nodesIdxsToSearchStack[64];  // Given this is a binary search, 64 should be a sufficent size for
//							     // the stack
//    int stackSize = 1;
//    nodesIdxsToSearchStack[0] = 0;
//
//	float closestHitDistance = 1. / 0.;  // Infinity
//
//	float trianglesTested = 0;
//
//    while(stackSize != 0){
//		int nodeIdx = nodesIdxsToSearchStack[stackSize - 1]; // pop off the top of the stack
//		stackSize -= 1;
//		BVHNode node = triangleBvhNodes[nodeIdx];
//		if (rayIntersectsBVHNode(ray, node)) {
//			if(isBVHNodeLeaf(node)){
//				for (int i = 0; i < getBVHNodeCount(node); ++i) {
//					trianglesTested += 1;
//					Triangle triangle = triangles[node.rightOrOffset + i];
//					
//					RayHit hit;
//					if (intersectRayAndTriangle(ray, triangle, hit)) {
//						float distanceSquared = lengthSquared(ray.origin - hit.pointOfHit);
//						if (closestHitDistance > distanceSquared) {
//							closestHitDistance = distanceSquared;
//							result = hit;
//						}
//					}
//				}
//			} else {
//				nodesIdxsToSearchStack[stackSize] = node.leftOrCount;
//				stackSize += 1;
//				nodesIdxsToSearchStack[stackSize] = node.rightOrOffset;
//				stackSize += 1;
//			}
//		}
//    }
//
//	color.x += trianglesTested / 950.0f;
//
//    return !isinf(closestHitDistance);
//}

 bool castRay(Ray ray, out RayHit result) {
 
     float closestHitDistance = 1. / 0.;  // Infinity
     for (int i = 0; i < triangles.length(); ++i) {
         RayHit hit;
         if (intersectRayAndTriangle(ray, triangles[i], hit)) {
             float distanceSquared = lengthSquared(ray.origin - hit.pointOfHit);
             if (closestHitDistance > distanceSquared) {
                 closestHitDistance = distanceSquared;
                 result = hit;
             }
         }
     }
 
     for (int i = 0; i < spheres.length(); ++i) {
         RayHit hit;
         if (intersectRayAndSphere(ray, spheres[i], hit)) {
             float distanceSquared = lengthSquared(ray.origin - hit.pointOfHit);
             if (closestHitDistance > distanceSquared) {
                 closestHitDistance = distanceSquared;
                 result = hit;
             }
         }
     }
 
     return !isinf(closestHitDistance);
 }

vec3 getHitIllumination(RayHit hit) {
    vec3 illumination = vec3(0.2, 0.2, 0.2) * v3(hit.material.color);
    for (int i = 0; i < lightSources.length(); ++i) {
        vec3 vectorToLight = v3(lightSources[i].position) - hit.pointOfHit;

        RayHit hitTowardsLight;
        bool lightReached = !castRay(
            rayFromPoints(hit.pointOfHit + (hit.normal * EPSILON), v3(lightSources[i].position)),
            hitTowardsLight);
        lightReached =
            lightReached || (
                lengthSquared(hitTowardsLight.pointOfHit - hit.pointOfHit) >
                lengthSquared(vectorToLight));
        if (lightReached) {
            illumination += v3(hit.material.color) * v3(lightSources[i].intensity) *
                            dot(normalize(vectorToLight), hit.normal);
        }
    }
    return illumination;
}

vec3 getRayColor(Ray ray) {
    const int MAX_RAY_BOUNCE = 20;

    vec3 currentModifier = vec3(1.0, 1.0, 1.0);



    for (int bounce = 0; bounce < MAX_RAY_BOUNCE; ++bounce) {
        RayHit hit;
        if (castRay(ray, hit)) {
			if (hit.material.type == 1) {
				// Specular
				currentModifier *= v3(hit.material.specularReflectivity);
				ray.origin = hit.pointOfHit + (hit.normal * EPSILON);
				ray.dir = reflect(ray.dir, hit.normal);
			} else if (hit.material.type == 2) {
				// Refractive
				if (dot(hit.normal, ray.dir) < 0) {
					// Ray comes from the inside
					currentModifier *= v3(hit.material.specularReflectivity);
					ray.origin = hit.pointOfHit + (-hit.normal * EPSILON);
					ray.dir = refract(ray.dir, hit.normal, 1.0/hit.material.refractiveIndex);
				} else {
					// Ray comes from the outside
					currentModifier *= v3(hit.material.specularReflectivity);
					ray.origin = hit.pointOfHit + (hit.normal * EPSILON);
					ray.dir = refract(ray.dir, -hit.normal, 1.0/hit.material.refractiveIndex);
				}
			} else {
				// Diffuse
				return currentModifier * getHitIllumination(hit);

			}
		}
    }

    return vec3(0.0, 0.0, 0.0);
}

Camera getCamera() {
	return Camera(cameraTopLeft, cameraTopRight, cameraBottomLeft, cameraBottomRight, cameraEyePos);
}

void main(){
    Ray ray = getCameraRay(getCamera(), screenPos);
    vec3 c = getRayColor(ray);
	color += c;
}
