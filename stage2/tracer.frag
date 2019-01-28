#version 430 core
out vec3 color;
in vec2 screenPos;

uniform vec3 cameraTopLeft;
uniform vec3 cameraTopRight;
uniform vec3 cameraBottomLeft;
uniform vec3 cameraBottomRight;
uniform vec3 cameraEyePos;


const float EPSILON = 0.000001;

struct Material {
    vec4 color;
    vec4 specularReflectivity;
    int type; // 0 diffuse, 1 specular

	int padding1;
	int padding2;
	int padding3;
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
	Sphere(vec4(-0.5,0.3,0,0), 0.2, Material(vec4(0), vec4(0.77, 0.83, 0.81, 0), 1, 0,0,0)),
	Sphere(vec4(0,-0.3,0,0), 0.2, Material(vec4(0), vec4(0.77, 0.83, 0.81, 0), 1, 0,0,0)),
	Sphere(vec4(0.5,0.3,0,0), 0.2, Material(vec4(0), vec4(0.77, 0.83, 0.81, 0), 1, 0,0,0))
);

struct PointLightSource {
    vec4 position;
    vec4 intensity;  // vec4(r,g,b,_)
};

readonly layout(std430, binding = 2) buffer geometryLayout
{
    Triangle sceneGeometry[];
};

readonly layout(std430, binding = 3) buffer lightingLayout
{
    PointLightSource lightSources[];
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

bool castRay(Ray ray, out RayHit result) {
    float closestHitDistance = 1. / 0.;  // Infinity
    for (int i = 0; i < sceneGeometry.length(); ++i) {
        RayHit hit;
        if (intersectRayAndTriangle(ray, sceneGeometry[i], hit)) {
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
        castRay(ray, hit);
        if (hit.material.type == 1) {
            // Specular
            currentModifier *= v3(hit.material.specularReflectivity);
            ray.origin = hit.pointOfHit + (hit.normal * EPSILON);
            ray.dir = reflect(ray.dir, hit.normal);
        } else {
            // Diffuse
            return currentModifier * getHitIllumination(hit);
        }
    }

    return vec3(0.0, 0.0, 0.0);
}

Camera getCamera() {
	return Camera(cameraTopLeft, cameraTopRight, cameraBottomLeft, cameraBottomRight, cameraEyePos);
}

void main(){
    Ray ray = getCameraRay(getCamera(), screenPos);
    color = getRayColor(ray);
}