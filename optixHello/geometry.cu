
#include <optix.h>

#include "optixHello.h"
#include "helpers.h"

extern "C" {
	__constant__ Params params;
}

extern "C" __global__ void __intersection__parallelogram()
{
	const Parallelogram* floor = reinterpret_cast<Parallelogram*>(optixGetSbtDataPointer());

	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir = optixGetWorldRayDirection();
	const float ray_tmin = optixGetRayTmin(), ray_tmax = optixGetRayTmax();

	float3 n = make_float3(floor->plane);
	float dt = dot(ray_dir, n);
	float t = (floor->plane.w - dot(n, ray_orig)) / dt;
	if (t > ray_tmin && t < ray_tmax)
	{
		float3 p = ray_orig + ray_dir * t;
		float3 vi = p - floor->anchor;
		float a1 = dot(floor->v1, vi);


		if (a1 >= 0 && a1 <= 1)
		{
			float a2 = dot(floor->v2, vi);
			if (a2 >= 0 && a2 <= 1)
			{
				optixReportIntersection(
					t,
					0,
					float3_as_args(n),
					float_as_int(t)
				);
			}
		}


	}
}


extern "C" __device__ void intersect_sphere()
{

	const Sphere* sphere = reinterpret_cast<Sphere*>(optixGetSbtDataPointer());
	const float3  ray_orig = optixGetWorldRayOrigin();
	const float3  ray_dir = optixGetWorldRayDirection();
	const float   ray_tmin = optixGetRayTmin(), ray_tmax = optixGetRayTmax();

	float3 O = ray_orig - sphere->center;
	float  l = 1 / length(ray_dir);
	float3 D = ray_dir * l;
	float radius = sphere->radius;

	float b = dot(O, D);
	float c = dot(O, O) - radius * radius;
	float disc = b * b - c;
	if (disc > 0.0f)
	{
		float sdisc = sqrtf(disc);
		float root1 = (-b - sdisc);

		bool do_refine = false;

		float root11 = 0.0f;

		if (fabsf(root1) > 10.f * radius)
		{
			do_refine = true;
		}

		if (do_refine) {
			// refine root1
			float3 O1 = O + root1 * D;
			b = dot(O1, D);
			c = dot(O1, O1) - radius * radius;
			disc = b * b - c;

			if (disc > 0.0f)
			{
				sdisc = sqrtf(disc);
				root11 = (-b - sdisc);
			}
		}

		bool check_second = true;

		float  t;
		float3 normal;
		t = (root1 + root11) * l;
		if (t > ray_tmin && t < ray_tmax)
		{
			normal = (O + (root1 + root11)*D) / radius;
			if (optixReportIntersection(t, 0, float3_as_args(normal), float_as_int(t)))

				check_second = false;
		}

		if (check_second)
		{
			float root2 = (-b + sdisc) + (do_refine ? root1 : 0);
			t = root2 * l;
			normal = (O + root2 * D) / radius;
			if (t > ray_tmin && t < ray_tmax)
				optixReportIntersection(t, 0, float3_as_args(normal));
		}
	}
}


extern "C" __global__ void __intersection__sphere()
{
	intersect_sphere();
}

