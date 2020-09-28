#include <vector_types.h>
#include <optix_device.h>
#include "optixHello.h"
#include "random.h"
#include "helpers.h"


extern "C" {
	__constant__ Params params;
}


//___________________________________EXTRACT AND SAVE THE PER RAY DATA INFORMATION_________________________________________________________
static __device__ __inline__ RadiancePRD getRadiancePRD()
{
	RadiancePRD prd;
	prd.result.x = int_as_float(optixGetPayload_0());
	prd.result.y = int_as_float(optixGetPayload_1());
	prd.result.z = int_as_float(optixGetPayload_2());
	prd.light_number = int_as_float(optixGetPayload_3());
	prd.depth = optixGetPayload_4();
	return prd;
}
static __device__ __inline__ void setRadiancePRD(const RadiancePRD &prd)
{
	optixSetPayload_0(float_as_int(prd.result.x));
	optixSetPayload_1(float_as_int(prd.result.y));
	optixSetPayload_2(float_as_int(prd.result.z));
	optixSetPayload_3(float_as_int(prd.light_number));
	optixSetPayload_4(prd.depth);
}
//____________________________________________________________________________________________________________________________________________



//--------------------------------------------------------------------------
//-------------------ONB-----------------------------------------------------
//--------------------------------------------------------------------------

struct Onb
{
	__forceinline__ __device__ Onb(const float3& normal)
	{
		m_normal = normal;

		if (fabs(m_normal.x) > fabs(m_normal.z))
		{
			m_binormal.x = -m_normal.y;
			m_binormal.y = m_normal.x;
			m_binormal.z = 0;
		}
		else
		{
			m_binormal.x = 0;
			m_binormal.y = -m_normal.z;
			m_binormal.z = m_normal.y;
		}

		m_binormal = normalize(m_binormal);
		m_tangent = cross(m_binormal, m_normal);
	}

	__forceinline__ __device__ void inverse_transform(float3& p) const
	{
		p = p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
	}

	float3 m_tangent;
	float3 m_binormal;
	float3 m_normal;
};

static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, float3& p)
{
	// Uniformly sample disk.
	const float r = sqrtf(u1);
	const float phi = 2.0f * M_PIf * u2;
	p.x = r * cosf(phi);
	p.y = r * sinf(phi);
	// Project up to hemisphere.
	p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
}
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------


extern "C" __global__ void __raygen__genVPL()
{

	const uint3    idx = optixGetLaunchIndex();
	const uint3    dim = optixGetLaunchDimensions();
	const uint32_t linear_idx = idx.x;//Get x index


	unsigned int index_random = linear_idx;
	index_random = lcg(index_random);
	index_random = linear_idx % params.number_of_lights; //Select random light
	BasicLight light = params.lights[index_random];


	float3 ray_origin = light.pos; //Light position/origin

	//////Generate random direction from the begining assume point light
	uint32_t seed_x = tea<16>(idx.x, 0);
	uint32_t seed_y = tea<16>(idx.x, 1);
	uint32_t seed_z = tea<16>(idx.x, 2);

	float xx = (rnd(seed_x) - 0.5f) * 2;
	float yy = (rnd(seed_y) - 0.5f) * 2;
	float zz = (rnd(seed_z) - 0.5f) * 2;

	const float3 direction = make_float3(xx, yy, zz);
	float3 ray_direction = normalize(direction);

	//Declare per ray data
	RadiancePRD prd;
	prd.result = light.color*4.f * M_PIf;//Divide the contribution by the probability of choosing	
	prd.light_number = index_random;//Light Number;		
	prd.depth = 0;//First bounce is 0

	//Generate first ray from random Light_soruce
	optixTrace(
		params.handle,
		ray_origin,
		ray_direction,
		params.scene_epsilon,
		1e16f,
		0.0f,
		OptixVisibilityMask(1),
		OPTIX_RAY_FLAG_NONE,
		RAY_TYPE_RADIANCE,
		1,
		RAY_TYPE_RADIANCE,
		float3_as_args(prd.result),
		reinterpret_cast<uint32_t&>(prd.light_number),
		reinterpret_cast<uint32_t&>(prd.depth)
	);


}

static __device__ __inline__ float3
traceNewRay(
	float3 origin,
	float3 direction,
	int depth,
	float light_number,
	float3 result)
{
	RadiancePRD prd;
	prd.depth = depth;
	prd.light_number = light_number;
	prd.result = result;


	optixTrace(
		params.handle,
		origin,
		direction,
		params.scene_epsilon,
		1e16f,
		0.0f,
		OptixVisibilityMask(1),
		OPTIX_RAY_FLAG_NONE,
		RAY_TYPE_RADIANCE,
		1,
		RAY_TYPE_RADIANCE,
		float3_as_args(prd.result),
		/* Can't use float_as_int() because it returns rvalue but payload requires a lvalue */
		reinterpret_cast<uint32_t&>(prd.light_number),
		reinterpret_cast<uint32_t&>(prd.depth));

	//return prd.result;
}

extern "C" __global__ void __closesthit__vpl_pos()
{
	//Get thread indiex
	const uint3    idx = optixGetLaunchIndex();
	const uint32_t linear_idx = idx.x;

	//Get ray information
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir = optixGetWorldRayDirection();
	const float  ray_t = optixGetRayTmax();

	//Get ray Data
	RadiancePRD prd = getRadiancePRD();

	//Compute index where VPL will be stored
	int bounce = prd.depth;
	unsigned int index;
	index = ((linear_idx * (params.max_bounces + 1)) + bounce);

	//Compute hit point of the ray
	float3 hit_point = ray_orig + ray_t * ray_dir;

	//Extract hit information and compute the hit point normal
	const HitGroupData* sbt_data = (HitGroupData*)optixGetSbtDataPointer();
	const Phong &phong = sbt_data->shading.diffuse;
	float3 object_normal = make_float3(
		int_as_float(optixGetAttribute_0()),
		int_as_float(optixGetAttribute_1()),
		int_as_float(optixGetAttribute_2()));
	float3 world_normal = normalize(optixTransformNormalFromObjectToWorldSpace(object_normal));
	float3 ffnormal = faceforward(world_normal, -optixGetWorldRayDirection(), world_normal);

	//Compute basic Vectors to compute the VPL contribution and generate the new ray
	float3 L = -ray_dir;
	float nDl = dot(ffnormal, L);

	//Compute and store VPL information
	VPL vpl_cu;//Generate the VPL
	vpl_cu.pos = hit_point;//Save its intersection point
	vpl_cu.normal = ffnormal;//Save its normal	
	float3 thisContribution = prd.result *phong.Kd * nDl;//Compute its contribution
	vpl_cu.color = thisContribution;//Save its contribution
	vpl_cu.bounces = bounce;//Save its level of recursion
	vpl_cu.hit = true;//Add a boolean to know is the VLP has hit the scene

	//Store the VPl inside the VPL Buffer
	params.vpls_raw[index] = vpl_cu;

	//We send another ray of the maximum number of bounces hasnt been reached. 
	if (bounce < params.max_bounces) {

		//Generate random index and the new vectro to compute the new direction
		const float z1 = rnd(index);
		const float z2 = rnd(index);
		float3 new_ray_direction;

		//Generate new direction (cosine sample)
		cosine_sample_hemisphere(z1, z2, new_ray_direction);
		Onb onb(ffnormal);
		onb.inverse_transform(new_ray_direction);
		new_ray_direction = normalize(new_ray_direction);

		//Contribution of the new VPL
		float3 thisContribution_next = thisContribution * M_PIf / dot(ffnormal, new_ray_direction);
		//Generate new ray
		traceNewRay(hit_point, new_ray_direction, bounce + 1, prd.light_number, thisContribution_next);

	}

}

extern "C" __global__ void __miss__vpl()
{

	//Get ray information
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir = optixGetWorldRayDirection();
	const float  ray_t = optixGetRayTmax();
	RadiancePRD prd = getRadiancePRD();
	//Get thread information
	const uint3    idx = optixGetLaunchIndex();
	const uint32_t linear_idx = idx.x;

	//Fill the VPL buffer as to know that the VPL have missed the scene. 
	int index = 0;
	for (int j = prd.depth; j <= params.max_bounces; j++) {
		//Compute the index
		index = ((linear_idx * (params.max_bounces + 1)) + j);
		//Create and save the VPL with the relevant information, that is, to know if it has hit the scene. 
		VPL vpl_cu;
		vpl_cu.hit = false;
		params.vpls_raw[index] = vpl_cu;
	}
}

