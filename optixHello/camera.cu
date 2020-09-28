//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <vector_types.h>
#include <optix_device.h>
#include "optixHello.h"
#include "random.h"
#include "helpers.h"

extern "C" {
	__constant__ Params params;
}

static
__device__ void compute_image() {

	const uint3 idx = optixGetLaunchIndex();//Thread index
	const uint3 dim = optixGetLaunchDimensions();

	const CameraData* camera = (CameraData*)optixGetSbtDataPointer();//Camera informatioj

	const uint32_t image_index = params.width*idx.y + idx.x; //Image index Buffer


	float2 d = make_float2(idx.x, idx.y) / make_float2(params.width, params.height) * 2.f - 1.f; //Direction ray based on the pixel
	float3 ray_origin = camera->eye;
	float3 ray_direction = normalize(d.x*camera->U + d.y*camera->V + camera->W);//Ray direction (local coordinates)

	RadiancePRD prd; //Per RAY data
	prd.light_number = 0.f; //Not necesary
	prd.depth = 0;

	//Trace the ray
	optixTrace(
		params.handle,
		ray_origin,
		ray_direction,
		params.scene_epsilon,
		1e16f,
		0,
		OptixVisibilityMask(1),
		OPTIX_RAY_FLAG_NONE,
		RAY_TYPE_RADIANCE,
		RAY_TYPE_COUNT,
		RAY_TYPE_RADIANCE,
		float3_as_args(prd.result),
		reinterpret_cast<uint32_t&>(prd.light_number),
		reinterpret_cast<uint32_t&>(prd.depth));

	float4 acc_val = params.accum_buffer[image_index];

	acc_val = make_float4(prd.result, 0.f);

	//**
	if (idx.x < params.num_hit_vpl && idx.y < K_POINTS_CLUSTER && params.show_R_matrix) {

		float4 acc_val_2 = make_float4(params.R_matrix[idx.y*params.num_hit_vpl + idx.x], 0.f);

		params.frame_buffer[image_index] = make_color(acc_val_2);
		params.accum_buffer[image_index] = acc_val_2;
	}
	else {
		params.frame_buffer[image_index] = make_color(acc_val);
		params.accum_buffer[image_index] = acc_val;
	}

	//**



}

static
__device__ void select_points() {

	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();

	const CameraData* camera = (CameraData*)optixGetSbtDataPointer();

	const uint32_t image_index = params.width*idx.y + idx.x;

	// Subpixel jitter: send the ray through a different position inside the pixel each time,
	// to provide antialiasing.

	float2 d = make_float2(idx.x, idx.y)  / make_float2(params.width, params.height) * 2.f - 1.f;
	float3 ray_origin = camera->eye;
	float3 ray_direction = normalize(d.x*camera->U + d.y*camera->V + camera->W);

	RadiancePRD prd;
	prd.light_number = 0.f; //Not necesary
	prd.depth = 0;

	optixTrace(
		params.handle,
		ray_origin,
		ray_direction,
		params.scene_epsilon,
		1e16f,
		0,
		OptixVisibilityMask(1),
		OPTIX_RAY_FLAG_NONE,
		RAY_TYPE_RADIANCE,
		RAY_TYPE_COUNT,
		RAY_TYPE_RADIANCE,
		float3_as_args(prd.result),
		reinterpret_cast<uint32_t&>(prd.light_number),
		reinterpret_cast<uint32_t&>(prd.depth));
}

static
__device__ void compute_R() {

	const uint3 idx = optixGetLaunchIndex();
	int index = idx.x;
	RadiancePRD prd;

	const CameraData* camera = (CameraData*)optixGetSbtDataPointer();

	float2 d = (make_float2(params.selected_point_index_x[index], params.selected_point_index_y[index])) / make_float2(params.width, params.height) * 2.f - 1.f;
	float3 ray_origin = camera->eye;
	float3 ray_direction = normalize(d.x*camera->U + d.y*camera->V + camera->W);

	prd.light_number = 0.f; //Not necesary
	prd.depth = 0;

	optixTrace(
		params.handle,
		ray_origin,
		ray_direction,
		params.scene_epsilon,
		1e16f,
		0,
		OptixVisibilityMask(1),
		OPTIX_RAY_FLAG_NONE,
		RAY_TYPE_RADIANCE,
		RAY_TYPE_COUNT,
		RAY_TYPE_RADIANCE,
		float3_as_args(prd.result),
		reinterpret_cast<uint32_t&>(prd.light_number),
		reinterpret_cast<uint32_t&>(prd.depth));
}

extern "C" __global__ void __raygen__pinhole_camera()
{
	if (params.compute_image) {
		compute_image();
	}
	if (params.select_space_points) {
		select_points();
	}
	if (params.compute_R) {
		compute_R();
	}
}




