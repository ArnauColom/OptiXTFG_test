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

#include <glad/glad.h> // Needs to be included before gl_interop

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <sutil/Camera.h>
#include <sutil/Trackball.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>

#include <GLFW/glfw3.h>
#include <iomanip>
#include <cstring>

#include "optixHello.h"




//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

bool              resize_dirty = false;

// Camera state
bool              camera_changed = true;
sutil::Camera     camera;
sutil::Trackball  trackball;

// Mouse state
int32_t           mouse_button = -1;

const int         max_trace = 10;

// Button Triggers


//Diferent lightinh
bool			  show_indirect = false; //Show indirect lighr
bool			  show_direct = true; //Show direc light
//Vpl options
bool			  show_vpl = false; //Show where is every VPL
//K means options
bool			  show_K_space = false; //Show how thw space is clustered
bool			  apply_k_means = false; // Bool to determine if apply k_means
//R matrix options
bool			  show_R_matrix = false;
bool			  show_cluster_VPL_bool = false;

//Smothstep Params
float			  SSmin = 0.f; //Variable to control the stepfunction
float		      SSmax = 0.f; //Variable to control the stepfunction


//------------------------------------------------------------------------------
//
// Local types
// TODO: some of these should move to sutil or optix util header
//
//------------------------------------------------------------------------------

template <typename T>
struct Record
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT)

		char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

typedef Record<CameraData>      RayGenRecord;
typedef Record<MissData>        MissRecord;
typedef Record<HitGroupData>    HitGroupRecord;
//typedef Record				VPLRecord;

struct EmptyData {};

typedef Record<EmptyData> EmptyRecord;



struct WhittedState
{
	OptixDeviceContext          context = 0;

	//////////////////////Handle//////////////////////////////

	OptixTraversableHandle      gas_handle = {};
	CUdeviceptr                 d_gas_output_buffer = {};

	OptixTraversableHandle      triangle_gas_handle = 0;  // Traversable handle for triangle AS
	CUdeviceptr                 d_triangle_gas_output_buffer = 0;  // Triangle AS memory
	CUdeviceptr                 d_vertices = 0;
	CUdeviceptr                 d_tex_coords = 0;

	OptixTraversableHandle      ias_handle = 0;  // Traversable handle for instance AS
	CUdeviceptr                 d_ias_output_buffer = 0;  // Instance AS memory

	//////////////////////Modules//////////////////////////////

	OptixModule                 geometry_module = 0;
	OptixModule                 camera_module = 0;
	OptixModule                 shading_module = 0;
	OptixModule                 VPL_module = 0; //Vpl Module
	OptixModule                 K_means_module = 0; //K_means Module Space


	//////////////////////Direct Illumination Program Group///////////////////////

	//Ray gen 
	OptixProgramGroup           raygen_prog_group = 0;


	//Miss 
	OptixProgramGroup           radiance_miss_prog_group = 0;
	OptixProgramGroup           occlusion_miss_prog_group = 0;

	//Radiance
	OptixProgramGroup           radiance_sphere_prog_group = 0;
	OptixProgramGroup           radiance_plane_prog_group = 0;

	//Oclussion
	OptixProgramGroup           occlusion_sphere_prog_group = 0;
	OptixProgramGroup           occlusion_plane_prog_group = 0;

	//////////////////////Compute  VPL Program Group//////////////////////

	//VPL Program
	OptixProgramGroup           raygen_VPL_prog_group = 0;
	OptixProgramGroup           sphere_vpl_prog_group = 0;
	OptixProgramGroup           plane_vpl_prog_group = 0;
	OptixProgramGroup           miss_vpl_prog_group = 0;	
	
	//////////////////////Compute  K_means Space Program Group//////////////////////

	//VPL Program
	OptixProgramGroup           raygen_K_means_prog_group = 0;
	OptixProgramGroup           sphere_K_means_prog_group = 0;
	OptixProgramGroup           plane_K_means_prog_group = 0;
	OptixProgramGroup           miss_K_means_prog_group = 0;

	/////////////////////Pipeline/////////////////////////////   	 

	OptixPipeline               pipeline = 0;
	//OptixPipeline               pipeline_2 = 0;

	OptixPipelineCompileOptions pipeline_compile_options = {};

	///////////////////Stream////////////////////////////////

	CUstream                    stream = 0;
	//CUstream                    stream_2 = 0;//VPL

	///////////////////Params////////////////////////////////

	Params                      params;
	Params*                     d_params = nullptr;

	//////////////////////////SBTs////////////////////////////77

	OptixShaderBindingTable     sbt = {};
	OptixShaderBindingTable     sbt_VPL = {}; //SBT for VPL generation
	OptixShaderBindingTable     sbt_K_means = {}; //SBT for K_means cimputations generation


};

//------------------------------------------------------------------------------
//
//  Geometry and Camera data
//
//------------------------------------------------------------------------------

const uint32_t OBJ_COUNT = 7; //Number of Objects in Scene


// SPHERES
const Sphere g_sphere = {
	{ 5.0f, 1.5f, -4.5f }, // center
	1.5f                   // radius
};


const Sphere g_sphere_2 = {
	{ 1.0f, 1.5f, 1.5f }, // center
	1.5f                   // radius
};



// OPEN CUBE

const Parallelogram g_floor(
	make_float3(14.0f, .0f, 0.0f),    // v1
	make_float3(0.0f, 0.0f, 14.0f),    // v2
	make_float3(-4.0f, 0.01f, -8.0f)  // anchor
);

const Parallelogram g_floor_back(
	make_float3(0.0f, 8.0f, 0.0f),    // v1
	make_float3(0.0f, 0.0f, 14.0f),    // v2
	make_float3(-4.0f, 0.01f, -8.0f)  // anchor
);

const Parallelogram g_left_wall(
	make_float3(14.0f, 0.0f, 0.0f),    // v1
	make_float3(0.0f, 8.0f, 0.0f),    // v2
	make_float3(-4.0f, 0.01f, 6.0f)  // anchor
);

const Parallelogram g_right_wall(
	make_float3(14.0f, 0.0f, 0.0f),    // v1
	make_float3(0.0f, 8.0f, 0.0f),    // v2
	make_float3(-4.0f, 0.01f, -8.0f)  // anchor
);

const Parallelogram g_ceil(
	make_float3(14.0f, .0f, 0.0f),    // v1
	make_float3(0.0f, 0.0f, 14.0f),    // v2
	make_float3(-4.0f, 8.01f, -8.0f)  // anchor
);

//----BOX

const Box g_box = {
	make_float3(0.0f, 0.0f, 0.0f),    // v2
	make_float3(1.f, 2.f, 3.f)  // anchor
};

//---LIGHTS

BasicLight g_light = {
	//make_float3(4.5f, 6.8f, 3.0f),   // pos  0 1 -2
	make_float3(8.0f, 6.8f, -1.0f),   // pos  0 1 -2
	make_float3(0.3f, 0.3f, 0.3f)      // color
	//make_float3(0.8f, 0.8f, 0.8f)
};

BasicLight g_light_2 = {
	make_float3(4.5f, 6.8f, -1.0f),   // pos
	make_float3(0.3f, 0.3f, 0.3f)      // color
};

BasicLight g_light_3 = {
	//make_float3(4.5f, 6.8f, -5.0f),   // pos
	make_float3(0.0f, 6.8f, -1.0f),   // pos
	make_float3(0.3f, 0.3f, 0.3f)      // color
};

//VPL_____________________________________-



//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);

	if (action == GLFW_PRESS)
	{
		mouse_button = button;
		trackball.startTracking(static_cast<int>(xpos), static_cast<int>(ypos));
	}
	else
	{
		mouse_button = -1;
	}
}


static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
	Params* params = static_cast<Params*>(glfwGetWindowUserPointer(window));

	if (mouse_button == GLFW_MOUSE_BUTTON_LEFT)
	{
		trackball.setViewMode(sutil::Trackball::LookAtFixed);
		trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), params->width, params->height);
		camera_changed = true;
	}
	else if (mouse_button == GLFW_MOUSE_BUTTON_RIGHT)
	{
		trackball.setViewMode(sutil::Trackball::EyeFixed);
		trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), params->width, params->height);
		camera_changed = true;
	}
}


static void windowSizeCallback(GLFWwindow* window, int32_t res_x, int32_t res_y)
{
	Params* params = static_cast<Params*>(glfwGetWindowUserPointer(window));
	params->width = res_x;
	params->height = res_y;
	camera_changed = true;
	resize_dirty = true;
}


static void keyCallback(GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/)
{
	if (action == GLFW_PRESS)
	{
		if (key == GLFW_KEY_Q ||
			key == GLFW_KEY_ESCAPE)
		{
			glfwSetWindowShouldClose(window, true);
		}

	}
	else if (key == GLFW_KEY_D) //Show direct illumination
	{
		if (show_direct) {
			show_direct = false;
			//camera_changed = true;


		}
		else {
			show_direct = true;
			//camera_changed = true;

		}

	}
	else if (key == GLFW_KEY_I)//Show indirect illumination by VPL
	{
		if (show_indirect) {
			show_indirect = false;
			camera_changed = true;


		}
		else {
			show_indirect = true;
			camera_changed = true;

		}

	}
	else if (key == GLFW_KEY_V) //View VPL position
	{
		if (show_vpl) {
			show_vpl = false;
			camera_changed = true;


		}
		else {
			show_vpl = true;
			camera_changed = true;

		}
	}else if (key == GLFW_KEY_C) //View VPL position cluster
	{
		if (show_cluster_VPL_bool) {
			show_cluster_VPL_bool = false;
			camera_changed = true;


		}
		else {
			show_cluster_VPL_bool = true;
			camera_changed = true;

		}
	}



	else if (key == GLFW_KEY_L) { //SHOW K MEANS SPACE 
		if (show_K_space) {
			show_K_space = false;			
		}
		else {
			show_K_space = true;
		}
	}else if (key == GLFW_KEY_K) { //APPLY K MEANS
		if (apply_k_means) {
			apply_k_means = false;
		}
		else {
			apply_k_means = true;
		}
	}
	else if (key == GLFW_KEY_R) { //SHOW R 
		if (show_R_matrix) {
			show_R_matrix = false;
		}
		else {
			show_R_matrix = true;
		}
	}




	else if (key == GLFW_KEY_O) //SSmin
	{

		SSmin = SSmin - 1;
		camera_changed = true;

	}
	else if (key == GLFW_KEY_P)
	{

		SSmin = SSmin + 1;
		camera_changed = true;

	}
	else if (key == GLFW_KEY_N)// SSmax
	{

		SSmax = SSmax - 1;
		camera_changed = true;

	}
	else if (key == GLFW_KEY_M)
	{

		SSmax = SSmax + 1;
		camera_changed = true;

	}

}


static void scrollCallback(GLFWwindow* window, double xscroll, double yscroll)
{
	if (trackball.wheelEvent((int)yscroll))
		camera_changed = true;
}

//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

void printUsageAndExit(const char* argv0)
{
	std::cerr << "Usage  : " << argv0 << " [options]\n";
	std::cerr << "Options: --file | -f <filename>      File for image output\n";
	std::cerr << "         --launch-samples | -s       Number of samples per pixel per launch (default 16)\n";
	std::cerr << "         --no-gl-interop             Disable GL interop for display\n";
	std::cerr << "         --help | -h                 Print this usage message\n";
	exit(0);
}

void SetBuffers(WhittedState& state) {

	//Image ---------------------------------------------------------------------------
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.accum_buffer),
		state.params.width*state.params.height * sizeof(float4)
	)); //Save space for each pixel

	//VPL --------------------------------------------------------------------------------

	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.vpls_raw),
		state.params.num_vpl *(state.params.max_bounces + 1) * sizeof(VPL) //Memory for VPL
	)); //Save space for the vpl	

	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.number_hit_vpl),
		1 * sizeof(int) //Memory for VPL
	)); //Save space one int, numner of hit vpl	
	


	//K_MEANS POINTS---------------------------------------------------------------------

	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.pos),
		state.params.width*state.params.height * sizeof(float3) //Memory for all points
	));
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.normal),
		state.params.width*state.params.height * sizeof(float3) //Memory for all normals
	));


	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.params.assing_cluster_vector),
		state.params.width*state.params.height * sizeof(int) //Save respective cluster per pixel
		));


	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.pos_cent),
		K_POINTS_CLUSTER * sizeof(float3) //Memory for all position centroid
	));
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.normal_cent),
		K_POINTS_CLUSTER * sizeof(float3) //Memory for all normals centroid 
	));

	//Points Selected per each cluster
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.selected_points_pos),
		K_POINTS_CLUSTER * sizeof(float3) //Memory to store position of selected position
	));
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.selected_points_norm),
		K_POINTS_CLUSTER * sizeof(float3) //Memory to store position of selected position
	));

	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.selected_point_index_x),
		K_POINTS_CLUSTER * sizeof(int) //Memory to store position of selected position
	));
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.selected_point_index_y),
		K_POINTS_CLUSTER * sizeof(int) //Memory to store position of selected position
	));	

	//Local Clusterin select closest cluster
		//local clustering

	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.distances_slides),
		K_POINTS_CLUSTER * K_POINTS_CLUSTER * sizeof(float) //Save respective cluster
	));
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.L_closest_clusters),
		K_POINTS_CLUSTER * L_NEAR_CLUSTERS * sizeof(int) //Save respective cluster
	));

	//QT CLUSTERNG count_points_per_cluster

	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.count_points_per_cluster),
		state.params.width*state.params.height * sizeof(int) //Memory for all normals
	));
 
}

void SetBuffers_VPL(WhittedState& state) {

	//VPL----------------------------------------------
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.vpls),
		state.params.num_hit_vpl * sizeof(VPL) //Memory for VPL
	)); //Save space for the vpl

	//R mantrix -----------------------------------------
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.R_matrix),
		state.params.num_hit_vpl * K_POINTS_CLUSTER * sizeof(float3) //Save respective cluster
	)); //Memory to store the R matrix. VPLSxSPACE_CLUSTER memory.
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.VPL_assing_cluster),
		state.params.num_hit_vpl * sizeof(int) //Save respective cluster
	));	
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.VPL_initial_cent),
		K_MEANS_VPL * sizeof(VPL) //Save respective cluster
	));//Save initial VPL centroids
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.VPL_cent),
		K_MEANS_VPL * sizeof(VPL) //Save respective cluster
	));//Save initial VPL centroids


	//Local Clustering Slice
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.L_i_modules),
		K_POINTS_CLUSTER * state.params.num_hit_vpl * sizeof(float) //Save respective cluster
	));
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.distances_clusters),
		K_POINTS_CLUSTER * K_MEANS_VPL * sizeof(float) //Save respective cluster  distances_clusters
	));

	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.closest_VPL),
		K_POINTS_CLUSTER * MAX_VPL_CLUSTERS * sizeof(int) //Save respective cluster  distances_clusters
	));
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.selected_VPL_pos),
		K_POINTS_CLUSTER * MAX_VPL_CLUSTERS * sizeof(int) //Save respective cluster  distances_clusters
	));

}

void initLaunchParams(WhittedState& state)
{

	state.params.num_vpl = NUM_OF_VPL; //Number of VPL per light
	state.params.max_bounces = NUM_OF_BOUNCES; //Number of bounces per vpl
	state.params.number_of_lights = MAX_LIGHT; //Number of lighdvt

	state.params.num_hit_vpl = 0;
	//SET BUFFERS
	SetBuffers(state);

	state.params.frame_buffer = nullptr; // Will be set when output buffer is mapped

	state.params.subframe_index = 0u;

	//Set lights
	//state.params.light = g_light;

	state.params.lights[0] = g_light;
	state.params.lights[1] = g_light_2;
	state.params.lights[2] = g_light_3;



	state.params.ambient_light_color = make_float3(0.0f, 0.0f, 0.0f);//set ambient color to 0, the indirect computed by the vpl
	state.params.max_depth = max_trace;
	state.params.scene_epsilon = 1.e-4f;

	//Parametres
	state.params.minSS = 0;
	state.params.maxSS = 0;

	CUDA_CHECK(cudaStreamCreate(&state.stream));
	//CUDA_CHECK(cudaStreamCreate(&state.stream_2));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_params), sizeof(Params)));

	state.params.handle = state.gas_handle;
}

//Sphere HitBox (AABB)
static void sphere_bound(float3 center, float radius, float result[6])
{
	OptixAabb *aabb = reinterpret_cast<OptixAabb*>(result);

	float3 m_min = center - radius;
	float3 m_max = center + radius;

	*aabb = {
		m_min.x, m_min.y, m_min.z,
		m_max.x, m_max.y, m_max.z
	};

}
//Paralelogram HitBox (AABB)
static void parallelogram_bound(float3 v1, float3 v2, float3 anchor, float result[6])
{
	// v1 and v2 are scaled by 1./length^2.  Rescale back to normal for the bounds computation.
	const float3 tv1 = v1 / dot(v1, v1);
	const float3 tv2 = v2 / dot(v2, v2);
	const float3 p00 = anchor;
	const float3 p01 = anchor + tv1;
	const float3 p10 = anchor + tv2;
	const float3 p11 = anchor + tv1 + tv2;

	OptixAabb* aabb = reinterpret_cast<OptixAabb*>(result);

	float3 m_min = fminf(fminf(p00, p01), fminf(p10, p11));
	float3 m_max = fmaxf(fmaxf(p00, p01), fmaxf(p10, p11));
	*aabb = {
		m_min.x, m_min.y, m_min.z,
		m_max.x, m_max.y, m_max.z
	};
}

static void box_bound(float3 min, float3 max, float result[6])
{
	OptixAabb *aabb = reinterpret_cast<OptixAabb*>(result);



	*aabb = {
		min.x, min.y, min.z,
		max.x, max.y, max.z
	};

}


static void buildGas(
	const WhittedState &state,
	const OptixAccelBuildOptions &accel_options,
	const OptixBuildInput &build_input,
	OptixTraversableHandle &gas_handle,
	CUdeviceptr &d_gas_output_buffer
)
{
	OptixAccelBufferSizes gas_buffer_sizes;
	CUdeviceptr d_temp_buffer_gas;

	OPTIX_CHECK(optixAccelComputeMemoryUsage(
		state.context,
		&accel_options,
		&build_input,
		1,
		&gas_buffer_sizes));

	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&d_temp_buffer_gas),
		gas_buffer_sizes.tempSizeInBytes));

	// non-compacted output and size of compacted GAS
	CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
	size_t compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size),
		compactedSizeOffset + 8
	));

	OptixAccelEmitDesc emitProperty = {};
	emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

	OPTIX_CHECK(optixAccelBuild(
		state.context,
		0,
		&accel_options,
		&build_input,
		1,
		d_temp_buffer_gas,
		gas_buffer_sizes.tempSizeInBytes,
		d_buffer_temp_output_gas_and_compacted_size,
		gas_buffer_sizes.outputSizeInBytes,
		&gas_handle,
		&emitProperty,
		1));

	CUDA_CHECK(cudaFree((void*)d_temp_buffer_gas));

	size_t compacted_gas_size;
	CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

	if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
	{
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer), compacted_gas_size));

		// use handle as input and output
		OPTIX_CHECK(optixAccelCompact(state.context, 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle));

		CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
	}
	else
	{
		d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
	}
}

void createGeomety(WhittedState &state)
{
	//
	// Build Custom Primitives
	//

	// Load AABB into device memory for each object in the scene
	OptixAabb   aabb[OBJ_COUNT];
	CUdeviceptr d_aabb;

	sphere_bound(
		g_sphere.center, g_sphere.radius,
		reinterpret_cast<float*>(&aabb[0]));

	sphere_bound(
		g_sphere_2.center, g_sphere_2.radius,
		reinterpret_cast<float*>(&aabb[1]));

	parallelogram_bound(
		g_floor.v1, g_floor.v2, g_floor.anchor,
		reinterpret_cast<float*>(&aabb[2]));

	parallelogram_bound(
		g_floor_back.v1, g_floor_back.v2, g_floor_back.anchor,
		reinterpret_cast<float*>(&aabb[3]));

	parallelogram_bound(
		g_left_wall.v1, g_left_wall.v2, g_left_wall.anchor,
		reinterpret_cast<float*>(&aabb[4]));

	parallelogram_bound(
		g_right_wall.v1, g_right_wall.v2, g_right_wall.anchor,
		reinterpret_cast<float*>(&aabb[5]));

	parallelogram_bound(
		g_ceil.v1, g_ceil.v2, g_ceil.anchor,
		reinterpret_cast<float*>(&aabb[6]));

	//box_bound(g_box.min_v,g_box.max_v, reinterpret_cast<float*>(&aabb[8]));


	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aabb
		), OBJ_COUNT * sizeof(OptixAabb)));//Save memory for the aabb for each object
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_aabb),
		&aabb,
		OBJ_COUNT * sizeof(OptixAabb),
		cudaMemcpyHostToDevice
	));//Copy the information

	// Setup AABB build input
	uint32_t aabb_input_flags[] = {
		/* flags for metal sphere */
		OPTIX_GEOMETRY_FLAG_NONE,
		OPTIX_GEOMETRY_FLAG_NONE,
		OPTIX_GEOMETRY_FLAG_NONE,
		OPTIX_GEOMETRY_FLAG_NONE,
		OPTIX_GEOMETRY_FLAG_NONE,
		OPTIX_GEOMETRY_FLAG_NONE,
		OPTIX_GEOMETRY_FLAG_NONE,


	};
	/* TODO: This API cannot control flags for different ray type */

	const uint32_t sbt_index[] = { 0, 1,2,3,4,5,6 };
	CUdeviceptr    d_sbt_index;

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sbt_index), sizeof(sbt_index)));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_sbt_index),
		sbt_index,
		sizeof(sbt_index),
		cudaMemcpyHostToDevice));

	OptixBuildInput aabb_input = {};

	aabb_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
	aabb_input.aabbArray.aabbBuffers = &d_aabb;
	aabb_input.aabbArray.flags = aabb_input_flags;
	aabb_input.aabbArray.numSbtRecords = OBJ_COUNT;
	aabb_input.aabbArray.numPrimitives = OBJ_COUNT;
	aabb_input.aabbArray.sbtIndexOffsetBuffer = d_sbt_index;
	aabb_input.aabbArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
	aabb_input.aabbArray.primitiveIndexOffset = 0;


	OptixAccelBuildOptions accel_options = {
		OPTIX_BUILD_FLAG_ALLOW_COMPACTION,  // buildFlags
		OPTIX_BUILD_OPERATION_BUILD         // operation
	};

	buildGas(
		state,
		accel_options,
		aabb_input,
		state.gas_handle,
		state.d_gas_output_buffer);

	CUDA_CHECK(cudaFree((void*)d_aabb));
}

void createModules(WhittedState &state)
{
	OptixModuleCompileOptions module_compile_options = {
		100,                                    // maxRegisterCount
		OPTIX_COMPILE_OPTIMIZATION_DEFAULT,     // optLevel
		OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO      // debugLevel
	};
	char log[2048];
	size_t sizeof_log = sizeof(log);

	{
		const std::string ptx = sutil::getPtxString(OPTIX_SAMPLE_NAME, "geometry.cu");
		OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
			state.context,
			&module_compile_options,
			&state.pipeline_compile_options,
			ptx.c_str(),
			ptx.size(),
			log,
			&sizeof_log,
			&state.geometry_module));
	}

	{
		const std::string ptx = sutil::getPtxString(OPTIX_SAMPLE_NAME, "camera.cu");
		OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
			state.context,
			&module_compile_options,
			&state.pipeline_compile_options,
			ptx.c_str(),
			ptx.size(),
			log,
			&sizeof_log,
			&state.camera_module));
	}

	{
		const std::string ptx = sutil::getPtxString(OPTIX_SAMPLE_NAME, "draw_solid_color.cu");
		OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
			state.context,
			&module_compile_options,
			&state.pipeline_compile_options,
			ptx.c_str(),
			ptx.size(),
			log,
			&sizeof_log,
			&state.shading_module));
	}

	{
		const std::string ptx = sutil::getPtxString(OPTIX_SAMPLE_NAME, "genVPL.cu");
		OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
			state.context,
			&module_compile_options,
			&state.pipeline_compile_options,
			ptx.c_str(),
			ptx.size(),
			log,
			&sizeof_log,
			&state.VPL_module));
	}

	{
		const std::string ptx = sutil::getPtxString(OPTIX_SAMPLE_NAME, "K_means_space.cu");
		OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
			state.context,
			&module_compile_options,
			&state.pipeline_compile_options,
			ptx.c_str(),
			ptx.size(),
			log,
			&sizeof_log,
			&state.K_means_module));
	}


}

static void createVPLProgram(WhittedState &state, std::vector<OptixProgramGroup> &program_groups)
{
	OptixProgramGroup           VPL_prog_group;
	OptixProgramGroupOptions    VPL_prog_group_options = {};
	OptixProgramGroupDesc       VPL_prog_group_desc = {};
	VPL_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	VPL_prog_group_desc.raygen.module = state.VPL_module;
	VPL_prog_group_desc.raygen.entryFunctionName = "__raygen__genVPL";

	char    log[2048];
	size_t  sizeof_log = sizeof(log);
	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		state.context,
		&VPL_prog_group_desc,
		1,
		&VPL_prog_group_options,
		log,
		&sizeof_log,
		&VPL_prog_group));

	program_groups.push_back(VPL_prog_group);
	state.raygen_VPL_prog_group = VPL_prog_group;

	OptixProgramGroup           set_vpl_sphere_prog_group;
	OptixProgramGroupOptions    set_vpl_sphere_prog_group_options = {};
	OptixProgramGroupDesc       set_vpl_sphere_prog_group_desc = {};
	set_vpl_sphere_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	set_vpl_sphere_prog_group_desc.hitgroup.moduleIS = state.geometry_module;
	set_vpl_sphere_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
	set_vpl_sphere_prog_group_desc.hitgroup.moduleCH = state.VPL_module;
	set_vpl_sphere_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__vpl_pos";
	set_vpl_sphere_prog_group_desc.hitgroup.moduleAH = nullptr;
	set_vpl_sphere_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;


	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		state.context,
		&set_vpl_sphere_prog_group_desc,
		1,
		&set_vpl_sphere_prog_group_options,
		log,
		&sizeof_log,
		&set_vpl_sphere_prog_group));

	program_groups.push_back(set_vpl_sphere_prog_group);
	state.sphere_vpl_prog_group = set_vpl_sphere_prog_group;

	OptixProgramGroup           set_vpl_plane_prog_group;
	OptixProgramGroupOptions    set_vpl_plane_prog_group_options = {};
	OptixProgramGroupDesc       set_vpl_plane_prog_group_desc = {};
	set_vpl_plane_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	set_vpl_plane_prog_group_desc.hitgroup.moduleIS = state.geometry_module;
	set_vpl_plane_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__parallelogram";
	set_vpl_plane_prog_group_desc.hitgroup.moduleCH = state.VPL_module;
	set_vpl_plane_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__vpl_pos";
	set_vpl_plane_prog_group_desc.hitgroup.moduleAH = nullptr;
	set_vpl_plane_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;


	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		state.context,
		&set_vpl_plane_prog_group_desc,
		1,
		&set_vpl_plane_prog_group_options,
		log,
		&sizeof_log,
		&set_vpl_plane_prog_group));

	program_groups.push_back(set_vpl_plane_prog_group);
	state.plane_vpl_prog_group = set_vpl_plane_prog_group;


	OptixProgramGroupOptions    miss_vpl_prog_group_options = {};
	OptixProgramGroupDesc       miss_vpl_prog_group_desc = {};
	miss_vpl_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	miss_vpl_prog_group_desc.miss.module = state.VPL_module;
	miss_vpl_prog_group_desc.miss.entryFunctionName = "__miss__vpl";


	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		state.context,
		&miss_vpl_prog_group_desc,
		1,
		&miss_vpl_prog_group_options,
		log,
		&sizeof_log,
		&state.miss_vpl_prog_group));
}



static void createKmeansProgram(WhittedState &state, std::vector<OptixProgramGroup> &program_groups)
{
	OptixProgramGroup           K_means_prog_group;
	OptixProgramGroupOptions    K_means_prog_group_options = {};
	OptixProgramGroupDesc       K_means_prog_group_desc = {};
	K_means_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	K_means_prog_group_desc.raygen.module = state.K_means_module;
	K_means_prog_group_desc.raygen.entryFunctionName = "__raygen__K_means_space";

	char    log[2048];
	size_t  sizeof_log = sizeof(log);
	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		state.context,
		&K_means_prog_group_desc,
		1,
		&K_means_prog_group_options,
		log,
		&sizeof_log,
		&K_means_prog_group));

	program_groups.push_back(K_means_prog_group);
	state.raygen_K_means_prog_group = K_means_prog_group;

	OptixProgramGroup           set_K_means_sphere_prog_group;
	OptixProgramGroupOptions    set_K_means_sphere_prog_group_options = {};
	OptixProgramGroupDesc       set_K_means_sphere_prog_group_desc = {};
	set_K_means_sphere_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	set_K_means_sphere_prog_group_desc.hitgroup.moduleIS = state.geometry_module;
	set_K_means_sphere_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
	set_K_means_sphere_prog_group_desc.hitgroup.moduleCH = state.K_means_module;
	set_K_means_sphere_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__K_means_points";
	set_K_means_sphere_prog_group_desc.hitgroup.moduleAH = nullptr;
	set_K_means_sphere_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;


	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		state.context,
		&set_K_means_sphere_prog_group_desc,
		1,
		&set_K_means_sphere_prog_group_options,
		log,
		&sizeof_log,
		&set_K_means_sphere_prog_group));

	program_groups.push_back(set_K_means_sphere_prog_group);
	state.sphere_K_means_prog_group = set_K_means_sphere_prog_group;

	OptixProgramGroup           set_K_means_plane_prog_group;
	OptixProgramGroupOptions    set_K_means_plane_prog_group_options = {};
	OptixProgramGroupDesc       set_K_means_plane_prog_group_desc = {};
	set_K_means_plane_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	set_K_means_plane_prog_group_desc.hitgroup.moduleIS = state.geometry_module;
	set_K_means_plane_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__parallelogram";
	set_K_means_plane_prog_group_desc.hitgroup.moduleCH = state.K_means_module;
	set_K_means_plane_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__K_means_points";
	set_K_means_plane_prog_group_desc.hitgroup.moduleAH = nullptr;
	set_K_means_plane_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;


	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		state.context,
		&set_K_means_plane_prog_group_desc,
		1,
		&set_K_means_plane_prog_group_options,
		log,
		&sizeof_log,
		&set_K_means_plane_prog_group));

	program_groups.push_back(set_K_means_plane_prog_group);
	state.plane_K_means_prog_group = set_K_means_plane_prog_group;


	OptixProgramGroupOptions    miss_K_means_prog_group_options = {};
	OptixProgramGroupDesc       miss_K_means_prog_group_desc = {};
	miss_K_means_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	miss_K_means_prog_group_desc.miss.module = state.K_means_module;
	miss_K_means_prog_group_desc.miss.entryFunctionName = "__miss__K_means";


	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		state.context,
		&miss_K_means_prog_group_desc,
		1,
		&miss_K_means_prog_group_options,
		log,
		&sizeof_log,
		&state.miss_K_means_prog_group));
}


static void createCameraProgram(WhittedState &state, std::vector<OptixProgramGroup> &program_groups)
{
	OptixProgramGroup           cam_prog_group;
	OptixProgramGroupOptions    cam_prog_group_options = {};
	OptixProgramGroupDesc       cam_prog_group_desc = {};
	cam_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	cam_prog_group_desc.raygen.module = state.camera_module;
	cam_prog_group_desc.raygen.entryFunctionName = "__raygen__pinhole_camera";

	char    log[2048];
	size_t  sizeof_log = sizeof(log);
	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		state.context,
		&cam_prog_group_desc,
		1,
		&cam_prog_group_options,
		log,
		&sizeof_log,
		&cam_prog_group));

	program_groups.push_back(cam_prog_group);
	state.raygen_prog_group = cam_prog_group;
}

static void createSphereProgram(WhittedState &state, std::vector<OptixProgramGroup> &program_groups)
{
	OptixProgramGroup           radiance_sphere_prog_group;
	OptixProgramGroupOptions    radiance_sphere_prog_group_options = {};
	OptixProgramGroupDesc       radiance_sphere_prog_group_desc = {};
	radiance_sphere_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
		radiance_sphere_prog_group_desc.hitgroup.moduleIS = state.geometry_module;
	radiance_sphere_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
	radiance_sphere_prog_group_desc.hitgroup.moduleCH = state.shading_module;
	radiance_sphere_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__diffuse_radiance";
	radiance_sphere_prog_group_desc.hitgroup.moduleAH = nullptr;
	radiance_sphere_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;

	char    log[2048];
	size_t  sizeof_log = sizeof(log);
	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		state.context,
		&radiance_sphere_prog_group_desc,
		1,
		&radiance_sphere_prog_group_options,
		log,
		&sizeof_log,
		&radiance_sphere_prog_group));

	program_groups.push_back(radiance_sphere_prog_group);
	state.radiance_sphere_prog_group = radiance_sphere_prog_group;

	OptixProgramGroup           occlusion_sphere_prog_group;
	OptixProgramGroupOptions    occlusion_sphere_prog_group_options = {};
	OptixProgramGroupDesc       occlusion_sphere_prog_group_desc = {};
	occlusion_sphere_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
		occlusion_sphere_prog_group_desc.hitgroup.moduleIS = state.geometry_module;
	occlusion_sphere_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
	occlusion_sphere_prog_group_desc.hitgroup.moduleCH = nullptr;
	occlusion_sphere_prog_group_desc.hitgroup.entryFunctionNameCH = nullptr;
	occlusion_sphere_prog_group_desc.hitgroup.moduleAH = state.shading_module;
	occlusion_sphere_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__full_occlusion";

	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		state.context,
		&occlusion_sphere_prog_group_desc,
		1,
		&occlusion_sphere_prog_group_options,
		log,
		&sizeof_log,
		&occlusion_sphere_prog_group));

	program_groups.push_back(occlusion_sphere_prog_group);
	state.occlusion_sphere_prog_group = occlusion_sphere_prog_group;

}

static void createPlaneProgram(WhittedState &state, std::vector<OptixProgramGroup> &program_groups)
{
	OptixProgramGroup           radiance_plane_prog_group;
	OptixProgramGroupOptions    radiance_plane_prog_group_options = {};
	OptixProgramGroupDesc       radiance_plane_prog_group_desc = {};
	radiance_plane_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	radiance_plane_prog_group_desc.hitgroup.moduleIS = state.geometry_module;
	radiance_plane_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__parallelogram";
	radiance_plane_prog_group_desc.hitgroup.moduleCH = state.shading_module;
	radiance_plane_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__diffuse_radiance";
	radiance_plane_prog_group_desc.hitgroup.moduleAH = nullptr;
	radiance_plane_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;

	char    log[2048];
	size_t  sizeof_log = sizeof(log);
	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		state.context,
		&radiance_plane_prog_group_desc,
		1,
		&radiance_plane_prog_group_options,
		log,
		&sizeof_log,
		&radiance_plane_prog_group));

	program_groups.push_back(radiance_plane_prog_group);
	state.radiance_plane_prog_group = radiance_plane_prog_group;

	OptixProgramGroup           occlusion_plane_prog_group;
	OptixProgramGroupOptions    occlusion_plane_prog_group_options = {};
	OptixProgramGroupDesc       occlusion_plane_prog_group_desc = {};
	occlusion_plane_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	occlusion_plane_prog_group_desc.hitgroup.moduleIS = state.geometry_module;
	occlusion_plane_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__parallelogram";
	occlusion_plane_prog_group_desc.hitgroup.moduleCH = nullptr;
	occlusion_plane_prog_group_desc.hitgroup.entryFunctionNameCH = nullptr;
	occlusion_plane_prog_group_desc.hitgroup.moduleAH = state.shading_module;
	occlusion_plane_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__full_occlusion";

	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		state.context,
		&occlusion_plane_prog_group_desc,
		1,
		&occlusion_plane_prog_group_options,
		log,
		&sizeof_log,
		&occlusion_plane_prog_group));

	program_groups.push_back(occlusion_plane_prog_group);
	state.occlusion_plane_prog_group = occlusion_plane_prog_group;

}////////////////////////////////////////////////////////////

static void createMissProgram(WhittedState &state, std::vector<OptixProgramGroup> &program_groups)
{
	OptixProgramGroupOptions    miss_prog_group_options = {};
	OptixProgramGroupDesc       miss_prog_group_desc = {};
	miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	miss_prog_group_desc.miss.module = state.shading_module;
	miss_prog_group_desc.miss.entryFunctionName = "__miss__constant_bg";

	char    log[2048];
	size_t  sizeof_log = sizeof(log);
	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		state.context,
		&miss_prog_group_desc,
		1,
		&miss_prog_group_options,
		log,
		&sizeof_log,
		&state.radiance_miss_prog_group));

	miss_prog_group_desc.miss = {
		nullptr,    // module
		nullptr     // entryFunctionName
	};
	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		state.context,
		&miss_prog_group_desc,
		1,
		&miss_prog_group_options,
		log,
		&sizeof_log,
		&state.occlusion_miss_prog_group));
}

void createPipeline(WhittedState &state)
{
	std::vector<OptixProgramGroup> program_groups;

	state.pipeline_compile_options = {
		false,                                                  // usesMotionBlur
		OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,          // traversableGraphFlags
		5,    /* RadiancePRD uses 5 payloads */                 // numPayloadValues
		5,    /* Parallelogram intersection uses 5 attrs */     // numAttributeValues
		OPTIX_EXCEPTION_FLAG_NONE,                              // exceptionFlags
		"params"                                                // pipelineLaunchParamsVariableName
	};

	// Prepare program groups
	createModules(state);

	createVPLProgram(state, program_groups);	

	createKmeansProgram(state, program_groups);

	createCameraProgram(state, program_groups);
	createSphereProgram(state, program_groups);
	createPlaneProgram(state, program_groups);
	createMissProgram(state, program_groups);



	// Link program groups to pipeline
	OptixPipelineLinkOptions pipeline_link_options = {
		max_trace,                          // maxTraceDepth
		OPTIX_COMPILE_DEBUG_LEVEL_FULL,     // debugLevel
		false                               // overrideUsesMotionBlur
	};
	char    log[2048];
	size_t  sizeof_log = sizeof(log);
	OPTIX_CHECK_LOG(optixPipelineCreate(
		state.context,
		&state.pipeline_compile_options,
		&pipeline_link_options,
		program_groups.data(),
		static_cast<unsigned int>(program_groups.size()),
		log,
		&sizeof_log,
		&state.pipeline));


}

void syncCameraDataToSbt(WhittedState &state, const CameraData& camData)
{
	RayGenRecord rg_sbt;

	optixSbtRecordPackHeader(state.raygen_prog_group, &rg_sbt);
	rg_sbt.data = camData;

	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(state.sbt.raygenRecord),
		&rg_sbt,
		sizeof(RayGenRecord),
		cudaMemcpyHostToDevice
	));
}

void createSBT_K_means(WhittedState &state) {

	// Raygen program record for VPL
	//It could be an error coused by the initilization of the ray_gen vpl Program.
	{
		CUdeviceptr d_raygen_record = 0;
		size_t sizeof_raygen_record = sizeof(EmptyRecord);
		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&d_raygen_record),
			sizeof_raygen_record));


		EmptyRecord K_means_sbt;
		optixSbtRecordPackHeader(state.raygen_K_means_prog_group, &K_means_sbt);


		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_raygen_record),
			&K_means_sbt,
			sizeof_raygen_record,
			cudaMemcpyHostToDevice
		));

		state.sbt_K_means.raygenRecord = d_raygen_record;

	}
	// Miss program record
	{
		CUdeviceptr d_miss_record;
		size_t sizeof_miss_record = sizeof(EmptyRecord);
		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&d_miss_record),
			sizeof_miss_record));

		EmptyRecord ms_K_means_sbt;
		optixSbtRecordPackHeader(state.miss_K_means_prog_group, &ms_K_means_sbt);

		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_miss_record),
			&ms_K_means_sbt,
			sizeof_miss_record,
			cudaMemcpyHostToDevice
		));

		state.sbt_K_means.missRecordBase = d_miss_record;
		state.sbt_K_means.missRecordCount = 1;
		state.sbt_K_means.missRecordStrideInBytes = static_cast<uint32_t>(sizeof_miss_record);
	}

	// Hitgroup program record
	{
		const size_t count_records = OBJ_COUNT;
		HitGroupRecord hitgroup_records[count_records];

		// Note: Fill SBT record array the same order like AS is built.
		int sbt_idx = 0;

		// Hit Sphere
		OPTIX_CHECK(optixSbtRecordPackHeader(
			state.sphere_K_means_prog_group,
			&hitgroup_records[sbt_idx]));
		hitgroup_records[sbt_idx].data.geometry.sphere = g_sphere;
		hitgroup_records[sbt_idx].data.shading.diffuse = {
			{ 1.f, 1.f, 1.f },   // Kd	
		};

		sbt_idx++;

		//---------------------------------------------------------------

		OPTIX_CHECK(optixSbtRecordPackHeader(
			state.sphere_K_means_prog_group,
			&hitgroup_records[sbt_idx]));
		hitgroup_records[sbt_idx].data.geometry.sphere = g_sphere_2;
		hitgroup_records[sbt_idx].data.shading.diffuse = {
			{  1.f, 1.f, 1.f },   // Kd			                
		};
		sbt_idx++;

		//---------------------------------------------------------------


		//Floor
		OPTIX_CHECK(optixSbtRecordPackHeader(
			state.plane_K_means_prog_group,
			&hitgroup_records[sbt_idx]));
		hitgroup_records[sbt_idx].data.geometry.plane = g_floor;
		hitgroup_records[sbt_idx].data.shading.diffuse = {
			{ 0.8f, 0.8f, 0.8f },      // Kd1			
		};
		sbt_idx++;

		//---------------------------------------------------------------


		// Wall back
		OPTIX_CHECK(optixSbtRecordPackHeader(
			state.plane_K_means_prog_group,
			&hitgroup_records[sbt_idx]));
		hitgroup_records[sbt_idx].data.geometry.plane = g_floor_back;
		hitgroup_records[sbt_idx].data.shading.diffuse = {
			{ 0.8f, 0.8f, 0.8f },      // Kd1			
		};
		sbt_idx++;


		//---------------------------------------------------------------
		// Wall left
		OPTIX_CHECK(optixSbtRecordPackHeader(
			state.plane_K_means_prog_group,
			&hitgroup_records[sbt_idx]));
		hitgroup_records[sbt_idx].data.geometry.plane = g_left_wall;
		hitgroup_records[sbt_idx].data.shading.diffuse = {
			{ 0.f, 0.8f, 0.f },      // Kd1			

		};
		sbt_idx++;


		// Wall right
		OPTIX_CHECK(optixSbtRecordPackHeader(
			state.plane_K_means_prog_group,
			&hitgroup_records[sbt_idx]));
		hitgroup_records[sbt_idx].data.geometry.plane = g_right_wall;
		hitgroup_records[sbt_idx].data.shading.diffuse = {
			 { 0.8f, 0.f, 0.f },      // Kd1		
		};
		sbt_idx++;


		// Wall ceil
		OPTIX_CHECK(optixSbtRecordPackHeader(
			state.plane_K_means_prog_group,
			&hitgroup_records[sbt_idx]));
		hitgroup_records[sbt_idx].data.geometry.plane = g_ceil;
		hitgroup_records[sbt_idx].data.shading.diffuse = {
			{ 0.8f, 0.8f, 0.8f },      // Kd1			
		};
		sbt_idx++;

		CUdeviceptr d_hitgroup_records;
		size_t      sizeof_hitgroup_record = sizeof(HitGroupRecord);
		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&d_hitgroup_records),
			sizeof_hitgroup_record*count_records
		));

		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_hitgroup_records),
			hitgroup_records,
			sizeof_hitgroup_record*count_records,
			cudaMemcpyHostToDevice
		));

		state.sbt_K_means.hitgroupRecordBase = d_hitgroup_records;
		state.sbt_K_means.hitgroupRecordCount = count_records;
		state.sbt_K_means.hitgroupRecordStrideInBytes = static_cast<uint32_t>(sizeof_hitgroup_record);
	}

}


void createSBT_VPL(WhittedState &state) {

	// Raygen program record for VPL
	//It could be an error coused by the initilization of the ray_gen vpl Program.
	{
		CUdeviceptr d_raygen_record = 0;
		size_t sizeof_raygen_record = sizeof(EmptyRecord);
		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&d_raygen_record),
			sizeof_raygen_record));


		EmptyRecord vpl_sbt;
		optixSbtRecordPackHeader(state.raygen_VPL_prog_group, &vpl_sbt);


		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_raygen_record),
			&vpl_sbt,
			sizeof_raygen_record,
			cudaMemcpyHostToDevice
		));

		state.sbt_VPL.raygenRecord = d_raygen_record;

	}
	// Miss program record
	{
		CUdeviceptr d_miss_record;
		size_t sizeof_miss_record = sizeof(EmptyRecord);
		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&d_miss_record),
			sizeof_miss_record));

		EmptyRecord ms_vpl_sbt;
		optixSbtRecordPackHeader(state.miss_vpl_prog_group, &ms_vpl_sbt);

		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_miss_record),
			&ms_vpl_sbt,
			sizeof_miss_record,
			cudaMemcpyHostToDevice
		));

		state.sbt_VPL.missRecordBase = d_miss_record;
		state.sbt_VPL.missRecordCount = 1;
		state.sbt_VPL.missRecordStrideInBytes = static_cast<uint32_t>(sizeof_miss_record);
	}

	// Hitgroup program record
	{
		const size_t count_records = OBJ_COUNT;
		HitGroupRecord hitgroup_records[count_records];

		// Note: Fill SBT record array the same order like AS is built.
		int sbt_idx = 0;

		// Hit Sphere
		OPTIX_CHECK(optixSbtRecordPackHeader(
			state.sphere_vpl_prog_group,
			&hitgroup_records[sbt_idx]));
		hitgroup_records[sbt_idx].data.geometry.sphere = g_sphere;
		hitgroup_records[sbt_idx].data.shading.diffuse = {
			{ 1.f, 1.f, 1.f },   // Kd	
		};

		sbt_idx++;

		//---------------------------------------------------------------

		OPTIX_CHECK(optixSbtRecordPackHeader(
			state.sphere_vpl_prog_group,
			&hitgroup_records[sbt_idx]));
		hitgroup_records[sbt_idx].data.geometry.sphere = g_sphere_2;
		hitgroup_records[sbt_idx].data.shading.diffuse = {
			{  1.f, 1.f, 1.f },   // Kd			                
		};
		sbt_idx++;

		//---------------------------------------------------------------


		//Floor
		OPTIX_CHECK(optixSbtRecordPackHeader(
			state.plane_vpl_prog_group,
			&hitgroup_records[sbt_idx]));
		hitgroup_records[sbt_idx].data.geometry.plane = g_floor;
		hitgroup_records[sbt_idx].data.shading.diffuse = {
			{ 0.8f, 0.8f, 0.8f },      // Kd1			
		};
		sbt_idx++;

		//---------------------------------------------------------------


		// Wall back
		OPTIX_CHECK(optixSbtRecordPackHeader(
			state.plane_vpl_prog_group,
			&hitgroup_records[sbt_idx]));
		hitgroup_records[sbt_idx].data.geometry.plane = g_floor_back;
		hitgroup_records[sbt_idx].data.shading.diffuse = {
			{ 0.8f, 0.8f, 0.8f },      // Kd1			
		};
		sbt_idx++;


		//---------------------------------------------------------------
		// Wall left
		OPTIX_CHECK(optixSbtRecordPackHeader(
			state.plane_vpl_prog_group,
			&hitgroup_records[sbt_idx]));
		hitgroup_records[sbt_idx].data.geometry.plane = g_left_wall;
		hitgroup_records[sbt_idx].data.shading.diffuse = {
			{ 0.f, 0.8f, 0.f },      // Kd1			

		};
		sbt_idx++;


		// Wall right
		OPTIX_CHECK(optixSbtRecordPackHeader(
			state.plane_vpl_prog_group,
			&hitgroup_records[sbt_idx]));
		hitgroup_records[sbt_idx].data.geometry.plane = g_right_wall;
		hitgroup_records[sbt_idx].data.shading.diffuse = {
			 { 0.8f, 0.f, 0.f },      // Kd1		
		};
		sbt_idx++;


		// Wall ceil
		OPTIX_CHECK(optixSbtRecordPackHeader(
			state.plane_vpl_prog_group,
			&hitgroup_records[sbt_idx]));
		hitgroup_records[sbt_idx].data.geometry.plane = g_ceil;
		hitgroup_records[sbt_idx].data.shading.diffuse = {
			{ 0.8f, 0.8f, 0.8f },      // Kd1			
		};
		sbt_idx++;

		CUdeviceptr d_hitgroup_records;
		size_t      sizeof_hitgroup_record = sizeof(HitGroupRecord);
		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&d_hitgroup_records),
			sizeof_hitgroup_record*count_records
		));

		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_hitgroup_records),
			hitgroup_records,
			sizeof_hitgroup_record*count_records,
			cudaMemcpyHostToDevice
		));

		state.sbt_VPL.hitgroupRecordBase = d_hitgroup_records;
		state.sbt_VPL.hitgroupRecordCount = count_records;
		state.sbt_VPL.hitgroupRecordStrideInBytes = static_cast<uint32_t>(sizeof_hitgroup_record);
	}

}

void createSBT(WhittedState &state)
{
	// Raygen program record
	{
		CUdeviceptr d_raygen_record;
		size_t sizeof_raygen_record = sizeof(RayGenRecord);
		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&d_raygen_record),
			sizeof_raygen_record));

		state.sbt.raygenRecord = d_raygen_record;

	}

	// Miss program record
	{
		CUdeviceptr d_miss_record;
		size_t sizeof_miss_record = sizeof(MissRecord);
		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&d_miss_record),
			sizeof_miss_record*RAY_TYPE_COUNT));

		MissRecord ms_sbt[RAY_TYPE_COUNT];
		optixSbtRecordPackHeader(state.radiance_miss_prog_group, &ms_sbt[0]);
		optixSbtRecordPackHeader(state.occlusion_miss_prog_group, &ms_sbt[1]);
		ms_sbt[1].data = ms_sbt[0].data = { 0.f, 0.f, 0.f };

		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_miss_record),
			ms_sbt,
			sizeof_miss_record*RAY_TYPE_COUNT,
			cudaMemcpyHostToDevice
		));

		state.sbt.missRecordBase = d_miss_record;
		state.sbt.missRecordCount = RAY_TYPE_COUNT;
		state.sbt.missRecordStrideInBytes = static_cast<uint32_t>(sizeof_miss_record);
	}

	// Hitgroup program record
	{
		const size_t count_records = OBJ_COUNT * RAY_TYPE_COUNT;
		HitGroupRecord hitgroup_records[count_records];

		// Note: Fill SBT record array the same order like AS is built.
		int sbt_idx = 0;

		// Hit Sphere
		OPTIX_CHECK(optixSbtRecordPackHeader(
			state.radiance_sphere_prog_group,
			&hitgroup_records[sbt_idx]));
		hitgroup_records[sbt_idx].data.geometry.sphere = g_sphere;
		hitgroup_records[sbt_idx].data.shading.diffuse = {
			{ 1.f, 1.f, 1.f },   // Kd

		};
		sbt_idx++;
		OPTIX_CHECK(optixSbtRecordPackHeader(
			state.occlusion_sphere_prog_group,
			&hitgroup_records[sbt_idx]));
		hitgroup_records[sbt_idx].data.geometry.sphere = g_sphere;
		sbt_idx++;
		//---------------------------------------------------------------

		OPTIX_CHECK(optixSbtRecordPackHeader(
			state.radiance_sphere_prog_group,
			&hitgroup_records[sbt_idx]));
		hitgroup_records[sbt_idx].data.geometry.sphere = g_sphere_2;
		hitgroup_records[sbt_idx].data.shading.diffuse = {
			{ 1.f, 1.f, 1.f },   // Kd
		};
		sbt_idx++;
		OPTIX_CHECK(optixSbtRecordPackHeader(
			state.occlusion_sphere_prog_group,
			&hitgroup_records[sbt_idx]));
		hitgroup_records[sbt_idx].data.geometry.sphere = g_sphere_2;
		sbt_idx++;
		//---------------------------------------------------------------


		//Floor
		OPTIX_CHECK(optixSbtRecordPackHeader(
			state.radiance_plane_prog_group,
			&hitgroup_records[sbt_idx]));
		hitgroup_records[sbt_idx].data.geometry.plane = g_floor;
		hitgroup_records[sbt_idx].data.shading.diffuse = {
			{ 0.8f, 0.8f, 0.8f },      // Kd1				
		};
		sbt_idx++;
		OPTIX_CHECK(optixSbtRecordPackHeader(
			state.occlusion_plane_prog_group,
			&hitgroup_records[sbt_idx]));
		hitgroup_records[sbt_idx].data.geometry.plane = g_floor;
		sbt_idx++;
		//---------------------------------------------------------------


		// Wall back
		OPTIX_CHECK(optixSbtRecordPackHeader(
			state.radiance_plane_prog_group,
			&hitgroup_records[sbt_idx]));
		hitgroup_records[sbt_idx].data.geometry.plane = g_floor_back;
		hitgroup_records[sbt_idx].data.shading.diffuse = {
			{ 0.8f, 0.8f, 0.8f },      // Kd1		
		};
		sbt_idx++;

		OPTIX_CHECK(optixSbtRecordPackHeader(
			state.occlusion_plane_prog_group,
			&hitgroup_records[sbt_idx]));
		hitgroup_records[sbt_idx].data.geometry.plane = g_floor_back;
		sbt_idx++;
		//---------------------------------------------------------------
		// Wall left
		OPTIX_CHECK(optixSbtRecordPackHeader(
			state.radiance_plane_prog_group,
			&hitgroup_records[sbt_idx]));
		hitgroup_records[sbt_idx].data.geometry.plane = g_left_wall;
		hitgroup_records[sbt_idx].data.shading.diffuse = {
			{ 0.f, 0.8f, 0.f },      // Kd1		
		};
		sbt_idx++;

		OPTIX_CHECK(optixSbtRecordPackHeader(
			state.occlusion_plane_prog_group,
			&hitgroup_records[sbt_idx]));
		hitgroup_records[sbt_idx].data.geometry.plane = g_left_wall;
		sbt_idx++;
		//---------------------------------------------------------------
		// Wall right
		OPTIX_CHECK(optixSbtRecordPackHeader(
			state.radiance_plane_prog_group,
			&hitgroup_records[sbt_idx]));
		hitgroup_records[sbt_idx].data.geometry.plane = g_right_wall;
		hitgroup_records[sbt_idx].data.shading.diffuse = {
			{ 0.8f, 0.f, 0.f },      // Kd1		
		};
		sbt_idx++;

		OPTIX_CHECK(optixSbtRecordPackHeader(
			state.occlusion_plane_prog_group,
			&hitgroup_records[sbt_idx]));
		hitgroup_records[sbt_idx].data.geometry.plane = g_right_wall;
		sbt_idx++;
		//---------------------------------------------------------------
		// Wall ceil
		OPTIX_CHECK(optixSbtRecordPackHeader(
			state.radiance_plane_prog_group,
			&hitgroup_records[sbt_idx]));
		hitgroup_records[sbt_idx].data.geometry.plane = g_ceil;
		hitgroup_records[sbt_idx].data.shading.diffuse = {
			{ 0.8f, 0.8f, 0.8f },      // Kd	
		};
		sbt_idx++;

		OPTIX_CHECK(optixSbtRecordPackHeader(
			state.occlusion_plane_prog_group,
			&hitgroup_records[sbt_idx]));
		hitgroup_records[sbt_idx].data.geometry.plane = g_ceil;
		sbt_idx++;
		//---------------------------------------------------------------

		//---------------------------------------------------------------

		CUdeviceptr d_hitgroup_records;
		size_t      sizeof_hitgroup_record = sizeof(HitGroupRecord);
		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&d_hitgroup_records),
			sizeof_hitgroup_record*count_records
		));

		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_hitgroup_records),
			hitgroup_records,
			sizeof_hitgroup_record*count_records,
			cudaMemcpyHostToDevice
		));

		state.sbt.hitgroupRecordBase = d_hitgroup_records;
		state.sbt.hitgroupRecordCount = count_records;
		state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(sizeof_hitgroup_record);
	}
}

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
	std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
		<< message << "\n";
}

void createContext(WhittedState& state)
{
	// Initialize CUDA
	CUDA_CHECK(cudaFree(0));

	OptixDeviceContext context;
	CUcontext          cuCtx = 0;  // zero means take the current context
	OPTIX_CHECK(optixInit());
	OptixDeviceContextOptions options = {};
	options.logCallbackFunction = &context_log_cb;
	options.logCallbackLevel = 4;
	OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));

	state.context = context;
}


void initCameraState()
{
	camera.setEye(make_float3(14.0f, 3.3f, -1.0f));
	camera.setLookat(make_float3(4.0f, 3.f, -1.0f));
	camera.setUp(make_float3(0.0f, 1.0f, 0.0f));
	camera.setFovY(60.0f);
	camera_changed = true;

	trackball.setCamera(&camera);
	trackball.setMoveSpeed(10.0f);
	trackball.setReferenceFrame(make_float3(1.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 1.0f), make_float3(0.0f, 1.0f, 0.0f));
	trackball.setGimbalLock(true);
}

void handleCameraUpdate(WhittedState &state)
{
	if (!camera_changed)
		return;
	camera_changed = false;

	camera.setAspectRatio(static_cast<float>(state.params.width) / static_cast<float>(state.params.height));
	CameraData camData;
	camData.eye = camera.eye();
	camera.UVWFrame(camData.U, camData.V, camData.W);

	syncCameraDataToSbt(state, camData);
}

void handleResize(sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params)
{
	if (!resize_dirty)
		return;
	resize_dirty = false;

	output_buffer.resize(params.width, params.height);


	// Realloc accumulation buffer
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.accum_buffer)));
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&params.accum_buffer),
		params.width*params.height * sizeof(float4)
	));
}

void updateState(sutil::CUDAOutputBuffer<uchar4>& output_buffer, WhittedState &state)
{
	// Update params on device
	if (camera_changed || resize_dirty)
		state.params.subframe_index = 0;

	state.params.s_d = show_direct; //Boolean to show the direct illumination
	state.params.s_i = show_indirect; //Boolean to show the indirect illumiantion
	state.params.s_v = show_vpl; //Boolean to show where is the diferent VPL

	state.params.s_k = show_K_space; //Bolean to show the clusterization of the space

	state.params.show_R_matrix = show_R_matrix;//Boolean to show the R matrix


	state.params.minSS = SSmin; //Variable to determinato how the stepfunction will behave
	state.params.maxSS = SSmax; //Variable to determinato how the stepfunction will behave


	state.params.result_K_means = apply_k_means;
	state.params.show_cluster_VPL = show_cluster_VPL_bool;

	handleCameraUpdate(state);
	handleResize(output_buffer, state.params);

}


void compute_image(sutil::CUDAOutputBuffer<uchar4>& output_buffer, WhittedState& state) {
	state.params.compute_image = true;

	uchar4* result_buffer_data = output_buffer.map();
	state.params.frame_buffer = result_buffer_data;
	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.d_params),
		&state.params,
		sizeof(Params),
		cudaMemcpyHostToDevice,
		state.stream
	));
	//Launch one ray per pixel
	OPTIX_CHECK(optixLaunch(
		state.pipeline,
		state.stream,
		reinterpret_cast<CUdeviceptr>(state.d_params),
		sizeof(Params),
		&state.sbt,
		state.params.width,  // launch width
		state.params.height, // launch height
		1                    // launch depth
	));
	output_buffer.unmap();
	CUDA_SYNC_CHECK();

	state.params.compute_image = false;


}

void k_means_select_points(WhittedState& state) {

	state.params.select_space_points  = true;

	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.d_params),
		&state.params,
		sizeof(Params),
		cudaMemcpyHostToDevice,
		state.stream
	));

	//Launch one ray per pixel
	OPTIX_CHECK(optixLaunch(
		state.pipeline,
		state.stream,
		reinterpret_cast<CUdeviceptr>(state.d_params),
		sizeof(Params),
		&state.sbt,
		state.params.width,  // launch width
		state.params.height, // launch height
		1                    // launch depth
	));
	CUDA_SYNC_CHECK();

	state.params.select_space_points = false;
}

extern "C" void create_centroids_space_device(
	cudaStream_t stream, Params* params, int32_t  width, int32_t  height);

void create_points_centroids(WhittedState& state) {

	state.params.init_centroid_points = true;

	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.d_params),
		&state.params,
		sizeof(Params),
		cudaMemcpyHostToDevice,
		state.stream
	));

	//Launch one ray per pixel
	OPTIX_CHECK(optixLaunch(
		state.pipeline,
		state.stream,
		reinterpret_cast<CUdeviceptr>(state.d_params),
		sizeof(Params),
		&state.sbt_K_means,
		K_POINTS_CLUSTER,  // launch width
		1, // launch height
		1                    // launch depth
	));
	CUDA_SYNC_CHECK();

	state.params.init_centroid_points = false;



	//create_centroids_space_device(state.stream, state.d_params, K_POINTS_CLUSTER, 1);

}

void assing_points_cluster(WhittedState& state) {

	state.params.assing_cluster_points = true;
	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.d_params),
		&state.params,
		sizeof(Params),
		cudaMemcpyHostToDevice,
		state.stream
	));

	//Launch one ray per pixel
	OPTIX_CHECK(optixLaunch(
		state.pipeline,
		state.stream,
		reinterpret_cast<CUdeviceptr>(state.d_params),
		sizeof(Params),
		&state.sbt_K_means,
		state.params.width,  // launch width
		state.params.height, // launch height
		1                    // launch depth
	));

	CUDA_SYNC_CHECK();
	state.params.assing_cluster_points = false;

}

void recompute_centroids(WhittedState& state) {

	state.params.recompute_cluster_points = true;

	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.d_params),
		&state.params,
		sizeof(Params),
		cudaMemcpyHostToDevice,
		state.stream
	));

	//Launch one ray per pixel
	OPTIX_CHECK(optixLaunch(
		state.pipeline,
		state.stream,
		reinterpret_cast<CUdeviceptr>(state.d_params),
		sizeof(Params),
		&state.sbt_K_means,
		K_POINTS_CLUSTER,  // launch width
		1, // launch height
		1                    // launch depth
	));
	CUDA_SYNC_CHECK();
	state.params.recompute_cluster_points = false;
}

void compute_R_matrix(WhittedState& state) {

	state.params.compute_R = true;
	//Columns: VPL---- rows : points
	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.d_params),
		&state.params,
		sizeof(Params),
		cudaMemcpyHostToDevice,
		state.stream
	));

	//Launch one ray per pixel
	OPTIX_CHECK(optixLaunch(
		state.pipeline,
		state.stream,
		reinterpret_cast<CUdeviceptr>(state.d_params),
		sizeof(Params),
		&state.sbt,
		K_POINTS_CLUSTER,  // launch width
		1, // launch height
		1                    // launch depth
	));
	CUDA_SYNC_CHECK();

	state.params.compute_R = false;
}

void assing_VPL(WhittedState& state){

	//Columns: VPL---- rows : points
	state.params.assing_VPL_bool = true;
	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.d_params),
		&state.params,
		sizeof(Params),
		cudaMemcpyHostToDevice,
		state.stream
	));

	//Launch one ray per pixel
	OPTIX_CHECK(optixLaunch(
		state.pipeline,
		state.stream,
		reinterpret_cast<CUdeviceptr>(state.d_params),
		sizeof(Params),
		&state.sbt_K_means,
		state.params.num_hit_vpl,  // launch width
		1, // launch height
		1                    // launch depth
	));
	CUDA_SYNC_CHECK();
	state.params.assing_VPL_bool = false;

}



void compute_cluster_distances(WhittedState& state) {

	state.params.local_slice_compute_distances_bool = true;
	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.d_params),
		&state.params,
		sizeof(Params),
		cudaMemcpyHostToDevice,
		state.stream
	));

	//Launch one ray per pixel
	OPTIX_CHECK(optixLaunch(
		state.pipeline,
		state.stream,
		reinterpret_cast<CUdeviceptr>(state.d_params),
		sizeof(Params),
		&state.sbt_K_means,
		K_POINTS_CLUSTER,  // launch width
		1, // launch height
		1                    // launch depth
	));
	CUDA_SYNC_CHECK();
	state.params.local_slice_compute_distances_bool = false;
}

void select_closest_clusters(WhittedState& state) {
	state.params.select_closest_clusters_bool = true;
	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.d_params),
		&state.params,
		sizeof(Params),
		cudaMemcpyHostToDevice,
		state.stream
	));

	//Launch one ray per pixel
	OPTIX_CHECK(optixLaunch(
		state.pipeline,
		state.stream,
		reinterpret_cast<CUdeviceptr>(state.d_params),
		sizeof(Params),
		&state.sbt,
		K_POINTS_CLUSTER,  // launch width
		1, // launch heightstate.params.num_vpl *(state.params.max_bounces + 1)
		1                    // launch depth
	));
	CUDA_SYNC_CHECK();
	state.params.select_closest_clusters_bool = false;
}

void compute_L_i_modules(WhittedState& state) {
	//Columns: VPL---- rows : points
	state.params.compute__Li_modules_bool = true;
	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.d_params),
		&state.params,
		sizeof(Params),
		cudaMemcpyHostToDevice,
		state.stream
	));

	//Launch one ray per pixel
	OPTIX_CHECK(optixLaunch(
		state.pipeline,
		state.stream,
		reinterpret_cast<CUdeviceptr>(state.d_params),
		sizeof(Params),
		&state.sbt_K_means,
		K_POINTS_CLUSTER,  // launch width
		1, // launch heightstate.params.num_vpl *(state.params.max_bounces + 1)
		1                    // launch depth
	));
	CUDA_SYNC_CHECK();

	state.params.compute__Li_modules_bool = false;
}

void select_cheap_clusters(WhittedState& state) {
	state.params.select_cheap_cluster_bool = true;
	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.d_params),
		&state.params,
		sizeof(Params),
		cudaMemcpyHostToDevice,
		state.stream
	));

	//Launch one ray per pixel
	OPTIX_CHECK(optixLaunch(
		state.pipeline,
		state.stream,
		reinterpret_cast<CUdeviceptr>(state.d_params),
		sizeof(Params),
		&state.sbt_K_means,
		K_POINTS_CLUSTER,  // launch width
		1, // launch heightstate.params.num_vpl *(state.params.max_bounces + 1)
		1                    // launch depth
	));
	CUDA_SYNC_CHECK();

	state.params.select_cheap_cluster_bool = false;
}

void QT_clustering(WhittedState& state) {
	state.params.compute_QT = true;


	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.d_params),
		&state.params,
		sizeof(Params),
		cudaMemcpyHostToDevice,
		state.stream
	));

	//Launch one ray per pixel
	OPTIX_CHECK(optixLaunch(
		state.pipeline,
		state.stream,
		reinterpret_cast<CUdeviceptr>(state.d_params),
		sizeof(Params),
		&state.sbt_K_means,
		state.params.width ,  // launch width
		state.params.height, // launch heightstate.params.num_vpl *(state.params.max_bounces + 1)
		1                    // launch depth
	));
	CUDA_SYNC_CHECK();


	state.params.compute_QT = false;

}

void k_means_space(WhittedState& state) {

	if (state.params.subframe_index == 0) {
		k_means_select_points(state);
		create_points_centroids(state);
	}


	//assing_points_cluster(state);
	//recompute_centroids(state);
	state.params.Num_points = state.params.width *state.params.height;
	QT_clustering(state);

	//compute_R_matrix(state);

	//assing_VPL(state);

	//compute_cluster_distances(state);
	//select_closest_clusters(state);

	//compute_L_i_modules(state);
	//select_cheap_clusters(state);


}

void launchSubframe(sutil::CUDAOutputBuffer<uchar4>& output_buffer, WhittedState& state)
{
	
	if (apply_k_means) {
		k_means_space(state);
	}

	compute_image(output_buffer, state);	
	//state.params.show_reduced_indirect = false;
}

void count_VPLs(WhittedState& state) {


	int num_hit_vpl;

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_params), sizeof(Params)));

	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.d_params),
		&state.params,
		sizeof(Params),
		cudaMemcpyHostToDevice,
		state.stream
	));

	CUDA_CHECK(cudaMemcpyAsync((&num_hit_vpl),
		(state.params.number_hit_vpl),
		sizeof(int),
		cudaMemcpyDeviceToHost,
		state.stream
	));

	state.params.num_hit_vpl = num_hit_vpl;

}

void launchVPL(WhittedState& state) {

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_params), sizeof(Params)));

	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.d_params),
		&state.params,
		sizeof(Params),
		cudaMemcpyHostToDevice,
		state.stream
	));
	//Launch as many rays as VPL
	OPTIX_CHECK(optixLaunch(
		state.pipeline,
		state.stream,
		reinterpret_cast<CUdeviceptr>(state.d_params),
		sizeof(Params),
		&state.sbt_VPL,
		state.params.num_vpl,  
		1,						
		1						
	));
	CUDA_SYNC_CHECK();

	state.params.count_VPL_hit = true;
	//Count VPL
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_params), sizeof(Params)));

	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.d_params),
		&state.params,
		sizeof(Params),
		cudaMemcpyHostToDevice,
		state.stream
	));

	//Launch as many rays as VPL
	OPTIX_CHECK(optixLaunch(
		state.pipeline,
		state.stream,
		reinterpret_cast<CUdeviceptr>(state.d_params),
		sizeof(Params),
		&state.sbt_K_means,
		1,
		1,
		1
	));
	CUDA_SYNC_CHECK();

	state.params.count_VPL_hit = false;
	count_VPLs(state);

	SetBuffers_VPL(state);

	//Save VPL that hit the scene
	state.params.save_relevant_VPL = true;
	

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_params), sizeof(Params)));

	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.d_params),
		&state.params,
		sizeof(Params),
		cudaMemcpyHostToDevice,
		state.stream
	));

	//Launch as many rays as VPL
	OPTIX_CHECK(optixLaunch(
		state.pipeline,
		state.stream,
		reinterpret_cast<CUdeviceptr>(state.d_params),
		sizeof(Params),
		&state.sbt_K_means,
		1,
		1,
		1
	));

	CUDA_SYNC_CHECK();
	state.params.save_relevant_VPL = false;
}

void createClusterColors(WhittedState& state) {

	for (int i = 0; i < K_POINTS_CLUSTER; i++) {

		float mod_x = rand() % 1000;
		float num_x = mod_x / 1000;

		float mod_y = rand() % 1000;
		float num_y = mod_y / 1000;

		float mod_z = rand() % 1000;
		float num_z = mod_z / 1000;

		state.params.cluster_color[i] = make_float3(num_x, num_y, num_z);
	}
}


void create_centroid(WhittedState& state) {

	for (int i = 0; i < K_POINTS_CLUSTER; i++) {

		int rn_x = rand() % state.params.width;
		int rn_y = rand() % state.params.height;

		int centroid_indx = state.params.width*rn_y + rn_x;
		state.params.position_cluster[i] = centroid_indx;

	}
}


void create_VPL_init_cluster(WhittedState& state) {

	for (int i = 0; i < K_MEANS_VPL; i++) {

		int rn = rand() % state.params.num_hit_vpl;
		state.params.first_VPL_cluster[i] = rn;
	}


}


void displaySubframe(
	sutil::CUDAOutputBuffer<uchar4>&  output_buffer,
	sutil::GLDisplay&                 gl_display,
	GLFWwindow*                       window)
{
	// Display
	int framebuf_res_x = 0;   // The display's resolution (could be HDPI res)
	int framebuf_res_y = 0;   //
	glfwGetFramebufferSize(window, &framebuf_res_x, &framebuf_res_y);
	gl_display.display(
		output_buffer.width(),
		output_buffer.height(),
		framebuf_res_x,
		framebuf_res_y,
		output_buffer.getPBO()
	);
}




void cleanupState(WhittedState& state)
{
	OPTIX_CHECK(optixPipelineDestroy(state.pipeline));

	OPTIX_CHECK(optixProgramGroupDestroy(state.raygen_prog_group));

	OPTIX_CHECK(optixProgramGroupDestroy(state.radiance_sphere_prog_group));
	OPTIX_CHECK(optixProgramGroupDestroy(state.radiance_plane_prog_group));
	OPTIX_CHECK(optixProgramGroupDestroy(state.radiance_miss_prog_group));

	OPTIX_CHECK(optixProgramGroupDestroy(state.occlusion_sphere_prog_group));
	OPTIX_CHECK(optixProgramGroupDestroy(state.occlusion_plane_prog_group));
	OPTIX_CHECK(optixProgramGroupDestroy(state.occlusion_miss_prog_group));

	//VPL
	OPTIX_CHECK(optixProgramGroupDestroy(state.raygen_VPL_prog_group));
	OPTIX_CHECK(optixProgramGroupDestroy(state.sphere_vpl_prog_group));
	OPTIX_CHECK(optixProgramGroupDestroy(state.plane_vpl_prog_group));
	OPTIX_CHECK(optixProgramGroupDestroy(state.miss_vpl_prog_group));

	//K_means
	OPTIX_CHECK(optixProgramGroupDestroy(state.raygen_K_means_prog_group));
	OPTIX_CHECK(optixProgramGroupDestroy(state.sphere_K_means_prog_group));
	OPTIX_CHECK(optixProgramGroupDestroy(state.plane_K_means_prog_group));
	OPTIX_CHECK(optixProgramGroupDestroy(state.miss_K_means_prog_group));

	//MOdules
	OPTIX_CHECK(optixModuleDestroy(state.shading_module));
	OPTIX_CHECK(optixModuleDestroy(state.geometry_module));
	OPTIX_CHECK(optixModuleDestroy(state.camera_module));
	OPTIX_CHECK(optixModuleDestroy(state.VPL_module));
	OPTIX_CHECK(optixModuleDestroy(state.K_means_module));

	OPTIX_CHECK(optixDeviceContextDestroy(state.context));


	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.raygenRecord)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.missRecordBase)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.hitgroupRecordBase)));

	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt_K_means.raygenRecord)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt_K_means.missRecordBase)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt_K_means.hitgroupRecordBase)));

	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt_VPL.raygenRecord)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt_VPL.missRecordBase)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt_VPL.hitgroupRecordBase)));


	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_gas_output_buffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.params.accum_buffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_params)));
}




int main(int argc, char* argv[])
{
	WhittedState state;
	state.params.width = 1000;
	state.params.height = 720;
	sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;
	//cleanupState(state);


	//
	// Parse command line options
	//
	std::string outfile;

	for (int i = 1; i < argc; ++i)
	{
		const std::string arg = argv[i];
		if (arg == "--help" || arg == "-h")
		{
			printUsageAndExit(argv[0]);
		}
		else if (arg == "--no-gl-interop")
		{
			output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
		}
		else if (arg == "--file" || arg == "-f")
		{
			if (i >= argc - 1)
				printUsageAndExit(argv[0]);
			outfile = argv[++i];
		}
		else
		{
			std::cerr << "Unknown option '" << argv[i] << "'\n";
			printUsageAndExit(argv[0]);
		}
	}
	int i = 0;
	try
	{
		initCameraState();	
		

		//
		// Set up OptiX state
		//
		createContext(state);
		createGeomety(state);
		createPipeline(state);
		createSBT(state);
		createSBT_VPL(state);//SBT for VPL
		createSBT_K_means(state);//SBT FOR K MEANS


		initLaunchParams(state);		


		//Init booleans
		state.params.compute_image = false;
		state.params.count_VPL_hit = false;
		state.params.save_relevant_VPL = false;

		state.params.k_means_space = false;
		state.params.select_space_points = false;
		state.params.init_centroid_points = false;

		state.params.assing_cluster_points = false;
		state.params.recompute_cluster_points = false;
		state.params.compute_R = false;

		state.params.assing_VPL_bool = false;
		state.params.init_vpl_centroids_bool = false;//**No use now
		state.params.recompute_VPL_cent_bool = false;//** No use now

		state.params.local_slice_compute_distances_bool = false;
		state.params.select_closest_clusters_bool = false;

		state.params.compute__Li_modules_bool = false;
		state.params.select_cheap_cluster_bool = false;
		state.params.result_K_means = false;

		state.params.compute_QT = false;



		launchVPL(state);//Launch vpl

		createClusterColors(state);
		create_centroid(state);
		create_VPL_init_cluster(state);

		




		//
		// Render loop
		//
		if (outfile.empty())
		{
			GLFWwindow* window = sutil::initUI("optixHello", state.params.width, state.params.height);
			glfwSetMouseButtonCallback(window, mouseButtonCallback);
			glfwSetCursorPosCallback(window, cursorPosCallback);
			glfwSetWindowSizeCallback(window, windowSizeCallback);
			glfwSetKeyCallback(window, keyCallback);
			glfwSetScrollCallback(window, scrollCallback);
			glfwSetWindowUserPointer(window, &state.params);


			{
				// output_buffer needs to be destroyed before cleanupUI is called
				sutil::CUDAOutputBuffer<uchar4> output_buffer(
					output_buffer_type,
					state.params.width,
					state.params.height
				);

				output_buffer.setStream(state.stream);
				sutil::GLDisplay gl_display;		


				std::chrono::duration<double> state_update_time(0.0);
				std::chrono::duration<double> render_time(0.0);
				std::chrono::duration<double> display_time(0.0);



				do
				{
					auto t0 = std::chrono::steady_clock::now();
					glfwPollEvents();

					updateState(output_buffer, state);//bool light :)
					auto t1 = std::chrono::steady_clock::now();
					state_update_time += t1 - t0;
					t0 = t1;

					launchSubframe(output_buffer, state);
					t1 = std::chrono::steady_clock::now();
					render_time += t1 - t0;
					t0 = t1;

					displaySubframe(output_buffer, gl_display, window);
					t1 = std::chrono::steady_clock::now();
					display_time += t1 - t0;
					sutil::displayStats(state_update_time, render_time, display_time);


					glfwSwapBuffers(window);

					++state.params.subframe_index;


				} while (!glfwWindowShouldClose(window));

			}
			sutil::cleanupUI(window);
		}
		else
		{
			if (output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP)
			{
				sutil::initGLFW(); // For GL context
				sutil::initGL();
			}




			sutil::CUDAOutputBuffer<uchar4> output_buffer(
				output_buffer_type,
				state.params.width,
				state.params.height
			);

			handleCameraUpdate(state);
			handleResize(output_buffer, state.params);
			launchSubframe(output_buffer, state);

			sutil::ImageBuffer buffer;
			buffer.data = output_buffer.getHostPointer();
			buffer.width = output_buffer.width();
			buffer.height = output_buffer.height();
			buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

			sutil::displayBufferFile(outfile.c_str(), buffer, false);
			char str_min = (char)state.params.minSS;
			char str_max = (char)state.params.maxSS;


			if (output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP)
			{
				glfwTerminate();
			}
		}

		cleanupState(state);
	}
	catch (std::exception& e)
	{
		std::cerr << "Caught exception: " << e.what() << "\n";
		return 1;
	}

	return 0;
}
