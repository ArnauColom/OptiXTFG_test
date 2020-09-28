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

#include <stdint.h>
#include <vector_types.h>
#include <optix_types.h>
#include <sutil/vec_math.h>
//#include <List>
#include <vector>

#define NUM_OF_VPL 500
#define NUM_OF_BOUNCES 2

#define MAX_LIGHT 3

#define K_POINTS_CLUSTER 500

#define K_MEANS_VPL 25
#define L_NEAR_CLUSTERS 10

#define MAX_VPL_CLUSTERS 10





enum RayType
{
	RAY_TYPE_RADIANCE = 0,
	RAY_TYPE_OCCLUSION = 1,
	RAY_TYPE_COUNT
};

struct  BasicLight//Light data
{
	float3  pos;//Position
	float3  color;
};

struct VPL {
	float3 pos; //Position
	float3 normal;//Normal
	float3 color; //color
	int bounces;//Bounce of the VPL, (recursion)
	bool hit = false;//If hit with the scene or is a miss
};


struct Params
{
	//Image-------------------------------------------------------------
	uint32_t     subframe_index;
	float4*      accum_buffer;
	uchar4*      frame_buffer;
	uint32_t     width;
	uint32_t     height;	
	
	//Light info-----------------------------------------------------------------------
	int number_of_lights; //Number of light
	BasicLight lights[MAX_LIGHT]; //Vector store lights	
	
	float3       ambient_light_color; //ambient light SEt to 0 
	int          max_depth;
	float        scene_epsilon;

	//VPL configuration and info-----------------------------------------------------
	VPL*		 vpls_raw;//Where the vpl array will be stored
	VPL*         vpls;//Only the VPL that hit the scene

	int*		 number_hit_vpl;//Num of VPL that hits the scene
	int			 num_hit_vpl;
	int			 num_vpl;//Number of VPL starting from the light
	int			 max_bounces;//Bounces per VPL

	   
	//SPACE_K-MEANS_VARIABLES and BUFFERS-----------------------------------------
	float3* pos; // Buffer that store the position of all the p ixels
	float3* normal; // Buffer that store the normal of all the pixels

	float3* pos_cent; //Buffer to store the position of the centroid for the space clustering
	float3* normal_cent; //Buffer to store the normal of the centroid for the space clustering

	int* assing_cluster_vector; //Buffer that for every pixel store the cluster where it belongs

	//Points selected per each cluster
	float3* selected_points_pos;
	float3* selected_points_norm;
	int* selected_point_index_x;
	int* selected_point_index_y;

	int position_cluster[K_POINTS_CLUSTER]; //Store the position of the initial centroid of the clustering

	//R MATRIX -------------------------------------------------------------------------------------
	float3*		 R_matrix;	

	//VPL CLUSTER --------------------------------------------------------------
	int first_VPL_cluster[K_MEANS_VPL];
	int* VPL_assing_cluster;

	VPL* VPL_initial_cent;
	VPL* VPL_cent;

	//Local Slice Cluster
		//Select Closest Cluser
		float* distances_slides;//Matrix KxK showing distance btw all the slides.
		int* L_closest_clusters;

	float* L_i_modules;
	float* distances_clusters;

	int* closest_VPL;
	int* selected_VPL_pos;

	//Cluster visualization
	float3 cluster_color[K_POINTS_CLUSTER]; //Colors to shw the differents clusters


	//QT_CLUSTERONG
	float3* pos_QT; // Buffer that store the position of all the p ixels
	float3* normal_QT; // Buffer that store the normal of all the pixels

	int* clusters;
	int* clusters_position;

	int* count_points_per_cluster;

	int Num_points;
	
	bool compute_QT;
	//

	//Launch bool

	bool compute_image;
	bool k_means_space;
	bool save_relevant_VPL;

	//K_means space
	
	bool select_space_points;
	bool init_centroid_points;
	bool assing_cluster_points;
	bool recompute_cluster_points;

	//R Matrix Booleas
	bool compute_R;

	//VPL clustering
	bool init_vpl_centroids_bool;
	bool assing_VPL_bool;
	bool init_VPL_cluster_bool;
	bool recompute_VPL_cent_bool;

	//Local Slice Cluster
		//Seelct closest cluster
		bool local_slice_compute_distances_bool;
		bool select_closest_clusters_bool;

	bool compute__Li_modules_bool;
	bool select_cheap_cluster_bool;


	//Debug booleans
	bool		s_d; //Boolean if show direct light
	bool		s_i;//Boolean if show indirect light
	bool		s_v;//Boolean if show vpl hit position
	bool		s_k;//Boolean show k means **only works if k means is applied
	bool show_R_matrix; // Boolean to show the R_matrix
	bool show_cluster_VPL;

	bool result_K_means;

	//**AUX BOOL
	bool count_VPL_hit;

	//SmoothStep
	float minSS; //Min param
	float maxSS; //Max param



	OptixTraversableHandle  handle;
};

struct CameraData
{
	float3       eye;
	float3       U;
	float3       V;
	float3       W;
};



struct MissData
{
	float3 bg_color; //Color miss ray
};

struct Sphere
{
	float3	center;
	float 	radius;
};

struct Parallelogram
{
	Parallelogram() = default;
	Parallelogram(float3 v1, float3 v2, float3 anchor) :
		v1(v1), v2(v2), anchor(anchor)
	{
		float3 normal = normalize(cross(v1, v2));
		float d = dot(normal, anchor);
		this->v1 *= 1.0f / dot(v1, v1);
		this->v2 *= 1.0f / dot(v2, v2);
		plane = make_float4(normal, d);
	}
	float4	plane;
	float3 	v1;
	float3 	v2;
	float3 	anchor;
};

struct Box
{
	float3 min_v;
	float3 max_v;
};

struct Phong
{
	float3 Kd;
};


struct HitGroupData
{
	union
	{
		Sphere          sphere;
		Parallelogram   plane;

	} geometry;

	union
	{
		Phong			diffuse;
	} shading;
};

//Per ray Data
struct RadiancePRD
{
	float3 result;
	int	   light_number;
	int    depth;//Store the bounce as depth
};


struct OcclusionPRD
{
	float3 attenuation;
};

