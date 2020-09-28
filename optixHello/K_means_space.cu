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


static
__device__ void create_centroids() {

	const uint3 idx = optixGetLaunchIndex();

	int clus_pos = params.position_cluster[idx.x];

	params.normal_cent[idx.x] = params.normal[clus_pos];
	params.pos_cent[idx.x] = params.pos[clus_pos];
}




static
__device__ void assing_cluster() {


	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();

	const uint32_t image_index = params.width*idx.y + idx.x;

	float dist_min = 999999.0f;
	float zero_dot = 0.0;

	float3 normal = params.normal[image_index];
	float3 point = params.pos[image_index];

	int cluster = 0;

	for (int i = 0; i < K_POINTS_CLUSTER; i++) {

		float3 p_centr = params.pos_cent[i];
		float3 n_centr = params.normal_cent[i];
		float dot_pro = dot(normal, n_centr);

		if (dot_pro > zero_dot) {

			float3 diff = point - p_centr;
			float p_distance = length(diff);

			if (p_distance <= dist_min) {
				dist_min = p_distance;
				cluster = i;

			}
		}
	}
	params.assing_cluster_vector[image_index] = cluster;
}

static
__device__ void recompute_centroids() {

	const uint3 idx = optixGetLaunchIndex();
	int index = idx.x;
	int num_clusters = 0;
	float3 sum_p = make_float3(0);
	float3 sum_n = make_float3(0);

	bool point_selected = false;

	float3 selected = make_float3(0.f);
	float3 selected_normal = make_float3(0.f);

	for (int i = 0; i < params.width; i++) {
		for (int j = 0; j < params.height; j++) {
			const uint32_t image_index = params.width*j + i;

			if (params.assing_cluster_vector[image_index] == index)
			{
				if (point_selected == false) {
					selected = params.pos[image_index];
					selected_normal = params.normal[image_index];
					point_selected = true;

					params.selected_point_index_x[index] = i;
					params.selected_point_index_y[index] = j;
				}

				num_clusters = num_clusters + 1;
				sum_p = sum_p + params.pos[image_index];
				sum_n = sum_n + params.normal[image_index];
			}
		}
	}

	sum_p = sum_p / num_clusters;
	sum_n = sum_n / num_clusters;

	params.pos_cent[index] = sum_p;
	params.normal_cent[index] = sum_n;

	params.selected_points_pos[index] = selected;
	params.selected_points_norm[index] = selected_normal;

}

static
__device__ void count_VPL_hit() {	
	int count = 0;
	for (int i = 0; i < params.num_vpl*(params.max_bounces + 1); i++) {
		VPL vpl = params.vpls_raw[i];
		if (vpl.hit) {
			count = count + 1;
		}
	}
	params.number_hit_vpl[0] = count;
}


static
__device__ void save_relevant_VPL() {
	int count = 0;
	for (int i = 0; i < params.num_vpl*(params.max_bounces + 1); i++) {
		VPL vpl = params.vpls_raw[i];
		if (vpl.hit) {
			params.vpls[count] = vpl;
			count = count + 1;
		}
	}
}


static
__device__ void assing_VPL() {

	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();

	int VPL_index_R = params.num_hit_vpl;

	float dist_min = 9999999.f;

	int p = idx.x;
	int cluster = 0;

	float Rp_module = 0.f;
	float Rq_module;


	float distance_p_q = 0.f;

	//Compute module of each columns
	for (int i = 0; i < K_MEANS_VPL; i++) {

		Rp_module = 0.f;
		Rq_module = 0.f;
		float diff_norm_sq = 0.f;

		int vpl_centroid_idx = params.first_VPL_cluster[i];

		int q = vpl_centroid_idx;

			for (int j = 0; j < K_POINTS_CLUSTER; j++) {

				//Lightness
				float luminance_point_p = (params.R_matrix[(j*VPL_index_R) + p].x + params.R_matrix[(j * VPL_index_R) + p].y + params.R_matrix[(j * VPL_index_R) + p].z) / 3;
				float luminance_point_q = (params.R_matrix[(j*VPL_index_R) + q].x + params.R_matrix[(j * VPL_index_R) + q].y + params.R_matrix[(j * VPL_index_R) + q].z) / 3;

				//Luminence
				//float luminance_point_p = (0.21*params.R_matrix[(j*VPL_index_R) + p].x + 0.72*params.R_matrix[(j * VPL_index_R) + p].y + 0.07*params.R_matrix[(j * VPL_index_R) + p].z);				
				//float luminance_point_q = (0.21*params.R_matrix[(j*VPL_index_R) + q].x + 0.72*params.R_matrix[(j * VPL_index_R) + q].y + 0.07*params.R_matrix[(j * VPL_index_R) + q].z);

				Rp_module = Rp_module + (luminance_point_p* luminance_point_p);
				Rq_module = Rq_module + (luminance_point_q* luminance_point_q);
			}

			float norm_Rp = sqrt(Rp_module);
			float norm_Rq = sqrt(Rq_module);

			for (int j = 0; j < K_POINTS_CLUSTER; j++) {

				//LIghness
				float luminance_point_p = (params.R_matrix[j * VPL_index_R + p].x + params.R_matrix[j * VPL_index_R + p].y + params.R_matrix[j * VPL_index_R + p].z) / 3;
				float luminance_point_q = (params.R_matrix[j * VPL_index_R + q].x + params.R_matrix[j * VPL_index_R + q].y + params.R_matrix[j * VPL_index_R + q].z) / 3;

				//Luminocity
				//float luminance_point_p = (0.21*params.R_matrix[j * VPL_index_R + p].x + 0.72*params.R_matrix[j * VPL_index_R + p].y + 0.07*params.R_matrix[j * VPL_index_R + p].z) ;				
				//float luminance_point_q = (0.21*params.R_matrix[j * VPL_index_R + q].x + 0.72* params.R_matrix[j * VPL_index_R + q].y + 0.07*params.R_matrix[j * VPL_index_R + q].z);

				float diff_q_p = (luminance_point_p / norm_Rp) - (luminance_point_q / norm_Rq);
				diff_norm_sq = diff_norm_sq + (diff_q_p * diff_q_p);
			}

			distance_p_q = norm_Rp * norm_Rq * diff_norm_sq;

			if (distance_p_q < dist_min) {
				dist_min = distance_p_q;
				cluster = i;
			}	

	}
	params.VPL_assing_cluster[p] = cluster;

}



void recoumpte_VPL_cent() {//**NOT USE



}


//COmpute matrix of all the distances of all the colums to all the colums(symmetric)
static
__device__ void compute_distances() {

	const uint3 idx = optixGetLaunchIndex();

	for (int i = 0; i < K_POINTS_CLUSTER; i++) {
		if (dot(params.selected_points_norm[idx.x], params.selected_points_norm[i]) <= 0.f) {
			params.distances_slides[idx.x*K_POINTS_CLUSTER + i] = 999999.0f;
		}
		else {
			float3 diff = params.selected_points_pos[idx.x] - params.selected_points_pos[i];
			float p_distance = length(diff);
			params.distances_slides[idx.x*K_POINTS_CLUSTER + i] = p_distance;
		}
	}

	//if (dot(params.selected_points_norm[idx.x], params.selected_points_norm[idx.y]) <= 0.f) {
	//	params.distances_slides[idx.x*K_POINTS_CLUSTER + idx.y] = 999999.0f;
	//}
	//else {
	//	float3 diff = params.selected_points_pos[idx.x] - params.selected_points_pos[idx.y];
	//	float p_distance = length(diff);
	//	params.distances_slides[idx.x*K_POINTS_CLUSTER + idx.y] = p_distance;
	//}

}

static
__device__ void closest_clusters() {

	//MAX HEAP ALGORITHM

	const uint3 idx = optixGetLaunchIndex();
	int index = idx.x;

	float max_dist_bag = 0.f;
	int pos_max_dist = 0;

	for (int l = 0; l < L_NEAR_CLUSTERS; l++) {

		params.L_closest_clusters[index * L_NEAR_CLUSTERS + l] = l;

		if (params.distances_slides[idx.x*K_POINTS_CLUSTER + l] > max_dist_bag) {
			max_dist_bag = params.distances_slides[idx.x*K_POINTS_CLUSTER + l];
			pos_max_dist = l;
		}

	}

	for (int i = L_NEAR_CLUSTERS; i < K_POINTS_CLUSTER; i++) {

		float curr_dist = params.distances_slides[idx.x*K_POINTS_CLUSTER + i];

		//Interchange for the larger number if necesary
		if (curr_dist < max_dist_bag) {
			params.L_closest_clusters[index * L_NEAR_CLUSTERS + pos_max_dist] = i;

			//To know if is smaller than the selected. 
			max_dist_bag = 0.f;
			pos_max_dist = 0;

			for (int l = 0; l < L_NEAR_CLUSTERS; l++) {
				int index_l_cluster = params.L_closest_clusters[index * L_NEAR_CLUSTERS + l];

				if (params.distances_slides[idx.x*K_POINTS_CLUSTER + index_l_cluster] > max_dist_bag) {
					max_dist_bag = params.distances_slides[idx.x*K_POINTS_CLUSTER + index_l_cluster];
					pos_max_dist = l;
				}
			}
		}
	}

}



static
__device__ void compute_L_i_modules() {

	const uint3 idx = optixGetLaunchIndex();
	int VPL_index_R = params.num_hit_vpl;
	int point_idx = idx.x;


	for (int v = 0; v < VPL_index_R; v++) {

		int vpl_idx = v;

		float current_module = 0.f;

		//  R_matrix_luminance   --   L_closest_clusters
		for (int i = 0; i < L_NEAR_CLUSTERS; i++) {

			int R_index_pos = params.L_closest_clusters[point_idx * L_NEAR_CLUSTERS + i];
			float3 color_ill = params.R_matrix[R_index_pos*VPL_index_R + vpl_idx];
			float luminance = (color_ill.x + color_ill.y + color_ill.z) / 3;

			current_module = current_module + (luminance* luminance);
		}

		current_module = sqrt(current_module);
		params.L_i_modules[point_idx * VPL_index_R + vpl_idx] = current_module;
	}

	//For each cluster
	for (int c = 0; c < K_MEANS_VPL; c++) {

		params.distances_clusters[point_idx*K_MEANS_VPL + c] = 0.f;

		for (int i = 0; i < VPL_index_R; i++) {


			if (params.VPL_assing_cluster[i] == c) {

				for (int j = 0; j < VPL_index_R; j++) {

					float distance_p_q = 0.f;

					if (params.VPL_assing_cluster[j] == c) {

						float diff_norm_sq = 0.f;

						for (int l = 0; l < L_NEAR_CLUSTERS; l++) {

							int point_R = params.L_closest_clusters[point_idx * L_NEAR_CLUSTERS + l];

							//LIGHTNESS
							float luminance_point_p = (params.R_matrix[point_R * VPL_index_R + i].x + params.R_matrix[point_R * VPL_index_R + i].y + params.R_matrix[point_R * VPL_index_R + i].z) / 3;
							float luminance_point_q = (params.R_matrix[point_R * VPL_index_R + j].x + params.R_matrix[point_R * VPL_index_R + j].y + params.R_matrix[point_R * VPL_index_R + j].z) / 3;

							//Luminocity
							/*float luminance_point_p = (0.21*params.R_matrix[point_R * VPL_index_R + i].x + 0.72*params.R_matrix[point_R * VPL_index_R + i].y + 0.07*params.R_matrix[point_R * VPL_index_R + i].z) ;							
							float luminance_point_q = (0.21*params.R_matrix[point_R * VPL_index_R + j].x + 0.72*params.R_matrix[point_R * VPL_index_R + j].y + 0.07*params.R_matrix[point_R * VPL_index_R + j].z);*/

							float diff_q_p = (luminance_point_p / params.L_i_modules[point_idx*VPL_index_R + i]) - (luminance_point_q / params.L_i_modules[point_idx*VPL_index_R + j]);
							diff_norm_sq = diff_norm_sq + (diff_q_p * diff_q_p);

						}

						distance_p_q = params.L_i_modules[point_idx*VPL_index_R + i] * params.L_i_modules[point_idx*VPL_index_R + j] * diff_norm_sq;
					}

					params.distances_clusters[point_idx*K_MEANS_VPL + c] = params.distances_clusters[point_idx*K_MEANS_VPL + c] + distance_p_q;
				}
			}
		}
	}


}

static
__device__ void select_cheap_clusters() {

	//MAX HEAP ALGORITHM

	const uint3 idx = optixGetLaunchIndex();
	int index_R = idx.x;
	int VPL_index_R = params.num_hit_vpl;


	float max_dist_bag = 0.f;
	int pos_max_dist = 0;

	//First clusters
	for (int l = 0; l < MAX_VPL_CLUSTERS; l++) {

		params.closest_VPL[index_R * MAX_VPL_CLUSTERS + l] = l;

		if (params.distances_clusters[idx.x*K_MEANS_VPL + l] > max_dist_bag) {
			max_dist_bag = params.distances_clusters[idx.x*K_MEANS_VPL + l];
			pos_max_dist = l;
		}

	}

	//-->

	for (int i = MAX_VPL_CLUSTERS; i < K_MEANS_VPL; i++) {

		float curr_dist = params.distances_clusters[idx.x*K_POINTS_CLUSTER + i];

		//Interchange for the larger number if necesary
		if (curr_dist < max_dist_bag) {
			params.closest_VPL[index_R * MAX_VPL_CLUSTERS + pos_max_dist] = i;

			//To know if is smaller than the selected. 
			max_dist_bag = 0.f;
			pos_max_dist = 0;

			for (int l = 0; l < MAX_VPL_CLUSTERS; l++) {
				int index_l_cluster = params.closest_VPL[index_R * MAX_VPL_CLUSTERS + l];

				if (params.distances_clusters[idx.x*K_MEANS_VPL + index_l_cluster] > max_dist_bag) {
					max_dist_bag = params.distances_clusters[idx.x*K_MEANS_VPL + index_l_cluster];
					pos_max_dist = l;

				}
			}
		}
	}

	//int VPL_count = 0;
	int pos_vpl = 0;
	bool stop = false;
	int count = 0;

	for (int k = 0; k < MAX_VPL_CLUSTERS; k++) {
		params.selected_VPL_pos[index_R*MAX_VPL_CLUSTERS + k] = -1;
	}


	for (int y = 0; y < VPL_index_R; y++)
	{

		int current_pos_cluster = params.VPL_assing_cluster[y];


		for (int i = 0; i < MAX_VPL_CLUSTERS; i++) {

			int current_cluster = params.closest_VPL[index_R * MAX_VPL_CLUSTERS + i];

			if (current_pos_cluster == current_cluster) {
				params.selected_VPL_pos[index_R*MAX_VPL_CLUSTERS + i] = y;
				params.closest_VPL[index_R * MAX_VPL_CLUSTERS + i] = 99999;
				count = count + 1;
			}

		}
		if (count >= MAX_VPL_CLUSTERS) {
			break;
		}
	}
}

static
__device__ void compute_QT() {

	const uint3 idx = optixGetLaunchIndex();
	int index_x = idx.x;
	int index_y = idx.y;
	const uint32_t image_index = params.width*idx.y + idx.x;


	float zero_dot = 0.0;
	float dist_min = 1.0;

	int count = 0;

	for (int i = 0; i < params.Num_points;i++) {

		
		
		float dot_pro = dot(params.normal[i], params.normal[image_index]);

		if (dot_pro > zero_dot) {

			float3 diff = params.pos[i] - params.pos[image_index];
			float p_distance = length(diff);

			if (p_distance <= dist_min) {
				
				count = count + 1;
			}
		}
		
	}
	params.count_points_per_cluster[image_index] = count;

}

extern "C" __global__ void __raygen__K_means_space() {


	
	if (params.init_centroid_points) {
		create_centroids();
	}
	if (params.assing_cluster_points) {
		assing_cluster();
	}
	if (params.recompute_cluster_points) {
		recompute_centroids();
	}
	if (params.count_VPL_hit) {
		count_VPL_hit();
	}
	if (params.save_relevant_VPL) {
		save_relevant_VPL();
	}
	if (params.init_vpl_centroids_bool) {
		//init_VPL_centroid();
	}
	if (params.assing_VPL_bool) {
		assing_VPL();
	}
	if (params.local_slice_compute_distances_bool) {
		compute_distances();
	}
	if (params.select_closest_clusters_bool) {
		closest_clusters();
	}
	if (params.compute__Li_modules_bool) {
		compute_L_i_modules();
	}
	if (params.select_cheap_cluster_bool) {
		select_cheap_clusters();
	}
	if (params.compute_QT) {
		compute_QT();
	}
}


extern "C" __global__ void __closesthit__K_means_points()
{//**DOES NOT WORK
	float3 object_normal = make_float3(
		int_as_float(optixGetAttribute_0()),
		int_as_float(optixGetAttribute_1()),
		int_as_float(optixGetAttribute_2()));

		float3 world_normal = normalize(optixTransformNormalFromObjectToWorldSpace(object_normal));
		float3 ffnormal = faceforward(world_normal, -optixGetWorldRayDirection(), world_normal);
	
		const uint3    idx = optixGetLaunchIndex();
		const float3 ray_orig = optixGetWorldRayOrigin();
		const float3 ray_dir = optixGetWorldRayDirection();
		const float  ray_t = optixGetRayTmax();

		const uint32_t image_index = params.width*idx.y + idx.x;

		float3 hit_point = ray_orig + ray_t * ray_dir;
		params.normal[image_index] = make_float3(ffnormal.x, ffnormal.y, ffnormal.z);
		params.pos[image_index] = make_float3(hit_point.x, hit_point.y, hit_point.z);
	
}

extern "C" __global__ void __miss__K_means()
{//**DOES NOT WORK

	const uint3 idx = optixGetLaunchIndex();
	const uint32_t image_index = params.width*idx.y + idx.x;

	params.normal[image_index] = make_float3(0, 0, 0);
	params.pos[image_index] = make_float3(1000, 1000, 1000);
}