/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

__device__ float3 tileCenterFisheyeWorldRay(
	int tile_x, int tile_y, int width, int height,
	const float* fisheye_params, float focal_x, float focal_y,
	const float* viewmatrix)
{
	const float px_image = fminf(tile_x * BLOCK_X + 0.5f * (BLOCK_X - 1), width - 1.0f);
	const float py_image = fminf(tile_y * BLOCK_Y + 0.5f * (BLOCK_Y - 1), height - 1.0f);
	const float fx = fisheye_params ? fisheye_params[0] : focal_x;
	const float fy = fisheye_params ? fisheye_params[1] : focal_y;
	const float cx = fisheye_params ? fisheye_params[2] : 0.5f * (width - 1);
	const float cy = fisheye_params ? fisheye_params[3] : 0.5f * (height - 1);
	const float px = (px_image - cx) / fx;
	const float py = (py_image - cy) / fy;
	const float theta_d = sqrtf(px * px + py * py);
	float theta = theta_d;
	if (fisheye_params)
	{
		const float k1 = fisheye_params[4], k2 = fisheye_params[5];
		const float k3 = fisheye_params[6], k4 = fisheye_params[7];
		for (int iteration = 0; iteration < 8; ++iteration)
		{
			const float theta2 = theta * theta;
			const float theta4 = theta2 * theta2;
			const float theta6 = theta4 * theta2;
			const float theta8 = theta4 * theta4;
			const float value = theta * (1.0f + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8) - theta_d;
			const float derivative = 1.0f + 3.0f * k1 * theta2 + 5.0f * k2 * theta4 + 7.0f * k3 * theta6 + 9.0f * k4 * theta8;
			theta -= value / fmaxf(fabsf(derivative), 1e-8f);
		}
	}
	const float radial = theta_d > 1e-7f ? sinf(theta) / theta_d : 1.0f;
	const float3 ray_camera = {radial * px, radial * py, cosf(theta)};
	float3 ray_world = {
		viewmatrix[0] * ray_camera.x + viewmatrix[1] * ray_camera.y + viewmatrix[2] * ray_camera.z,
		viewmatrix[4] * ray_camera.x + viewmatrix[5] * ray_camera.y + viewmatrix[6] * ray_camera.z,
		viewmatrix[8] * ray_camera.x + viewmatrix[9] * ray_camera.y + viewmatrix[10] * ray_camera.z};
	const float inv_norm = rsqrtf(ray_world.x * ray_world.x + ray_world.y * ray_world.y + ray_world.z * ray_world.z + 1e-12f);
	ray_world.x *= inv_norm; ray_world.y *= inv_norm; ray_world.z *= inv_norm;
	return ray_world;
}

__device__ float gaussianRayTau(
	const float3 origin, const float3 direction,
	const float3 mean, const float* covariance, const float fallback)
{
	const float a = covariance[0], b = covariance[1], c = covariance[2];
	const float d = covariance[3], e = covariance[4], f = covariance[5];
	const float cof00 = d * f - e * e, cof01 = c * e - b * f, cof02 = b * e - c * d;
	const float cof11 = a * f - c * c, cof12 = b * c - a * e, cof22 = a * d - b * b;
	const float determinant = a * cof00 + b * cof01 + c * cof02;
	if (!isfinite(determinant) || determinant <= 1e-18f) return fallback;
	const float inv_det = 1.0f / determinant;
	const float3 ad = {
		(cof00 * direction.x + cof01 * direction.y + cof02 * direction.z) * inv_det,
		(cof01 * direction.x + cof11 * direction.y + cof12 * direction.z) * inv_det,
		(cof02 * direction.x + cof12 * direction.y + cof22 * direction.z) * inv_det};
	const float denominator = direction.x * ad.x + direction.y * ad.y + direction.z * ad.z;
	if (!isfinite(denominator) || denominator <= 1e-12f) return fallback;
	const float3 offset = {origin.x - mean.x, origin.y - mean.y, origin.z - mean.z};
	const float tau = -(offset.x * ad.x + offset.y * ad.y + offset.z * ad.z) / denominator;
	return isfinite(tau) && tau > 0.0f ? tau : fallback;
}

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	const float* candidate_means2D,
	const float* candidate_cov2D,
	const int* candidate_counts,
	dim3 grid,
	const float3* means3D,
	const float* cov3D,
	const float3* cam_pos,
	const float* fisheye_params,
	const float* viewmatrix,
	int width, int height,
	float focal_x, float focal_y,
	bool sort_by_tile_tau)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 first_rect_min, first_rect_max;
		const int component_count = candidate_counts == nullptr ? 1 : max(1, min(2, candidate_counts[idx]));

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int component = 0; component < component_count; ++component)
		{
			float2 component_point = points_xy[idx];
			float component_radius = radii[idx];
			if (candidate_counts != nullptr)
			{
				const float* mean = candidate_means2D + (idx * 2 + component) * 2;
				const float* cov = candidate_cov2D + (idx * 2 + component) * 3;
				component_point = {mean[0], mean[1]};
				const float determinant = cov[0] * cov[2] - cov[1] * cov[1];
				const float middle = 0.5f * (cov[0] + cov[2]);
				const float lambda_max = middle + sqrtf(max(0.1f, middle * middle - determinant));
				component_radius = ceilf(3.0f * sqrtf(max(0.1f, lambda_max)));
			}
			uint2 rect_min, rect_max;
			getRect(component_point, component_radius, rect_min, rect_max, grid);
			if (component == 0) { first_rect_min = rect_min; first_rect_max = rect_max; }
			for (int y = rect_min.y; y < rect_max.y; ++y)
			{
				for (int x = rect_min.x; x < rect_max.x; ++x)
				{
					if (component == 1 && x >= first_rect_min.x && x < first_rect_max.x &&
						y >= first_rect_min.y && y < first_rect_max.y)
						continue;
					uint64_t key = y * grid.x + x;
					key <<= 32;
					float sort_depth = depths[idx];
					if (sort_by_tile_tau)
					{
						const float3 ray = tileCenterFisheyeWorldRay(
							x, y, width, height, fisheye_params, focal_x, focal_y, viewmatrix);
						sort_depth = gaussianRayTau(*cam_pos, ray, means3D[idx], cov3D + 6 * idx, sort_depth);
					}
					key |= *((uint32_t*)&sort_depth);
					gaussian_keys_unsorted[off] = key;
					gaussian_values_unsorted[off] = idx;
					++off;
				}
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* means2D_precomp,
	const float* cov2D_precomp,
	const float* candidate_means2D,
	const float* candidate_cov2D,
	const int* candidate_counts,
	const float* ut_gate,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float* fisheye_params,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	int* radii,
	bool is_fisheye,
	bool use_ut,
	bool use_3d_response,
	bool sort_by_camera_distance,
	bool sort_by_tile_tau,
	int tau_k_buffer_size,
	bool debug,
	bool fisheye_no_center_cull)
{
	if (fisheye_no_center_cull && (!is_fisheye || !use_ut || !use_3d_response || tau_k_buffer_size == 0))
		throw std::runtime_error("No-center-cull requires fisheye UT, 3D response and pixel-tau sorting.");
	if (use_3d_response && (!use_ut || !is_fisheye)){
		throw std::runtime_error(
			"3D response requires UT fisheye rendering.");
	}
	if (tau_k_buffer_size < 0 || tau_k_buffer_size > 16) {
		throw std::runtime_error("tau_k_buffer_size must be in [0, 16].");
	}
	if (tau_k_buffer_size > 0 && !use_3d_response) {
		throw std::runtime_error("Per-pixel tau K-buffer requires 3D response rendering.");
	}


	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		means2D_precomp,
		cov2D_precomp,
		candidate_means2D,
		candidate_cov2D,
		candidate_counts,
		ut_gate,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		fisheye_params,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
			is_fisheye,
			use_ut,
			sort_by_camera_distance,
			prefiltered, fisheye_no_center_cull
	), debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	const float* response_cov3D_ptr =
		(cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		candidate_means2D,
		candidate_cov2D,
		candidate_counts,
		tile_grid,
		(const float3*)means3D,
		response_cov3D_ptr,
		(const float3*)cam_pos,
		fisheye_params,
		viewmatrix,
		width, height,
		focal_x, focal_y,
		sort_by_tile_tau)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,

		(const float3*)means3D,
		response_cov3D_ptr,
		viewmatrix,
		(const glm::vec3*)cam_pos,
		fisheye_params,
		focal_x,
		focal_y,
		use_3d_response,
		ut_gate,
		tau_k_buffer_size,

		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color), debug)

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* cov2D_precomp,
	const float* ut_gate,
	const float* shs,
	const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float* fisheye_params,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	float* dL_dmean2D,
	float* dL_dcov2D,
	float* dL_dut_gate,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	bool is_fisheye,
	bool use_ut,
	bool use_3d_response,
	int tau_k_buffer_size,
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;

	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,

		(const float3*)means3D,
		cov3D_ptr,
		viewmatrix,
		(const glm::vec3*)campos,
		fisheye_params,
		focal_x,
		focal_y,
		use_3d_response,
		ut_gate,
		tau_k_buffer_size,

		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
	
		(glm::vec3*)dL_dmean3D,
		dL_dcov3D,
		dL_dut_gate), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		cov2D_precomp,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		width, height,
		(float3*)dL_dmean2D,
		dL_dcov2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		is_fisheye,
		use_ut,
		use_3d_response), debug)
}
