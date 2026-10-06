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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix, const bool is_fisheye)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 9982444353.f;
	const float limy = 9982444353.f;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J;
	if (is_fisheye) // fisheye
	{
		float eps = 0.0000001f;
		float x2 = t.x * t.x + eps;
		float y2 = t.y * t.y;
		float xy = t.x * t.y;
		float x2y2 = x2 + y2 ;
		float len_xy = length(glm::vec2({t.x, t.y})) + eps;
		float x2y2z2_inv = 1.f / (x2y2 + t.z * t.z);

		float b = glm::atan(len_xy, t.z) / len_xy / x2y2;
		float a = t.z * x2y2z2_inv / (x2y2);
		J = glm::mat3(
			focal_x * (x2 * a + y2 * b), focal_x * xy * (a - b),    - focal_x * t.x * x2y2z2_inv,
			focal_y * xy  * (a - b),    focal_y * (y2 * a + x2 * b), - focal_y * t.y * x2y2z2_inv,
			0, 0, 0
		);
	}
	else
	{
		J = glm::mat3(
			focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
			0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
			0, 0, 0);
	}
	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

__device__ float2 projectOpenCVFisheye(
	const float3 p, const float* params, float J[6])
{
	const float fx = params[0], fy = params[1];
	const float cx = params[2], cy = params[3];
	const float k1 = params[4], k2 = params[5], k3 = params[6], k4 = params[7];
	const float rho2 = p.x * p.x + p.y * p.y;
	const float rho = sqrtf(fmaxf(rho2, 1e-20f));
	const float theta = atan2f(rho, p.z);
	const float t2 = theta * theta;
	const float t4 = t2 * t2, t6 = t4 * t2, t8 = t4 * t4;
	const float theta_d = theta * (1.0f + k1*t2 + k2*t4 + k3*t6 + k4*t8);
	const bool axis = rho2 < 1e-12f;
	const float radial = axis ? 1.0f / fmaxf(p.z, 1e-8f) : theta_d / rho;
	const float2 uv = {fx * radial * p.x + cx, fy * radial * p.y + cy};
	if (J == nullptr)
		return uv;
	if (axis)
	{
		J[0] = fx / fmaxf(p.z, 1e-8f); J[1] = 0.0f; J[2] = 0.0f;
		J[3] = 0.0f; J[4] = fy / fmaxf(p.z, 1e-8f); J[5] = 0.0f;
		return uv;
	}
	const float dtheta_d = 1.0f + 3*k1*t2 + 5*k2*t4 + 7*k3*t6 + 9*k4*t8;
	const float den = fmaxf(rho2 + p.z*p.z, 1e-20f);
	const float dtx = p.z*p.x/(rho*den), dty = p.z*p.y/(rho*den), dtz = -rho/den;
	const float rho3 = rho*rho2;
	const float drx = dtheta_d*dtx/rho - theta_d*p.x/rho3;
	const float dry = dtheta_d*dty/rho - theta_d*p.y/rho3;
	const float drz = dtheta_d*dtz/rho;
	J[0] = fx*(radial + p.x*drx); J[1] = fx*p.x*dry; J[2] = fx*p.x*drz;
	J[3] = fy*p.y*drx; J[4] = fy*(radial + p.y*dry); J[5] = fy*p.y*drz;
	return uv;
}

__device__ void computeLearnedAdaptiveProjection(
	const float3 center, const glm::vec3 scale, const float scale_modifier,
	const glm::vec4 q, const float* viewmatrix, const float* fisheye_params,
	const float gate, float2& mean, float3& covariance)
{
	const float r=q.x, x=q.y, y=q.z, z=q.w;
	const float R00=1-2*(y*y+z*z), R01=2*(x*y-r*z), R02=2*(x*z+r*y);
	const float R10=2*(x*y+r*z), R11=1-2*(x*x+z*z), R12=2*(y*z-r*x);
	const float R20=2*(x*z-r*y), R21=2*(y*z+r*x), R22=1-2*(x*x+y*y);
	float3 axes[3] = {
		{scale_modifier*scale.x*R00, scale_modifier*scale.x*R10, scale_modifier*scale.x*R20},
		{scale_modifier*scale.y*R01, scale_modifier*scale.y*R11, scale_modifier*scale.y*R21},
		{scale_modifier*scale.z*R02, scale_modifier*scale.z*R12, scale_modifier*scale.z*R22}
	};
	for (int a=0; a<3; ++a)
		axes[a] = transformVec4x3(axes[a], viewmatrix);

	float J[6];
	const float2 ewa_mean = projectOpenCVFisheye(center, fisheye_params, J);
	float e00=0.3f, e01=0.0f, e11=0.3f;
	for (int a=0; a<3; ++a)
	{
		const float du=J[0]*axes[a].x+J[1]*axes[a].y+J[2]*axes[a].z;
		const float dv=J[3]*axes[a].x+J[4]*axes[a].y+J[5]*axes[a].z;
		e00 += du*du; e01 += du*dv; e11 += dv*dv;
	}

	float2 sigma[7];
	sigma[0] = ewa_mean;
	const float delta = 1.7320508075688772f;
	for (int a=0; a<3; ++a)
	{
		const float3 d = {delta*axes[a].x, delta*axes[a].y, delta*axes[a].z};
		sigma[1+a] = projectOpenCVFisheye(
			{center.x+d.x, center.y+d.y, center.z+d.z}, fisheye_params, nullptr);
		sigma[4+a] = projectOpenCVFisheye(
			{center.x-d.x, center.y-d.y, center.z-d.z}, fisheye_params, nullptr);
	}
	float2 ut_mean = {0.0f, 0.0f};
	for (int s=1; s<7; ++s) {ut_mean.x += sigma[s].x/6.0f; ut_mean.y += sigma[s].y/6.0f;}
	float u00=0.3f, u01=0.0f, u11=0.3f;
	for (int s=0; s<7; ++s)
	{
		const float weight = s==0 ? 2.0f : 1.0f/6.0f;
		const float du=sigma[s].x-ut_mean.x, dv=sigma[s].y-ut_mean.y;
		u00 += weight*du*du; u01 += weight*du*dv; u11 += weight*dv*dv;
	}

	mean = {(1-gate)*ewa_mean.x+gate*ut_mean.x, (1-gate)*ewa_mean.y+gate*ut_mean.y};
	const float edx=ewa_mean.x-mean.x, edy=ewa_mean.y-mean.y;
	const float udx=ut_mean.x-mean.x, udy=ut_mean.y-mean.y;
	covariance = {
		(1-gate)*(e00+edx*edx)+gate*(u00+udx*udx),
		(1-gate)*(e01+edx*edy)+gate*(u01+udx*udy),
		(1-gate)*(e11+edy*edy)+gate*(u11+udy*udy)
	};
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const float* means2D_precomp,
	const float* cov2D_precomp,
	const float* candidate_means2D,
	const float* candidate_cov2D,
	const int* candidate_counts,
	const float* ut_gate,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,//world2cam
	const float* projmatrix, //world2ndc, not cam2ndc
	const float* fisheye_params,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool is_fisheye,
	bool use_ut,
	bool sort_by_camera_distance,
	bool prefiltered,
	bool fisheye_no_center_cull)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (fisheye_no_center_cull)
	{
		// Conservative diagnostic: let projected tile overlap and the actual
		// forward-ray response decide visibility, not a pinhole center-z test.
		const float3 center = {orig_points[3*idx], orig_points[3*idx+1], orig_points[3*idx+2]};
		p_view = transformPoint4x3(center, viewmatrix);
		if (!isfinite(p_view.x) || !isfinite(p_view.y) || !isfinite(p_view.z)) return;
	}
	else if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	// manual projection matrix
	// float3 p_proj;
	// if (is_fisheye) // panorama
	// {
	// 	float xy_len = glm::length(glm::vec2({p_view.x, p_view.y})) + 0.000001f;
	// 	float theta = glm::atan(xy_len, p_view.z + 0.0000001f);
	// 	if (abs(theta) > 3.14 * 0.403)
	// 		return; 
	// 	p_proj.x = 2 * p_view.x * focal_x * theta / (xy_len * W);
	// 	p_proj.y = 2 * p_view.y * focal_y * theta / (xy_len * H);
	// 	p_proj.z = 0;
	// }
	// else
	// {
	// 	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	// 	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	// 	p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	// }
	if (is_fisheye && !use_ut)
	{
		const float xy_len =
			glm::length(glm::vec2({p_view.x, p_view.y}))
			+ 0.000001f;

		const float theta =
			glm::atan(xy_len, p_view.z + 0.0000001f);

		if (fabsf(theta) > 3.14f * 0.403f)
			return;
	}

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}
	////
	float2 point_image;
	float3 cov;

	if (use_ut && ut_gate != nullptr && fisheye_params != nullptr && scales != nullptr)
	{
		computeLearnedAdaptiveProjection(
			p_view, scales[idx], scale_modifier, rotations[idx], viewmatrix,
			fisheye_params, fminf(1.0f, fmaxf(0.0f, ut_gate[idx])),
			point_image, cov);
	}
	else if (use_ut)
	{
		// means2D_precomp: [P,3], NDC
		point_image = {
			ndc2Pix(means2D_precomp[3 * idx], W),
			ndc2Pix(means2D_precomp[3 * idx + 1], H)
		};

		// cov2D_precomp: [P,3], pixel covariance
		cov = {
			cov2D_precomp[3 * idx],
			cov2D_precomp[3 * idx + 1],
			cov2D_precomp[3 * idx + 2]
		};
	}
	else
	{
		float3 p_proj;

		if (is_fisheye)
		{
			const float xy_len =
				glm::length(glm::vec2({p_view.x, p_view.y}))
				+ 0.000001f;

			const float theta =
				glm::atan(xy_len, p_view.z + 0.0000001f);

			p_proj.x =
				2.0f * p_view.x * focal_x * theta
				/ (xy_len * W);

			p_proj.y =
				2.0f * p_view.y * focal_y * theta
				/ (xy_len * H);

			p_proj.z = 0.0f;
		}
		else
		{
			const float4 p_hom =
				transformPoint4x4(p_orig, projmatrix);

			const float p_w =
				1.0f / (p_hom.w + 0.0000001f);

			p_proj = {
				p_hom.x * p_w,
				p_hom.y * p_w,
				p_hom.z * p_w
			};
		}

		point_image = {
			ndc2Pix(p_proj.x, W),
			ndc2Pix(p_proj.y, H)
		};

		cov = computeCov2D(
			p_orig,
			focal_x,
			focal_y,
			tan_fovx,
			tan_fovy,
			cov3D,
			viewmatrix,
			is_fisheye
		);
	}
	////
	// Compute 2D screen-space covariance matrix
	// float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix, is_fisheye);

	// Invert covariance (EWA algorithm)
	// float det = (cov.x * cov.z - cov.y * cov.y);
	// if (det == 0.0f)
		// return;
	const float det =
		cov.x * cov.z - cov.y * cov.y;

	if (!isfinite(det) || det <= 1e-12f)
		return;

	const float det_inv = 1.0f / det;

	const float3 conic = {
		cov.z * det_inv,
		-cov.y * det_inv,
		cov.x * det_inv
	};
	// float det_inv = 1.f / det;
	// float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	// float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	uint32_t tile_count = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);
	if (candidate_counts != nullptr)
	{
		const int count = max(1, min(2, candidate_counts[idx]));
		uint2 candidate_min[2], candidate_max[2];
		float max_candidate_radius = 0.0f;
		for (int component = 0; component < count; ++component)
		{
			const float* candidate_mean = candidate_means2D + (idx * 2 + component) * 2;
			const float* candidate_cov = candidate_cov2D + (idx * 2 + component) * 3;
			const float candidate_det = candidate_cov[0] * candidate_cov[2] - candidate_cov[1] * candidate_cov[1];
			const float candidate_mid = 0.5f * (candidate_cov[0] + candidate_cov[2]);
			const float candidate_lambda = candidate_mid + sqrtf(max(0.1f, candidate_mid * candidate_mid - candidate_det));
			const float candidate_radius = ceilf(3.0f * sqrtf(max(0.1f, candidate_lambda)));
			const float2 candidate_point = {candidate_mean[0], candidate_mean[1]};
			getRect(candidate_point, candidate_radius, candidate_min[component], candidate_max[component], grid);
			max_candidate_radius = max(max_candidate_radius, candidate_radius);
		}
		const uint32_t area0 = (candidate_max[0].x - candidate_min[0].x) * (candidate_max[0].y - candidate_min[0].y);
		tile_count = area0;
		if (count == 2)
		{
			const uint32_t area1 = (candidate_max[1].x - candidate_min[1].x) * (candidate_max[1].y - candidate_min[1].y);
			const uint32_t ix0 = max(candidate_min[0].x, candidate_min[1].x);
			const uint32_t iy0 = max(candidate_min[0].y, candidate_min[1].y);
			const uint32_t ix1 = min(candidate_max[0].x, candidate_max[1].x);
			const uint32_t iy1 = min(candidate_max[0].y, candidate_max[1].y);
			const uint32_t overlap = (ix1 > ix0 && iy1 > iy0) ? (ix1 - ix0) * (iy1 - iy0) : 0;
			tile_count += area1 - overlap;
		}
		my_radius = max_candidate_radius;
	}
	if (tile_count == 0)
	 	return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	// Camera-distance ordering accounts for the angle between a fisheye ray
	// and the optical axis: ||p_view|| = p_view.z / cos(theta).  It remains a
	// single representative depth per Gaussian, unlike per-pixel tau sorting.
	depths[idx] = (sort_by_camera_distance || fisheye_no_center_cull)
		? sqrtf(p_view.x * p_view.x + p_view.y * p_view.y + p_view.z * p_view.z)
		: p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = tile_count;
}

__device__ float3 fisheyePixelToWorldRay(
	const float2 pixel, const int W, const int H,
	const float* fisheye_params, const float focal_x, const float focal_y,
	const float* viewmatrix);
__device__ bool evaluate3DGaussianResponse(
	const float3 ray_origin, const float3 ray_direction, const float3 mean,
	const float* covariance, float& power, float& tau_max);

constexpr int MAX_TAU_K_BUFFER_SIZE = 16;

__device__ inline void insertTauHit(
	int* ids, float* taus, float* alphas, int& count, int capacity,
	int id, float tau, float alpha)
{
	int pos = count;
	while (pos > 0 && taus[pos - 1] > tau)
	{
		if (pos < capacity)
		{
			ids[pos] = ids[pos - 1];
			taus[pos] = taus[pos - 1];
			alphas[pos] = alphas[pos - 1];
		}
		--pos;
	}
	ids[pos] = id;
	taus[pos] = tau;
	alphas[pos] = alpha;
	++count;
}

template <uint32_t CHANNELS>
__device__ inline bool blendTauHit(
	int id, float alpha, const float* features,
	float& T, float* C)
{
	const float test_T = T * (1.0f - alpha);
	if (test_T < 0.0001f)
		return false;
	for (int ch = 0; ch < CHANNELS; ++ch)
		C[ch] += features[id * CHANNELS + ch] * alpha * T;
	T = test_T;
	return true;
}

// 3DGUT-style per-pixel MLAB path. The tile list remains a coarse ordering;
// accepted hits are reordered by their ray-wise tau_max in a local K-buffer.
template <uint32_t CHANNELS>
__global__ void renderTauKBufferCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	const float3* __restrict__ means3D,
	const float* __restrict__ cov3D,
	const float* __restrict__ viewmatrix,
	const glm::vec3* __restrict__ cam_pos,
	const float* __restrict__ fisheye_params,
	float focal_x, float focal_y,
	const float* __restrict__ ut_gate,
	int k_buffer_size,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color)
{
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
	const uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
	const bool inside = pix.x < W && pix.y < H;
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = {(float)pix.x, (float)pix.y};
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	float3 ray_origin = {0.0f, 0.0f, 0.0f};
	float3 ray_direction = {0.0f, 0.0f, 1.0f};
	if (inside)
	{
		ray_origin = {cam_pos->x, cam_pos->y, cam_pos->z};
		ray_direction = fisheyePixelToWorldRay(
			pixf, W, H, fisheye_params, focal_x, focal_y, viewmatrix);
	}

	int hit_ids[MAX_TAU_K_BUFFER_SIZE];
	float hit_taus[MAX_TAU_K_BUFFER_SIZE];
	float hit_alphas[MAX_TAU_K_BUFFER_SIZE];
	int hit_count = 0;
	float T = 1.0f;
	float C[CHANNELS] = {0};
	uint32_t blended_count = 0;
	bool done = !inside;
	int toDo = range.y - range.x;

	for (int batch = 0; batch < rounds; ++batch, toDo -= BLOCK_SIZE)
	{
		if (__syncthreads_count(done) == BLOCK_SIZE)
			break;
		const int progress = batch * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = id;
			collected_xy[block.thread_rank()] = points_xy_image[id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[id];
		}
		block.sync();

		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); ++j)
		{
			const int id = collected_id[j];
			const float4 con_o = collected_conic_opacity[j];
			float power3D, tau;
			if (!evaluate3DGaussianResponse(
				ray_origin, ray_direction, means3D[id], cov3D + 6 * id,
				power3D, tau) || power3D <= -4.482f)
				continue;

			float G3D = expf(power3D);
			float G = G3D;
			if (ut_gate != nullptr)
			{
				const float gate = fminf(1.0f, fmaxf(0.0f, ut_gate[id]));
				const float2 xy = collected_xy[j];
				const float2 delta = {xy.x - pixf.x, xy.y - pixf.y};
				const float power2D = -0.5f * (
					con_o.x * delta.x * delta.x + con_o.z * delta.y * delta.y)
					- con_o.y * delta.x * delta.y;
				const float G2D = power2D <= 0.0f ? expf(power2D) : 0.0f;
				G = (1.0f - gate) * G2D + gate * G3D;
			}
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			if (hit_count == k_buffer_size)
			{
				if (!blendTauHit<CHANNELS>(hit_ids[0], hit_alphas[0], features, T, C))
				{
					done = true;
					continue;
				}
				++blended_count;
				for (int q = 1; q < hit_count; ++q)
				{
					hit_ids[q - 1] = hit_ids[q];
					hit_taus[q - 1] = hit_taus[q];
					hit_alphas[q - 1] = hit_alphas[q];
				}
				--hit_count;
			}
			insertTauHit(hit_ids, hit_taus, hit_alphas, hit_count,
				k_buffer_size, id, tau, alpha);
		}
	}

	for (int q = 0; !done && q < hit_count; ++q)
	{
		if (!blendTauHit<CHANNELS>(hit_ids[q], hit_alphas[q], features, T, C))
		{
			done = true;
			break;
		}
		++blended_count;
	}
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = blended_count;
		for (int ch = 0; ch < CHANNELS; ++ch)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
// template <uint32_t CHANNELS>
// __global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)

__device__ float3 fisheyePixelToWorldRay(
    const float2 pixel,
    const int W,
    const int H,
    const float* fisheye_params,
    const float focal_x,
    const float focal_y,
    const float* viewmatrix)
{
    const float fx = fisheye_params ? fisheye_params[0] : focal_x;
    const float fy = fisheye_params ? fisheye_params[1] : focal_y;
    const float cx = fisheye_params ? fisheye_params[2] : 0.5f * (W - 1);
    const float cy = fisheye_params ? fisheye_params[3] : 0.5f * (H - 1);

    const float px = (pixel.x - cx) / fx;
    const float py = (pixel.y - cy) / fy;

    const float theta_d = sqrtf(px * px + py * py);
    float theta = theta_d;
    if (fisheye_params)
    {
        const float k1 = fisheye_params[4];
        const float k2 = fisheye_params[5];
        const float k3 = fisheye_params[6];
        const float k4 = fisheye_params[7];
        for (int iteration = 0; iteration < 8; ++iteration)
        {
            const float theta2 = theta * theta;
            const float theta4 = theta2 * theta2;
            const float theta6 = theta4 * theta2;
            const float theta8 = theta4 * theta4;
            const float value = theta * (
                1.0f + k1 * theta2 + k2 * theta4 +
                k3 * theta6 + k4 * theta8) - theta_d;
            const float derivative =
                1.0f + 3.0f * k1 * theta2 + 5.0f * k2 * theta4 +
                7.0f * k3 * theta6 + 9.0f * k4 * theta8;
            theta -= value / fmaxf(fabsf(derivative), 1e-8f);
        }
    }

    float3 ray_camera;

    if (theta_d > 1e-7f)
    {
        const float sin_theta_over_theta_d =
            sinf(theta) / theta_d;

        ray_camera = {
            sin_theta_over_theta_d * px,
            sin_theta_over_theta_d * py,
            cosf(theta)
        };
    }
    else
    {
        ray_camera = {0.0f, 0.0f, 1.0f};
    }

    // viewmatrix는 world→camera이므로 transpose 회전으로
    // camera direction을 world direction으로 변환
    float3 ray_world =
        transformVec4x3Transpose(
            ray_camera,
            viewmatrix);

    const float inv_norm = rsqrtf(
        ray_world.x * ray_world.x +
        ray_world.y * ray_world.y +
        ray_world.z * ray_world.z +
        1e-12f);

    ray_world.x *= inv_norm;
    ray_world.y *= inv_norm;
    ray_world.z *= inv_norm;

    return ray_world;
}

__device__ bool evaluate3DGaussianResponse(
    const float3 ray_origin,
    const float3 ray_direction,
    const float3 mean,
    const float* covariance,
    float& power,
    float& tau_max)
{
    // Symmetric covariance:
    //
    // [ a b c ]
    // [ b d e ]
    // [ c e f ]

    const float a = covariance[0];
    const float b = covariance[1];
    const float c = covariance[2];
    const float d = covariance[3];
    const float e = covariance[4];
    const float f = covariance[5];

    // Cofactors of covariance
    const float cof00 = d * f - e * e;
    const float cof01 = c * e - b * f;
    const float cof02 = b * e - c * d;
    const float cof11 = a * f - c * c;
    const float cof12 = b * c - a * e;
    const float cof22 = a * d - b * b;

    const float determinant =
        a * cof00 +
        b * cof01 +
        c * cof02;

    if (!isfinite(determinant) ||
        determinant <= 1e-18f)
    {
        return false;
    }

    const float inv_det = 1.0f / determinant;

    // Sigma^{-1}
    const float A00 = cof00 * inv_det;
    const float A01 = cof01 * inv_det;
    const float A02 = cof02 * inv_det;
    const float A11 = cof11 * inv_det;
    const float A12 = cof12 * inv_det;
    const float A22 = cof22 * inv_det;

    const float3 o_minus_mean = {
        ray_origin.x - mean.x,
        ray_origin.y - mean.y,
        ray_origin.z - mean.z
    };

    // A d
    const float3 A_direction = {
        A00 * ray_direction.x +
            A01 * ray_direction.y +
            A02 * ray_direction.z,

        A01 * ray_direction.x +
            A11 * ray_direction.y +
            A12 * ray_direction.z,

        A02 * ray_direction.x +
            A12 * ray_direction.y +
            A22 * ray_direction.z
    };

    const float denominator =
        ray_direction.x * A_direction.x +
        ray_direction.y * A_direction.y +
        ray_direction.z * A_direction.z;

    if (!isfinite(denominator) ||
        denominator <= 1e-12f)
    {
        return false;
    }

    // Equation (11):
    // tau = -(o-mu)^T Sigma^{-1} d
    //       / d^T Sigma^{-1} d
    const float numerator =
        o_minus_mean.x * A_direction.x +
        o_minus_mean.y * A_direction.y +
        o_minus_mean.z * A_direction.z;

    tau_max = -numerator / denominator;

    // Maximum response point must be in front of the camera.
    if (!isfinite(tau_max) || tau_max <= 0.0f)
    {
        return false;
    }

    const float3 hit = {
        o_minus_mean.x + tau_max * ray_direction.x,
        o_minus_mean.y + tau_max * ray_direction.y,
        o_minus_mean.z + tau_max * ray_direction.z
    };

    // A hit
    const float3 A_hit = {
        A00 * hit.x + A01 * hit.y + A02 * hit.z,
        A01 * hit.x + A11 * hit.y + A12 * hit.z,
        A02 * hit.x + A12 * hit.y + A22 * hit.z
    };

    float gray_distance =
        hit.x * A_hit.x +
        hit.y * A_hit.y +
        hit.z * A_hit.z;

    if (!isfinite(gray_distance))
    {
        return false;
    }

    // Numerical roundoff may produce a tiny negative value.
    gray_distance = fmaxf(gray_distance, 0.0f);

    // 3DGUT kernel degree 2:
    // response = exp(-0.5 * gray_distance)
    power = -0.5f * gray_distance;

    return isfinite(power) && power <= 0.0f;
}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,

	const float3* __restrict__ means3D,
	const float* __restrict__ cov3D,
	const float* __restrict__ viewmatrix,
	const glm::vec3* __restrict__ cam_pos,
	const float* __restrict__ fisheye_params,
	float focal_x,
	float focal_y,
	bool use_3d_response,
	const float* __restrict__ ut_gate,
	int tau_k_buffer_size,

	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	float3 ray_origin = {0.0f, 0.0f, 0.0f};
	float3 ray_direction = {0.0f, 0.0f, 1.0f};

	if (inside && use_3d_response)
	{
		ray_origin = {
			cam_pos->x,
			cam_pos->y,
			cam_pos->z
		};

		ray_direction =
			fisheyePixelToWorldRay(
				pixf,
				W,
				H,
				fisheye_params,
				focal_x,
				focal_y,
				viewmatrix);
	}
	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			// float2 xy = collected_xy[j];
			// float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			// float4 con_o = collected_conic_opacity[j];
			// float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;


			const int gaussian_id = collected_id[j];
			const float4 con_o = collected_conic_opacity[j];

			float power = 0.0f;
			float G = 0.0f;

			if (use_3d_response && ut_gate != nullptr)
			{
				const float gate = fminf(1.0f, fmaxf(0.0f, ut_gate[gaussian_id]));
				float G2D = 0.0f;
				float G3D = 0.0f;
				if (gate < 1.0f)
				{
					const float2 xy = collected_xy[j];
					const float2 d = {xy.x - pixf.x, xy.y - pixf.y};
					const float power2D = -0.5f * (
						con_o.x * d.x * d.x + con_o.z * d.y * d.y)
						- con_o.y * d.x * d.y;
					G2D = power2D <= 0.0f ? expf(power2D) : 0.0f;
				}
				if (gate > 0.0f)
				{
					float power3D;
					float tau_max;
					const bool valid3D = evaluate3DGaussianResponse(
						ray_origin, ray_direction, means3D[gaussian_id],
						cov3D + 6 * gaussian_id, power3D, tau_max);
					G3D = (valid3D && power3D > -4.482f) ? expf(power3D) : 0.0f;
				}
				G = (1.0f - gate) * G2D + gate * G3D;
			}
			else if (use_3d_response)
			{
				float tau_max;

				const bool valid_response =
					evaluate3DGaussianResponse(
						ray_origin,
						ray_direction,
						means3D[gaussian_id],
						cov3D + 6 * gaussian_id,
						power,
						tau_max);

				if (!valid_response)
				{
					continue;
				}
			}
			else
			{
				const float2 xy = collected_xy[j];

				const float2 d = {
					xy.x - pixf.x,
					xy.y - pixf.y
				};

				power =
					-0.5f * (
						con_o.x * d.x * d.x +
						con_o.z * d.y * d.y)
					- con_o.y * d.x * d.y;
			}

			if (!(use_3d_response && ut_gate != nullptr))
			{
				if (power > 0.0f)
					continue;
				if (use_3d_response && power <= -4.482f)
					continue;
				G = expf(power);
			}

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,

	const float3* means3D,
	const float* cov3D,
	const float* viewmatrix,
	const glm::vec3* cam_pos,
	const float* fisheye_params,
	float focal_x,
	float focal_y,
	bool use_3d_response,
	const float* ut_gate,
	int tau_k_buffer_size,

	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color)
{
	if (tau_k_buffer_size > 0)
	{
		renderTauKBufferCUDA<NUM_CHANNELS> << <grid, block >> > (
			ranges, point_list, W, H, means2D, colors, conic_opacity,
			means3D, cov3D, viewmatrix, cam_pos, fisheye_params,
			focal_x, focal_y, ut_gate, tau_k_buffer_size,
			final_T, n_contrib, bg_color, out_color);
		return;
	}
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,

		means3D,
		cov3D,
		viewmatrix,
		cam_pos,
		fisheye_params,
		focal_x,
		focal_y,
		use_3d_response,
		ut_gate,
		tau_k_buffer_size,

		final_T,
		n_contrib,
		bg_color,
		out_color);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const float* means2D_precomp,
	const float* cov2D_precomp,
	const float* candidate_means2D,
	const float* candidate_cov2D,
	const int* candidate_counts,
	const float* ut_gate,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* fisheye_params,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool is_fisheye,
	bool use_ut,
	bool sort_by_camera_distance,
	bool prefiltered,
	bool fisheye_no_center_cull)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		means2D_precomp,
		cov2D_precomp,
		candidate_means2D,
		candidate_cov2D,
		candidate_counts,
		ut_gate,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		fisheye_params,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
			is_fisheye,
			use_ut,
			sort_by_camera_distance,
			prefiltered, fisheye_no_center_cull
		);
}
