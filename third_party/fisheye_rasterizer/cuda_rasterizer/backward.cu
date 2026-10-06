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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__global__ void computeCov2DPrecompGradCUDA(
    int P,
    const int* radii,
    const float* cov2D,
    const float* dL_dconics,
    float* dL_dcov2D)
{
    auto idx = cg::this_grid().thread_rank();

    if (idx >= P)
        return;

    if (radii[idx] <= 0)
    {
        dL_dcov2D[3 * idx + 0] = 0.0f;
        dL_dcov2D[3 * idx + 1] = 0.0f;
        dL_dcov2D[3 * idx + 2] = 0.0f;
        return;
    }

    const float a = cov2D[3 * idx + 0];
    const float b = cov2D[3 * idx + 1];
    const float c = cov2D[3 * idx + 2];

    const float dL_dconic_a = dL_dconics[4 * idx + 0];
    const float dL_dconic_b = dL_dconics[4 * idx + 1];
    const float dL_dconic_c = dL_dconics[4 * idx + 3];

    const float det = a * c - b * b;

    if (!isfinite(det) || fabsf(det) <= 1e-12f)
    {
        dL_dcov2D[3 * idx + 0] = 0.0f;
        dL_dcov2D[3 * idx + 1] = 0.0f;
        dL_dcov2D[3 * idx + 2] = 0.0f;
        return;
    }

    const float det2inv = 1.0f / (det * det + 1e-7f);

    dL_dcov2D[3 * idx + 0] =
        det2inv * (
            -c * c * dL_dconic_a
            + 2.0f * b * c * dL_dconic_b
            + (det - a * c) * dL_dconic_c);

    dL_dcov2D[3 * idx + 1] =
        det2inv * 2.0f * (
            b * c * dL_dconic_a
            - (det + 2.0f * b * b) * dL_dconic_b
            + a * b * dL_dconic_c);

    dL_dcov2D[3 * idx + 2] =
        det2inv * (
            -a * a * dL_dconic_c
            + 2.0f * a * b * dL_dconic_b
            + (det - a * c) * dL_dconic_a);
}

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DFisheyeCUDA(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* dL_dconics,
	float3* dL_dmeans,
	float* dL_dcov)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, view_matrix);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	const float eps = 0.0000001f;
	const float x2 = t.x * t.x + eps;
	const float y2 = t.y * t.y;
	const float xy = t.x * t.y;
	const float x2y2 = x2 + y2 ;
	const float len_xy = length(glm::vec2({t.x, t.y})) + eps;
	const float x2y2z2 = x2y2 + t.z * t.z;
	const float x2y2z2_inv = 1.f / x2y2z2;
	const float theta = glm::atan(len_xy, t.z);

	const float J_b = theta / len_xy / x2y2;
	const float J_a = t.z * x2y2z2_inv / (x2y2);
	glm::mat3 J = glm::mat3(
		h_x * (x2 * J_a + y2 * J_b), h_x * xy * (J_a - J_b),  - h_x * t.x * x2y2z2_inv,
		h_y * xy  * (J_a - J_b),  h_y * (y2 * J_a + x2 * J_b), - h_y * t.y * x2y2z2_inv,
		0, 0, 0
	);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// printf("dL_dconic = (%.10f, %.10f, %.10f)\b", dL_dconic.x, dL_dconic.y, dL_dconic.z);
	// printf("dL_dcov2d = (%.10f, %.10f, %.10f)\n", dL_da, dL_db / 2.0, dL_dc);
	// printf("dL_dcov3d = (%.10f, %.10f, %.10f, %.10f, %.10f, %.10f)\n", dL_dcov[6 * idx + 0], dL_dcov[6 * idx + 1], dL_dcov[6 * idx + 2], dL_dcov[6 * idx + 3], dL_dcov[6 * idx + 4], dL_dcov[6 * idx + 5]);
	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ01 = W[1][0] * dL_dT00 + W[1][1] * dL_dT01 + W[1][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ10 = W[0][0] * dL_dT10 + W[0][1] * dL_dT11 + W[0][2] * dL_dT12;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	// printf("dL_dJ(cov2d) = \n[%.10f, %.10f, %.10f\n %.10f, %.10f, %.10f]\n", dL_dJ00, dL_dJ01,dL_dJ02, dL_dJ10, dL_dJ11, dL_dJ12);
	// printf("t = %.10f, %.10f, %.10f\n", t.x, t.y, t.z);
	// printf("focal = %.10f, %.10f\n", h_x, h_y);

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	float l4 = x2y2z2 * x2y2z2;

	float E = - l4 * x2y2 * theta + x2y2z2 * x2y2 * len_xy * t.z;
	float F = 3 * l4 * theta - 3 * x2y2z2 * len_xy * t.z - 2 * x2y2 * len_xy * t.z;

	float A = t.x * (3 * E + x2 * F);
	float B = t.y * (E + x2 * F);
	float C = t.x * (E + y2 * F);
	float D = t.y * (3 * E + y2 * F);
	
	float S1 = x2 - y2 - t.z * t.z;
	float S2 = y2 - x2 - t.z * t.z;
	float inv1 = x2y2z2_inv * x2y2z2_inv;
	float inv2 = inv1 / (x2y2 * x2y2 * len_xy);

	float dJ_dx00 = h_x * A * inv2;
	float dJ_dx01 = h_x * B * inv2;
	float dJ_dx02 = h_x * S1 * inv1;
	float dJ_dx10 = h_y * B * inv2;
	float dJ_dx11 = h_y * C * inv2;
	float dJ_dx12 = 2.f * h_y * xy * inv1;

	float dJ_dy00 = dJ_dx01;
	float dJ_dy01 = h_x * C * inv2;
	float dJ_dy02 = 2.f * h_x * xy * inv1;
	float dJ_dy10 = dJ_dx11;
	float dJ_dy11 = h_y * D * inv2;
	float dJ_dy12 = h_y * S2 * inv1;

	float dJ_dz00 = dJ_dx02;
	float dJ_dz01 = dJ_dy02;
	float dJ_dz02 = 2.f * h_x * t.x * t.z * inv1;
	float dJ_dz10 = dJ_dx12;
	float dJ_dz11 = dJ_dy12;
	float dJ_dz12 = 2.f * h_y * t.y * t.z * inv1;

	float dL_dtx_raw = dJ_dx00 * dL_dJ00 + dJ_dx01 * dL_dJ01 + dJ_dx02 * dL_dJ02 + dJ_dx10 * dL_dJ10 + dJ_dx11 * dL_dJ11 + dJ_dx12 * dL_dJ12;
	float dL_dty_raw = dJ_dy00 * dL_dJ00 + dJ_dy01 * dL_dJ01 + dJ_dy02 * dL_dJ02 + dJ_dy10 * dL_dJ10 + dJ_dy11 * dL_dJ11 + dJ_dy12 * dL_dJ12;
	float dL_dtz_raw = dJ_dz00 * dL_dJ00 + dJ_dz01 * dL_dJ01 + dJ_dz02 * dL_dJ02 + dJ_dz10 * dL_dJ10 + dJ_dz11 * dL_dJ11 + dJ_dz12 * dL_dJ12;
	
	float dL_dtx = x_grad_mul * dL_dtx_raw;
	float dL_dty = y_grad_mul * dL_dty_raw;
	float dL_dtz = dL_dtz_raw;

	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	dL_dmeans[idx] = dL_dmean;
}


__global__ void computeCov2DCUDA(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* dL_dconics,
	float3* dL_dmeans,
	float* dL_dcov)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, view_matrix);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx] = dL_dmean;
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* proj,
	const float* view_matrix,
	const float focal_x, float focal_y,
	const glm::vec3* campos,
	const int W,int H,
	const float3* dL_dmean2D, //NDC Coordinate
	glm::vec3* dL_dmeans, 
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	bool is_fisheye,
	bool use_ut,
	bool use_3d_response)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];
	//bool is_fisheye = false;
	// printf("dL_dmeans2D=(%.10f, %.10f)\n", dL_dmean2D[idx].x, dL_dmean2D[idx].y);
	if (!use_ut && !use_3d_response)
	{
		glm::vec3 dL_dmean;
		if (is_fisheye)
		{
			float3 t = transformPoint4x3(m, view_matrix);
			//const float eps = 0.001f;
			const float eps = 0.0000001f;
			float x2 = t.x * t.x + eps;
			float y2 = t.y * t.y;
			float xy = t.x * t.y;
			float x2y2 = x2 + y2 ;
			float len_xy = length(glm::vec2({t.x, t.y})) + eps;
			float x2y2z2_inv = 1.f / (x2y2 + t.z * t.z);
			float b = glm::atan(len_xy, t.z) / len_xy / x2y2;
			float a = t.z * x2y2z2_inv / (x2y2);
			float fx = 2.0 * focal_x / W;
			float fy = 2.0 * focal_y / H;
			// float fx = focal_x;
			// float fy = focal_y;
			dL_dmean.x = fx * (x2 * a + y2 * b) * dL_dmean2D[idx].x + fy * xy * (a - b) * dL_dmean2D[idx].y;
			dL_dmean.y = fx * xy * (a - b) * dL_dmean2D[idx].x + fy * (y2 * a + x2 * b) * dL_dmean2D[idx].y;
			dL_dmean.z = - fx * t.x * x2y2z2_inv * dL_dmean2D[idx].x - fy * t.y * x2y2z2_inv * dL_dmean2D[idx].y;
			float3 dL_dmean1 = transformVec4x3Transpose({dL_dmean.x, dL_dmean.y, dL_dmean.z}, view_matrix);
			dL_dmean = {dL_dmean1.x, dL_dmean1.y, dL_dmean1.z};
		}
		else
		{
		// Taking care of gradients from the screenspace points
			float4 m_hom = transformPoint4x4(m, proj);
			float m_w = 1.0f / (m_hom.w + 0.0000001f);

			// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
			// from rendering procedure
			float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
			float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
			dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
			dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
			dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

			// That's the second part of the mean gradient. Previous computation
			// of cov2D and following SH conversion also affects it.
		}
		// printf("dL_dt(means) = (%.10f, %.10f, %.10f)\n", dL_dmean.x, dL_dmean.y, dL_dmean.z);
		dL_dmeans[idx] += dL_dmean;
	}
	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales && (!use_ut || use_3d_response))
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

// Backward version of the rendering procedure.
__device__ float3 fisheyePixelToWorldRayBackward(
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

__device__ bool evaluate3DGaussianResponseBackward(
    const float3 ray_origin,
    const float3 ray_direction,
    const float3 mean,
    const float* covariance,
    float& power,
    float3& precision_hit)
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
	
	float tau_max;
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

	precision_hit = A_hit;

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
constexpr int MAX_TAU_K_BUFFER_SIZE_BWD = 16;

__device__ inline bool tauForGaussian(
	const float3 origin, const float3 direction, const float3 mean,
	const float* covariance, float& tau)
{
	const float a = covariance[0], b = covariance[1], c = covariance[2];
	const float d = covariance[3], e = covariance[4], f = covariance[5];
	const float cof00 = d * f - e * e, cof01 = c * e - b * f, cof02 = b * e - c * d;
	const float cof11 = a * f - c * c, cof12 = b * c - a * e, cof22 = a * d - b * b;
	const float det = a * cof00 + b * cof01 + c * cof02;
	if (!isfinite(det) || det <= 1e-18f) return false;
	const float inv_det = 1.0f / det;
	const float3 ad = {
		(cof00 * direction.x + cof01 * direction.y + cof02 * direction.z) * inv_det,
		(cof01 * direction.x + cof11 * direction.y + cof12 * direction.z) * inv_det,
		(cof02 * direction.x + cof12 * direction.y + cof22 * direction.z) * inv_det};
	const float denominator = direction.x * ad.x + direction.y * ad.y + direction.z * ad.z;
	if (!isfinite(denominator) || denominator <= 1e-12f) return false;
	const float3 offset = {origin.x - mean.x, origin.y - mean.y, origin.z - mean.z};
	tau = -(offset.x * ad.x + offset.y * ad.y + offset.z * ad.z) / denominator;
	return isfinite(tau) && tau > 0.0f;
}

__device__ inline void insertTauHitBackward(
	int* ids, float* taus, float* alphas, int& count,
	int id, float tau, float alpha)
{
	int pos = count;
	while (pos > 0 && taus[pos - 1] > tau)
	{
		ids[pos] = ids[pos - 1];
		taus[pos] = taus[pos - 1];
		alphas[pos] = alphas[pos - 1];
		--pos;
	}
	ids[pos] = id;
	taus[pos] = tau;
	alphas[pos] = alpha;
	++count;
}

__device__ inline bool evaluateTauCandidateBackward(
	int id, const float2 pixf, const float2 xy, const float4 con_o,
	const float3 ray_origin, const float3 ray_direction,
	const float3* means3D, const float* cov3D, const float* ut_gate,
	float& alpha, float& tau)
{
	float power3D;
	float3 precision_hit;
	if (!evaluate3DGaussianResponseBackward(
		ray_origin, ray_direction, means3D[id], cov3D + 6 * id,
		power3D, precision_hit) || power3D <= -4.482f ||
		!tauForGaussian(ray_origin, ray_direction, means3D[id], cov3D + 6 * id, tau))
		return false;
	const float G3D = expf(power3D);
	float G = G3D;
	if (ut_gate != nullptr)
	{
		const float gate = fminf(1.0f, fmaxf(0.0f, ut_gate[id]));
		const float2 delta = {xy.x - pixf.x, xy.y - pixf.y};
		const float power2D = -0.5f * (
			con_o.x * delta.x * delta.x + con_o.z * delta.y * delta.y)
			- con_o.y * delta.x * delta.y;
		const float G2D = power2D <= 0.0f ? expf(power2D) : 0.0f;
		G = (1.0f - gate) * G2D + gate * G3D;
	}
	alpha = min(0.99f, con_o.w * G);
	return alpha >= 1.0f / 255.0f;
}

template <uint32_t C>
__device__ inline bool accumulateTauHitBackwardPass(
	int id, float alpha, const float* colors, const float* bg_color,
	const float* dL_dpixel, const float* final_color,
	const float2 pixf, const float2 xy, const float4 con_o,
	const float3 ray_origin, const float3 ray_direction,
	const float3* means3D, const float* cov3D, const float* ut_gate,
	float& T, float* prefix_color,
	float3* dL_dmean2D, float4* dL_dconic2D,
	float* dL_dopacity, float* dL_dcolors,
	glm::vec3* dL_dmean3D, float* dL_dcov3D, float* dL_dut_gate,
	float ddelx_dx, float ddely_dy)
{
	const float test_T = T * (1.0f - alpha);
	if (test_T < 0.0001f)
		return false;

	float dL_dalpha = 0.0f;
	for (int ch = 0; ch < C; ++ch)
	{
		const float color = colors[id * C + ch];
		const float prefix_after = prefix_color[ch] + T * alpha * color;
		const float remainder_after = (final_color[ch] - prefix_after) / test_T;
		dL_dalpha += T * (color - remainder_after) * dL_dpixel[ch];
		atomicAdd(&dL_dcolors[id * C + ch], T * alpha * dL_dpixel[ch]);
		prefix_color[ch] = prefix_after;
	}

	const bool learned_adaptive = ut_gate != nullptr;
	const float gate = learned_adaptive
		? fminf(1.0f, fmaxf(0.0f, ut_gate[id])) : 1.0f;
	const float2 delta = {xy.x - pixf.x, xy.y - pixf.y};
	const float power2D = -0.5f * (
		con_o.x * delta.x * delta.x + con_o.z * delta.y * delta.y)
		- con_o.y * delta.x * delta.y;
	const float G2D = learned_adaptive && power2D <= 0.0f ? expf(power2D) : 0.0f;
	float power3D;
	float3 precision_hit;
	const bool valid3D = evaluate3DGaussianResponseBackward(
		ray_origin, ray_direction, means3D[id], cov3D + 6 * id,
		power3D, precision_hit);
	const float G3D = valid3D && power3D > -4.482f ? expf(power3D) : 0.0f;
	const float G = learned_adaptive ? (1.0f - gate) * G2D + gate * G3D : G3D;
	const float raw_alpha = con_o.w * G;
	const float dL_dG = raw_alpha < 0.99f ? con_o.w * dL_dalpha : 0.0f;

	if (learned_adaptive)
		atomicAdd(&dL_dut_gate[id], dL_dG * (G3D - G2D));
	if (G3D > 0.0f)
	{
		const float dL_dpower = dL_dG * (learned_adaptive ? gate : 1.0f) * G3D;
		const float vx = precision_hit.x, vy = precision_hit.y, vz = precision_hit.z;
		atomicAdd(&dL_dmean3D[id].x, dL_dpower * vx);
		atomicAdd(&dL_dmean3D[id].y, dL_dpower * vy);
		atomicAdd(&dL_dmean3D[id].z, dL_dpower * vz);
		float* dcov = dL_dcov3D + 6 * id;
		atomicAdd(&dcov[0], 0.5f * dL_dpower * vx * vx);
		atomicAdd(&dcov[1], dL_dpower * vx * vy);
		atomicAdd(&dcov[2], dL_dpower * vx * vz);
		atomicAdd(&dcov[3], 0.5f * dL_dpower * vy * vy);
		atomicAdd(&dcov[4], dL_dpower * vy * vz);
		atomicAdd(&dcov[5], 0.5f * dL_dpower * vz * vz);
	}
	if (learned_adaptive && G2D > 0.0f)
	{
		const float branch_dL_dG = dL_dG * (1.0f - gate);
		const float gdx = G2D * delta.x, gdy = G2D * delta.y;
		const float dGdx = -gdx * con_o.x - gdy * con_o.y;
		const float dGdy = -gdy * con_o.z - gdx * con_o.y;
		atomicAdd(&dL_dmean2D[id].x, branch_dL_dG * dGdx * ddelx_dx);
		atomicAdd(&dL_dmean2D[id].y, branch_dL_dG * dGdy * ddely_dy);
		atomicAdd(&dL_dconic2D[id].x, -0.5f * gdx * delta.x * branch_dL_dG);
		atomicAdd(&dL_dconic2D[id].y, -0.5f * gdx * delta.y * branch_dL_dG);
		atomicAdd(&dL_dconic2D[id].w, -0.5f * gdy * delta.y * branch_dL_dG);
	}
	atomicAdd(&dL_dopacity[id], G * dL_dalpha);
	T = test_T;
	return true;
}

template <uint32_t C>
__global__ void renderTauKBufferBackwardCUDA(
	const uint2* __restrict__ ranges, const uint32_t* __restrict__ point_list,
	int W, int H, const float* bg_color,
	const float2* points_xy_image, const float4* conic_opacity, const float* colors,
	const float3* means3D, const float* cov3D, const float* viewmatrix,
	const glm::vec3* cam_pos, const float* fisheye_params,
	float focal_x, float focal_y, const float* ut_gate, int k_buffer_size,
	const float* dL_dpixels,
	float3* dL_dmean2D, float4* dL_dconic2D, float* dL_dopacity,
	float* dL_dcolors, glm::vec3* dL_dmean3D, float* dL_dcov3D,
	float* dL_dut_gate)
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
		ray_direction = fisheyePixelToWorldRayBackward(
			pixf, W, H, fisheye_params, focal_x, focal_y, viewmatrix);
	}
	float dL_dpixel[C] = {0};
	if (inside)
		for (int ch = 0; ch < C; ++ch) dL_dpixel[ch] = dL_dpixels[ch * H * W + pix_id];

	float final_color[C] = {0};
	float final_T = 1.0f;
	for (int pass = 0; pass < 2; ++pass)
	{
		int ids[MAX_TAU_K_BUFFER_SIZE_BWD];
		float taus[MAX_TAU_K_BUFFER_SIZE_BWD];
		float alphas[MAX_TAU_K_BUFFER_SIZE_BWD];
		int count = 0;
		float T = 1.0f;
		float prefix[C] = {0};
		bool done = !inside;
		int toDo = range.y - range.x;
		for (int batch = 0; batch < rounds; ++batch, toDo -= BLOCK_SIZE)
		{
			if (__syncthreads_count(done) == BLOCK_SIZE) break;
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
				float alpha, tau;
				if (!evaluateTauCandidateBackward(id, pixf, collected_xy[j], collected_conic_opacity[j],
					ray_origin, ray_direction, means3D, cov3D, ut_gate, alpha, tau)) continue;
				if (count == k_buffer_size)
				{
					const int pop_id = ids[0];
					if (pass == 0)
					{
						const float test_T = T * (1.0f - alphas[0]);
						if (test_T < 0.0001f) done = true;
						else {
							for (int ch = 0; ch < C; ++ch) prefix[ch] += T * alphas[0] * colors[pop_id * C + ch];
							T = test_T;
						}
					}
					else if (!accumulateTauHitBackwardPass<C>(pop_id, alphas[0], colors, bg_color,
						dL_dpixel, final_color, pixf, points_xy_image[pop_id], conic_opacity[pop_id],
						ray_origin, ray_direction, means3D, cov3D, ut_gate, T, prefix,
						dL_dmean2D, dL_dconic2D, dL_dopacity, dL_dcolors,
						dL_dmean3D, dL_dcov3D, dL_dut_gate, 0.5f * W, 0.5f * H)) done = true;
					if (done) continue;
					for (int q = 1; q < count; ++q) { ids[q-1]=ids[q]; taus[q-1]=taus[q]; alphas[q-1]=alphas[q]; }
					--count;
				}
				insertTauHitBackward(ids, taus, alphas, count, id, tau, alpha);
			}
		}
		for (int q = 0; !done && q < count; ++q)
		{
			const int id = ids[q];
			if (pass == 0)
			{
				const float test_T = T * (1.0f - alphas[q]);
				if (test_T < 0.0001f) { done = true; break; }
				for (int ch = 0; ch < C; ++ch) prefix[ch] += T * alphas[q] * colors[id * C + ch];
				T = test_T;
			}
			else if (!accumulateTauHitBackwardPass<C>(id, alphas[q], colors, bg_color,
				dL_dpixel, final_color, pixf, points_xy_image[id], conic_opacity[id],
				ray_origin, ray_direction, means3D, cov3D, ut_gate, T, prefix,
				dL_dmean2D, dL_dconic2D, dL_dopacity, dL_dcolors,
				dL_dmean3D, dL_dcov3D, dL_dut_gate, 0.5f * W, 0.5f * H)) { done = true; break; }
		}
		if (pass == 0 && inside)
		{
			final_T = T;
			for (int ch = 0; ch < C; ++ch) final_color[ch] = prefix[ch] + T * bg_color[ch];
		}
		block.sync();
	}
}

template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,

	const float3* __restrict__ means3D,
	const float* __restrict__ cov3D,
	const float* __restrict__ viewmatrix,
	const glm::vec3* __restrict__ cam_pos,
	const float* __restrict__ fisheye_params,
	float focal_x,
	float focal_y,
	bool use_3d_response,
	const float* __restrict__ ut_gate,

	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	float3* __restrict__ dL_dmean2D, //NDC Coordinate
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,

	glm::vec3* __restrict__ dL_dmean3D,
	float* __restrict__ dL_dcov3D,
	float* __restrict__ dL_dut_gate)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;

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
			fisheyePixelToWorldRayBackward(
				pixf,
				W,
				H,
				fisheye_params,
				focal_x,
				focal_y,
				viewmatrix);
	}

	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];
	if (inside)
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];

	float last_alpha = 0;
	float last_color[C] = { 0 };

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			// const float2 xy = collected_xy[j];
			// const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			// const float4 con_o = collected_conic_opacity[j];
			// const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			const int global_id = collected_id[j];
			const float4 con_o = collected_conic_opacity[j];

			const bool learned_adaptive = use_3d_response && ut_gate != nullptr;
			float2 d = {0.0f, 0.0f};
			float3 precision_hit = {0.0f, 0.0f, 0.0f};
			float G2D = 0.0f;
			float G3D = 0.0f;
			float gate = learned_adaptive
				? fminf(1.0f, fmaxf(0.0f, ut_gate[global_id]))
				: (use_3d_response ? 1.0f : 0.0f);

			if (!use_3d_response || learned_adaptive)
			{
				const float2 xy = collected_xy[j];
				d = {xy.x - pixf.x, xy.y - pixf.y};
				const float power2D = -0.5f * (
					con_o.x * d.x * d.x + con_o.z * d.y * d.y)
					- con_o.y * d.x * d.y;
				if (power2D <= 0.0f)
					G2D = expf(power2D);
			}

			if (use_3d_response)
			{
				float power3D;
				const bool valid3D = evaluate3DGaussianResponseBackward(
					ray_origin, ray_direction, means3D[global_id],
					cov3D + 6 * global_id, power3D, precision_hit);
				if (valid3D && power3D > -4.482f)
					G3D = expf(power3D);
			}

			const float G = learned_adaptive
				? (1.0f - gate) * G2D + gate * G3D
				: (use_3d_response ? G3D : G2D);
			if (G <= 0.0f)
				continue;
			const float raw_alpha = con_o.w * G;
			const float alpha = min(0.99f, raw_alpha);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			// const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}
			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


			// Helpful reusable temporary variables
			// The forward alpha is clamped at 0.99, so its derivative is zero
			// on the clamped branch.
			const float dL_dG = raw_alpha < 0.99f ? con_o.w * dL_dalpha : 0.0f;

			if (learned_adaptive)
				atomicAdd(&dL_dut_gate[global_id], dL_dG * (G3D - G2D));

			if (use_3d_response && G3D > 0.0f)
			{
				const float branch_weight = learned_adaptive ? gate : 1.0f;
				const float dL_dpower = dL_dG * branch_weight * G3D;

				const float vx = precision_hit.x;
				const float vy = precision_hit.y;
				const float vz = precision_hit.z;

				// d power / d mean = Sigma^{-1} hit
				atomicAdd(
					&dL_dmean3D[global_id].x,
					dL_dpower * vx);
				atomicAdd(
					&dL_dmean3D[global_id].y,
					dL_dpower * vy);
				atomicAdd(
					&dL_dmean3D[global_id].z,
					dL_dpower * vz);

				float* dL_dcov =
					dL_dcov3D + 6 * global_id;

				// Packed covariance order:
				// [xx, xy, xz, yy, yz, zz]
				atomicAdd(
					&dL_dcov[0],
					0.5f * dL_dpower * vx * vx);
				atomicAdd(
					&dL_dcov[1],
					dL_dpower * vx * vy);
				atomicAdd(
					&dL_dcov[2],
					dL_dpower * vx * vz);
				atomicAdd(
					&dL_dcov[3],
					0.5f * dL_dpower * vy * vy);
				atomicAdd(
					&dL_dcov[4],
					dL_dpower * vy * vz);
				atomicAdd(
					&dL_dcov[5],
					0.5f * dL_dpower * vz * vz);
			}
			if ((!use_3d_response || learned_adaptive) && G2D > 0.0f)
			{
				const float branch_weight = learned_adaptive ? (1.0f - gate) : 1.0f;
				const float branch_dL_dG = dL_dG * branch_weight;
				const float gdx = G2D * d.x;
				const float gdy = G2D * d.y;

				const float dG_ddelx =
					-gdx * con_o.x - gdy * con_o.y;
				const float dG_ddely =
					-gdy * con_o.z - gdx * con_o.y;

				atomicAdd(
					&dL_dmean2D[global_id].x,
					branch_dL_dG * dG_ddelx * ddelx_dx);
				atomicAdd(
					&dL_dmean2D[global_id].y,
					branch_dL_dG * dG_ddely * ddely_dy);

				atomicAdd(
					&dL_dconic2D[global_id].x,
					-0.5f * gdx * d.x * branch_dL_dG);
				atomicAdd(
					&dL_dconic2D[global_id].y,
					-0.5f * gdx * d.y * branch_dL_dG);
				atomicAdd(
					&dL_dconic2D[global_id].w,
					-0.5f * gdy * d.y * branch_dL_dG);
			}

			// const float gdx = G * d.x;
			// const float gdy = G * d.y;
			// const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			// const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// // Update gradients w.r.t. 2D mean position of the Gaussian
			// atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			// atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

			// // Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			// atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			// atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
			// atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* cov2D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const int W,int H,
	const float3* dL_dmean2D,
	float* dL_dcov2D,
	const float* dL_dconic,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	bool is_fisheye,
	bool use_ut,
	bool use_3d_response)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	if (!use_3d_response)
	{
		if (use_ut && cov2D_precomp != nullptr)
		{
			computeCov2DPrecompGradCUDA<<<
				(P + 255) / 256, 256>>>(
				P,
				radii,
				cov2D_precomp,
				dL_dconic,
				dL_dcov2D);
		}
		else if(is_fisheye)
		{
			computeCov2DFisheyeCUDA << <(P + 255) / 256, 256 >> > (
				P,
				means3D,
				radii,
				cov3Ds,
				focal_x,
				focal_y,
				tan_fovx,
				tan_fovy,
				viewmatrix,
				dL_dconic,
				(float3*)dL_dmean3D,
				dL_dcov3D);
		}
		else
		{
			computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
				P,
				means3D,
				radii,
				cov3Ds,
				focal_x,
				focal_y,
				tan_fovx,
				tan_fovy,
				viewmatrix,
				dL_dconic,
				(float3*)dL_dmean3D,
				dL_dcov3D);
		}
	}

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		projmatrix,
		viewmatrix,
		focal_x, focal_y,
		campos,
		W, H,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_drot,
		is_fisheye,
		use_ut,
		use_3d_response);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* bg_color,
	const float2* means2D,
	const float4* conic_opacity,
	const float* colors,

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

	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors,
	
	glm::vec3* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dut_gate)
{
	if (tau_k_buffer_size > 0)
	{
		renderTauKBufferBackwardCUDA<NUM_CHANNELS> << <grid, block >> >(
			ranges, point_list, W, H, bg_color, means2D, conic_opacity, colors,
			means3D, cov3D, viewmatrix, cam_pos, fisheye_params,
			focal_x, focal_y, ut_gate, tau_k_buffer_size, dL_dpixels,
			dL_dmean2D, dL_dconic2D, dL_dopacity, dL_dcolors,
			dL_dmean3D, dL_dcov3D, dL_dut_gate);
		return;
	}
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		conic_opacity,
		colors,

		means3D,
		cov3D,
		viewmatrix,
		cam_pos,
		fisheye_params,
		focal_x,
		focal_y,
		use_3d_response,
		ut_gate,

		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors,

		dL_dmean3D,
		dL_dcov3D,
		dL_dut_gate
		);
}
