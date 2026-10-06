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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& means2D_precomp,
	const torch::Tensor& cov2D_precomp,
	const torch::Tensor& candidate_means2D,
	const torch::Tensor& candidate_cov2D,
	const torch::Tensor& candidate_counts,
	const torch::Tensor& ut_gate,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& fisheye_params,
	const bool is_fisheye,
	const bool use_ut,
	const bool use_3d_response,
	const bool sort_by_camera_distance,
	const bool sort_by_tile_tau,
	const int tau_k_buffer_size,
	const bool prefiltered,
	const bool debug,
	const bool fisheye_no_center_cull)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  if (use_ut)
  {
	if (means2D_precomp.ndimension() != 2 ||
		means2D_precomp.size(0) != means3D.size(0) ||
		means2D_precomp.size(1) != 3)
	{
		AT_ERROR("means2D_precomp must have shape (num_points, 3)");
	}

	if (cov2D_precomp.ndimension() != 2 ||
		cov2D_precomp.size(0) != means3D.size(0) ||
		cov2D_precomp.size(1) != 3)
	{
		AT_ERROR("cov2D_precomp must have shape (num_points, 3)");
	}

	if (!means2D_precomp.is_cuda() || !cov2D_precomp.is_cuda())
	{
		AT_ERROR("UT precomputed tensors must be CUDA tensors");
	}
  }
  if (fisheye_params.numel() != 0 && fisheye_params.numel() != 8)
  {
	AT_ERROR("fisheye_params must be empty or contain [fx, fy, cx, cy, k1..k4]");
  }
  if (candidate_counts.numel() != 0)
  {
	if (candidate_means2D.sizes() != torch::IntArrayRef({means3D.size(0), 2, 2}) ||
		candidate_cov2D.sizes() != torch::IntArrayRef({means3D.size(0), 2, 3}) ||
		candidate_counts.size(0) != means3D.size(0))
		AT_ERROR("split candidates must have shapes [P,2,2], [P,2,3], [P]");
	if (!candidate_means2D.is_cuda() || !candidate_cov2D.is_cuda() || !candidate_counts.is_cuda())
		AT_ERROR("split candidate tensors must be CUDA tensors");
  }
  if (fisheye_params.numel() == 8 && !fisheye_params.is_cuda())
  {
	AT_ERROR("fisheye_params must be a CUDA tensor");
  }
  if (ut_gate.numel() != 0)
  {
	if (ut_gate.ndimension() != 1 || ut_gate.size(0) != means3D.size(0))
		AT_ERROR("ut_gate must be empty or have shape (num_points,)");
	if (!ut_gate.is_cuda())
		AT_ERROR("ut_gate must be a CUDA tensor");
  }
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		use_ut
			? means2D_precomp.contiguous().data_ptr<float>()
			: nullptr,
		use_ut
			? cov2D_precomp.contiguous().data_ptr<float>()
			: nullptr,
		candidate_counts.numel() == P ? candidate_means2D.contiguous().data_ptr<float>() : nullptr,
		candidate_counts.numel() == P ? candidate_cov2D.contiguous().data_ptr<float>() : nullptr,
		candidate_counts.numel() == P ? candidate_counts.contiguous().data_ptr<int>() : nullptr,
		ut_gate.numel() == P ? ut_gate.contiguous().data_ptr<float>() : nullptr,
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		fisheye_params.numel() == 8
			? fisheye_params.contiguous().data_ptr<float>()
			: nullptr,
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		is_fisheye,
			use_ut,
			use_3d_response,
			sort_by_camera_distance,
			sort_by_tile_tau,
			tau_k_buffer_size,
			debug, fisheye_no_center_cull);
  }
  return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& cov2D_precomp,
	const torch::Tensor& ut_gate,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& fisheye_params,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool is_fisheye,
	const bool use_ut,
	const bool use_3d_response,
	const int tau_k_buffer_size,
	const bool debug) 
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcov2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dut_gate = torch::zeros({P}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  use_ut
		? cov2D_precomp.contiguous().data_ptr<float>()
		: nullptr,
	  ut_gate.numel() == P ? ut_gate.contiguous().data_ptr<float>() : nullptr,
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  fisheye_params.numel() == 8
		? fisheye_params.contiguous().data_ptr<float>()
		: nullptr,
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dcov2D.contiguous().data<float>(),
	  dL_dut_gate.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  is_fisheye,
	  use_ut,
	  use_3d_response,
	  tau_k_buffer_size,
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcov2D, dL_dut_gate, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}
