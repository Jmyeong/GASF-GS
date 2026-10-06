#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
void launch_pair_trace(int,int,int,int,int,const float*,const float*,const float*,const float*,
 const int*,const int*,const int*,const float*,const float*,const float*,const float*,
 int*,float*,int*,float*,cudaStream_t);
std::vector<torch::Tensor> trace(torch::Tensor xyz,torch::Tensor cov,torch::Tensor opacity,
 torch::Tensor rgb,torch::Tensor ids,torch::Tensor ranges,torch::Tensor uv,torch::Tensor view,
 torch::Tensor p,torch::Tensor center,torch::Tensor bg,int W,int H,int K,int M){
 TORCH_CHECK(xyz.is_cuda() && xyz.scalar_type()==torch::kFloat32,"CUDA float32 required");
 TORCH_CHECK(K>=0 && K<=16 && M>0,"Invalid trace bounds");
 for(auto t:{xyz,cov,opacity,rgb,ids,ranges,uv,view,p,center,bg})
  TORCH_CHECK(t.is_contiguous() && t.device()==xyz.device(),"Contiguous tensors on the same device required");
 c10::cuda::CUDAGuard guard(xyz.device());int Q=uv.size(0);
 auto oi=torch::full({Q,M},-1,ids.options());auto oa=torch::zeros({Q,M},xyz.options());
 auto counts=torch::zeros({Q},ids.options());auto color=torch::zeros({Q,3},xyz.options());
 launch_pair_trace(Q,M,W,H,K,xyz.data_ptr<float>(),cov.data_ptr<float>(),opacity.data_ptr<float>(),rgb.data_ptr<float>(),
  ids.data_ptr<int>(),ranges.data_ptr<int>(),uv.data_ptr<int>(),view.data_ptr<float>(),p.data_ptr<float>(),
  center.data_ptr<float>(),bg.data_ptr<float>(),oi.data_ptr<int>(),oa.data_ptr<float>(),counts.data_ptr<int>(),color.data_ptr<float>(),c10::cuda::getCurrentCUDAStream());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {oi,oa,counts,color};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("trace",&trace);}
