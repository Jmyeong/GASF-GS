// Ray/response functions shared with the fisheye CUDA rasterizer.
// Copyright (C) 2023, Inria, GRAPHDECO. Non-commercial research/evaluation only.
// See the accompanying LICENSE.md.
#include <cuda_runtime.h>
#include <cmath>
namespace pair_trace {
__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

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


__global__ void trace_kernel(int Q,int M,int W,int H,int K,
 const float* xyz,const float* cov,const float* opacity,const float* rgb,
 const int* ids,const int* ranges,const int* uv,const float* view,const float* p,
 const float* center,const float* bg,int* out_ids,float* out_alpha,int* counts,float* out_rgb) {
 int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=Q)return;
 int x=uv[2*q],y=uv[2*q+1],tile=(y/16)*((W+15)/16)+x/16;
 float3 origin={center[0],center[1],center[2]};
 float3 d=fisheyePixelToWorldRay(make_float2(x,y),W,H,p,p[0],p[1],view);
 int hid[16],hc=0;float ht[16],ha[16];int count=0;
 float T=1.f,C[3]={0,0,0};bool stopped=false;
 // Append every hit for counterfactual graph construction; RGB retains native early stop.
 #define APPEND(ID,ALPHA) { int gi=(ID);float trace_alpha=(ALPHA); \
   if(count<M){out_ids[q*M+count]=gi;out_alpha[q*M+count]=trace_alpha;} ++count; \
   if(!stopped){float next=T*(1.f-trace_alpha);if(next<.0001f)stopped=true; \
     else {for(int c=0;c<3;++c)C[c]+=rgb[3*gi+c]*trace_alpha*T;T=next;}} }
 for(int j=ranges[2*tile];j<ranges[2*tile+1];++j){
  int id=ids[j];float power,tau;float3 mean={xyz[3*id],xyz[3*id+1],xyz[3*id+2]};
  if(!evaluate3DGaussianResponse(origin,d,mean,cov+6*id,power,tau)||power<=-4.482f)continue;
  float al=fminf(.99f,opacity[id]*expf(power));if(al<1.f/255.f)continue;
  if(K==0){APPEND(id,al);continue;}
  if(hc==K){APPEND(hid[0],ha[0]);for(int k=1;k<hc;++k){hid[k-1]=hid[k];ht[k-1]=ht[k];ha[k-1]=ha[k];}--hc;}
  int pos=hc;while(pos>0&&ht[pos-1]>tau){hid[pos]=hid[pos-1];ht[pos]=ht[pos-1];ha[pos]=ha[pos-1];--pos;}
  hid[pos]=id;ht[pos]=tau;ha[pos]=al;++hc;
 }
 for(int k=0;k<hc;++k){APPEND(hid[k],ha[k]);}
 counts[q]=count;for(int c=0;c<3;++c)out_rgb[3*q+c]=C[c]+T*bg[c];
 #undef APPEND
}
}
void launch_pair_trace(int Q,int M,int W,int H,int K,
 const float* xyz,const float* cov,const float* opacity,const float* rgb,
 const int* ids,const int* ranges,const int* uv,const float* view,const float* p,
 const float* center,const float* bg,int* out_ids,float* out_alpha,int* counts,float* out_rgb,cudaStream_t stream){
 pair_trace::trace_kernel<<<(Q+63)/64,64,0,stream>>>(Q,M,W,H,K,xyz,cov,opacity,rgb,ids,ranges,uv,view,p,center,bg,out_ids,out_alpha,counts,out_rgb);
}
