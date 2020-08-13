__kernel void sum_mul(
    __global const float<VECSIZE> *a_g, __global const float<VECSIZE> *b_g, 
    __global float<VECSIZE>  *res_add, __global float<VECSIZE> *res_mul)
{
  int gid = get_global_id(0);
  res_add[gid] = a_g[gid] + b_g[gid];
  res_mul[gid] = a_g[gid] * b_g[gid];
}