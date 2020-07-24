__kernel void sum(__global float16* a_g, __global const float16* b_g, __global float16* res_g)
{
  int gid = get_global_id(0);
  res_g[gid] = a_g[gid] + b_g[gid];
}
