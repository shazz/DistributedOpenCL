__kernel void test_sqrt(
    __global const float<VECSIZE> *a_g, __global const float<VECSIZE> *b_g, 
    __global float<VECSIZE>  *res_add, __global float<VECSIZE> *res_mul)
{
  int gid = get_global_id(0);
  __private float<VECSIZE> t1;
  __private float<VECSIZE> t2;

  t1 = sqrt(a_g[gid]);
  t2 = sqrt(b_g[gid]);

  for(int i=0; i<100000; i++){
    t1 = sqrt(a_g[gid] * t2);
    t2 = sqrt(b_g[gid] * t1);
  }

  res_add[gid] = t1 + t2;
  res_mul[gid] = t1 * t2;
}