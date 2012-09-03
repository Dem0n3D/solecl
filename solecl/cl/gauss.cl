kernel void gauss_fwd_pre(global float *A, int n, int i)
{
    int y = get_global_id(0) + i + 1;

    A[i+y*n] /= A[i+i*n];
}

kernel void gauss_fwd(global float *A, int n, int i)
{
    int x = get_global_id(0) + i + 1;
    int y = get_global_id(1) + i + 1;

    A[x+y*n] -= A[x+i*n]*A[i+y*n];
}

kernel void gauss_bwd_prepare(global float *A, global float *x, int n, int i)
{
    x[i] = A[(n-1)+i*n]/A[i+i*n];
}

kernel void gauss_bwd(global float *A, global float *x, int n, int i)
{
    int y = get_global_id(0);

    A[(n-1)+y*n] -= x[i]*A[i+y*n];
}
