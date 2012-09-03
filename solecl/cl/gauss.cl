kernel void gauss_fwd_pre(global float *A, int n, int i)
{
    int y = get_global_id(1);

    if(y > i)
    {
        A[i+y*n] /= A[i+i*n];
    }
}

kernel void gauss_fwd(global float *A, int n, int i)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(y > i && x > i)
    {
        A[x+y*n] -= A[x+i*n]*A[i+y*n];
    }
}

////////////////////////////////////////////////////////////

kernel void gauss_bwd_prepare(global float *A, global float *x, int n, int i)
{
    int y = get_global_id(1);

    if(y == i)
    {
        x[i] = A[(n-1)+i*n]/A[i+i*n];
    }
}

kernel void gauss_bwd(global float *A, global float *x, int n, int i)
{
    int y = get_global_id(1);

    if(y < i)
    {
        A[(n-1)+y*n] -= x[i]*A[i+y*n];
    }
}
