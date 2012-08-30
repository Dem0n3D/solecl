kernel void mult(global float *A, global float *B, global float *C, int n)
{
    int i = get_global_id(1);
    int j = get_global_id(0);

    float v = 0;

    for(int k = 0; k < n; k++) {
        v += A[i*n+k] * B[k*n+j];
    }

    C[i*n+j] = v;
}
