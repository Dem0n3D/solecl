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

kernel void multTransp(global float *A, global float *C, int n)
{
    int i = get_global_id(1);
    int j = get_global_id(0);

    if(j >= i) {
        float v = 0;

        for(int k = 0; k < n; k++) {
            v += A[k*(n+1)+i] * A[k*(n+1)+j];
        }

        C[i*(n+1)+j] = v;
        C[j*(n+1)+i] = v;
    }
}

kernel void multTranspB(global float *A, global float *C, int n)
{
    int i = get_global_id(0);

    float v = 0;

    for(int k = 0; k < n; k++) {
        v += A[k*(n+1)+i] * A[k*(n+1)+n];
    }

    C[i*(n+1)+n] = v;
}
