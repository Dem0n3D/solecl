kernel void square_fwd1(global float *A, int n, int i)
{
    A[i+i*n] = sqrt(A[i+i*n]);
}

kernel void square_fwd2(global float *A, int n, int i)
{
    int j = get_global_id(0);

    if(j > i) {
        float s = A[j+i*n];

        for(int k = 0; k < i; k++) {
            s -= A[i+k*n]*A[j+k*n];
        }

        s /= A[i+i*n];
        A[j+i*n] = s;

        A[j+j*n] -= s*s;
    }
}
