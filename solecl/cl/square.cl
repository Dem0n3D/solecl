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

kernel void square_y1(global float *U, global float *x, int n, int i)
{
    x[i] = U[n-1+i*n] / U[i+i*n];
}

kernel void square_y2(global float *U, global float *x, int n, int i)
{
    local float y;
    y = x[i];

    int k = get_global_id(0);

    if(k > i) {
        U[n-1+k*n] -= U[k+i*n] * y;
    }
}

kernel void square_x1(global float *U, global float *x, int n, int i)
{
    x[i] /= U[i+i*n];
}

kernel void square_x2(global float *U, global float *x, int n, int i)
{
    local float y;
    y = x[i];

    int k = get_global_id(0);

    x[k] -= y * U[i+k*n];
}
