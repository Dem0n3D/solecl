kernel void zeidel_pre(global float *A, global float *x, int n, int i)
{
    int j = get_global_id(0);

    float v = A[i*(n+1)+j];

    if(j != i) {
        v *= x[j];
    }

    A[i*(n+1)+j] = v;
}

kernel void zeidel(global float *A, global float *x, int n, int i)
{
    float v = 0;

    for(int j = 0; j < i; j++) {
        v += A[i*(n+1)+j];
    }

    // Пропуск i

    for(int j = i+1; j < n; j++) {
        v += A[i*(n+1)+j];
    }

    x[i] = (A[i*(n+1)+n] - v) / A[i*(n+1)+i];
}

kernel void zeidel_pre2(global float *A, global float *A2, global float *x, int n, int i)
{
    int j = get_global_id(0);

    if(j != i) {
        A2[i*(n+1)+j] = A[i*(n+1)+j]*x[j];
    }
}

kernel void zeidel2(global float *A, global float *A2, global float *x, int n, int i)
{
    A2[i*(n+1)+n] = A[i*(n+1)+n];

    for(int j = 0; j < i; j++) {
        A2[i*(n+1)+n] -= A2[i*(n+1)+j];
    }

    // Пропуск i

    for(int j = i+1; j < n; j++) {
        A2[i*(n+1)+n] -= A2[i*(n+1)+j];
    }

    x[i] = A2[i*(n+1)+n] / A[i*(n+1)+i];
}
