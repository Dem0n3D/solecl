kernel void zeidel_pre(global float *A, global float *x, int n, int i)
{
    int j = get_global_id(0);

    if(j != i) {
        A[i*(n+1)+j] *= x[j];
    }
}

kernel void zeidel(global float *A, global float *x, int n, int i)
{
    for(int j = 0; j < i; j++) {
        A[i*(n+1)+n] -= A[i*(n+1)+j];
    }

    // Пропуск i

    for(int j = i+1; j < n; j++) {
        A[i*(n+1)+n] -= A[i*(n+1)+j];
    }

    x[i] = A[i*(n+1)+n] / A[i*(n+1)+i];
}
