kernel void mult(global float *A, global float *B, global float *C, int n, local float *v)
{
    int i = get_group_id(1);
    int j = get_group_id(2);
    int k = get_local_id(0);

    int g = get_local_size(0);

    local int sem;
    sem = 0;

    v[k] = A[i*n+k] * B[k*n+j];

    atomic_inc(&sem);

    if(sem == g) {
        float sum = 0;
        for(int l = 0; l < g; l++) {
            sum += v[l];
        }
        C[i*n+j] += sum;
    }
}
