#include "jacobi.h"

#include <math.h>

#include <QDebug>
#include <QTime>

float normMax(float *x1, float *x2, int n)
{
    float max = 0;
    for(int i = 0; i < n; i++) {
        float norm = fabs(x1[i] - x2[i]);
        max = (max > norm) ? max : norm;
    }
    return max;
}

int Jacobi(QVector< QVector<float> > A, int n, float *x, float eps)
{
    float *x2 = new float[n];

    memset(x, 0, n*sizeof(float));
    memset(x2, 0, n*sizeof(float));

    QTime t;
    t.start();

    int it = 0;
    float norm = 1;
    while(norm > eps) {
        memcpy(x2, x, n*sizeof(float));
        for(int i = 0; i < n; i++) {
            float sum = A[i][n];
            for(int j = 0; j < n; j++) {
                sum -= (i == j) ? 0 : A[i][j]*x2[j];
            }
            x[i] = sum / A[i][i];
        }
        norm = normMax(x, x2, n);
        qDebug() << it++ << norm;
    }

    return t.elapsed();
}
