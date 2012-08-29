#include "util.h"

#include <QDebug>

QVector<float> outX(float *x, int n, bool show)
{
    QVector<float> v(n);
    for(int i = 0; i < n; i++) {
        v[i] = x[i];
    }

    if(show)
        qDebug() << v;

    return v;
}

QVector< QVector<float> > outM(float *A, int m, int n, bool show)
{
    QVector< QVector<float> > M(m, QVector<float>(n));
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            M[i][j] = A[i*n+j];
        }
        if(show)
            qDebug() <<M[i];
    }

    return M;
}

float normMax(float *x1, float *x2, int n) // Норма-максимум разности векторов
{
    float max = 0;
    for(int i = 0; i < n; i++) {
        float norm = fabs(x1[i] - x2[i]);
        max = (max > norm) ? max : norm;
    }
    return max;
}
