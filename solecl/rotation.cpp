#include "rotation.h"

#include "util.h"

int Rotation(const QVector< QVector<float> > &A, int n, QVector<float> &x)
{
    QVector< QVector<float> > U(n, QVector<float>(n, 0));

    QTime t;
    t.start();

    float c,s;

    for(int k = 0; k < n; k++) {
        for(int i = k+1; i < n; i++) {
            //c = A[k][k] / sqrt();
        }
    }

    return t.elapsed();
}
