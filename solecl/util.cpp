#include "util.h"

#include <QDebug>

void outX(float *x, int n)
{
    QVector<float> v(n);
    for(int i = 0; i < n; i++) {
        v[i] = x[i];
    }
    qDebug() << v;
}
