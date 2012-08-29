#ifndef UTIL_H
#define UTIL_H

#include <QVector>

QVector<float> outX(float *x, int n, bool show = true);
QVector< QVector<float> > outM(float *A, int m, int n, bool show = true);

#endif // UTIL_H
