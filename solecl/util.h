#ifndef UTIL_H
#define UTIL_H

#include <math.h>

#include <QVector>

QVector<float> outX(float *x, int n, bool show = true);
QVector< QVector<float> > outM(float *A, int m, int n, bool show = true);

float normMax(float *x1, float *x2, int n);

#endif // UTIL_H
