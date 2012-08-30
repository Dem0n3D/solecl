#ifndef UTIL_H
#define UTIL_H

#include <opencl/qclcontext.h>

#include <math.h>

#include <QVector>
#include <QTime>

QVector<float> outX(float *x, int n, bool show = true);
QVector< QVector<float> > outM(float *A, int m, int n, bool show = true);

float normMax(float *x1, float *x2, int n);

int multMatrix(QCLBuffer buffA, QCLBuffer buffB, float *C, int n, QCLContext *context);
int multMatrix(QVector< QVector<float> > A, QVector< QVector<float> > B, float *C, int n, QCLContext *context = NULL);

#endif // UTIL_H
