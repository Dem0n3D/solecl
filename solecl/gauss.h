#ifndef GAUSS_H
#define GAUSS_H

#include <opencl/qclcontext.h>

#include <QVector>

int Gauss(QVector< QVector<float> > A, int n, QVector<float> &x, float *D = NULL);

int GaussCL(QCLBuffer buffA, int n, QCLVector<float> &xcl, int gSize, QCLContext *context = NULL);

#endif // GAUSS_H
