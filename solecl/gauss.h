#ifndef GAUSS_H
#define GAUSS_H

#include <opencl/qclcontext.h>

#include <QVector>

int Gauss(QVector< QVector<float> > A, int n, QVector<float> &x, float *D = NULL);

int GaussCL(QCLBuffer buffA, int n, float *x = NULL, QCLContext *context = NULL, float *D = NULL);
int GaussCL(const QVector< QVector<float> > &A, int n, QVector<float> &x, QCLContext *context = NULL, float *D = NULL);

#endif // GAUSS_H
