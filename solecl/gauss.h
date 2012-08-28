#ifndef GAUSS_H
#define GAUSS_H

#include <opencl/qclcontext.h>

#include <QVector>

int Gauss(QVector< QVector<float> > A, QVector<float> b, float *x = NULL, float *D = NULL);
int Gauss(QVector< QVector<float> > A, int n, float *x = NULL, float *D = NULL);

int GaussCL(QCLBuffer buffA, int n, QCLContext *context = NULL, float *x = NULL, float *D = NULL);
int GaussCL(QVector< QVector<float> >, int n, QCLContext *context = NULL, float *x = NULL, float *D = NULL);

#endif // GAUSS_H
