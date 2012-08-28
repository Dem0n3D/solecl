#ifndef GAUSS_H
#define GAUSS_H

#include <time.h>
#include <opencl/qclcontext.h>

#include <QVector>

QVector<float> Gauss(QVector< QVector<float> > A, QVector<float> b, int *r = NULL);
QVector<float> Gauss(QVector< QVector<float> > A, int n, int *r = NULL);

QVector<float> GaussCL(QCLBuffer buffA, int n, QCLContext *context = NULL, int *r = NULL);
QVector<float> GaussCL(QVector< QVector<float> >, int n, QCLContext *context = NULL, int *r = NULL);

#endif // GAUSS_H
