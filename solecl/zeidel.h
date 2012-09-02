#ifndef ZEIDEL_H
#define ZEIDEL_H

#include <opencl/qclcontext.h>

#include <QVector>

int multTransp(const QVector< QVector<float> > &A, QVector< QVector<float> > &AtA);

int Zeidel(const QVector< QVector<float> > &A, int n, QVector<float> &x, float eps = 0.001);

int ZeidelCL(QVector< QVector<float> > A, int n, QCLContext *context = NULL, float *x = NULL, float eps = 0.001);

int ZeidelCL2(QCLBuffer buffA, int n, QCLContext *context, float *x = NULL, float eps = 0.001);
int ZeidelCL2(QVector< QVector<float> > A, int n, QCLContext *context = NULL, float *x = NULL, float eps = 0.001);

#endif // ZEIDEL_H
