#ifndef ZEIDEL_H
#define ZEIDEL_H

#include <opencl/qclcontext.h>

#include <QVector>

int Zeidel(QVector< QVector<float> > A, int n, float *x = NULL, float eps = 0.001, int *tmult = NULL);

int ZeidelCL(QVector< QVector<float> > A, int n, QCLContext *context = NULL, float *x = NULL, float eps = 0.001);

int ZeidelCL2(QCLBuffer buffA, int n, QCLContext *context, float *x = NULL, float eps = 0.001);
int ZeidelCL2(QVector< QVector<float> > A, int n, QCLContext *context = NULL, float *x = NULL, float eps = 0.001);

#endif // ZEIDEL_H
