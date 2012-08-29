#ifndef JACOBI_H
#define JACOBI_H

#include <opencl/qclcontext.h>

#include <QVector>

int Jacobi(QVector< QVector<float> > A, int n, float *x = NULL, float eps = 0.001);

int JacobiCL(QVector< QVector<float> > A, int n, QCLContext *context = NULL, float *x = NULL, float eps = 0.001);

#endif // JACOBI_H
