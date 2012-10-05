#ifndef SQUARE_H
#define SQUARE_H

#include <opencl/qclcontext.h>

#include <QVector>

int Square(QVector< QVector<float> > &A, int n, float *x = NULL);

int SquareCL(QCLBuffer buffA, int n, float *x = NULL, QCLContext *context = NULL);
int SquareCL(QVector<QVector<float> > &A, int n, float *x = NULL, QCLContext *context = NULL);

#endif // SQUARE_H
