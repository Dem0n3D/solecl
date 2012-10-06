#ifndef SQUARE_H
#define SQUARE_H

#include <opencl/qclcontext.h>

#include <QVector>

int Square(QVector< QVector<float> > &A, int n, QVector<float> &x);

int SquareCL(QCLBuffer buffA, int n, QVector<float> &x, QCLContext *context = NULL);
int SquareCL(QVector<QVector<float> > &A, int n, QVector<float> &x, QCLContext *context = NULL);

#endif // SQUARE_H
