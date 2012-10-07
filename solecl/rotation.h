#ifndef ROTATION_H
#define ROTATION_H

#include <opencl/qclcontext.h>

#include <QVector>

int Rotation(const QVector< QVector<float> > &A, int n, QVector<float> &x);

//int RotationCL(QCLBuffer buffA, int n, QVector<float> &x, QCLContext *context = NULL);
//int RotationCL(const QVector<QVector<float> > &A, int n, QVector<float> &x, QCLContext *context = NULL);

#endif // ROTATION_H
