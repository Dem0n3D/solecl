#ifndef UTIL_H
#define UTIL_H

#include <opencl/qclcontext.h>

#include <math.h>

#include <QVector>

#include <QTime>

void outX(QVector<float> x);
void outM(QVector< QVector<float> > M);

float normMax(float *x1, float *x2, int n);
float normMax(const QVector<float> &x1, const QVector<float> &x2);

float maxError(const QVector< QVector<float> > &A, const QVector<float> &x);

void matrix2CLBuff(const QVector< QVector<float> > &M, QCLBuffer buff);
void CLBuff2matrix(QCLBuffer buff, QVector< QVector<float> > &M);

int multMatrix(QCLBuffer buffA, QCLBuffer buffB, float *C, int n, QCLContext *context);
int multMatrix(QVector< QVector<float> > A, QVector< QVector<float> > B, float *C, int n, QCLContext *context = NULL);

int multTranspCL(const QVector< QVector<float> > &A, QVector< QVector<float> > &C, int n, QCLContext *context, QCLBuffer *buffC);

#endif // UTIL_H
