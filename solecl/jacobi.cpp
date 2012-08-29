#include "jacobi.h"

#include <math.h>

#include <QDebug>
#include <QTime>

float normMax(float *x1, float *x2, int n)
{
    float max = 0;
    for(int i = 0; i < n; i++) {
        float norm = fabs(x1[i] - x2[i]);
        max = (max > norm) ? max : norm;
    }
    return max;
}

int Jacobi(QVector< QVector<float> > A, int n, float *x, float eps)
{
    float *x2 = new float[n];

    memset(x, 0, n*sizeof(float));
    memset(x2, 0, n*sizeof(float));

    QTime t;
    t.start();

    int it = 0;
    float norm = 1;
    while(norm > eps) {
        memcpy(x2, x, n*sizeof(float));
        for(int i = 0; i < n; i++) {
            float sum = A[i][n];
            for(int j = 0; j < n; j++) {
                sum -= (i == j) ? 0 : A[i][j]*x2[j];
            }
            x[i] = sum / A[i][i];
        }
        norm = normMax(x, x2, n);
        qDebug() << "J:"<< it++ << norm;
    }

    return t.elapsed();
}

int JacobiCL(float *A, int n, QCLContext *context, float *x, float eps)
{
    QCLProgram program;

    program = context->buildProgramFromSourceFile(QLatin1String("cl/jacobi.cl"));

    QCLKernel jacobi_pre = program.createKernel("jacobi_pre");
    QCLKernel jacobi = program.createKernel("jacobi");

    jacobi_pre.setGlobalWorkSize(n, n);
    jacobi.setGlobalWorkSize(n, 1);

    QCLBuffer buffA = context->createBufferDevice(n*(n+1)*sizeof(float), QCLMemoryObject::ReadWrite);

    QCLVector<float> xcl = context->createVector<float>(n, QCLMemoryObject::ReadWrite);

    float *x2 = new float[n];

    memset(x, 0, n*sizeof(float));
    memset(x2, 0, n*sizeof(float));

    xcl.write(x, n);

    QTime t;
    t.start();

    int it = 0;
    float norm = 1;
    while(norm > eps)
    {
        buffA.write(A, n*(n+1)*sizeof(float)); // Необходимо на каждой итерации перезаписывать матрицу исходной, т.к. она модифицируется в процессе вычислений
        memcpy(x2, x, n*sizeof(float));

        jacobi_pre(buffA, xcl, n).waitForFinished();
        jacobi(buffA, xcl, n).waitForFinished();

        xcl.read(x, n);
        norm = normMax(x, x2, n);

        qDebug() << "JCL:"<< it++ << norm;
    }

    int r = t.elapsed();

    xcl.read(x, n);

    return r;
}

int JacobiCL(QVector< QVector<float> > A, int n, QCLContext *context, float *x, float eps)
{
    if(!context) {
        context = new QCLContext();

        if(!context->create(QCLDevice::GPU)) {
            qFatal("Could not create OpenCL context");
        }
    }

    float *A2 = new float[n*(n+1)];
    for(int i = 0; i < n; i++) {
        memcpy(&A2[i*(n+1)], A[i].data(), (n+1)*sizeof(float));
    }

    return JacobiCL(A2, n, context, x, eps);
}
