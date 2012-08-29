#include "jacobi.h"

#include "util.h"

#include <QDebug>
#include <QTime>

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
            for(int j = 0; j < i; j++) {
                sum -= A[i][j]*x2[j];
            }
            for(int j = i+1; j < n; j++) {
                sum -= A[i][j]*x2[j];
            }
            x[i] = sum / A[i][i];
        }
        norm = normMax(x, x2, n);
        qDebug() << "J:"<< it++ << norm;
    }

    return t.elapsed();
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
        buffA.write(A2, n*(n+1)*sizeof(float)); // Необходимо на каждой итерации перезаписывать матрицу исходной, т.к. она модифицируется в процессе вычислений
        memcpy(x2, x, n*sizeof(float));

        jacobi_pre(buffA, xcl, n).waitForFinished();
        jacobi(buffA, xcl, n).waitForFinished();

        xcl.read(x, n);
        norm = normMax(x, x2, n);

        qDebug() << "JCL:"<< it++ << norm;
    }

    return t.elapsed();
}

int JacobiCL2(QVector< QVector<float> > A, int n, QCLContext *context, float *x, float eps)
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

    QCLProgram program;

    program = context->buildProgramFromSourceFile(QLatin1String("cl/jacobi.cl"));

    QCLKernel jacobi_pre2 = program.createKernel("jacobi_pre2");
    QCLKernel jacobi2 = program.createKernel("jacobi2");

    jacobi_pre2.setGlobalWorkSize(n, n);
    jacobi2.setGlobalWorkSize(n, 1);

    QCLBuffer buffA = context->createBufferDevice(n*(n+1)*sizeof(float), QCLMemoryObject::ReadWrite);
    QCLBuffer buffA2 = context->createBufferDevice(n*(n+1)*sizeof(float), QCLMemoryObject::ReadWrite);

    buffA.write(A2, n*(n+1)*sizeof(float));

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
        memcpy(x2, x, n*sizeof(float));

        jacobi_pre2(buffA, buffA2, xcl, n).waitForFinished();
        jacobi2(buffA, buffA2, xcl, n).waitForFinished();

        xcl.read(x, n);
        norm = normMax(x, x2, n);

        qDebug() << "JCL2:"<< it++ << norm;
    }

    return t.elapsed();
}
