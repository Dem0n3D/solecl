#include "zeidel.h"

#include "util.h"

#include <QDebug>
#include <QTime>

int Zeidel(QVector< QVector<float> > A, int n, float *x, float eps)
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
                sum -= A[i][j]*x[j];
            }
            for(int j = i+1; j < n; j++) {
                sum -= A[i][j]*x2[j];
            }
            x[i] = sum / A[i][i];
        }
        norm = normMax(x, x2, n);
        qDebug() << "Z:"<< it++ << norm;
    }

    return t.elapsed();
}

int ZeidelCL(QVector< QVector<float> > A, int n, QCLContext *context, float *x, float eps)
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

    program = context->buildProgramFromSourceFile(QLatin1String("cl/zeidel.cl"));

    QCLKernel zeidel_pre = program.createKernel("zeidel_pre");
    QCLKernel zeidel = program.createKernel("zeidel");

    zeidel_pre.setGlobalWorkSize(n, 1);
    zeidel.setGlobalWorkSize(n, 1);

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

        for(int i = 0; i < n; i++) {
            zeidel_pre(buffA, xcl, n, i).waitForFinished();
            zeidel(buffA, xcl, n, i).waitForFinished();
        }

        xcl.read(x, n);
        norm = normMax(x, x2, n);

        qDebug() << "ZCL:"<< it++ << norm;
    }

    return t.elapsed();
}

int ZeidelCL2(QVector< QVector<float> > A, int n, QCLContext *context, float *x, float eps)
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

    program = context->buildProgramFromSourceFile(QLatin1String("cl/zeidel.cl"));

    QCLKernel zeidel_pre2 = program.createKernel("zeidel_pre2");
    QCLKernel zeidel2 = program.createKernel("zeidel2");

    zeidel_pre2.setGlobalWorkSize(n, 1);
    zeidel2.setGlobalWorkSize(n, 1);

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

        for(int i = 0; i < n; i++) {
            zeidel_pre2(buffA, buffA2, xcl, n, i).waitForFinished();
            zeidel2(buffA, buffA2, xcl, n, i).waitForFinished();
        }

        xcl.read(x, n);
        norm = normMax(x, x2, n);

        qDebug() << "ZCL2:"<< it++ << norm;
    }

    return t.elapsed();
}
