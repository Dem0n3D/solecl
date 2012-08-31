#include "util.h"

#include <QDebug>

QVector<float> outX(float *x, int n, bool show)
{
    QVector<float> v(n);
    for(int i = 0; i < n; i++) {
        v[i] = x[i];
    }

    if(show)
        qDebug() << v;

    return v;
}

QVector< QVector<float> > outM(float *A, int m, int n, bool show)
{
    QVector< QVector<float> > M(m, QVector<float>(n));
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            M[i][j] = A[i*n+j];
        }
        if(show)
            qDebug() <<M[i];
    }

    return M;
}

float normMax(float *x1, float *x2, int n) // Норма-максимум разности векторов
{
    float max = 0;
    for(int i = 0; i < n; i++) {
        float norm = fabs(x1[i] - x2[i]);
        max = (max > norm) ? max : norm;
    }
    return max;
}

int multMatrix(QCLBuffer buffA, QCLBuffer buffB, float *C, int n, QCLContext *context)
{
    size_t mSize = n*n*sizeof(float);
    size_t gSize = context->defaultDevice().maximumWorkItemSize().height();
    gSize = (n < gSize) ? n : gSize;

    QCLProgram program;
    program = context->buildProgramFromSourceFile(QLatin1String("cl/matrix.cl"));

    QCLKernel mult = program.createKernel("mult");
    mult.setGlobalWorkSize(n, n, n);
    mult.setLocalWorkSize(gSize);

    QCLBuffer buffC = context->createBufferDevice(mSize, QCLMemoryObject::ReadWrite);
    memset(C, 0, mSize);
    buffC.write(C, mSize);

    clSetKernelArg(mult.kernelId(), 4, gSize*sizeof(cl_float), NULL);

    QTime t;
    t.start();

    mult(buffA, buffB, buffC, n);

    buffC.read(C, mSize);

    return t.elapsed();
}


int multMatrix(QVector< QVector<float> > A, QVector< QVector<float> > B, float *C, int n, QCLContext *context)
{
    if(!context) {
        context = new QCLContext();

        if(!context->create(QCLDevice::GPU)) {
            qFatal("Could not create OpenCL context");
        }
    }

    size_t mSize = n*n*sizeof(float);

    float *A2 = new float[n*n];
    for(int i = 0; i < n; i++) {
        memcpy(&A2[i*n], A[i].data(), n*sizeof(float));
    }

    float *B2 = new float[n*n];
    for(int i = 0; i < n; i++) {
        memcpy(&B2[i*n], B[i].data(), n*sizeof(float));
    }

    memset(C, 0, mSize);

    QCLBuffer buffA = context->createBufferDevice(mSize, QCLMemoryObject::ReadWrite);
    QCLBuffer buffB = context->createBufferDevice(mSize, QCLMemoryObject::ReadWrite);

    buffA.write(A2, mSize);
    buffB.write(B2, mSize);

    return multMatrix(buffA, buffB, C, n, context);
}
