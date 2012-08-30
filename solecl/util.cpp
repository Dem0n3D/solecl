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
    size_t Msize = n*n*sizeof(float);

    memset(C, 0, Msize);

    QCLProgram program;

    program = context->buildProgramFromSourceFile(QLatin1String("cl/matrix.cl"));

    QCLKernel mult = program.createKernel("mult");

    mult.setGlobalWorkSize(n, n);

    QCLBuffer buffC = context->createBufferDevice(Msize, QCLMemoryObject::ReadWrite);

    QTime t;
    t.start();

    mult(buffA, buffB, buffC, n);

    buffC.read(C, Msize);

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

    size_t Msize = n*n*sizeof(float);

    float *A2 = new float[n*n];
    for(int i = 0; i < n; i++) {
        memcpy(&A2[i*n], A[i].data(), n*sizeof(float));
    }

    float *B2 = new float[n*n];
    for(int i = 0; i < n; i++) {
        memcpy(&B2[i*n], B[i].data(), n*sizeof(float));
    }

    memset(C, 0, Msize);

    QCLBuffer buffA = context->createBufferDevice(Msize, QCLMemoryObject::ReadWrite);
    QCLBuffer buffB = context->createBufferDevice(Msize, QCLMemoryObject::ReadWrite);

    buffA.write(A2, Msize);
    buffB.write(B2, Msize);

    return multMatrix(buffA, buffB, C, n, context);
}
