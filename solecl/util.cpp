#include "util.h"

#include <sstream>

#include <QDebug>

void outX(QVector<float> x)
{
    qDebug() << "OutX:";
    qDebug() << x;
}

void outM(QVector< QVector<float> > M)
{
    qDebug() << "OutM:";

    for(int i = 0; i < M.size(); i++) {
            qDebug() << M[i];
    }
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

float normMax(const QVector<float> &x1, const QVector<float> &x2) // Норма-максимум разности векторов
{
    float max = 0;
    for(int i = 0; i < x1.size(); i++) {
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

int multTransp(const QVector< QVector<float> > &A, QVector< QVector<float> > &C, int n, QCLContext *context, QCLBuffer *buffC)
{
    size_t Msize = n*(n+1)*sizeof(float);

    QCLBuffer buffA = context->createBufferDevice(Msize, QCLMemoryObject::ReadWrite);
    for(int i = 0; i < n; i++)
        buffA.write(i*(n+1)*sizeof(float), A[i].data(), (n+1)*sizeof(float));

    QCLProgram program;

    program = context->buildProgramFromSourceFile(QLatin1String("cl/matrix.cl"));

    QCLKernel multTransp = program.createKernel("multTransp");
    QCLKernel multTranspB = program.createKernel("multTranspB");

    multTransp.setGlobalWorkSize(n, n);
    multTranspB.setGlobalWorkSize(n);

    if(!buffC)
        buffC = new QCLBuffer();

    *buffC = context->createBufferDevice(Msize, QCLMemoryObject::ReadWrite);

    QTime t;
    t.start();

    multTransp(buffA, *buffC, n);
    multTranspB(buffA, *buffC, n);

    for(int i = 0; i < n; i++)
        buffC->read(i*(n+1)*sizeof(float), &C[i][0], (n+1)*sizeof(float));

    return t.elapsed();
}

