#include "gauss.h"

#include "util.h"

#include <QDebug>

int Gauss(QVector< QVector<float> > A, int n, QVector<float> &x, float *D)
{
    QTime t;
    t.start();

    // Прямой ход

    for(int i = 0; i < n-1; i++)
    {
        for(int j = i+1; j < n; j++)
        {
            A[j][i] /= A[i][i];
            for(int k = i+1; k < n; k++)
                A[j][k] -= A[i][k]*A[j][i];
            A[j][n] -= A[i][n]*A[j][i];

            // Не будем обнулять элементы ниже главной диагонали (A[j][i]), мы их просто не будем потом использовать
        }

        qDebug() << "fwd:" << i;
    }

    if(D) {
        *D = 1;
        for(int i = 0; i < n-1; i++) {
            *D *= A[i][i];
        }
    }

    // Обратный ход

    for(int i = n-1; i >= 0; i--)
    {
        float s = 0;
        for(int j = i+1; j < n; j++)
            s += A[i][j]*x[j];
        x[i] = (A[i][n]-s)/A[i][i];

        qDebug() << "bck:" << i;
    }

    return t.elapsed();
}

int GaussCL(QCLBuffer buffA, int n, QVector<float> &x, QCLContext *context, float *D)
{
    if(!context) {
        context = new QCLContext();

        if(!context->create(QCLDevice::GPU)) {
            qFatal("Could not create OpenCL context");
        }
    }

    QCLProgram program;

    program = context->buildProgramFromSourceFile(QLatin1String("cl/gauss.cl"));

    QCLKernel gauss_f_pre = program.createKernel("gauss_fwd_pre");
    QCLKernel gauss_f = program.createKernel("gauss_fwd");
    QCLKernel gauss_f2 = program.createKernel("gauss_fwd");

    QCLKernel gauss_bp = program.createKernel("gauss_bwd_prepare");
    QCLKernel gauss_b = program.createKernel("gauss_bwd");

    int m = 192 * ((n+1) / 192);

    gauss_f_pre.setGlobalWorkSize(1, n);
    gauss_f.setGlobalWorkSize(m, n);
    gauss_f2.setGlobalWorkSize((n+1) % 192, n);
    gauss_f2.setGlobalWorkOffset(m, 0, 0);

    gauss_f.setLocalWorkSize(192);

    gauss_b.setGlobalWorkSize(1,n);
    gauss_bp.setGlobalWorkSize(1,n);

    QCLVector<float> xcl = context->createVector<float>(n, QCLMemoryObject::ReadWrite);

    QTime t;
    t.start();

    for(int i = 0; i < n-1; i++)
    {
        gauss_f_pre(buffA, n+1, i).waitForFinished();
        gauss_f(buffA, n+1, i).waitForFinished();
        gauss_f2(buffA, n+1, i).waitForFinished();

        qDebug() << "cl fwd:" << i;
    }

    if(D) {
        float *A = new float[n*(n+1)];
        buffA.read(0, A, n*(n+1)*sizeof(float));
        *D = 1;
        for(int i = 0; i < n-1; i++) {
            *D *= A[i*(n+1)+i];
        }
    }

    for(int i = n-1; i >= 0; i--)
    {
        gauss_bp(buffA, xcl, n+1, i).waitForFinished();
        gauss_b(buffA, xcl, n+1, i).waitForFinished();

        qDebug() << "cl bck:" << n-i;
    }

    int r = t.elapsed();

    xcl.read(&x[0], n);

    return r;
}

int GaussCL(const QVector< QVector<float> > &A, int n, QVector<float> &x, QCLContext *context, float *D)
{
    if(!context) {
        context = new QCLContext();

        if(!context->create(QCLDevice::GPU)) {
            qFatal("Could not create OpenCL context");
        }
    }

    QCLBuffer buffA = context->createBufferDevice(n*(n+1)*sizeof(float), QCLMemoryObject::ReadWrite);
    matrix2CLBuff(A, buffA);

    return GaussCL(buffA, n, x, context, D);
}
