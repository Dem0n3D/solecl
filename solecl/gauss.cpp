#include "gauss.h"

#include <QDebug>
#include <QTime>

int Gauss(QVector< QVector<float> > A, QVector<float> b, float *x, float *D)
{
    int n = A.size();

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
            b[j] -= b[i]*A[j][i];

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
        x[i] = (b[i]-s)/A[i][i];

        qDebug() << "bck:" << i;
    }

    return t.elapsed();
}

int Gauss(QVector< QVector<float> > A, int n, float *x, float *D)
{
    QVector<float> b(n);

    for(int i = 0; i < n; i++)
    {
        b[i] = A[i][n];
        A[i].remove(n);
    }

    return Gauss(A, b, x, D);
}

int GaussCL(QCLBuffer buffA, int n, QCLContext *context, float *x, float *D)
{
    if(!context) {
        context = new QCLContext();

        if(!context->create(QCLDevice::GPU)) {
            qWarning("Could not create OpenCL context");
        }
    }

    QCLProgram program;

    program = context->buildProgramFromSourceFile(QLatin1String("cl/gauss.cl"));

    QCLKernel gauss_f_pre = program.createKernel("gauss_fwd_pre");
    QCLKernel gauss_f = program.createKernel("gauss_fwd");

    QCLKernel gauss_bp = program.createKernel("gauss_bwd_prepare");
    QCLKernel gauss_b = program.createKernel("gauss_bwd");

    gauss_f_pre.setGlobalWorkSize(1, n);
    gauss_f.setGlobalWorkSize(n+1, n);

    gauss_b.setGlobalWorkSize(1,n);
    gauss_bp.setGlobalWorkSize(1,n);

    QCLVector<float> xcl = context->createVector<float>(n, QCLMemoryObject::ReadWrite);

    QTime t;
    t.start();

    for(int i = 0; i < n-1; i++)
    {
        gauss_f_pre(buffA, n+1, i).waitForFinished();
        gauss_f(buffA, n+1, i).waitForFinished();

        //qDebug() << "cl fwd:" << i;
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

    xcl.read(x, n);

    return r;
}

int GaussCL(QVector< QVector<float> > A, int n, QCLContext *context, float *x, float *D)
{
    if(!context) {
        context = new QCLContext();

        if(!context->create(QCLDevice::GPU)) {
            qWarning("Could not create OpenCL context");
        }
    }

    float *A2 = new float[n*(n+1)];
    for(int i = 0; i < n; i++) {
        memcpy(&A2[i*(n+1)], A[i].data(), (n+1)*sizeof(float));
    }

    QCLBuffer buffA = context->createBufferDevice(n*(n+1)*sizeof(float), QCLMemoryObject::ReadWrite);
    buffA.write(A2, n*(n+1)*sizeof(float));

    return GaussCL(buffA, n, context, x, D);
}
