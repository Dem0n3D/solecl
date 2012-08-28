#include "gauss.h"

#include <QDebug>

QVector<float> Gauss(QVector< QVector<float> > A, QVector<float> b, int *r)
{
    int n = A.size();

    int t = clock();

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

    // Обратный ход

    QVector<float> x = QVector<float>(n);

    for(int i = n-1; i >= 0; i--)
    {
        float s = 0;
        for(int j = i+1; j < n; j++)
            s += A[i][j]*x[j];
        x[i] = (b[i]-s)/A[i][i];

        qDebug() << "bck:" << i;
    }

    if(r)
        *r = (clock() - t)/float(CLOCKS_PER_SEC);

    return x;
}

QVector<float> Gauss(QVector< QVector<float> > A, int n, int *r)
{
    QVector<float> b(n);

    for(int i = 0; i < n; i++)
    {
        b[i] = A[i][n];
        A[i].remove(n);
    }

    return Gauss(A, b, r);
}

QVector<float> GaussCL(QCLBuffer buffA, int n, QCLContext *context, int *r)
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

    QCLVector<float> x = context->createVector<float>(n, QCLMemoryObject::ReadWrite);

    int t = clock();

    for(int i = 0; i < n-1; i++)
    {
        gauss_f_pre(buffA, n+1, i).waitForFinished();
        gauss_f(buffA, n+1, i).waitForFinished();

        qDebug() << "cl fwd:" << i;
    }

    for(int i = n-1; i >= 0; i--)
    {
        gauss_bp(buffA, x, n+1, i).waitForFinished();
        gauss_b(buffA, x, n+1, i).waitForFinished();

        qDebug() << "cl bck:" << n-i;
    }

    if(r)
        *r = (clock() - t)/float(CLOCKS_PER_SEC);

    QVector<float> x2(n);

    x.read(&x2[0], n);

    return x2;
}

QVector<float> GaussCL(float *A, int n, QCLContext *context, int *r)
{
    if(!context) {
        context = new QCLContext();

        if(!context->create(QCLDevice::GPU)) {
            qWarning("Could not create OpenCL context");
        }
    }

    QCLBuffer buffA = context->createBufferDevice(n*(n+1)*sizeof(float), QCLMemoryObject::ReadWrite);
    buffA.write(A, n*(n+1)*sizeof(float));

    return GaussCL(buffA, n, context, r);
}
