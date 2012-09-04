#include "gauss.h"

#include "util.h"

#include <fstream>

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

int GaussCL(QCLBuffer buffA, int n, QCLVector<float> &xcl, int gSize, QCLContext *context)
{
    QCLProgram program;

    program = context->buildProgramFromSourceFile(QLatin1String("cl/gauss.cl"));

    QCLKernel gauss_f_pre = program.createKernel("gauss_fwd_pre");
    QCLKernel gauss_f = program.createKernel("gauss_fwd");
    QCLKernel gauss_f2 = program.createKernel("gauss_fwd");

    QCLKernel gauss_bp = program.createKernel("gauss_bwd_prepare");
    QCLKernel gauss_b = program.createKernel("gauss_bwd");

    int m = ((n+1) / gSize) * gSize;

    gauss_f_pre.setGlobalWorkSize(1, n);
    gauss_f.setGlobalWorkSize(m, n);
    gauss_f2.setGlobalWorkSize(n+1-m, n);
    gauss_f2.setGlobalWorkOffset(m, 0, 0);

    gauss_f.setLocalWorkSize(gSize);

    gauss_b.setGlobalWorkSize(1,n);
    gauss_bp.setGlobalWorkSize(1,n);

    std::ofstream log("log.txt");

    QTime t;
    t.start();

    for(int i = 0; i < n-1; i++)
    {
        gauss_f_pre(buffA, n+1, i).waitForFinished();
        gauss_f(buffA, n+1, i).waitForFinished();
        if(gSize > 1)
            gauss_f2(buffA, n+1, i).waitForFinished();

        log << "cl fwd:" << i << std::endl;
    }

    for(int i = n-1; i >= 0; i--)
    {
        gauss_bp(buffA, xcl, n+1, i).waitForFinished();
        gauss_b(buffA, xcl, n+1, i).waitForFinished();

        log << "cl bck:" << n-i << std::endl;
    }

    log.close();

    return t.elapsed();
}
