#-------------------------------------------------
#
# Project created by QtCreator 2012-08-28T00:51:34
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = solecl
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += main.cpp \
    gauss.cpp

HEADERS += \
    gauss.h

OTHER_FILES += \
    cl/gauss.cl
