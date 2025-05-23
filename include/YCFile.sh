#!/bin/bash

mkdir_if_not_exist() {
    local newdirFile=${1}
    if [ ! ${newdirFile} ]; then
        echo_erro "${FUNCNAME} argument is nil!"
        exit 0
    fi
    if [ ! -d ${newdirFile} ]; then
        echo_back "mkdir ${newdirFile}"
    fi
}

rmdir_if_exist() {
    local newdirFile=${1}
    if [ ! ${newdirFile} ]; then
        echo_erro "${FUNCNAME}: argument is nil!"
        exit 0
    fi
    if [ -d ${newdirFile} ]; then
        echo_back "rm -r ${newdirFile}"
    fi
}

touch_if_not_exist() {
    local newFile=${1}
    if [ ! ${newFile} ]; then
        echo_erro "${FUNCNAME}: argument is nil!"
        exit 0
    fi
    if [ ! -f ${newFile} ]; then
        echo_back "touch ${newFile}"
    fi
}

rm_if_exist() {
    local newFile=${1}
    if [ ! ${newFile} ]; then
        echo_erro "${FUNCNAME}: argument is nil!"
        exit 0
    fi
    if [ -f ${newFile} ]; then
        echo_back "rm ${newFile}"
    fi
}
