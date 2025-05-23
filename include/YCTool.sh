#!/bin/bash

# Para :
#     ${1} : time (seconds)
#
progress_bar() {
    if [ ! ${1} ]; then
        echo_erro "usage: ${FUNCNAME} time"
	exit 0
    fi
    if [ ${1} -le 0 ]; then
        echo_erro "usage: ${FUNCNAME} time"
	exit 0
    fi
    bTool=''
    curTool=0
    intervalTool=1
    maxlenTool=50
    temptimeTool=$((${1}-1))
    intervalTool=`expr ${temptimeTool} / ${maxlenTool}`
    intervalTool=$((${intervalTool}+1))
    barlenTool=$((${1}/${intervalTool}))
    while [ ${curTool} -le ${1} ]
    do
        printf "[${color_yellow}WAIT${color_reset}] [${color_green}%-${barlenTool}s${color_reset}] [%2d/%d] \r" "$bTool" "${curTool}" "${1}";
        bTool+="#"
        ((curTool=curTool+${intervalTool}))
        sleep ${intervalTool}
    done
    echo ""
}

get_terminal_width() {
    local _width_=`stty size|awk '{print $2}'`
    echo ${_width_}
}

##
# This function will block until it cannot find $1 for more than $2 seconds
#
# $1: process name
# $2: interval
#
block_on_proc() {
    local proc_name=$1
    local time_interval=$2
    local interval_counter=0
    while true
    do
        if [ ${interval_counter} -gt ${time_interval} ]; then
            break
        fi
        local pids=(`pgrep ${proc_name}`)
        if [ ${#pids[@]} -eq 0 ]; then
            ((interval_counter=interval_counter+1))
        else
            ((interval_counter=0))
        fi
        sleep 1
    done
}

##
# Kill one or more processes on the local machine or remote server
#
# @para: $1:
#   process name
# @para: $2 (optional)
#   Login infomation for a remote access, e.g., bob@192.168.0.1
#
kill_proc() {
    local _proc=${1}
    if [ ! ${_proc} ]; then
        echo_info "NULL parameter!"
        return
    fi
    local _user_info=$2       # e.g., bob@192.168.0.1
    if [ ${_user_info} ]; then
        local _pids=`ssh ${_user_info} 'pgrep iperf'`
	if [ ${#_pids[@]} -eq 0 ]; then
            echo_info "Cannot find ${_proc} on ${_user_info}"
        else
            for _proc_pid in ${_pids[@]}
            do
                echo_back "ssh ${_user_info} 'sudo kill -9 ${_proc_pid}'"
            done
        fi
    else
        local _pids=(`pgrep ${_proc}`)
	if [ ${#_pids[@]} -eq 0 ]; then
            echo_info "Cannot find ${_proc}"
        else
            for _proc_pid in ${_pids[@]}
            do
                echo_back "sudo kill -9 ${_proc_pid}"
            done
        fi
    fi
}
