#!/bin/bash

source $(dirname "$0")/include/YCLog.sh

show_usage() {
    appname=$0
    echo "Usage: ${appname} [command], e.g., ${appname} limit 1"
    echo "  -- limit [device_id]                limit the gpu memory"
    echo "  -- reset                           reset the gpu memory"
    echo "  -- monitor                         monitor the gpu memory"
}

if (( $# == 0 )); then
    echo_warn "Argument cannot be NULL!"
    show_usage
    exit 0
fi

global_choice=$1
case ${global_choice} in
    "limit")
        device_id=$2
        freq=$3
        echo_back "nvidia-smi -lgc 0,${freq} -i ${device_id}"
        ;;
    "reset")
        echo_back "nvidia-smi -rgc"
        ;;
    "monitor")
        echo_back "nvidia-smi dmon"
        ;;
    *)
        echo "Unrecognized argument!"
        show_usage
esac
