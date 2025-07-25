#!/bin/bash

# Copyright 2025 Suanzhi Future Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

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
