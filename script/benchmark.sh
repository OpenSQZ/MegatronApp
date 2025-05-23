#!/bin/bash

SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd ${SHELL_FOLDER}

source ../include/YCFile.sh
source ../include/YCLog.sh
source ../include/YCTool.sh
source ../include/YCOS.sh

####################################################
####################################################

show_usage() {
    appname=$0
    echo_info "Usage: ${appname} [command], e.g., ${appname} install v3"
    echo_info "  -- install [v3|v4]"
    echo_info "  -- setup [v3|v4]"
    echo_info "  -- train"
    echo_info "  -- help                          show help message"
}


install(){
    local version=${1} 
    case ${version} in
        "v3")
            install_v3.0
            ;;
        "v4")
            install_v4.0
            ;;
        *)
            show_usage
            ;;
    esac
}

install_v3.0() {
    echo_back "cd ${SHELL_FOLDER}"
    echo_info "install megatron v3.02, v3.02 requires cuda11+torch1.0, please check your dependency"
    echo_back "wget --no-check-certificate --content-disposition https://github.com/NVIDIA/Megatron-LM/archive/refs/tags/v3.0.2.zip"
    echo_back "unzip Megatron-LM-3.0.2.zip"
    echo_back  "cd ../"
    echo_back "mkdir Megatron"
    echo_back "mv script/Megatron-LM-3.0.2/*  Megatron"
    echo_back "rm -r script/Megatron-LM-3.0.2/"
    echo_info "install tracer"
    local _sw_list=(__init__.py schedules.py global_vars.py training.py)
    for _item in ${_sw_list[@]}
    do
        echo_back "cp script/megatron3.0/${_item} Megatron/megatron/"
    done
}

install_v4.0() {
    echo_back "cd ${SHELL_FOLDER}"
    echo_info "install megatron v4.0, v3.02 requires cuda12+torch2.1, please check your dependency"
    echo_back "wget --no-check-certificate --content-disposition https://github.com/NVIDIA/Megatron-LM/archive/refs/tags/core_v0.4.0.zip"
    echo_back "unzip Megatron-LM-core_v0.4.0.zip"
    echo_back  "cd ../"
    echo_back "mkdir Megatron"
    echo_back "mv script/Megatron-LM-core_v0.4.0/*  Megatron"
    echo_back "rm -r script/Megatron-LM-core_v0.4.0/"
    echo_info "install nltk"
    echo_back "pip install nltk"
    echo_info "install tracer"
    local _sw_list=(__init__.py global_vars.py training.py)
    for _item in ${_sw_list[@]}
    do
        echo_back "cp script/megatron4.0/${_item} Megatron/megatron/"
    done
    echo_back "cp script/megatron4.0/schedules.py Megatron/megatron/core/pipeline_parallel/"
}

setup() {
    local version=${1} 
    local prefix_dir=""
    if [[ ${version} == "v3" ]]; then 
        prefix_dir="megatron3.0"
    else
        prefix_dir="megatron4.0"
    fi
    echo_back "cd ${SHELL_FOLDER}"
    echo_back "cd ../Megatron"
    echo_info "we only provide a small dataset for correctness validation"
    echo_back "cp -r ../datasets ./"
    echo_back "cp ../examples/${prefix_dir}/* examples/"
    mkdir_if_not_exist ngc_models
    echo_back "bash examples/pretrain_gpt_distributed_dataprep_small.sh"
}

train() {
    echo_back "cd ${SHELL_FOLDER}"
    echo_info "adjust parameters in the pretrain bash before training"
    echo_back "cd ../Megatron"
    echo_info "clear checkpoint"
    rmdir_if_exist ngc_models/release_gpt_base
    echo_back "bash examples/pretrain_gpt_distributed_small.sh"
}

################################################################
####################    * Main Process *    ####################
################################################################
export LC_ALL=C

if (( $# == 0 )); then
    echo_warn "Argument cannot be NULL!"
    show_usage
    exit 0
fi

username=`whoami | awk '{print $1}'`

global_choice=${1}
case ${global_choice} in
    "install")
        install ${2}
        ;;
    "setup")
        setup ${2}
        ;;
    "train")
        train
        ;;
    "help")
        show_usage 
        ;;
    *)
        echo_erro "Unrecognized argument!"
        show_usage
        ;;
esac
