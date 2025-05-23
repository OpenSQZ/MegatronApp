#!/bin/bash

##
# Check if a command exists
#
# @para _cmd_
#   target command
#
# @return
#   0: exist
#   -1: otherwise
#
check_cmd() {
    local _cmd_=${1}
    if [ ! ${_cmd_} ]; then
        echo -1
    fi
    local _loc_=`which ${_cmd_}`
    if [ ! ${_loc_} ]; then
        echo -1
    else
        echo 0
    fi
}

##
# Find the number of logical cores ( = num_phy_core * num_thread_per_core)
##
get_total_lcore() {
    local _ret_=`grep -c ^processor /proc/cpuinfo`
    echo ${_ret_}
}

##
# Update kernel with apt 
#
# @para _target_kernel_ (optional)
#   target linux kernel version (output from `uname -r`) 
##
update_kernel() {
    local _target_kernel_=${1}
    if [ ! ${_target_kernel_} ]; then
        return
    fi
    local _cur_ker_=`uname -r`
    if [ ${_target_kernel_} == ${_cur_ker_} ]; then
        echo_info "Nothing to do! Already at target kernel version number"
        return
    fi
    echo_back "sudo apt-get install -y linux-image-${_target_kernel_}"
    echo_back "sudo apt-get install -y linux-headers-${_target_kernel_}"
    echo_warn "  -- Denote the line number (from 0) in which the target kernel version first appears in the text below as x"
    echo_warn "  -- Open /etc/default/grub and set GRUB_DEFAULT=\"1> x\""
    echo_warn "  -- Please note that there must be a space between 1 and x"
    echo_warn "  -- Then run 'sudo update-grub' and 'sudo reboot'"
    echo_back "cat /boot/grub/grub.cfg | grep -E 'menuentry.+class.+class.+menuentry_id_option'"
    exit 0
}

##
# Get GPU Product Name
##
get_nvidia_gpu_name() {
    local _ret_=`check_cmd`
    if [[ ${_ret_} -eq -1 ]]; then
       echo_erro "Cannot find nvidia-smi!"
       exit 0
    fi
    local _info_=`nvidia-smi -q | grep "Product Name"` 
    echo "${_info_}"
}

