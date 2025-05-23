#!/bin/bash

##
# Get the bus address of a specified NIC
#
# @para: $1
#   NIC name, e.g., eth0
# @para: $2 (optional)
#   Login infomation for a remote access, e.g., bob@192.168.0.1
#
# @ret
#   bus-info, e.g., 0000:b2:00.0
##
get_nic_businfo()
{
    local _bus_info=""
    local _nic_name=$1        # e.g., eth0
    if [ ! ${_nic_name} ]; then
        echo_warn "NULL parameter!"
        return
    fi

    local _user_info=$2       # e.g., bob@192.168.0.1
    if [ ${_user_info} ]; then
        local _temp_info=`ssh ${_user_info} ethtool -i ${_nic_name}`
        _bus_info=`echo ${_temp_info} | grep -Eo ....\:..\:..\..`
    else
        _bus_info=`ethtool -i ${_nic_name} | grep -Eo ....\:..\:..\..`
    fi
    # when invoking this function, use
    #   'var=`get_nic_businfo nic_name {log_info}`'
    # to receive this return value
    echo ${_bus_info}
}

##
# Get the cpu list of the nearest socket (NUMA node) to the target bus-info
#
# @para: $1
#   bus-info, e.g., 0000:b2:00.0
# @para: $2 (optional)
#   Login infomation for a remote access, e.g., bob@192.168.0.1
#
# @ret
#   cpu list, e.g, NUMA node0 CPU(s): 0-13,28-41
##
get_cpu_list()
{
    local _bus_info=$1        # e.g., 0000:b2:00.0
    if [ ! ${_bus_info} ]; then
        echo "NULL parameter!"
        return
    fi

    local _numa_index=0
    local _lscpu_info=""
    local _user_info=$2       # e.g., bob@192.168.0.1
    local _total_core=0
    if [ ${_user_info} ]; then
        _numa_index=`ssh ${_user_info} cat /sys/bus/pci/devices/${_bus_info}/numa_node`
        _lscpu_info=`ssh ${_user_info} lscpu`
        _total_core=`ssh ${_user_info} grep -c ^processor /proc/cpuinfo`
    else
        _numa_index=`cat /sys/bus/pci/devices/${_bus_info}/numa_node`
        _lscpu_info=`lscpu`
        _total_core=`grep -c ^processor /proc/cpuinfo`
    fi
    local _ret=`echo ${_lscpu_info} | grep -Eo "NUMA node${_numa_index} CPU\(s\):\s+[0-9]+-[0-9]+(,[0-9]+-[0-9]+)*"`
    local _ret=`echo ${_ret} | awk '{print $4}'`
    # NUMA is not supported
    if [ ! ${_ret} ]; then
        local _max_id=$((_total_core-1))
        _ret="0-${_max_id}"
    fi
    echo ${_ret}
}

##
# Get the core mask. For example, if we have 28 cores (14-27,42-55) and we wanna use 16 of them, then the core mask will be 0xc000fffc000
# @para: $1
#   Available cores, e.g., 14-27,42-55
# @para: $2
#   Number of cores will be used, e.g., 16
#
# @ret
#   The corresponding core mask, e.g., 0xc000fffc000
get_cpu_mask()
{
    local _cpu_info=${1}
    local _num_core=${2}
    if [ ! ${_cpu_info} ]; then
        echo "NULL parameter!"
        return
    fi
    if [ ! ${_num_core} ]; then
        echo "NULL parameter!"
        return
    fi

    local _cpu_mask=0
    local _count=0
    local _arr=(`echo ${_cpu_info} | tr ',' ' '`)
    for _item in ${_arr[@]}
    do
        local _temp_arr=(`echo ${_item} | tr '-' ' '`)
        local _min_index=${_temp_arr[0]}
        local _max_index=${_temp_arr[1]}
        for _index in `seq ${_min_index} ${_max_index}`
        do
            if [ ${_index} -ge 64 ]; then
                continue
            fi
            # Note: if _max_index >= 64, the following operation would overflow
            ((count=count+1))
            if [[ ${count} -le ${_num_core} ]]; then
                local var_1=$((1<<${_index}))
                ((_cpu_mask=${_cpu_mask}+${var_1}))
            fi
        done
    done
    printf 0x%x ${_cpu_mask}
}

##
# Update driver for NIC
# 
# @para file_name 
#   Name of OFED file 
# @para work_dir
#   work directory 
# @para remote_path 
#   Where to fetch the file if it is not exists in the local host, e.g., username@host_ip:DIRECTORY_TO_OFED
# @para ofed_options
#   e.g., --upstream-libs, --dpdk
##
update_ofed() {
    local _mnt_dir_=/mnt/MLNX_OFED
    local _file_name_=$1
    local _work_dir_=$2
    local _remote_path_=$3
    local _oos_1_=$4
    local _oos_2_=$5
    local _oos_3_=$6
    local _oos_4_=$7
    local _oos_5_=$8
    local _oos_6_=$9

    if [ ! ${_file_name_} ]; then
        echo_warn "NULL parameter in ${FUNCNAME}"
        return
    fi
    if [ ! ${_work_dir_} ]; then
        echo_warn "NULL parameter in ${FUNCNAME}"
        return
    fi
    if [ ! ${_remote_path_} ]; then
        echo_warn "NULL parameter in ${FUNCNAME}"
        return
    fi

    if [ ! -f ${_work_dir_}/${_file_name_} ]; then
        echo_back "scp ${_remote_path_}/${_file_name_} ${_work_dir_}"
    fi  
    if [ ! -f ${_work_dir_}/${_file_name_} ]; then
        echo_warn "Cannot find ${_file_name_}"
        exit 0 
    fi

    if [ ! -d ${_mnt_dir_} ]; then
        echo_back "sudo mkdir ${_mnt_dir_}"
    elif mountpoint -q ${_mnt_dir_}; then
        echo_back "sudo umount ${_mnt_dir_}"
    fi  

    echo_back "sudo mount -o ro,loop ${_work_dir_}/${_file_name_} ${_mnt_dir_}"
    echo_back "sudo /mnt/MLNX_OFED/mlnxofedinstall ${_oos_1_} ${_oos_2_} ${_oos_3_} ${_oos_4_} ${_oos_5_} ${_oos_6_}"
    echo_back "sudo /etc/init.d/openibd restart"
}

