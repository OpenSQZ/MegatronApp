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

from dataclasses import dataclass
import json
from typing import Any, Dict, List, Tuple


@dataclass
class WaitingEvent:
    expect_arrivals: int
    arrived: List[int]


def dependency(traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    waiting: Dict[Tuple[str, str], WaitingEvent] = {}
    for i, event in enumerate(traces):
        event["args"]["id"] = i
        if "group" in event["args"] and event["args"]["group"] is not None:
            # print(event["name"], event["args"]["group"])
            # group: List[int] = list(map(int, event["args"]["group"].split(" ")))
            group = event["args"]["group"]
            group.append(event["args"]["g_rk"])
            group.sort()
            group_str = " ".join(str(r) for r in group)

            key = (event["name"], group_str)
            if key not in waiting:
                waiting[(event["args"]["expect"], group_str)] = WaitingEvent(
                    len(group), [i]
                )
            else:
                waiting[key].arrived.append(i)
                if len(waiting[key].arrived) == waiting[key].expect_arrivals:
                    for j in waiting[key].arrived:
                        traces[j]["args"]["related_sync_op"] = " ".join(
                            str(r) for r in waiting[key].arrived
                        )
                    del waiting[key]
    return traces


def amendP2P(traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for i in range(len(traces)):
        if (
            traces[i]["name"]
            in ["send-warmup", "recv-warmup", "exchange-next", "exchange-prev"]
            and "related_sync_op" in traces[i]["args"]
        ):
            related_sync_op_pair: List[str] = traces[i]["args"]["related_sync_op"].split(" ")
            related_sync_op_pair.remove(str(i))
            related_sync_op_idx = int(related_sync_op_pair[0])
            if traces[i]["dur"] != traces[related_sync_op_idx]["dur"]:
                actual_duration = min(
                    traces[i]["dur"], traces[related_sync_op_idx]["dur"]
                )
                traces[i]["ts"] += traces[i]["dur"] - actual_duration
                traces[i]["dur"] = actual_duration
                traces[related_sync_op_idx]["ts"] += (
                    traces[related_sync_op_idx]["dur"] - actual_duration
                )
                traces[related_sync_op_idx]["dur"] = actual_duration

                if (
                    "bandwidth" in traces[i]["args"]
                    and "bandwidth" in traces[related_sync_op_idx]["args"]
                ):
                    traces[i]["args"]["bandwidth"] = max(
                        traces[i]["args"]["bandwidth"],
                        traces[related_sync_op_idx]["args"]["bandwidth"],
                    )
                    traces[related_sync_op_idx]["args"]["bandwidth"] = traces[i][
                        "args"
                    ]["bandwidth"]
    return traces
