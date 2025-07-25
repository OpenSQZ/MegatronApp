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

import json
import threading
import queue
import time
from websockets.sync.server import serve
from websockets.exceptions import ConnectionClosed

from megatron.training import get_args
from megatron.core.tensor_tracer import get_tt_flags
from megatron.core.tensor_disturbance import get_disturbance
from megatron.core.tensor_tracer import get_compressor
import megatron.virtual_tensor_parallel_communication as dist

start_training_event = threading.Event()
_shutdown_event = threading.Event()

_request_configs = {}
_request_lock = threading.Lock()

_websocket_connection = None
_websocket_lock = threading.Lock()

data_queue = queue.Queue()

def _data_sender_thread():
    print("Rank 0: Data sender thread started.", flush=True)

    while not _shutdown_event.is_set():
        try:
            name_tuple, report_args, tensor_data = data_queue.get()
            tensor_cpu = tensor_data
            payload = {
                "type": "update",
                "update_type": name_tuple[1].value,
                "layer_id": name_tuple[0],
                "args": report_args,
                "result": tensor_cpu.tolist()
            }
            ws = get_websocket()
            if ws:
                try:
                    ws.send(json.dumps(payload))
                    data_queue.task_done()
                except ConnectionClosed:
                    print("Rank 0 WS Sender: Connection closed while sending. Dropping data.", flush=True)
                except Exception as e:
                    print(f"Rank 0 WS Sender: Error sending data: {e}", flush=True)
            else:
                pass

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Rank 0 WS Sender: Unexpected error in sender thread: {e}", flush=True)
            time.sleep(1)

class TrainingWSServer:
    def __init__(self, port):
        self.port = port
        self.server_instance = None
        self.server_thread = None
        self.sender_thread = None

    def _websocket_handler(self, websocket):
        global _websocket_connection, _request_configs
        
        print("Rank 0 WS: Frontend connected.", flush=True)
        with _websocket_lock:
            _websocket_connection = websocket
        
        try:
            for message in websocket:
                try:
                    request = json.loads(message)
                    req_type = request.get("type")

                    if req_type == "run_training_step":
                        print("Rank 0 WS: Received 'run_training_step' command. Starting training...", flush=True)
                        
                        with _request_lock:
                            _request_configs['visualization_flags'] = request.get("visualization_flags", {})
                            _request_configs['disturbance_configs'] = request.get("disturbance_configs", {})
                            _request_configs['compressor_config'] = request.get("compressor_config", {})

                        start_training_event.set()

                    elif req_type == "ping":
                        websocket.send(json.dumps({"type": "pong"}))
                    else:
                        print(f"Rank 0 WS: Received unknown request type: {req_type}", flush=True)

                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Rank 0 WS: Error processing message: {e}", flush=True)

        except ConnectionClosed:
            print("Rank 0 WS: Connection closed by frontend.", flush=True)
        finally:
            with _websocket_lock:
                _websocket_connection = None
            start_training_event.clear()
            print("Rank 0 WS: Frontend disconnected.", flush=True)

    def _run_server_thread(self):
        with serve(self._websocket_handler, "0.0.0.0", self.port, ping_interval=None) as server:
            self.server_instance = server
            print(f"Rank 0: Training WebSocket server started on ws://0.0.0.0:{self.port}", flush=True)
            server.serve_forever()
            while not _shutdown_event.is_set():
                try:
                    _shutdown_event.wait(timeout=10.0)
                except KeyboardInterrupt:
                    break
        print("Rank 0: WebSocket server has shut down.", flush=True)

    def start(self):
        if self.port is None:
            print("Rank 0: No training-ws-port specified, WebSocket server will not start.", flush=True)
            return
        self.server_thread = threading.Thread(target=self._run_server_thread)
        self.server_thread.start()
        print("Rank 0: WebSocket server thread started.", flush=True)
        self.sender_thread = threading.Thread(target=_data_sender_thread, daemon=True)
        self.sender_thread.start()
        return self.server_thread
    
    def shutdown(self):
        print("Rank 0: Signaling WebSocket server to shut down...", flush=True)
        _shutdown_event.set()

def get_websocket():
    with _websocket_lock:
        return _websocket_connection