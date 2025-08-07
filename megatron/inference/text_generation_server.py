# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import datetime
import json
import os
import sys

from flask import Flask, request, jsonify
from flask_restful import Resource, Api

from megatron.core.inference.sampling_params import SamplingParams
from megatron.inference.endpoints.common import send_do_generate, send_do_beam_search, LOCK
from megatron.inference.endpoints.completions import MegatronCompletions
from megatron.inference.text_generation import beam_search_and_post_process
from megatron.inference.text_generation.mcore_engine_server import run_mcore_engine

from megatron.training import get_tokenizer, get_args
from megatron.core.tensor_tracer import set_report, set_filter, get_tt_flags, FlagType, get_compressor
from megatron.core.tensor_disturbance import get_disturbance
import torch
import megatron.virtual_tensor_parallel_communication as dist
import threading
from websockets.sync.server import serve

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)


class InferenceGenerate:
    def __init__(self, engine, args):
        self.engine = engine
        self.args = args

    def query(self, request, websocket):
        args = get_args()

        if not "prompts" in request:
            return "prompts argument required", 400

        if "max_len" in request:
            return "max_len is no longer used.  Replace with tokens_to_generate", 400

        if "sentences" in request:
            return "sentences is no longer used.  Replace with prompts", 400

        prompts = request["prompts"]
        if not isinstance(prompts, list):
            return "prompts is not a list of strings", 400

        if len(prompts) == 0:
            return "prompts is empty", 400

        if len(prompts) > 128:
            return "Maximum number of prompts is 128", 400

        tokens_to_generate = 64  # Choosing hopefully sane default.  Full sequence is slow
        if "tokens_to_generate" in request:
            tokens_to_generate = request["tokens_to_generate"]
            if not isinstance(tokens_to_generate, int):
                return "tokens_to_generate must be an integer greater than 0"
            if tokens_to_generate < 0:
                return "tokens_to_generate must be an integer greater than or equal to 0"

        logprobs = False
        if "logprobs" in request:
            logprobs = request["logprobs"]
            if not isinstance(logprobs, bool):
                return "logprobs must be a boolean value"

        if tokens_to_generate == 0 and not logprobs:
            return "tokens_to_generate=0 implies logprobs should be True"

        temperature = 1.0
        if "temperature" in request:
            temperature = request["temperature"]
            if not (isinstance(temperature, (int, float))):
                return "temperature must be a positive number less than or equal to 1000.0"
            if not (0.0 < temperature <= 100.0):
                return "temperature must be a positive number less than or equal to 100.0"

        top_k = 0
        if "top_k" in request:
            top_k = request["top_k"]
            if not (isinstance(top_k, int)):
                return "top_k must be an integer equal to or greater than 0 and less than or equal to 1000"
            if not (0 <= top_k <= 1000):
                return "top_k must be equal to or greater than 0 and less than or equal to 1000"

        top_p = 0.0
        if "top_p" in request:
            top_p = request["top_p"]
            if not (isinstance(top_p, float)):
                return "top_p must be a positive float less than or equal to 1.0"
            if top_p > 0.0 and top_k > 0.0:
                return "cannot set both top-k and top-p samplings."
            if not (0 <= top_p <= 1.0):
                return "top_p must be less than or equal to 1.0"

        top_p_decay = 0.0
        if "top_p_decay" in request:
            top_p_decay = request["top_p_decay"]
            if not (isinstance(top_p_decay, float)):
                return "top_p_decay must be a positive float less than or equal to 1.0"
            if top_p == 0.0:
                return "top_p_decay cannot be set without top_p"
            if not (0 <= top_p_decay <= 1.0):
                return "top_p_decay must be less than or equal to 1.0"

        top_p_bound = 0.0
        if "top_p_bound" in request:
            top_p_bound = request["top_p_bound"]
            if not (isinstance(top_p_bound, float)):
                return "top_p_bound must be a positive float less than or equal to top_p"
            if top_p == 0.0:
                return "top_p_bound cannot be set without top_p"
            if not (0.0 < top_p_bound <= top_p):
                return "top_p_bound must be greater than 0 and less than top_p"

        add_BOS = False
        if "add_BOS" in request:
            add_BOS = request["add_BOS"]
            if not isinstance(add_BOS, bool):
                return "add_BOS must be a boolean value"

        if any([len(prompt) == 0 for prompt in prompts]) and not add_BOS:
            return "Empty prompts require add_BOS=true"

        stop_on_double_eol = False
        if "stop_on_double_eol" in request:
            stop_on_double_eol = request["stop_on_double_eol"]
            if not isinstance(stop_on_double_eol, bool):
                return "stop_on_double_eol must be a boolean value"

        stop_on_eol = False
        if "stop_on_eol" in request:
            stop_on_eol = request["stop_on_eol"]
            if not isinstance(stop_on_eol, bool):
                return "stop_on_eol must be a boolean value"

        prevent_newline_after_colon = False
        if "prevent_newline_after_colon" in request:
            prevent_newline_after_colon = request["prevent_newline_after_colon"]
            if not isinstance(prevent_newline_after_colon, bool):
                return "prevent_newline_after_colon must be a boolean value"

        random_seed = -1
        if "random_seed" in request:
            random_seed = request["random_seed"]
            if not isinstance(random_seed, int):
                return "random_seed must be integer"
            if random_seed < 0:
                return "random_seed must be a positive integer"

        no_log = False
        if "no_log" in request:
            no_log = request["no_log"]
            if not isinstance(no_log, bool):
                return "no_log must be a boolean value"

        beam_width = None
        if "beam_width" in request:
            beam_width = request["beam_width"]
            if not isinstance(beam_width, int):
                return "beam_width must be integer"
            if beam_width < 1:
                return "beam_width must be an integer > 1"
            if len(prompts) > 1:
                return "When doing beam_search, batch size must be 1"

        stop_token = 50256
        if "stop_token" in request:
            stop_token = request["stop_token"]
            if not isinstance(stop_token, int):
                return "stop_token must be an integer"

        length_penalty = 1
        if "length_penalty" in request:
            length_penalty = request["length_penalty"]
            if not isinstance(length_penalty, float):
                return "length_penalty must be a float"

        visualization_flags_config = request.get("visualization_flags", {})
        disturbance_configs = request.get("disturbance_configs", {})
        compressor_config = request.get("compressor_config", {})

        with LOCK:  # Need to get lock to keep multiple threads from hitting code

            get_tt_flags().set_by_configs(visualization_flags_config)
            dist.broadcast_object_list([visualization_flags_config], 0)
            get_disturbance().set_by_configs(disturbance_configs)
            dist.broadcast_object_list([disturbance_configs], 0)
            get_compressor().set_by_configs(compressor_config)

            set_filter()

            if not no_log:
                print("request IP: " + str(websocket.remote_address[0]))
                print(json.dumps(request), flush=True)
                print("start time: ", datetime.datetime.now())

            try:
                tokenizer = get_tokenizer()

                tokenized_prompts = [tokenizer.tokenize(p) for p in prompts]
                min_prompt_len = min(len(tp) for tp in tokenized_prompts)
                initial_display_tokens = []
                for batch_idx in range(len(tokenized_prompts)):
                    for i in range(min_prompt_len):
                        initial_display_tokens.append(tokenized_prompts[batch_idx][i])

                websocket.send(json.dumps({
                    "type": "start",
                    "prompts": [{"id": tid, "token": tokenizer.decoder.get(tid, str(tid))} for tid in initial_display_tokens],
                    "num_layers": args.num_layers
                }))

                def report_func(name_tuple, report_args, tensor_data):
                    """回调函数，用于将追踪到的数据发送到前端。"""
                    # name_tuple is (layer_id, FlagType)
                    # report_args are specific to the FlagType (e.g., [n,m] for attention)
                    # tensor_data is the actual data (list or tensor that can be .tolist())
                    if name_tuple[1] == FlagType.INVALID_FLAG:
                        return
                    if name_tuple[1] == FlagType.Result:
                        payload = {
                            "type": "update",
                            "update_type": name_tuple[1].value,
                            "result": tensor_data,
                            "sampled": report_args
                        }
                    else:
                        payload = {
                            "type": "update",
                            "update_type": name_tuple[1].value,
                            "layer_id": name_tuple[0],
                            "args": report_args,
                            "result": tensor_data.tolist()
                        }
                    websocket.send(json.dumps(payload))

                set_report(report_func)

                if beam_width is not None:
                    send_do_beam_search()  # Tell other ranks we're doing beam_search
                    response, response_seg, response_scores = beam_search_and_post_process(
                        self.model,
                        prompts=prompts,
                        tokens_to_generate=tokens_to_generate,
                        beam_size=beam_width,
                        add_BOS=add_BOS,
                        stop_token=stop_token,
                        num_return_gen=beam_width,  # Returning whole beam
                        length_penalty=length_penalty,
                        prevent_newline_after_colon=prevent_newline_after_colon,
                    )

                    return json.dumps(
                        {"text": response, "segments": response_seg, "scores": response_scores}
                    )
                else:
                    send_do_generate()  # Tell other ranks we're doing generate

                    response_dict = run_mcore_engine(self.engine, prompts, temperature, top_k, top_p, logprobs, tokens_to_generate)

                    response_dict["type"] = "finish"

                    return json.dumps(response_dict)

            except ValueError as ve:
                return ve.args[0]


class InferenceWSServer(object):
    def __init__(self, engine, args):
        self.generator=InferenceGenerate(engine, args)
    def parser(self, websocket):
        for message in websocket:
            try:
                request = json.loads(message)
                if "type" not in request:
                    raise ValueError("请求中需要 type 参数")
                if request["type"] == "generate":
                    result = self.generator.query(request, websocket)
                    websocket.send(result)
                elif request["type"] == "ping":
                    websocket.send(json.dumps({"type": "pong"})) # 回复一个 JSON pong
                    continue
                else:
                    raise ValueError(f"无法识别的请求类型: {request['type']}")
            except ValueError as ve:
                websocket.send("Value Error!")

    def runServer(self,url,port):
        with serve(self.parser, url, port, ping_interval=None) as server:
            server.serve_forever()
    def run(self, url, port): 
        wsserver=threading.Thread(self.runServer(url,port))
        wsserver.start()

class MegatronGenerate(Resource):
    def __init__(self, engine, args):
        self.engine = engine
        self.args = args

    def put(self):
        if not "prompts" in request.get_json():
            return "prompts argument required", 400

        if "max_len" in request.get_json():
            return "max_len is no longer used.  Replace with tokens_to_generate", 400

        if "sentences" in request.get_json():
            return "sentences is no longer used.  Replace with prompts", 400

        prompts = request.get_json()["prompts"]
        if not isinstance(prompts, list):
            return "prompts is not a list of strings", 400

        if len(prompts) == 0:
            return "prompts is empty", 400

        if len(prompts) > 128:
            return "Maximum number of prompts is 128", 400

        tokens_to_generate = 64  # Choosing hopefully sane default.  Full sequence is slow
        if "tokens_to_generate" in request.get_json():
            tokens_to_generate = request.get_json()["tokens_to_generate"]
            if not isinstance(tokens_to_generate, int):
                return "tokens_to_generate must be an integer greater than 0"
            if tokens_to_generate < 0:
                return "tokens_to_generate must be an integer greater than or equal to 0"

        logprobs = False
        if "logprobs" in request.get_json():
            logprobs = request.get_json()["logprobs"]
            if not isinstance(logprobs, bool):
                return "logprobs must be a boolean value"

        if tokens_to_generate == 0 and not logprobs:
            return "tokens_to_generate=0 implies logprobs should be True"

        temperature = 1.0
        if "temperature" in request.get_json():
            temperature = request.get_json()["temperature"]
            if not (isinstance(temperature, (int, float))):
                return "temperature must be a positive number less than or equal to 1000.0"
            if not (0.0 < temperature <= 100.0):
                return "temperature must be a positive number less than or equal to 100.0"

        top_k = 0
        if "top_k" in request.get_json():
            top_k = request.get_json()["top_k"]
            if not (isinstance(top_k, int)):
                return "top_k must be an integer equal to or greater than 0 and less than or equal to 1000"
            if not (0 <= top_k <= 1000):
                return "top_k must be equal to or greater than 0 and less than or equal to 1000"

        top_p = 0.0
        if "top_p" in request.get_json():
            top_p = request.get_json()["top_p"]
            if not (isinstance(top_p, float)):
                return "top_p must be a positive float less than or equal to 1.0"
            if top_p > 0.0 and top_k > 0.0:
                return "cannot set both top-k and top-p samplings."
            if not (0 <= top_p <= 1.0):
                return "top_p must be less than or equal to 1.0"

        top_p_decay = 0.0
        if "top_p_decay" in request.get_json():
            top_p_decay = request.get_json()["top_p_decay"]
            if not (isinstance(top_p_decay, float)):
                return "top_p_decay must be a positive float less than or equal to 1.0"
            if top_p == 0.0:
                return "top_p_decay cannot be set without top_p"
            if not (0 <= top_p_decay <= 1.0):
                return "top_p_decay must be less than or equal to 1.0"

        top_p_bound = 0.0
        if "top_p_bound" in request.get_json():
            top_p_bound = request.get_json()["top_p_bound"]
            if not (isinstance(top_p_bound, float)):
                return "top_p_bound must be a positive float less than or equal to top_p"
            if top_p == 0.0:
                return "top_p_bound cannot be set without top_p"
            if not (0.0 < top_p_bound <= top_p):
                return "top_p_bound must be greater than 0 and less than top_p"

        add_BOS = False
        if "add_BOS" in request.get_json():
            add_BOS = request.get_json()["add_BOS"]
            if not isinstance(add_BOS, bool):
                return "add_BOS must be a boolean value"

        if any([len(prompt) == 0 for prompt in prompts]) and not add_BOS:
            return "Empty prompts require add_BOS=true"

        stop_on_double_eol = False
        if "stop_on_double_eol" in request.get_json():
            stop_on_double_eol = request.get_json()["stop_on_double_eol"]
            if not isinstance(stop_on_double_eol, bool):
                return "stop_on_double_eol must be a boolean value"

        stop_on_eol = False
        if "stop_on_eol" in request.get_json():
            stop_on_eol = request.get_json()["stop_on_eol"]
            if not isinstance(stop_on_eol, bool):
                return "stop_on_eol must be a boolean value"

        prevent_newline_after_colon = False
        if "prevent_newline_after_colon" in request.get_json():
            prevent_newline_after_colon = request.get_json()["prevent_newline_after_colon"]
            if not isinstance(prevent_newline_after_colon, bool):
                return "prevent_newline_after_colon must be a boolean value"

        random_seed = -1
        if "random_seed" in request.get_json():
            random_seed = request.get_json()["random_seed"]
            if not isinstance(random_seed, int):
                return "random_seed must be integer"
            if random_seed < 0:
                return "random_seed must be a positive integer"

        no_log = False
        if "no_log" in request.get_json():
            no_log = request.get_json()["no_log"]
            if not isinstance(no_log, bool):
                return "no_log must be a boolean value"

        beam_width = None
        if "beam_width" in request.get_json():
            beam_width = request.get_json()["beam_width"]
            if not isinstance(beam_width, int):
                return "beam_width must be integer"
            if beam_width < 1:
                return "beam_width must be an integer > 1"
            if len(prompts) > 1:
                return "When doing beam_search, batch size must be 1"

        stop_token = 50256
        if "stop_token" in request.get_json():
            stop_token = request.get_json()["stop_token"]
            if not isinstance(stop_token, int):
                return "stop_token must be an integer"

        length_penalty = 1
        if "length_penalty" in request.get_json():
            length_penalty = request.get_json()["length_penalty"]
            if not isinstance(length_penalty, float):
                return "length_penalty must be a float"

        with LOCK:  # Need to get lock to keep multiple threads from hitting code

            if not no_log:
                print("request IP: " + str(request.remote_addr))
                print(json.dumps(request.get_json()), flush=True)
                print("start time: ", datetime.datetime.now())

            try:
                if beam_width is not None:
                    send_do_beam_search()  # Tell other ranks we're doing beam_search
                    response, response_seg, response_scores = beam_search_and_post_process(
                        self.model,
                        prompts=prompts,
                        tokens_to_generate=tokens_to_generate,
                        beam_size=beam_width,
                        add_BOS=add_BOS,
                        stop_token=stop_token,
                        num_return_gen=beam_width,  # Returning whole beam
                        length_penalty=length_penalty,
                        prevent_newline_after_colon=prevent_newline_after_colon,
                    )

                    return jsonify(
                        {"text": response, "segments": response_seg, "scores": response_scores}
                    )
                else:
                    send_do_generate()  # Tell other ranks we're doing generate

                    response_dict = run_mcore_engine(self.engine, prompts, temperature, top_k, top_p, logprobs, tokens_to_generate)

                    return jsonify(response_dict)

            except ValueError as ve:
                return ve.args[0]


class MegatronServer(object):
    def __init__(self, model, args=None):
        self.app = Flask(__name__, static_url_path='')
        api = Api(self.app)
        api.add_resource(MegatronGenerate, '/api', resource_class_args=[model, args])
        api.add_resource(MegatronCompletions, '/completions', resource_class_args=[model, args])

    def run(self, url, port):
        self.app.run(url, threaded=True, debug=False, port=port)
