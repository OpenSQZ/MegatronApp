# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import datetime
import torch
import json
import threading
from websockets.sync.server import serve
from megatron import get_args, get_tokenizer, get_flags, get_disturbance, FlagType
from megatron.text_generation import generate_and_post_process
from megatron.text_generation import beam_search_and_post_process
from megatron.global_vars import set_report, unset_report, set_filter

GENERATE_NUM = 0
BEAM_NUM = 1
lock = threading.Lock()

class MegatronGenerate:
    def __init__(self, model):
        self.model = model

    @staticmethod
    def send_do_generate():
        choice = torch.cuda.LongTensor([GENERATE_NUM])
        torch.distributed.broadcast(choice, 0)
     
    @staticmethod
    def send_do_beam_search():
        choice = torch.cuda.LongTensor([BEAM_NUM])
        torch.distributed.broadcast(choice, 0)
    
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
            if not (type(temperature) == int or type(temperature) == float):
                return "temperature must be a positive number less than or equal to 100.0"
            if not (0.0 < temperature <= 100.0):
                return "temperature must be a positive number less than or equal to 100.0"
        
        top_k = 0.0
        if "top_k" in request:
            top_k = request["top_k"]
            if not (type(top_k) == int):
                return "top_k must be an integer equal to or greater than 0 and less than or equal to 1000"
            if not (0 <= top_k <= 1000):
                return "top_k must be equal to or greater than 0 and less than or equal to 1000"
        
        top_p = 0.0
        if "top_p" in request:
            top_p = request["top_p"]
            if not (type(top_p) == float):
                return "top_p must be a positive float less than or equal to 1.0"
            if top_p > 0.0 and top_k > 0.0:
                return "cannot set both top-k and top-p samplings."
            if not (0 <= top_p <= 1.0):
                return "top_p must be less than or equal to 1.0"
        
        top_p_decay = 0.0
        if "top_p_decay" in request:
            top_p_decay = request["top_p_decay"]
            if not (type(top_p_decay) == float):
                return "top_p_decay must be a positive float less than or equal to 1.0"
            if top_p == 0.0:
                return "top_p_decay cannot be set without top_p"
            if not (0 <= top_p_decay <= 1.0):
                return "top_p_decay must be less than or equal to 1.0"
        
        top_p_bound = 0.0
        if "top_p_bound" in request:
            top_p_bound = request["top_p_bound"]
            if not (type(top_p_bound) == float):
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

        stop_token=50256
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

        with lock:

            get_flags().set_by_configs(visualization_flags_config)
            torch.distributed.broadcast_object_list([visualization_flags_config], 0)
            get_disturbance().set_by_configs(disturbance_configs)
            torch.distributed.broadcast_object_list([disturbance_configs], 0)

            set_filter() # This initializes mlp2_record based on num_layers

            if not no_log:
                print("request IP: " + str(websocket.remote_address[0]))
                print(json.dumps(request),flush=True)
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
                    MegatronGenerate.send_do_beam_search()  # Tell other ranks we're doing beam_search
                    response, response_seg, response_scores = \
                        beam_search_and_post_process(
                        self.model,
                        prompts=prompts,
                        tokens_to_generate=tokens_to_generate,
                        beam_size = beam_width,
                        add_BOS=add_BOS,
                        stop_token=stop_token,
                        num_return_gen=beam_width,  # Returning whole beam
                        length_penalty=length_penalty,
                        prevent_newline_after_colon=prevent_newline_after_colon,
                        )
                    
                    return json.dumps({"text": response,
                        "segments": response_seg,
                        "scores": response_scores})
                else:
                    MegatronGenerate.send_do_generate()  # Tell other ranks we're doing generate
                    response, response_seg, response_logprobs, _ = \
                        generate_and_post_process(
                        self.model,
                        prompts=prompts,
                        tokens_to_generate=tokens_to_generate,
                        return_output_log_probs=logprobs,
                        top_k_sampling=top_k,
                        top_p_sampling=top_p,
                        top_p_decay=top_p_decay,
                        top_p_bound=top_p_bound,
                        temperature=temperature,
                        add_BOS=add_BOS,
                        use_eod_token_for_early_termination=True,
                        stop_on_double_eol=stop_on_double_eol,
                        stop_on_eol=stop_on_eol,
                        prevent_newline_after_colon=prevent_newline_after_colon,
                        random_seed=random_seed,
                    )

                    return json.dumps({"type":"finish","text": response,
                        "segments": response_seg,
                        "logprobs": response_logprobs})

            except ValueError as ve:
                return ve.args[0]
            print("end time: ", datetime.datetime.now())
        

class MegatronWSServer(object):
    def __init__(self, model):
        self.generator=MegatronGenerate(model)
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
                    # 前端发送了 {"type": "ping"}
                    # 后端需要回复一个 pong 消息，让前端的 pongTimeout 机制知道连接是活跃的
                    # 根据前端代码，它期望一个 "PONG" 字符串，或者一个包含 "PONG" 的 Blob
                    # 最简单的是直接发送 "PONG" 文本，或者一个简单的 JSON 确认
                    # print("Received ping from client, sending pong.") # 调试打印
                    websocket.send(json.dumps({"type": "pong"})) # 回复一个 JSON pong
                    # 或者，如果前端严格期望纯文本 "PONG":
                    # websocket.send("PONG")
                    continue # 处理完 ping 后，等待下一条消息，不执行后续的 generate 逻辑
                else:
                    raise ValueError(f"无法识别的请求类型: {request['type']}")
            except ValueError as ve:
                websocket.send("Value Error!")

    def runServer(self,url,port):
        with serve(self.parser, url, port, ping_interval=60, ping_timeout=60) as server:
            server.serve_forever()
    def run(self, url, port): 
        wsserver=threading.Thread(self.runServer(url,port))
        wsserver.start()
