<!--
Copyright 2025 Suanzhi Future Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions
and limitations under the License.
-->

<script setup lang="ts">
import ColoredVector from "./components/ColoredVector.vue";
import AttentionMatrix from "./components/AttentionMatrix.vue";
import OutputProbs from "./components/OutputProbs.vue";
import PCAPlot from "./components/PCAPlot.vue";
import { ref, computed, h, watch, nextTick } from "vue";
import { NButton, NIcon, useMessage, NFlex } from "naive-ui"; // Added NFlex
import { FlashOutline, CodeDownloadOutline, AddCircleOutline, RemoveCircleOutline, LayersOutline, InformationCircleOutline, SparklesOutline } from "@vicons/ionicons5"; // Added more icons
import { useWebSocket } from "@vueuse/core";
import { headerProps } from "naive-ui/es/layout/src/LayoutHeader";
import { isArray } from "chart.js/helpers";

// --- 消息提示 ---
const message = useMessage();

// --- 类型定义 ---
interface TokenInfo {
  logit: number;
  id: number;
  token: string;
  probability: number;
}
interface OutputProbData {
  probs: Array<TokenInfo>;
  sampled: TokenInfo;
}
interface DisturbanceConfig {
  weight_perturbation: boolean;
  weight_perturbation_fn: string;
  weight_perturbation_coef: number;
  calculation_perturbation: boolean;
  calculation_perturbation_fn: string;
  calculation_perturbation_coef: number;
  system_perturbation: boolean; // 假设后端支持
  system_perturbation_fn: string;
  system_perturbation_coef: number;
}


// --- 常量与配置 ---
const visualizationSettings = ref({
  qkvPixels: 96,
  mlpPixels: 64,
  qkvCompressionMethod: "data.mean(dim=-1)",
  mlpCompressionMethod: "data.mean(dim=-1)",
})
const TOP_K_FOR_PROBS = 20; // 与后端 global_vars.py 中的 topk (Tracer.tik_result) 一致

// --- 颜色定义 ---
const q_color = [1, 0, 0]; // Red for Query
const k_color = [0, 1, 0]; // Green for Key
const v_color = [0, 0, 1]; // Blue for Value
const qkv_colors = computed(() => {
  const num_per_part = Math.floor(visualizationSettings.value.qkvPixels / 3);
  return [
    ...Array(num_per_part).fill(q_color),
    ...Array(num_per_part).fill(k_color),
    // 确保总数正确，将余数分配给最后一个
    ...Array(visualizationSettings.value.qkvPixels - 2 * num_per_part).fill(v_color),
  ];
});
const attention_color = [0.4235, 0.1686, 0.8509]; // Purple
const mlp1_color_single = [0.4235, 0.1686, 0.8509]; // Purple
const mlp2_color_single = [0, 0, 1]; // Blue
const mlp1_colors = computed(() => Array(visualizationSettings.value.mlpPixels).fill(mlp1_color_single));
const mlp2_colors = computed(() => Array(visualizationSettings.value.mlpPixels).fill(mlp2_color_single));


// --- 核心状态 ---
const isTrainingMode = ref(false);
const prompts = ref<string[]>(["The quick brown fox jumps over the lazy dog."]);
const current_batch_index_to_display = ref(0);
const batch_size = computed(() => prompts.value.length);
const actualNumLayers = ref(0); // 从后端获取
const loadingRef = ref(false);
const isInitializedAfterStart = ref(false); // 标记是否已收到 "start" 消息并初始化

// --- 可视化开关 ---
const visualizationSwitches = ref({
  QKV_mat_mul: true,
  RawAttentionScore_mat_mul: true,
  ContextLayer_mat_mul: false, // 通常不关注此项
  MLP1_mat_mul: true,
  MLP2_mat_mul: true,
  MLP2_Plot: true,
  Result: true, // 控制表格中输出概率的展开
});

// --- 扰动配置 ---
const disturbanceSettings = ref<DisturbanceConfig>({
  weight_perturbation: false, weight_perturbation_fn: "noise1", weight_perturbation_coef: 0.01,
  calculation_perturbation: false, calculation_perturbation_fn: "noise2", calculation_perturbation_coef: 0.05,
  system_perturbation: false, system_perturbation_fn: "noise1", system_perturbation_coef: 0.01,
});
const disturbanceFnOptions = ["noise1", "noise2"].map(o => ({label: o, value: o}));


// --- 数据存储 (支持 Batch) ---
// [layer_idx (0-based)][batch_idx (0-based)][token_idx (0-based)][feature_value]
const qkv_vectors_batched = ref<number[][][][]>([]);
const mlp1_vectors_batched = ref<number[][][][]>([]);
const mlp2_vectors_batched = ref<number[][][][]>([]);
// [layer_idx][batch_idx][num_heads][query_token_idx][key_token_idx]
const attention_matrix_batched = ref<number[][][][][]>([]);
// [layer_idx][batch_idx] -> current sequence length for this attention matrix
const attention_size_batched = ref<number[][]>([]);
// [layer_idx][batch_idx][token_idx][pca_pair [x, y]]
const mlp2_vectors_pca = ref<Array<Array<Array<[number, number]>>>>([]);
// [batch_idx][token_idx] -> TokenInfo object {id, token, logit, probability}
const tokens_per_batch = ref<TokenInfo[][]>([]);
// [batch_idx][token_idx] -> OutputProbData object {probs, sampled} for the generation step *after* this token
const results_per_batch = ref<OutputProbData[][]>([]);
// [batch_idx] -> final generated string
const result_text_batched = ref<string[]>([]);


// --- 初始化与清理 ---
function initializeDataStructures(numLayers: number, currentBatchSize: number) {
  console.log(`Initializing data structures for ${numLayers} layers and ${currentBatchSize} batches.`);
  const layers = Math.max(1, numLayers); // 确保至少为1，避免空数组访问问题
  const bs = Math.max(1, currentBatchSize);

  qkv_vectors_batched.value = Array.from({ length: layers }, () => Array.from({ length: bs }, () => []));
  mlp1_vectors_batched.value = Array.from({ length: layers }, () => Array.from({ length: bs }, () => []));
  mlp2_vectors_batched.value = Array.from({ length: layers }, () => Array.from({ length: bs }, () => []));
  attention_matrix_batched.value = Array.from({ length: layers }, () => Array.from({ length: bs }, () => []));
  attention_size_batched.value = Array.from({ length: layers }, () => Array(bs).fill(0));
  mlp2_vectors_pca.value = Array.from({ length: layers }, () => Array.from({ length: bs }, () => []));

  tokens_per_batch.value = Array.from({ length: bs }, () => []);
  results_per_batch.value = Array.from({ length: bs }, () => []);
  result_text_batched.value = Array(bs).fill("");

  isInitializedAfterStart.value = true;
  console.log("Data structures initialized.");
}

function cleanBeforeGenerate() {
  console.log("Cleaning data before new generation...");
  current_batch_index_to_display.value = 0; // 重置显示批次
  // actualNumLayers 将由 'start' 消息更新，此处不清零
  
  // 清空所有批处理数据容器
  qkv_vectors_batched.value = [];
  mlp1_vectors_batched.value = [];
  mlp2_vectors_batched.value = [];
  attention_matrix_batched.value = [];
  attention_size_batched.value = [];
  mlp2_vectors_pca.value = [];
  tokens_per_batch.value = [];
  results_per_batch.value = [];
  result_text_batched.value = [];

  isInitializedAfterStart.value = false; // 等待新的 "start" 消息
  console.log("Data cleaned. isInitializedAfterStart set to false.");
}

// --- WebSocket 连接与消息处理 ---
const { status: wsStatus, data: wsData, send: wsSend, open: wsOpen, close: wsClose } = useWebSocket(
  "ws://localhost:5000", // 后端 WebSocket 地址
  {
    autoReconnect: { retries: 3, delay: 3000, onFailed() { message.error("WebSocket auto-reconnect failed."); }},
    heartbeat: { message: JSON.stringify({ type: "ping" }), interval: 25000, pongTimeout: 10000 }, // 调整心跳间隔
    onConnected: () => { console.log("WebSocket connected"); message.success("WebSocket connected."); loadingRef.value = false; },
    onDisconnected: (ws, event) => { console.log("WebSocket disconnected:", event); message.warning(event.wasClean ? "WebSocket connection closed." : "WebSocket connection lost unexpectedly."); loadingRef.value = false; },
    onError: (ws, event) => { console.error("WebSocket error:", event); message.error("WebSocket connection error."); loadingRef.value = false; },
    onMessage: (ws, event) => {
      if (typeof event.data !== 'string') {
          console.log("Received non-string WS message (e.g., pong):", event.data);
          return;
      }
      let parsedData;
      try {
        parsedData = JSON.parse(event.data);
      } catch (e) {
        console.error("Failed to parse WebSocket message:", event.data, e);
        message.error("Received an unparsable WebSocket message.");
        return;
      }
      // console.log("Parsed WS Data:", parsedData);

      const type = parsedData.type;
      const currentBatchSize = batch_size.value; // 获取当前配置的批次大小

      if (type === "start") {
        actualNumLayers.value = parseInt(parsedData.num_layers) || 0;
        
        if (isTrainingMode.value) {
          // resize prompts to parsedData.micro_batch_size size
          const microBatchSize = parseInt(parsedData.micro_batch_size) || 1;
          if (microBatchSize > 0 && microBatchSize !== currentBatchSize) {
            console.log(`Received 'start' with micro_batch_size ${microBatchSize}, but current batch size is ${currentBatchSize}. Adjusting prompts.`);
            prompts.value = prompts.value.slice(0, microBatchSize);
            while (prompts.value.length < microBatchSize) {
              prompts.value.push(""); // Fill with empty prompts if needed
            }
          }
          initializeDataStructures(actualNumLayers.value, batch_size.value);
          const seqLength = parseInt(parsedData.seq_length) || 128;
          for (let bIdx = 0; bIdx < microBatchSize; bIdx++) {
            tokens_per_batch.value[bIdx] = Array.from({ length: seqLength }, () => ({ id: -1, token: "", logit: 0, probability: 0 }));
            results_per_batch.value[bIdx] = Array.from({ length: seqLength }, () => ({} as OutputProbData));
          }
        } else {
          initializeDataStructures(actualNumLayers.value, currentBatchSize);
          // 后端发送的 prompts 是一个扁平化的 token 对象列表
          // [{id, token}, {id, token}, ...]
          // 需要根据 batch_size 将其分配到 tokens_per_batch
          const flatInitialTokens: TokenInfo[] = parsedData.prompts || [];
          if (currentBatchSize > 0 && flatInitialTokens.length > 0) {
            // 假设所有初始 prompt 长度相同，这是后端目前的行为
            const numTokensPerInitialPrompt = Math.floor(flatInitialTokens.length / currentBatchSize);
            if (numTokensPerInitialPrompt * currentBatchSize !== flatInitialTokens.length) {
              console.warn(`Initial tokens length (${flatInitialTokens.length}) not divisible by batch size (${currentBatchSize}). Truncating.`);
              message.warning("The number of initial tokens is not divisible by the batch size. Some prompts may be truncated.");
            }

            for (let bIdx = 0; bIdx < currentBatchSize; bIdx++) {
              const batchPromptTokens = flatInitialTokens.slice(
                bIdx * numTokensPerInitialPrompt,
                (bIdx + 1) * numTokensPerInitialPrompt
              );
              tokens_per_batch.value[bIdx] = batchPromptTokens;
              // 为每个初始 token 准备空的 result 对象槽位
              results_per_batch.value[bIdx] = Array(batchPromptTokens.length).fill({});
            }
          } else if (currentBatchSize > 0) {
              console.warn("Received 'start' message but parsedData.prompts is empty or batch size issue.");
          }
          // loadingRef.value 应该在 "generate" 开始时设置，在 "finish" 或 "error" 时清除
          // "start" 消息本身不直接控制 loadingRef
        }
      } else if (type === "update") {
        if (!isInitializedAfterStart.value) {
          console.warn("Ignoring 'update': Data structures not yet initialized (no 'start' message received or data cleaned).");
          return;
        }
        const layer_id_from_backend = parseInt(parsedData.layer_id); // 1-indexed
        const layer_idx = layer_id_from_backend - 1; // 0-indexed for array access
        const update_type = parseInt(parsedData.update_type); // Corresponds to FlagType enum value

        // --- Vector Data (QKV, MLP1, MLP2) ---
        // Types 1 (QKV), 4 (MLP1), 5 (MLP2)
        if (update_type === 1 || update_type === 4 || update_type === 5) {
          if ((update_type === 1 && !visualizationSwitches.value.QKV_mat_mul) ||
              (update_type === 4 && !visualizationSwitches.value.MLP1_mat_mul) ||
              (update_type === 5 && !visualizationSwitches.value.MLP2_mat_mul)) {
            return; // Skip if visualization is off
          }
          
          const num_new_tokens_per_batch = parseInt(parsedData.args[0]);
          const feature_dim = (update_type === 1) ? visualizationSettings.value.qkvPixels : visualizationSettings.value.mlpPixels;
          const flat_data: number[] = parsedData.result;
          
          let target_array_ref;
          if (update_type === 1) target_array_ref = qkv_vectors_batched;
          else if (update_type === 4) target_array_ref = mlp1_vectors_batched;
          else target_array_ref = mlp2_vectors_batched;

          if (layer_idx < 0 || layer_idx >= actualNumLayers.value || !target_array_ref.value[layer_idx]) {
            console.error(`Vector Update: Invalid layer_idx ${layer_idx} for update_type ${update_type}. Total layers: ${actualNumLayers.value}`);
            return;
          }

          for (let bIdx = 0; bIdx < currentBatchSize; bIdx++) {
            if (!target_array_ref.value[layer_idx][bIdx]) {
                console.warn(`Vector Update: Target array for layer ${layer_idx}, batch ${bIdx} is undefined.`);
                target_array_ref.value[layer_idx][bIdx] = []; // Initialize if somehow missed
            }
            if (isTrainingMode.value) {
              target_array_ref.value[layer_idx][bIdx] = []; // Clear previous data for training mode
            }
            for (let t = 0; t < num_new_tokens_per_batch; t++) {
              const slice_start = (bIdx * num_new_tokens_per_batch + t) * feature_dim;
              const slice_end = slice_start + feature_dim;
              if (slice_end <= flat_data.length) {
                target_array_ref.value[layer_idx][bIdx].push(flat_data.slice(slice_start, slice_end));
              } else {
                console.warn(`Vector Update: Data slice out of bounds for L${layer_idx} B${bIdx} T_new${t}. Start: ${slice_start}, End: ${slice_end}, DataLen: ${flat_data.length}`);
              }
            }
          }
        }
        // --- Attention Matrix (Type 2) ---
        else if (update_type === 2 && visualizationSwitches.value.RawAttentionScore_mat_mul) {
          const num_heads = parseInt(parsedData.args[0]); // Number of attention heads
          const n_dim_queries = parseInt(parsedData.args[1]); // num new queries for this update
          const m_dim_keys = parseInt(parsedData.args[2]);    // total key sequence length now
          const flat_data: number[] = parsedData.result;
          numHeads.value = num_heads; // Update global numHeads

          if (layer_idx < 0 || layer_idx >= actualNumLayers.value || !attention_matrix_batched.value[layer_idx]) {
            console.error(`Attention Update: Invalid layer_idx ${layer_idx}. Total layers: ${actualNumLayers.value}`);
            return;
          }

          for (let bIdx = 0; bIdx < currentBatchSize; bIdx++) {
            const current_lb_data = attention_matrix_batched.value[layer_idx][bIdx];
            if (!current_lb_data || !isArray(current_lb_data) || current_lb_data.length !== num_heads) {
                 console.log(`Attention Update: Structures for L${layer_idx} B${bIdx} not initialized. Matrix: ${!!attention_matrix_batched.value[layer_idx][bIdx]}, Size: ${attention_size_batched.value[layer_idx][bIdx]}`);
                 attention_matrix_batched.value[layer_idx][bIdx] = Array.from({ length: num_heads }, () => []); // Initialize if needed
                 attention_size_batched.value[layer_idx][bIdx] = 0;
            }
            
            const current_matrix_size = attention_size_batched.value[layer_idx][bIdx];
            let batch_matrix = attention_matrix_batched.value[layer_idx][bIdx];

            // Expand matrix if m_dim_keys (new total size) is larger than current_matrix_size
            if (m_dim_keys > current_matrix_size) {
              for (let hIdx = 0; hIdx < num_heads; hIdx++) {
                let head_matrix = batch_matrix[hIdx];
                // Expand existing rows
                for (let r = 0; r < current_matrix_size; r++) {
                  head_matrix[r].push(...Array(m_dim_keys - current_matrix_size).fill(0));
                }
                // Add new rows
                for (let r = 0; r < (m_dim_keys - current_matrix_size); r++) {
                  head_matrix.push(Array(m_dim_keys).fill(0));
                }
              }
              attention_size_batched.value[layer_idx][bIdx] = m_dim_keys;
            }
            
            // Fill in the new attention block
            // The backend sends data for (n_dim_queries new queries) X (m_dim_keys total keys)
            // These new query rows start at index (m_dim_keys - n_dim_queries) in the full matrix
            const start_row_idx = m_dim_keys - n_dim_queries;
            const batch_base_offset = bIdx * num_heads * n_dim_queries * m_dim_keys;

            for (let hIdx = 0; hIdx < num_heads; hIdx++) {
              const head_matrix = batch_matrix[hIdx];
              const head_offset = hIdx * n_dim_queries * m_dim_keys + batch_base_offset;
              for (let i = 0; i < n_dim_queries; i++) { // Iterate over new query rows
                const matrix_row_idx = start_row_idx + i;
                if (matrix_row_idx < head_matrix.length) {
                  for (let j = 0; j < m_dim_keys; j++) { // Iterate over key columns
                    if (j < head_matrix[matrix_row_idx].length) {
                      const data_idx = head_offset + i * m_dim_keys + j;
                      if (data_idx < flat_data.length) {
                        head_matrix[matrix_row_idx][j] = flat_data[data_idx];
                      } else { console.warn(`Attn Update: Flat data OOB. Idx:${data_idx}, Len:${flat_data.length}`); }
                    } else { console.warn(`Attn Update: Key idx ${j} OOB for row ${matrix_row_idx}. Row len: ${head_matrix[matrix_row_idx].length}`);}
                  }
                } else { console.warn(`Attn Update: Query idx ${matrix_row_idx} OOB. Matrix rows: ${head_matrix.length}`); }
              }
            }
          }
        }
        // --- Output Probabilities (Type 6) ---
        else if (update_type === 6 && visualizationSwitches.value.Result) {
          // `sampled` is an array of TokenInfo, one for each batch
          const sampled_tokens_info_batched: TokenInfo[] = parsedData.sampled;
          // `result` is a flat list of TokenInfo for top_k, needs to be chunked by TOP_K_FOR_PROBS
          const top_k_probs_flat: TokenInfo[] = parsedData.result;

          for (let bIdx = 0; bIdx < currentBatchSize; bIdx++) {
            if (bIdx >= sampled_tokens_info_batched.length || !tokens_per_batch.value[bIdx] || !results_per_batch.value[bIdx]) {
              console.warn(`Probs Update: Batch index ${bIdx} out of bounds or uninitialized data.`);
              continue;
            }

            const sampled_for_this_batch = sampled_tokens_info_batched[bIdx];
            const probs_for_this_batch = top_k_probs_flat.slice(
              bIdx * TOP_K_FOR_PROBS,
              (bIdx + 1) * TOP_K_FOR_PROBS
            );
            
            const current_token_list_for_batch = tokens_per_batch.value[bIdx];
            const current_results_list_for_batch = results_per_batch.value[bIdx];
            
            // The probabilities received are for the token *before* the `sampled_for_this_batch`.
            // So, we store them at the index of the last token currently in `tokens_per_batch`.
            // If `tokens_per_batch` is [t0, t1], and `sampled_for_this_batch` is t2,
            // these probs are P(t2 | t0, t1). We store this at index 1 of `results_per_batch`.
            const result_idx_for_prev_token = current_token_list_for_batch.length - 1;

            if (result_idx_for_prev_token >= 0 && result_idx_for_prev_token < current_results_list_for_batch.length) {
              current_results_list_for_batch[result_idx_for_prev_token] = {
                probs: probs_for_this_batch,
                sampled: sampled_for_this_batch, // This 'sampled' is actually the token *following* the one whose probs are listed
              };
            } else {
               console.warn(`Probs Update: results_per_batch index ${result_idx_for_prev_token} for batch ${bIdx} is out of bounds. Len: ${current_results_list_for_batch.length}. This might happen for the very first set of probabilities if prompt tokens are not pre-filled in results_per_batch.`);
               // If it's for the prompt and results_per_batch was initialized for prompt length.
               if (result_idx_for_prev_token === -1 && current_results_list_for_batch.length > 0) { // No, this case should not happen if start is handled.
                    // This logic seems complex, usually results_per_batch should be pre-sized
               }
            }
            
            // Add the new sampled token to the list for this batch
            current_token_list_for_batch.push(sampled_for_this_batch);
            // Add an empty placeholder for the *next* token's probabilities
            current_results_list_for_batch.push({} as OutputProbData);
          }
        }
        // --- PCA Data (Type 7) ---
        else if (update_type === 7 && visualizationSwitches.value.MLP2_Plot) {
          const num_batches_pca = parseInt(parsedData.args[0]); // Should match currentBatchSize
          const n_tokens_in_pca_batch = parseInt(parsedData.args[1]); // Num tokens per batch in this PCA update
          const pca_data_flat: number[] = parsedData.result; // flat list of [x,y,x,y,...]

          if (layer_idx < 0 || layer_idx >= actualNumLayers.value || !mlp2_vectors_pca.value[layer_idx]) {
            console.error(`PCA Update: Invalid layer_idx ${layer_idx}. Total layers: ${actualNumLayers.value}`);
            return;
          }
          if (num_batches_pca !== currentBatchSize) {
            console.warn(`PCA Update: num_batches_pca (${num_batches_pca}) from backend doesn't match current UI batch_size (${currentBatchSize}).`);
            // Proceeding with num_batches_pca, but this might indicate a mismatch.
          }

          // Backend sends ALL PCA data for the sequence so far for this layer. Clear previous.
          for(let bIdx=0; bIdx < num_batches_pca; bIdx++){
              if (mlp2_vectors_pca.value[layer_idx][bIdx] !== undefined) {
                  mlp2_vectors_pca.value[layer_idx][bIdx] = [];
              } else {
                  // This case should ideally be prevented by proper initialization
                  console.warn(`PCA Update: mlp2_vectors_pca for L${layer_idx} B${bIdx} was undefined. Initializing.`);
                  if (!mlp2_vectors_pca.value[layer_idx]) mlp2_vectors_pca.value[layer_idx] = [];
                  mlp2_vectors_pca.value[layer_idx][bIdx] = [];
              }
          }
          
          let current_flat_idx = 0;
          for (let b = 0; b < num_batches_pca; b++) {
            if(b < mlp2_vectors_pca.value[layer_idx].length){ // Ensure batch array exists
              for (let t = 0; t < n_tokens_in_pca_batch; t++) {
                if (current_flat_idx + 1 < pca_data_flat.length) {
                  mlp2_vectors_pca.value[layer_idx][b].push(
                    [pca_data_flat[current_flat_idx], pca_data_flat[current_flat_idx + 1]]
                  );
                  current_flat_idx += 2;
                } else {
                  console.warn(`PCA Update: Flat data OOB. Idx:${current_flat_idx}, Required 2, Len:${pca_data_flat.length}. L${layer_idx} B${b} T${t}`);
                  break; // Stop processing this batch if data runs out
                }
              }
            } else {
                console.warn(`PCA Update: Batch index ${b} OOB for layer ${layer_idx} during fill. Max batches: ${mlp2_vectors_pca.value[layer_idx].length}`);
            }
          }
        }

      } else if (type === "finish") {
        result_text_batched.value = parsedData.text as string[] || Array(currentBatchSize).fill("[No text received]");
        loadingRef.value = false;
        message.success("Generation complete!");
      } else if (type === "finish-update") { // Not used by current 'generate' flow
        loadingRef.value = false;
        message.info("Update complete (perturbation or similar operation).");
      } else if (type === "error") {
        loadingRef.value = false;
        message.error(`Backend error: ${parsedData.message || "Unknown error"}`);
      } else if (type === "pong") {
        // console.log("Pong received from server."); // Handled by useWebSocket heartbeat
      } else {
        console.warn("Received unknown WebSocket message type:", type, parsedData);
      }
    }
  }
);

// --- UI 控制与交互 ---
const current_layer_id_for_display = ref<number>(1); // For table and PCA tab
const tokens_to_generate_num = ref<number>(10);
// const filterRange = ref(false); // Not used in this refactor, backend doesn't show it
// const range = ref<[number, number]>([1, 3]); // Not used
// const slice_start_offset = ref(0); // Not used, display all tokens

const addPrompt = () => {
  if (prompts.value.length < 8) { // Max 8 prompts for sanity
    prompts.value.push("Another example prompt.");
  } else {
    message.warning("A maximum of 8 prompts is supported.");
  }
};
const removePrompt = (index: number) => {
  if (prompts.value.length > 1) {
    prompts.value.splice(index, 1);
    // Adjust current_batch_index_to_display if it becomes out of bounds
    if (current_batch_index_to_display.value >= prompts.value.length) {
      current_batch_index_to_display.value = Math.max(0, prompts.value.length - 1);
    }
  } else {
    message.warning("At least 1 prompt is required.");
  }
};

function generate() {
  if (wsStatus.value !== 'OPEN') {
    message.error("WebSocket not connected! Please wait for the connection or check the backend service.");
    // Try to reconnect if not already connecting
    if (wsStatus.value === 'CLOSED') wsOpen();
    return;
  }
  if (prompts.value.some(p => p.trim() === "")) {
    message.warning("Please enter content for all prompts!");
    return;
  }
  if (tokens_to_generate_num.value <= 0) {
    message.warning("The number of tokens to generate must be greater than 0.");
    return;
  }

  loadingRef.value = true;
  cleanBeforeGenerate(); // 清理旧数据, isInitializedAfterStart is set to false

  // 确保 prompts.value 和 batch_size.value 是最新的
  // batch_size 是 computed property, 它会自己更新

  // 等待一个 tick 确保 cleanBeforeGenerate 的状态更新 (isInitializedAfterStart=false)
  // 已经传播，然后再发送请求。这有助于避免竞争条件。
  nextTick(() => {
    const vis_flags_payload: { [key: string]: string } = {};
    for (const key in visualizationSwitches.value) {
      vis_flags_payload[key] = (visualizationSwitches.value as any)[key] ? "True" : "False";
    }

    const dist_configs_payload: Partial<DisturbanceConfig> & {[key:string]:any} = {};
    // Only include coef if the perturbation is active, and ensure boolean is sent for inactive
    if (disturbanceSettings.value.weight_perturbation) {
      dist_configs_payload.weight_perturbation = true;
      dist_configs_payload.weight_perturbation_fn = disturbanceSettings.value.weight_perturbation_fn;
      dist_configs_payload.weight_perturbation_coef = Number(disturbanceSettings.value.weight_perturbation_coef);
    } else {
      dist_configs_payload.weight_perturbation = false;
    }
    if (disturbanceSettings.value.calculation_perturbation) {
      dist_configs_payload.calculation_perturbation = true;
      dist_configs_payload.calculation_perturbation_fn = disturbanceSettings.value.calculation_perturbation_fn;
      dist_configs_payload.calculation_perturbation_coef = Number(disturbanceSettings.value.calculation_perturbation_coef);
    } else {
      dist_configs_payload.calculation_perturbation = false;
    }
     if (disturbanceSettings.value.system_perturbation) {
      dist_configs_payload.system_perturbation = true;
      dist_configs_payload.system_perturbation_fn = disturbanceSettings.value.system_perturbation_fn;
      dist_configs_payload.system_perturbation_coef = Number(disturbanceSettings.value.system_perturbation_coef);
    } else {
      dist_configs_payload.system_perturbation = false;
    }

    const compressor_config_payload = {
        "QKV": {
            "pixels": visualizationSettings.value.qkvPixels,
            "method": visualizationSettings.value.qkvCompressionMethod
        },
        "MLP": {
            "pixels": visualizationSettings.value.mlpPixels,
            "method": visualizationSettings.value.mlpCompressionMethod
        }
    };

    let params = {
      type: "generate",
      prompts: prompts.value,
      tokens_to_generate: tokens_to_generate_num.value,
      visualization_flags: vis_flags_payload,
      disturbance_configs: dist_configs_payload,
      // top_k, top_p, temperature etc. can be added here if UI controls are provided
      compressor_config: compressor_config_payload,
    };
    console.log("Sending 'generate' request:", JSON.stringify(params, null, 2));
    wsSend(JSON.stringify(params));
  });
}

function runTrainingStep() {
  if (wsStatus.value !== 'OPEN') {
    message.error("WebSocket 未连接!");
    if (wsStatus.value === 'CLOSED') wsOpen();
    return;
  }
  
  loadingRef.value = true;
  cleanBeforeGenerate();

  nextTick(() => {
    visualizationSwitches.value.RawAttentionScore_mat_mul = false;
    visualizationSwitches.value.Result = false;
    visualizationSwitches.value.MLP2_Plot = false;
    const vis_flags_payload: { [key: string]: string } = {};
    for (const key in visualizationSwitches.value) {
      vis_flags_payload[key] = (visualizationSwitches.value as any)[key] ? "True" : "False";
    }

    const dist_configs_payload: Partial<DisturbanceConfig> & {[key:string]:any} = {};
    // Only include coef if the perturbation is active, and ensure boolean is sent for inactive
    if (disturbanceSettings.value.weight_perturbation) {
      dist_configs_payload.weight_perturbation = true;
      dist_configs_payload.weight_perturbation_fn = disturbanceSettings.value.weight_perturbation_fn;
      dist_configs_payload.weight_perturbation_coef = Number(disturbanceSettings.value.weight_perturbation_coef);
    } else {
      dist_configs_payload.weight_perturbation = false;
    }
    if (disturbanceSettings.value.calculation_perturbation) {
      dist_configs_payload.calculation_perturbation = true;
      dist_configs_payload.calculation_perturbation_fn = disturbanceSettings.value.calculation_perturbation_fn;
      dist_configs_payload.calculation_perturbation_coef = Number(disturbanceSettings.value.calculation_perturbation_coef);
    } else {
      dist_configs_payload.calculation_perturbation = false;
    }
     if (disturbanceSettings.value.system_perturbation) {
      dist_configs_payload.system_perturbation = true;
      dist_configs_payload.system_perturbation_fn = disturbanceSettings.value.system_perturbation_fn;
      dist_configs_payload.system_perturbation_coef = Number(disturbanceSettings.value.system_perturbation_coef);
    } else {
      dist_configs_payload.system_perturbation = false;
    }

    const compressor_config_payload = {
        "QKV": {
            "pixels": visualizationSettings.value.qkvPixels,
            "method": visualizationSettings.value.qkvCompressionMethod
        },
        "MLP": {
            "pixels": visualizationSettings.value.mlpPixels,
            "method": visualizationSettings.value.mlpCompressionMethod
        }
    };

    let params = {
      type: "run_training_step",
      visualization_flags: vis_flags_payload,
      disturbance_configs: dist_configs_payload,
      compressor_config: compressor_config_payload,
    };
    console.log("Sending 'run_training_step' request:", params);
    wsSend(JSON.stringify(params));
  });
}

// --- 表格数据与列定义 ---
const table_data_for_display = computed(() => {
  if (!isInitializedAfterStart.value || actualNumLayers.value === 0 || tokens_per_batch.value.length === 0) return [];
  
  const bIdx = current_batch_index_to_display.value;
  if (bIdx < 0 || bIdx >= batch_size.value || !tokens_per_batch.value[bIdx]) return [];

  const layerIdx = current_layer_id_for_display.value - 1; // 0-indexed for array access
  if (layerIdx < 0 || layerIdx >= actualNumLayers.value) return [];

  const tokens_for_selected_batch = tokens_per_batch.value[bIdx];
  const num_tokens_to_display = tokens_for_selected_batch.length;

  return Array.from({ length: num_tokens_to_display }, (_, token_array_idx) => {
    const token_obj = tokens_for_selected_batch[token_array_idx];
    const result_obj = results_per_batch.value[bIdx]?.[token_array_idx];

    return {
      key: token_array_idx, // Unique key for row
      token_render_idx: token_array_idx + 1, // 1-indexed for display
      token: token_obj?.token || `[T${token_array_idx + 1}]`,
      qkv: (visualizationSwitches.value.QKV_mat_mul && qkv_vectors_batched.value[layerIdx]?.[bIdx]?.[token_array_idx]) 
           ? qkv_vectors_batched.value[layerIdx][bIdx][token_array_idx] : [],
      mlp1: (visualizationSwitches.value.MLP1_mat_mul && mlp1_vectors_batched.value[layerIdx]?.[bIdx]?.[token_array_idx])
            ? mlp1_vectors_batched.value[layerIdx][bIdx][token_array_idx] : [],
      mlp2: (visualizationSwitches.value.MLP2_mat_mul && mlp2_vectors_batched.value[layerIdx]?.[bIdx]?.[token_array_idx])
            ? mlp2_vectors_batched.value[layerIdx][bIdx][token_array_idx] : [],
      // result_obj is OutputProbData for the generation step *after* this token.
      // So, row `i` (token `i`) will show probabilities P(token_{i+1} | token_0...token_i)
      // The `sampled` field in result_obj is token_{i+1}.
      result: (visualizationSwitches.value.Result && result_obj && Object.keys(result_obj).length > 0) 
              ? result_obj : null,
    };
  });
});

const columns = computed(() => {
  const baseColumns: any[] = [
    { 
      type: "expand", 
      expandable: (rowData: any) => visualizationSwitches.value.Result && rowData.result && rowData.result.probs && rowData.result.probs.length > 0,
      renderExpand: (rowData: any) => rowData.result ? h(OutputProbs, { data: rowData.result }) : h('span', 'No probability data for this step.'),
      align: "center", width: 60 
    },
    { title: "#", key: "token_render_idx", width: 60, align: "center"},
    { title: "Token", key: "token", ellipsis: { tooltip: true }, width: 120, align: "center" },
  ];
  if (visualizationSwitches.value.QKV_mat_mul) {
    baseColumns.push({ title: "QKV Vector", key: "qkv", align: "center", width: 250,
      render: (rowData: any) => rowData.qkv?.length 
        ? h(ColoredVector, { values: rowData.qkv, colors: qkv_colors.value, length: visualizationSettings.value.qkvPixels }) 
        : h('span', '-')
    });
  }
  if (visualizationSwitches.value.MLP1_mat_mul) {
    baseColumns.push({ title: "MLP1 Vector", key: "mlp1", align: "center", width: 200,
      render: (rowData: any) => rowData.mlp1?.length 
        ? h(ColoredVector, { values: rowData.mlp1, colors: mlp1_colors.value, length: visualizationSettings.value.mlpPixels }) 
        : h('span', '-')
    });
  }
  if (visualizationSwitches.value.MLP2_mat_mul) {
    baseColumns.push({ title: "MLP2 Vector", key: "mlp2", align: "center", width: 200,
      render: (rowData: any) => rowData.mlp2?.length 
        ? h(ColoredVector, { values: rowData.mlp2, colors: mlp2_colors.value, length: visualizationSettings.value.mlpPixels }) 
        : h('span', '-')
    });
  }
  return baseColumns;
});


// --- 注意力矩阵和PCA的层选择与数据获取 ---
const attentionLayersToDisplay = ref<number[]>([]); // Array of layer IDs (1-indexed)
const selectedHeadId = ref<number>(0); // 0-indexed head ID for attention matrix display
const numHeads = ref<number>(0); // Number of attention heads for the current layer
const layerOptionsForSelection = computed(() => 
  Array.from({ length: actualNumLayers.value }, (_, i) => ({ label: `Layer ${i + 1}`, value: i + 1 }))
);
const headOptions = computed(() => {
  if (numHeads.value === 0) {
    return [];
  }
  return Array.from({ length: numHeads.value }, (_, i) => ({
    label: `Head ${i + 1}`,
    value: i,
  }));
});

// Watch actualNumLayers to initialize/reset attentionLayersToDisplay and current_layer_id_for_display
watch(actualNumLayers, (newVal) => {
  if (newVal > 0) {
    // Default to showing the first layer for attention, if not already set or if previous selection is invalid
    if (attentionLayersToDisplay.value.length === 0 || attentionLayersToDisplay.value.some(l => l > newVal)) {
         attentionLayersToDisplay.value = [1];
    }
    // Ensure current_layer_id_for_display is valid
    if (current_layer_id_for_display.value > newVal) current_layer_id_for_display.value = newVal;
    if (current_layer_id_for_display.value < 1) current_layer_id_for_display.value = 1;
  } else {
    attentionLayersToDisplay.value = [];
    current_layer_id_for_display.value = 1; // Default, will be disabled if actualNumLayers is 0
  }
}, { immediate: true });


const tokens_for_selected_batch_attention = computed(() => {
    if (!isInitializedAfterStart.value) return [];
    const bIdx = current_batch_index_to_display.value;
    if (bIdx < 0 || !tokens_per_batch.value || bIdx >= tokens_per_batch.value.length ) return [];
    return tokens_per_batch.value[bIdx] || []; // Returns TokenInfo[]
});

const tokens_for_pca = computed(() => {
    if (!isInitializedAfterStart.value) return [];
    return tokens_per_batch.value || []; // Returns TokenInfo[]
});

// Getter for attention matrix data for a specific layer (1-indexed) and current_batch_index_to_display
const getAttentionMatrixForDisplay = (layerId: number, headId: number) => {
  if (!isInitializedAfterStart.value) return [];
  const layerIdx = layerId - 1;
  const bIdx = current_batch_index_to_display.value;
  if (layerIdx < 0 || layerIdx >= actualNumLayers.value ||
      bIdx < 0 || bIdx >= batch_size.value ||
      !attention_matrix_batched.value[layerIdx] || !attention_matrix_batched.value[layerIdx][bIdx] || headId < 0 || headId >= attention_matrix_batched.value[layerIdx][bIdx].length || !attention_matrix_batched.value[layerIdx][bIdx][headId]) {
    return [];
  }
  return attention_matrix_batched.value[layerIdx][bIdx][headId];
};

const getAttentionSizeForDisplay = (layerId: number) => {
  if (!isInitializedAfterStart.value) return 0;
  const layerIdx = layerId - 1;
  const bIdx = current_batch_index_to_display.value;
   if (layerIdx < 0 || layerIdx >= actualNumLayers.value ||
      bIdx < 0 || bIdx >= batch_size.value ||
      !attention_size_batched.value[layerIdx] || attention_size_batched.value[layerIdx][bIdx] === undefined ) {
    return 0;
  }
  return attention_size_batched.value[layerIdx][bIdx];
};

const pca_data_for_selected_layer_all_batches = computed(() => {
    if (!isInitializedAfterStart.value || actualNumLayers.value === 0) return [];
    const layerIdx = current_layer_id_for_display.value - 1;
    if (layerIdx < 0 || layerIdx >= actualNumLayers.value || !mlp2_vectors_pca.value[layerIdx]) {
        return [];
    }
    // Returns Array<Array<[number,number]>> which is [batch_idx][token_idx][pca_pair]
    // PCAPlot component expects this format for its `values` prop.
    return mlp2_vectors_pca.value[layerIdx];
});


const batchSelectorOptions = computed(() => 
  Array.from({length: batch_size.value}, (_, i) => ({ label: `Batch ${i+1}`, value: i }))
);

// --- Watchers for index safety ---
watch(batch_size, (newSize, oldSize) => {
    if (newSize === 0) {
        current_batch_index_to_display.value = 0; // Or -1 if you prefer to indicate no valid batch
    } else if (current_batch_index_to_display.value >= newSize) {
        current_batch_index_to_display.value = newSize - 1;
    }
    // If batch_size increases from 0, and index was 0, it's still valid.
});
// actualNumLayers watcher already handles current_layer_id_for_display updates.

</script>

<template>
  <n-space vertical style="width: 100%; gap: 16px;">
    <!-- Prompts 输入区域 -->
    <n-card title="Model Input Prompts" v-if="!isTrainingMode">
      <template #header-extra>
        <n-button @click="addPrompt" type="primary" size="small" :disabled="loadingRef" round>
          <template #icon><n-icon :component="AddCircleOutline"/></template> Add Prompt
        </n-button>
      </template>
      <n-space vertical>
        <n-input-group v-for="(prompt_text, index) in prompts" :key="index">
          <n-input
            v-model:value="prompts[index]"
            type="textarea"
            :autosize="{ minRows: 1, maxRows: 3 }"
            :placeholder="`Enter Prompt ${index + 1}`"
            style="flex-grow: 1;"
            :disabled="loadingRef"
          />
          <n-button
            quaternary
            circle
            type="error"
            size="small"
            :disabled="prompts.length === 1 || loadingRef"
            @click="removePrompt(index)"
            title="Remove this Prompt"
            style="margin-left: 8px;"
          >
            <template #icon><n-icon :component="RemoveCircleOutline"/></template>
          </n-button>
        </n-input-group>
      </n-space>
    </n-card>

    <!-- 生成控制与扰动 -->
    <n-grid :x-gap="12" :y-gap="12" :cols="2">
        <n-gi>
            <n-card title="Run Control">
                <n-space vertical>
                    <n-radio-group v-model:value="isTrainingMode" name="run-mode" style="margin-bottom: 10px;" :disabled="loadingRef">
                      <n-radio-button :value="false">Inference Mode</n-radio-button>
                      <n-radio-button :value="true">Training Mode</n-radio-button>
                    </n-radio-group>
                    <n-space v-if="!isTrainingMode" align="center">
                        <n-text>Tokens to Generate:</n-text>
                        <n-input-number v-model:value="tokens_to_generate_num" :min="1" :max="512" :disabled="loadingRef" style="width: 120px"/>
                    </n-space>
                    <n-button type="primary" block :loading="loadingRef" :disabled="loadingRef || wsStatus !== 'OPEN'" @click="isTrainingMode ? runTrainingStep() : generate()">
                        <template #icon><n-icon :component="FlashOutline" /></template>
                        {{ wsStatus !== 'OPEN' ? 'Connecting...' : (loadingRef ? (isTrainingMode ? 'Running...' : 'Generating...') : (isTrainingMode ? 'Run Training Step' : 'Start Generation')) }}
                    </n-button>
                    <n-alert title="WebSocket Status" :type="wsStatus === 'OPEN' ? 'success' : (wsStatus === 'CONNECTING' ? 'info' : 'error')" :show-icon="true">
                        Current Status: {{ wsStatus }}
                         <n-button v-if="wsStatus === 'CLOSED' || wsStatus === 'NONE'" @click="wsOpen" size="tiny" style="margin-left:10px;">Reconnect</n-button>
                    </n-alert>
                </n-space>
            </n-card>
        </n-gi>
        <n-gi>
            <n-card title="Visualization Toggles">
                <n-grid :x-gap="8" :y-gap="4" :cols="2">
                    <n-gi><n-checkbox v-model:checked="visualizationSwitches.QKV_mat_mul" :disabled="loadingRef">QKV Vector</n-checkbox></n-gi>
                    <n-gi><n-checkbox v-model:checked="visualizationSwitches.MLP1_mat_mul" :disabled="loadingRef">MLP1 Vector</n-checkbox></n-gi>
                    <n-gi><n-checkbox v-model:checked="visualizationSwitches.MLP2_mat_mul" :disabled="loadingRef">MLP2 Vector</n-checkbox></n-gi>
                    <n-gi><n-checkbox v-model:checked="visualizationSwitches.Result" :disabled="loadingRef || isTrainingMode">Output Probs</n-checkbox></n-gi>
                    <n-gi><n-checkbox v-model:checked="visualizationSwitches.RawAttentionScore_mat_mul" :disabled="loadingRef || isTrainingMode">Attention Matrix</n-checkbox></n-gi>
                    <n-gi><n-checkbox v-model:checked="visualizationSwitches.MLP2_Plot" :disabled="loadingRef || isTrainingMode">MLP2 PCA Plot</n-checkbox></n-gi>
                </n-grid>
            </n-card>
        </n-gi>
        <n-gi span="2">
          <n-card title="Visualization Config">
            <template #header-extra><n-icon :component="InformationCircleOutline" title="These values are sent to the backend to determine the number of pixels for compressed vectors."/></template>
            <n-grid :x-gap="12" :y-gap="8" :cols="2">
              <n-gi>
                <n-space align="center">
                  <n-text>QKV Vector Pixels:</n-text>
                  <n-input-number 
                    v-model:value="visualizationSettings.qkvPixels" 
                    :min="3" :step="3" 
                    :disabled="loadingRef" 
                    style="width: 120px"
                  />
                  <n-text>QKV Compression Method (Expr):</n-text>
                  <n-input
                    v-model:value="visualizationSettings.qkvCompressionMethod"
                    placeholder="e.g., data.mean(dim=-1)"
                    :disabled="loadingRef"
                  />
                </n-space>
              </n-gi>
              <n-gi>
                <n-space align="center">
                  <n-text>MLP Vector Pixels:</n-text>
                  <n-input-number 
                    v-model:value="visualizationSettings.mlpPixels" 
                    :min="1" 
                    :disabled="loadingRef" 
                    style="width: 120px"
                  />
                  <n-text>MLP Compression Method (Expr):</n-text>
                  <n-input
                    v-model:value="visualizationSettings.mlpCompressionMethod"
                    placeholder="e.g., data.mean(dim=-1)"
                    :disabled="loadingRef"
                  />
                </n-space>
              </n-gi>
            </n-grid>
          </n-card>
        </n-gi>
    </n-grid>

    <!-- 扰动配置 -->
    <n-collapse accordion :default-expanded-names="[]">
        <n-collapse-item title="Perturbation Config" name="disturbance">
            <template #header-extra><n-icon :component="CodeDownloadOutline" /></template>
            <n-grid :x-gap="12" :y-gap="12" :cols="3" item-responsive>
                <!-- 权重扰动卡片 -->
                <n-gi span="3 s:3 m:1 l:1">
                    <n-card title="Storage Perturbation" size="small">
                        <n-space align="center" justify="start" :wrap="false">
                            <n-switch 
                                v-model:value="disturbanceSettings.weight_perturbation" 
                                :disabled="loadingRef"
                            />
                            <n-select 
                              v-model:value="disturbanceSettings.weight_perturbation_fn" 
                              :options="disturbanceFnOptions" 
                              size="tiny"
                              placeholder="Function"
                              style="min-width: 100px; margin-left: 10px;"
                              :disabled="loadingRef || !disturbanceSettings.weight_perturbation"
                            />
                            <n-input-number 
                              v-model:value="disturbanceSettings.weight_perturbation_coef" 
                              :step="0.001" :min="0" placeholder="Coefficient" 
                              size="tiny" 
                              style="width: 100px; margin-left: 5px;"
                              :disabled="loadingRef || !disturbanceSettings.weight_perturbation"
                            />
                        </n-space>
                    </n-card>
                </n-gi>

                <!-- 计算过程扰动卡片 -->
                <n-gi span="3 s:3 m:1 l:1">
                    <n-card title="Calculation Perturbation" size="small">
                        <n-space align="center" justify="start" :wrap="false">
                            <n-switch 
                                v-model:value="disturbanceSettings.calculation_perturbation" 
                                :disabled="loadingRef"
                            />
                            <n-select 
                              v-model:value="disturbanceSettings.calculation_perturbation_fn" 
                              :options="disturbanceFnOptions" 
                              size="tiny" 
                              placeholder="Function"
                              style="min-width: 100px; margin-left: 10px;" 
                              :disabled="loadingRef || !disturbanceSettings.calculation_perturbation"
                            />
                            <n-input-number 
                              v-model:value="disturbanceSettings.calculation_perturbation_coef" 
                              :step="0.01" :min="0" placeholder="Coefficient" 
                              size="tiny" 
                              style="width: 100px; margin-left: 5px;" 
                              :disabled="loadingRef || !disturbanceSettings.calculation_perturbation"
                            />
                        </n-space>
                    </n-card>
                </n-gi>

                <!-- 系统扰动卡片 -->
                <n-gi span="3 s:3 m:1 l:1">
                     <n-card title="System Perturbation" size="small">
                        <n-space align="center" justify="start" :wrap="false">
                            <n-switch 
                                v-model:value="disturbanceSettings.system_perturbation" 
                                :disabled="loadingRef"
                            />
                            <n-select 
                              v-model:value="disturbanceSettings.system_perturbation_fn" 
                              :options="disturbanceFnOptions" 
                              size="tiny" 
                              placeholder="Function"
                              style="min-width: 100px; margin-left: 10px;" 
                              :disabled="loadingRef || !disturbanceSettings.system_perturbation"
                            />
                            <n-input-number 
                              v-model:value="disturbanceSettings.system_perturbation_coef" 
                              :step="0.001" :min="0" placeholder="Coefficient" 
                              size="tiny" 
                              style="width: 100px; margin-left: 5px;" 
                              :disabled="loadingRef || !disturbanceSettings.system_perturbation"
                            />
                        </n-space>
                    </n-card>
                </n-gi>
            </n-grid>
        </n-collapse-item>
    </n-collapse>
    
    <!-- 生成结果文本显示 -->
    <n-card title="Model Output Result" v-if="!isTrainingMode && !loadingRef && result_text_batched.some(t => t !== '')">
      <n-list bordered>
          <n-list-item v-for="(item, index) in result_text_batched" :key="`result-text-${index}`">
              <template #prefix><n-tag type="info">Batch {{ index + 1 }}</n-tag></template>
              <pre>{{ item }}</pre>
          </n-list-item>
      </n-list>
    </n-card>
    
    <!-- 可视化结果 -->
    <div v-if="isInitializedAfterStart && actualNumLayers > 0">
      <n-tabs type="line" animated display-directive="show" placement="top" style="margin-top: 16px;">
        <template #prefix>
          <NFlex align="center" style="margin-right: 20px;">
            <n-text strong>Select Batch to View:</n-text>
            <n-select 
              v-model:value="current_batch_index_to_display" 
              :options="batchSelectorOptions" 
              :disabled="batch_size <= 0" 
              size="small" 
              style="width: 130px;"
            />
          </NFlex>
        </template>

        <!-- Tab 1: 中间向量 & 输出概率表格 -->
        <n-tab-pane name="vectors_probs" tab="Intermediate Vectors & Output Probs">
          <NFlex align="center" justify="space-between" style="margin-bottom: 10px;">
            <NFlex align="center">
              <n-text><n-icon :component="LayersOutline" size="18" style="vertical-align: middle; margin-right: 5px;" />Select Layer to View (Total: {{ actualNumLayers }}):</n-text>
              <n-select 
                v-model:value="current_layer_id_for_display" 
                :options="layerOptionsForSelection"
                :disabled="actualNumLayers === 0" 
                size="small"
                style="width: 130px;"
              />
            </NFlex>
             <n-tag type="info" round size="small">Currently Displaying Batch: {{ current_batch_index_to_display + 1 }}</n-tag>
          </NFlex>
          <n-data-table
            :columns="columns"
            :data="table_data_for_display"
            :bordered="false"
            striped
            size="small"
            :max-height="600"
            :loading="loadingRef && table_data_for_display.length === 0"
            virtual-scroll
          />
          <n-empty v-if="!loadingRef && table_data_for_display.length === 0" description="No table data to display. Please check the selected layer or batch, or wait for data to load." style="margin-top: 20px;"/>
        </n-tab-pane>

        <!-- Tab 2: 注意力矩阵 -->
        <n-tab-pane name="attention" tab="Attention Matrix"  v-if="visualizationSwitches.RawAttentionScore_mat_mul">
            <NFlex align="center" justify="space-between" style="margin-bottom: 10px;">
                <NFlex align="center">
                    <n-text><n-icon :component="LayersOutline" size="18" style="vertical-align: middle; margin-right: 5px;" />Select Layers for Attention (Multi-select):</n-text>
                    <n-select 
                        v-model:value="attentionLayersToDisplay" 
                        multiple filterable tag 
                        :options="layerOptionsForSelection" 
                        clearable 
                        :disabled="actualNumLayers === 0"
                        placeholder="Select layers..."
                        size="small"
                        style="min-width: 200px;"
                    />
                    <n-text><n-icon :component="SparklesOutline" size="18" style="vertical-align: middle; margin-right: 5px;" />Select Attention Head:</n-text>
                    <n-select
                        v-model:value="selectedHeadId"
                        :options="headOptions"
                        :disabled="numHeads === 0"
                        placeholder="Select head..."
                        size="small"
                        style="min-width: 120px;"
                    />
                </NFlex>
                <n-tag type="info" round size="small">Currently Displaying Batch: {{ current_batch_index_to_display + 1 }}</n-tag>
            </NFlex>
            <n-grid v-if="attentionLayersToDisplay.length > 0" x-gap="12" y-gap="12" :cols="attentionLayersToDisplay.length === 1 ? 1 : 2" responsive="screen" item-responsive>
            <n-gi v-for="layerId in attentionLayersToDisplay" :key="`attn-${layerId}`" span="2 m:1">
                <AttentionMatrix
                v-if="getAttentionSizeForDisplay(layerId) > 0 && 
                        tokens_for_selected_batch_attention.length >= getAttentionSizeForDisplay(layerId) &&
                        getAttentionMatrixForDisplay(layerId, selectedHeadId).length === getAttentionSizeForDisplay(layerId)"
                :values="getAttentionMatrixForDisplay(layerId, selectedHeadId)"
                :size="getAttentionSizeForDisplay(layerId)"
                :color="attention_color"
                :layer_id="layerId"
                :tokens="tokens_for_selected_batch_attention.slice(0, getAttentionSizeForDisplay(layerId))"
                />
                <n-empty v-else :description="`Layer ${layerId} Head ${selectedHeadId} Batch ${current_batch_index_to_display + 1} attention data is unavailable or dimensions do not match (Matrix size: ${getAttentionSizeForDisplay(layerId)}, Tokens: ${tokens_for_selected_batch_attention.length}, Actual matrix size: ${getAttentionMatrixForDisplay(layerId, selectedHeadId).length})`" style="padding: 20px 0;" />
            </n-gi>
            </n-grid>
            <n-empty v-else description="Please select at least one layer to display the attention matrix." style="margin-top: 20px;" />
        </n-tab-pane>

        <!-- Tab 3: PCA 结果 -->
        <n-tab-pane name="pca" tab="PCA Result" v-if="visualizationSwitches.MLP2_Plot" :disabled="isTrainingMode">
            <NFlex align="center" justify="space-between" style="margin-bottom: 10px;">
                <NFlex align="center">
                    <n-text><n-icon :component="LayersOutline" size="18" style="vertical-align: middle; margin-right: 5px;" />Select Layer to View PCA:</n-text>
                     <n-select 
                        v-model:value="current_layer_id_for_display" 
                        :options="layerOptionsForSelection"
                        :disabled="actualNumLayers === 0" 
                        size="small"
                        style="width: 130px;"
                    />
                </NFlex>
                <!-- PCA图表已显示所有批次数据，无需特别注明当前批次 -->
            </NFlex>
            <PCAPlot
            v-if="pca_data_for_selected_layer_all_batches &&
                    pca_data_for_selected_layer_all_batches.length > 0 &&
                    pca_data_for_selected_layer_all_batches.some(batch_pca_data => batch_pca_data && batch_pca_data.length > 0)"
            :values="pca_data_for_selected_layer_all_batches"
            :layerId="current_layer_id_for_display"
            :tokens="tokens_for_pca"
            />
            <n-empty v-else :description="`Layer ${current_layer_id_for_display} PCA data is unavailable or empty. Please ensure that MLP2 vectors have been generated and PCA has been calculated. (PCA data batch count: ${pca_data_for_selected_layer_all_batches?.length || 0})`" style="margin-top: 20px;" />
        </n-tab-pane>
      </n-tabs>
    </div>
    <n-empty v-else-if="!loadingRef" description="No data to display. Please configure the prompts and click 'Start Generation'." style="margin-top: 30px;">
        <template #icon><n-icon :component="InformationCircleOutline" /></template>
    </n-empty>

  </n-space>
</template>

<style scoped>
.n-card {
  text-align: left; /* Naive UI 默认可能不是 left */
}
/* 微调 n-collapse-item 的头部，使其不那么拥挤 */
:deep(.n-collapse-item__header-main) {
  font-size: 1.05em !important;
}
</style>