<script setup lang="ts">
import { computed } from "vue";

// 定义 Token 和 Data 的接口，与 AppContent.vue 中的类型保持一致
interface TokenInfo {
  logit: number;
  id: number;
  token: string;
  probability: number;
}
interface OutputData {
  probs: Array<TokenInfo>; // 概率列表
  sampled: TokenInfo;    // 被采样到的 token
}

// 定义 props，并明确类型
const props = defineProps<{ data: OutputData | null | undefined }>(); // 允许 data 为 null 或 undefined

// 计算属性，检查数据是否有效
const isValidData = computed(() => {
  return props.data && props.data.probs && props.data.probs.length > 0 && props.data.sampled;
});

// 对概率进行排序，方便查看（只在数据有效时执行）
const sortedProbs = computed(() => {
  if (!isValidData.value || !props.data) return []; // 添加对 props.data 的检查
  // 创建副本进行排序，避免修改原始 prop 数据
  return [...props.data.probs].sort((a, b) => b.probability - a.probability);
});

</script>

<template>
  <!-- 使用 v-if 确保数据有效时才渲染 -->
  <n-space v-if="isValidData && props.data" vertical>
    <n-text strong>采样到的 Token:
      <!-- 使用 n-tag 显示采样到的 token 及其概率 -->
      <n-tag :type="'success'">
        {{ props.data.sampled.token }}: {{ (props.data.sampled.probability * 100).toFixed(2) }}% 🎯
      </n-tag>
    </n-text>
    <n-text strong>Top K 概率:</n-text>
    <n-space>
    <!-- 遍历排序后的概率列表 -->
    <n-tag
      v-for="item in sortedProbs"
      :key="item.id"
      :type="item.id === props.data.sampled.id ? 'success' : 'default'"
      :bordered="false"
      round
    >
      <!-- 显示 token 和其概率 -->
      {{ item.token }}: {{ (item.probability * 100).toFixed(2) }}%
      <!-- 如果是采样到的 token，则添加标记 -->
      <span v-if="item.id === props.data.sampled.id" style="margin-left: 4px;">🎯</span>
    </n-tag>
    </n-space>
  </n-space>
  <div v-else>
    <!-- 数据无效或加载中时的提示信息 -->
    <n-text type="info">等待或无输出概率数据...</n-text>
  </div>
</template>

<style scoped>
/* 为 n-tag 添加一些边距，使其在 n-space 中表现更好 */
.n-tag {
  margin: 2px;
}
</style>
