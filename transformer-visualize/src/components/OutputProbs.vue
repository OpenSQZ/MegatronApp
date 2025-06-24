<script setup lang="ts">
import { computed } from "vue";

// Token 和 Data 的接口，与 AppContent.vue 中的类型保持一致
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

const props = defineProps<{ data: OutputData | null | undefined }>();

// 计算属性，检查数据是否有效
const isValidData = computed(() => {
  return props.data && props.data.probs && props.data.probs.length > 0 && props.data.sampled;
});

// 对概率进行排序，方便查看（只在数据有效时执行）
const sortedProbs = computed(() => {
  if (!isValidData.value || !props.data) return [];
  return [...props.data.probs].sort((a, b) => b.probability - a.probability);
});

</script>

<template>
  <n-space v-if="isValidData && props.data" vertical>
    <n-text strong>采样到的 Token:
      <n-tag :type="'success'">
        {{ props.data.sampled.token }}: {{ (props.data.sampled.probability * 100).toFixed(2) }}% 🎯
      </n-tag>
    </n-text>
    <n-text strong>Top K 概率:</n-text>
    <n-space>
    <n-tag
      v-for="item in sortedProbs"
      :key="item.id"
      :type="item.id === props.data.sampled.id ? 'success' : 'default'"
      :bordered="false"
      round
    >
      {{ item.token }}: {{ (item.probability * 100).toFixed(2) }}%
      <span v-if="item.id === props.data.sampled.id" style="margin-left: 4px;">🎯</span>
    </n-tag>
    </n-space>
  </n-space>
  <div v-else>
    <n-text type="info">等待或无输出概率数据...</n-text>
  </div>
</template>

<style scoped>
.n-tag {
  margin: 2px;
}
</style>
