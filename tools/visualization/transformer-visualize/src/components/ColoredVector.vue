<script setup lang="ts">
import { computed } from "vue";

// 定义 Props
// length: 向量的维度
// colors: 一个颜色数组，每个元素是一个 [r, g, b] 数组 (0-1范围)，用于给向量的每个维度上色
// values: 实际的向量数值数组
const props = defineProps<{
  length: number;
  colors: Array<Array<number>>;
  values: Array<number>; // 使用 number[] 更精确
}>();

// 将0-1范围的颜色值和权重转换为十六进制颜色代码
function tohex(color: Array<number>, normalizedValue: number): string {
  const val = Math.max(0, Math.min(1, normalizedValue));
  return (
    "#" +
    color
      .map((x) =>
        Math.round(255 * (x * val + (1 - val)))
          .toString(16)
          .padStart(2, "0")
      )
      .join("")
  );
}

// 计算向量中所有元素的最小值和最大值，用于归一化
const minVal = computed(() => {
  if (!props.values || props.values.length === 0) return 0;
  return Math.min(...props.values);
});
const maxVal = computed(() => {
  if (!props.values || props.values.length === 0) return 1;
  return Math.max(...props.values);
});

// 计算每个维度的颜色
const segmentColors = computed(() => {
  if (!props.values || props.values.length === 0 || !props.colors || props.colors.length === 0) return [];
  const range = maxVal.value - minVal.value;
  const effectivelyConstant = range < 1e-6;

  return Array.from({ length: props.length }, (_, index) => {
    if (index >= props.values.length || index >= props.colors.length) {
      return "#CCCCCC";
    }
    const value = props.values[index];
    const normalizedValue = effectivelyConstant ? 0.5 : (value - minVal.value) / range;
    return tohex(props.colors[index], normalizedValue);
  });
});
</script>

<template>
  <div>
    <div style="display: flex; height: 25px; width: 100%;">
      <div
        v-for="i in props.length"
        :key="i"
        :style="{ flexGrow: 1, backgroundColor: segmentColors[i - 1], minWidth: '1px' }"
        :title="`Value: ${props.values[i-1]?.toFixed(4)}`"
      ></div>
    </div>
  </div>
</template>

<style scoped>
</style>
