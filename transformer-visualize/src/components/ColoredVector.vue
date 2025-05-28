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
// value 参数在这里是单个向量元素的值，需要归一化后再用于颜色插值
function tohex(color: Array<number>, normalizedValue: number): string {
  // 确保 normalizedValue 在 0-1 之间
  const val = Math.max(0, Math.min(1, normalizedValue));
  return (
    "#" +
    color
      .map((x) =>
        Math.round(255 * (x * val + (1 - val))) // 线性插值: color*value + white*(1-value)
          .toString(16)
          .padStart(2, "0") // 确保是两位十六进制
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
  // 如果所有值都相同 (range is 0), 则所有段都将是中间颜色 (normalizedValue = 0.5) 或基于单一值
  const effectivelyConstant = range < 1e-6;

  return Array.from({ length: props.length }, (_, index) => {
    if (index >= props.values.length || index >= props.colors.length) {
      return "#CCCCCC"; // 越界或数据不足时的默认颜色
    }
    const value = props.values[index];
    // 归一化当前值到 0-1 范围
    const normalizedValue = effectivelyConstant ? 0.5 : (value - minVal.value) / range;
    return tohex(props.colors[index], normalizedValue);
  });
});
</script>

<template>
  <!-- 容器 div -->
  <div>
    <!-- 使用 flex布局来横向排列颜色段 -->
    <div style="display: flex; height: 25px; width: 100%;">
      <!-- 遍历生成每个颜色段 -->
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
/* 无特定样式，基础样式由 inline style 提供 */
</style>
