<script setup lang="ts">
import { computed } from "vue";

// 定义Props
// size: 注意力矩阵的大小 (序列长度)
// color: 用于着色的基础颜色 [r, g, b] (0-1范围)
// values: 注意力权重矩阵 (二维数组, 代表一个batch中的一个layer的注意力)
// tokens: 当前序列的token对象数组 {id: number, token: string}
// layer_id: 当前层的ID (用于标题)
const props = defineProps<{
  size: number;
  color: Array<number>; // [r, g, b]
  values: Array<Array<number>>;
  tokens: Array<{id: number, token: string}>;
  layer_id: number;
}>();

// 将0-1范围的颜色值和权重转换为十六进制颜色代码
function tohex(baseColor: Array<number>, value: number): string {
  // 确保 value 在 0-1 之间，以防传入非法值
  const normalizedValue = Math.max(0, Math.min(1, value));
  return (
    "#" +
    baseColor
      .map((cVal) => // cVal 是基础颜色分量 (r,g,或b)
        Math.round(255 * (cVal * normalizedValue + (1 - normalizedValue))) // 线性插值: color*alpha + white*(1-alpha)
          .toString(16)
          .padStart(2, "0") // 确保是两位十六进制
      )
      .join("")
  );
}

// 计算每个单元格的颜色
// 这是一个二维数组，存储每个注意力单元格的十六进制颜色值
const cellColors = computed(() => {
  // 确保 props.values 和 props.color 有效，并且 props.values 非空
  if (!props.values || !props.color || props.values.length === 0 || props.values[0].length === 0) return [];
  // 遍历 props.values (注意力权重矩阵)
  return props.values.map((row) =>
    row.map((weight) =>
      tohex(props.color, weight) // 对每个权重应用 tohex 函数计算颜色
    )
  );
});

// 安全地获取 token 字符串的函数
// 如果 props.tokens 中没有对应索引的 token，则返回一个占位符
function getTokenString(index: number): string {
  return props.tokens?.[index]?.token ?? `[Token ${index + 1}]`;
}

// 检查数据是否有效，以便在模板中条件渲染
const isDataValid = computed(() => {
  return props.values &&
         props.values.length === props.size &&
         props.values.every(row => row && row.length === props.size) &&
         props.tokens &&
         props.tokens.length === props.size &&
         cellColors.value.length > 0;
});

</script>

<template>
  <!-- 使用 Naive UI 卡片包裹注意力矩阵，标题显示层ID -->
  <n-card :title="'Layer ' + props.layer_id + ' Attention Matrix'">
    <!-- 主容器，保持宽高比为1:1 -->
    <div v-if="isDataValid" style="width: 100%; height: auto; aspect-ratio: 1; margin: 0 auto;">
      <!-- Naive UI 网格布局，列数等于 props.size (序列长度) -->
      <!-- 外层循环创建列 (对应注意力矩阵的 key tokens, 即 j_idx) -->
      <div :style="{ display: 'grid', 'grid-template-columns': `repeat(${props.size}, 1fr)`, border: '1px solid #eee'}">
        <!-- 内层循环创建行 (对应注意力矩阵的 query tokens, 即 i_idx) -->
        <template v-for="i_idx in props.size" :key="`row-${i_idx}`">
          <!-- 单元格循环 (对应注意力矩阵的 key tokens, 即 j_idx) -->
          <div v-for="j_idx in props.size"
               :key="`cell-${i_idx}-${j_idx}`"
               style="aspect-ratio: 1; display: flex; align-items: center; justify-content: center; border-right: 1px solid #f0f0f0; border-bottom: 1px solid #f0f0f0;"
               :style="j_idx === props.size ? 'border-right: none;' : '' + i_idx === props.size ? 'border-bottom: none;' : ''"
          >
            <!-- 悬浮提示框，显示详细信息 -->
            <n-popover trigger="hover">
              <template #trigger>
                <!-- 实际的颜色块，背景色由 cellColors 计算得到 -->
                <div
                  :style="{ backgroundColor: cellColors[i_idx - 1]?.[j_idx - 1] || '#FFFFFF' }"
                  style="
                    width: 90%;
                    height: 90%;
                    border-radius: 3px; /* 方角或微圆角 */
                  "
                ></div>
              </template>
              <!-- 悬浮提示内容：查询Token, 键Token, 和对应的注意力概率 -->
              <span>
                Query: {{ getTokenString(i_idx - 1) }} (idx {{ i_idx - 1 }})<br/>
                Key: {{ getTokenString(j_idx - 1) }} (idx {{ j_idx - 1 }})<br/>
                Attention:
                {{ ((props.values[i_idx - 1]?.[j_idx - 1] ?? 0) * 100).toFixed(2) }}%
              </span>
            </n-popover>
          </div>
        </template>
      </div>
    </div>
    <!-- 如果数据无效，显示提示信息 -->
    <n-empty v-else :description="`Layer ${props.layer_id} attention data not available or mismatched dimensions.`" />
  </n-card>
</template>

<style scoped>
/* 可以根据需要添加更多样式调整网格线等 */
/* 使用 grid 代替 n-grid 以获得更细致的控制 */
</style>
