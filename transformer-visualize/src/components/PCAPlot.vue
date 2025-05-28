<script setup lang="ts">
import { ref, computed, onMounted, watch, onBeforeUnmount, nextTick } from "vue";
import {
  Chart,
  ScatterController,
  LinearScale,
  PointElement,
  Tooltip,
  Legend,
  Title, // 引入 Title 用于图表标题
} from "chart.js";

// 注册 Chart.js 组件
Chart.register(ScatterController, LinearScale, PointElement, Tooltip, Legend, Title);

// 定义 Props
// values: PCA降维后的数据点数组，结构为 [batch_index][token_index][pca_coordinate_pair]
// 即 Array<Array<[x: number, y: number]>>. 这是针对特定层的PCA数据。
// layerId: 当前图表对应的层ID，用于图表标题。
const props = defineProps<{
  values: Array<Array<[number, number]>>; // 每个批次一个数据点数组
  layerId: number; // 当前图表对应的层ID
}>();

const chartRef = ref<HTMLCanvasElement | null>(null); // Vue ref 引用 canvas 元素
let chartInstance: Chart | null = null; // Chart.js 实例

// 为不同批次生成不同颜色
function getColorForBatch(batchIndex: number, alpha: number = 1): string {
  const colors = [
    `rgba(75, 192, 192, ${alpha})`,   // 蓝绿色
    `rgba(255, 99, 132, ${alpha})`,   // 粉红色
    `rgba(54, 162, 235, ${alpha})`,   // 蓝色
    `rgba(255, 206, 86, ${alpha})`,   // 黄色
    `rgba(153, 102, 255, ${alpha})`,  // 紫色
    `rgba(255, 159, 64, ${alpha})`,   // 橙色
    `rgba(100, 100, 100, ${alpha})`,  // 灰色 (备用)
    `rgba(200, 100, 50, ${alpha})`,   // 棕色 (备用)
  ];
  return colors[batchIndex % colors.length];
}

// 将 props.values 转换为 Chart.js datasets 格式
// 每个 batch 对应一个 dataset
const chartDatasets = computed(() => {
  if (!props.values || props.values.length === 0) return [];

  return props.values.map((batchData, batchIndex) => {
    // 检查 batchData 是否存在且不为空
    if (!batchData || batchData.length === 0) {
      return {
        label: `Batch ${batchIndex + 1} (No data)`,
        data: [],
        backgroundColor: getColorForBatch(batchIndex, 0.1),
        borderColor: getColorForBatch(batchIndex, 0.3),
        pointRadius: 5,
        pointHoverRadius: 7,
      };
    }
    return {
      label: `Batch ${batchIndex + 1}`, // 数据集标签 (对应一个批次)
      data: batchData.map(point => ({ x: point[0], y: point[1] })), // 数据点
      backgroundColor: getColorForBatch(batchIndex, 0.7), // 数据点填充颜色
      borderColor: getColorForBatch(batchIndex, 1),     // 数据点边框颜色
      pointRadius: 5, // 数据点半径
      pointHoverRadius: 7, // 悬停时数据点半径
    };
  }).filter(dataset => dataset.data.length > 0); // 过滤掉完全没有数据的批次，避免图表库出问题
});


// 创建或更新图表的函数
const createOrUpdateChart = async () => {
  if (!chartRef.value) return; // 确保canvas元素存在
  await nextTick(); // 等待DOM更新完成

  const ctx = chartRef.value.getContext("2d");
  if (!ctx) return;

  // 销毁旧实例（如果存在）
  if (chartInstance) {
    chartInstance.destroy();
    chartInstance = null;
  }

  // 仅当有有效数据时才创建图表
  if (chartDatasets.value.length === 0) {
    console.warn(`PCA Plot (Layer ${props.layerId}): No valid data to plot.`);
    return;
  }

  // 找出所有数据点中的x和y的最小值和最大值，用于动态调整坐标轴范围
  let allX: number[] = [];
  let allY: number[] = [];
  // 从 chartDatasets 中提取数据，因为 props.values 可能包含空batch
  chartDatasets.value.forEach(dataset => {
    dataset.data.forEach(point => {
      allX.push(point.x);
      allY.push(point.y);
    });
  });

  // 如果没有有效数据点，使用默认范围
  const minX = allX.length > 0 ? Math.min(...allX) : -10;
  const maxX = allX.length > 0 ? Math.max(...allX) : 10;
  const minY = allY.length > 0 ? Math.min(...allY) : -10;
  const maxY = allY.length > 0 ? Math.max(...allY) : 10;

  // 添加一些padding以避免点在边缘
  const xPadding = (maxX - minX) * 0.1 || 1; // 如果maxX-minX为0，则padding为1
  const yPadding = (maxY - minY) * 0.1 || 1; // 如果maxY-minY为0，则padding为1

  chartInstance = new Chart(ctx, {
    type: "scatter", // 图表类型为散点图
    data: {
      datasets: chartDatasets.value, // 使用计算属性获取数据集
    },
    options: {
      responsive: true, // 响应式布局
      maintainAspectRatio: false, // 不保持宽高比，允许自定义画布大小
      scales: {
        x: {
          type: "linear",
          position: "bottom",
          title: { display: true, text: "PCA Dimension 1" },
          min: minX - xPadding,
          max: maxX + xPadding,
        },
        y: {
          type: "linear",
          title: { display: true, text: "PCA Dimension 2" },
          min: minY - yPadding,
          max: maxY + yPadding,
        },
      },
      plugins: {
        legend: {
          display: true, // 显示图例
          position: "top", // 图例位置
        },
        tooltip: {
          enabled: true, // 启用工具提示
          callbacks: { // 自定义工具提示内容
            label: function (context) {
              let label = context.dataset.label || "";
              if (label) {
                label += ": ";
              }
              if (context.parsed.x !== null && context.parsed.y !== null) {
                label += `(PCA1: ${context.parsed.x.toFixed(2)}, PCA2: ${context.parsed.y.toFixed(2)})`;
              }
              // 可以在这里添加 token 信息，如果能从 context 或 props 传递进来
              // 例如: label += `\nToken: ${getTokenForPoint(context.datasetIndex, context.dataIndex)}`;
              return label;
            },
          },
        },
        title: { // 图表标题
            display: true,
            text: `Layer ${props.layerId} MLP2 Output PCA`,
            font: { size: 16 }
        }
      },
    },
  });
};

// 组件挂载时，如果 props.values 有数据，则创建图表
onMounted(() => {
  // 检查 props.values 是否有实际数据点
  if (props.values && props.values.length > 0 && props.values.some(batch => batch && batch.length > 0)) {
    createOrUpdateChart();
  }
});

// 监听 props.values 的变化，以重新创建或更新图表
watch(() => props.values, (newValues) => {
  // 检查新值是否有实际数据点
  if (newValues && newValues.length > 0 && newValues.some(batch => batch && batch.length > 0)) {
    createOrUpdateChart();
  } else if (chartInstance) {
    // 如果新数据为空，但存在旧图表，则销毁图表实例
    chartInstance.destroy();
    chartInstance = null;
  }
}, { deep: true }); // 使用 deep watch 来监测数组内部复杂结构的变化

// 组件卸载前，销毁图表实例以防止内存泄漏
onBeforeUnmount(() => {
  if (chartInstance) {
    chartInstance.destroy();
    chartInstance = null;
  }
});
</script>

<template>
  <!-- Naive UI 卡片包裹图表 -->
  <!-- 标题已由 Chart.js 内部处理，n-card 的 title 可以移除或简化 -->
  <n-card content-style="padding: 0; height: 100%;" style="height: 800px; width: 800px;"> <!-- 给予一个固定高度 -->
    <!-- Canvas 容器 -->
    <div style="position: relative; width: 100%; height: 100%;">
      <canvas ref="chartRef"></canvas>
    </div>
  </n-card>
</template>

<style scoped>
/* 确保 canvas 元素能正确填充其容器 */
canvas {
  display: block; /* 消除 canvas 下方的额外空白 */
  width: 100% !important;  /* !important 确保覆盖 Chart.js 可能的内联样式 */
  height: 100% !important; /* !important 确保覆盖 Chart.js 可能的内联样式 */
}
</style>
