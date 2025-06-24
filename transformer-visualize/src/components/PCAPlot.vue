<script setup lang="ts">
import { ref, computed, onMounted, watch, onBeforeUnmount, nextTick } from "vue";
import {
  Chart,
  ScatterController,
  LinearScale,
  PointElement,
  Tooltip,
  Legend,
  Title,
} from "chart.js";

Chart.register(ScatterController, LinearScale, PointElement, Tooltip, Legend, Title);

// 定义 Props
// values: PCA降维后的数据点数组，结构为 [batch_index][token_index][pca_coordinate_pair]
// 即 Array<Array<[x: number, y: number]>>. 这是针对特定层的PCA数据。
// layerId: 当前图表对应的层ID，用于图表标题。
const props = defineProps<{
  values: Array<Array<[number, number]>>;
  layerId: number;
  tokens: Array<Array<{ id: number; token: string }>>;
}>();

const chartRef = ref<HTMLCanvasElement | null>(null);
let chartInstance: Chart | null = null;

// 为不同批次生成不同颜色
function getColorForBatch(batchIndex: number, alpha: number = 1): string {
  const colors = [
    `rgba(75, 192, 192, ${alpha})`,   // 蓝绿色
    `rgba(255, 99, 132, ${alpha})`,   // 粉红色
    `rgba(54, 162, 235, ${alpha})`,   // 蓝色
    `rgba(255, 206, 86, ${alpha})`,   // 黄色
    `rgba(153, 102, 255, ${alpha})`,  // 紫色
    `rgba(255, 159, 64, ${alpha})`,   // 橙色
    `rgba(100, 100, 100, ${alpha})`,  // 灰色
    `rgba(200, 100, 50, ${alpha})`,   // 棕色
  ];
  return colors[batchIndex % colors.length];
}

// 将 props.values 转换为 Chart.js datasets 格式
// 每个 batch 对应一个 dataset
const chartDatasets = computed(() => {
  if (!props.values || props.values.length === 0) return [];

  return props.values.map((batchData, batchIndex) => {
    const tokenBatch = props.tokens[batchIndex];
    if (!batchData || batchData.length === 0 || !tokenBatch) {
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
      label: `Batch ${batchIndex + 1}`,
      data: batchData.map((point, pointIndex) => {
        const tokenString = tokenBatch[pointIndex]?.token ?? `[Token ${pointIndex + 1}]`;
        return {
          x: point[0],
          y: point[1],
          token: tokenString,
        };
      }),
      backgroundColor: getColorForBatch(batchIndex, 0.7),
      borderColor: getColorForBatch(batchIndex, 1),
      pointRadius: 5,
      pointHoverRadius: 7,
    };
  }).filter(dataset => dataset.data.length > 0);
});


// 创建或更新图表的函数
const createOrUpdateChart = async () => {
  if (!chartRef.value) return;
  await nextTick();

  const ctx = chartRef.value.getContext("2d");
  if (!ctx) return;

  if (chartInstance) {
    chartInstance.destroy();
    chartInstance = null;
  }

  if (chartDatasets.value.length === 0) {
    console.warn(`PCA Plot (Layer ${props.layerId}): No valid data to plot.`);
    return;
  }

  let allX: number[] = [];
  let allY: number[] = [];
  chartDatasets.value.forEach(dataset => {
    dataset.data.forEach(point => {
      allX.push(point.x);
      allY.push(point.y);
    });
  });

  const minX = allX.length > 0 ? Math.min(...allX) : -10;
  const maxX = allX.length > 0 ? Math.max(...allX) : 10;
  const minY = allY.length > 0 ? Math.min(...allY) : -10;
  const maxY = allY.length > 0 ? Math.max(...allY) : 10;

  const xPadding = (maxX - minX) * 0.1 || 1;
  const yPadding = (maxY - minY) * 0.1 || 1;

  chartInstance = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: chartDatasets.value,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
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
          display: true,
          position: "top",
        },
        tooltip: {
          enabled: true,
          callbacks: {
            label: function (context) {
              let label = context.dataset.label || "";
              if (label) {
                label += ": ";
              }
              if (context.parsed.x !== null && context.parsed.y !== null) {
                label += `(PCA1: ${context.parsed.x.toFixed(2)}, PCA2: ${context.parsed.y.toFixed(2)})`;
              }
              label += `\nToken: ${(context.raw as { token: string })?.token}`;
              return label;
            },
          },
        },
        title: {
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
  if (props.values && props.values.length > 0 && props.values.some(batch => batch && batch.length > 0)) {
    createOrUpdateChart();
  }
});

// 监听 props.values 的变化，以重新创建或更新图表
watch(() => props.values, (newValues) => {
  if (newValues && newValues.length > 0 && newValues.some(batch => batch && batch.length > 0)) {
    createOrUpdateChart();
  } else if (chartInstance) {
    chartInstance.destroy();
    chartInstance = null;
  }
}, { deep: true });

// 组件卸载前，销毁图表实例以防止内存泄漏
onBeforeUnmount(() => {
  if (chartInstance) {
    chartInstance.destroy();
    chartInstance = null;
  }
});
</script>

<template>
  <n-card content-style="padding: 0; height: 100%;" style="height: 800px; width: 800px;">
    <div style="position: relative; width: 100%; height: 100%;">
      <canvas ref="chartRef"></canvas>
    </div>
  </n-card>
</template>

<style scoped>
canvas {
  display: block;
  width: 100% !important;
  height: 100% !important;
}
</style>
