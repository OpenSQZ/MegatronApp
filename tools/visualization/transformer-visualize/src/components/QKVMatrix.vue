<script setup lang="ts">
import { ref, computed } from "vue";

const props = defineProps({
  rows: Number,
  cols: Number,
  colors: Array<Array<Number>>,
  values: Array<Number>,
});
const svgWidth = computed(() => 10 * props.cols);
const svgHeight = computed(() => 10 * props.rows);
function tohex(color: Array<number>, value: number): string {
  return (
    "#" +
    color
      .map((x) =>
        Math.round(255 * (x * value + (1 - value)))
          .toString(16)
          .padStart(2, "0"),
      )
      .join("")
  );
}
const color = computed(() =>
  Array.from({ length: props.colors.length }, (_, index) =>
    tohex(props.colors[index], props.values[index]),
  ),
);
</script>

<template>
  <svg
    :width="svgWidth"
    :height="svgHeight"
    :viewBox="'0 0 ' + svgWidth + ' ' + svgHeight"
  >
    <g v-for="i in props.rows" :key="i">
      <rect
        v-for="j in props.cols"
        :key="j"
        :y="10 * (i - 1)"
        :x="10 * (j - 1)"
        width="10"
        height="10"
        :fill="color[(i - 1) * cols + (j - 1)]"
      />
    </g>
  </svg>
</template>

<style scoped></style>
