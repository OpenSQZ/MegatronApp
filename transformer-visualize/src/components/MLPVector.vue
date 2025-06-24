<script setup lang="ts">
import { ref, computed } from "vue";

const props = defineProps({
  length: Number,
  color: Array<Number>,
  values: Array<Float>,
});

const svgHeight = 10;
const svgWidth = computed(() => 2 * props.length);
function tohex(color: Array<number>, value: Float): string {
  // console.log(color, value);
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
const min = computed(() => Math.min(...props.values));
const max = computed(() => Math.max(...props.values));
const color = computed(() =>
  Array.from({ length: props.length }, (_, index) =>
    tohex(
      props.color,
      (props.values[index] - min.value) / (max.value - min.value),
    ),
  ),
);
</script>

<template>
  <svg
    :width="svgWidth"
    :height="svgHeight"
    :viewBox="'0 0 ' + svgWidth + ' ' + svgHeight"
  >
    <rect
      v-for="i in props.length"
      :key="i"
      :x="2 * (i - 1)"
      :y="0"
      width="2"
      :height="svgHeight"
      :fill="color[i - 1]"
    />
  </svg>
</template>

<style scoped></style>
