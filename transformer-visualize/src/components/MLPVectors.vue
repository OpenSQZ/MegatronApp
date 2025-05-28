<script setup lang="ts">
import { ref, computed } from "vue";
import MLPVector from "./MLPVector.vue";
const props = defineProps({
  color: Array<Number>,
  values: Array<Float>,
});
const n_tokens = computed(() => props.values.length / 64);
</script>

<template>
  <n-flex>
    <MLPVector
      v-for="i in n_tokens"
      :key="i"
      :color="props.color"
      :values="props.values.slice((i - 1) * 64, i * 64)"
      :length="64"
    />
  </n-flex>
</template>

<style scoped></style>
