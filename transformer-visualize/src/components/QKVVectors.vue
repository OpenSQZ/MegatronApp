<script setup lang="ts">
import { ref, computed } from "vue";
import QKVVector from "./QKVVector.vue";
const props = defineProps({
  colors: Array<Array<Number>>,
  values: Array<Float>,
});
const n_tokens = computed(() => props.values.length / 96);
</script>

<template>
  <n-flex>
    <QKVVector
      v-for="i in n_tokens"
      :key="i"
      :colors="props.colors"
      :values="props.values.slice((i - 1) * 96, i * 96)"
      :length="96"
    />
  </n-flex>
</template>

<style scoped></style>
