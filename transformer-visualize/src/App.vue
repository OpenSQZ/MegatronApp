<script setup lang="ts">
import AppContent from './AppContent.vue';
</script>

<template>
  <n-config-provider>
    <n-message-provider>
      <n-dialog-provider>
        <div id="app-container">
          <AppContent />
        </div>
      </n-dialog-provider>
    </n-message-provider>
  </n-config-provider>
</template>

<style>

#app-container {
  max-width: 1600px;
  width: 100%;
  margin: 0 auto;
  padding: 1rem;
  box-sizing: border-box;
  min-height: 100vh;
}
</style>
