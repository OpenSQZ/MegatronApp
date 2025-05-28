<script setup lang="ts">
// App.vue 现在很简单，只负责提供 message context 和全局布局
import AppContent from './AppContent.vue'; // 引入新的内容组件
</script>

<template>
  <!-- 使用 NConfigProvider 来全局设置主题 -->
  <n-config-provider>
    <n-message-provider>
      <n-dialog-provider>
        <!-- #app-container 是主要的布局容器 -->
        <div id="app-container">
          <AppContent />
        </div>
      </n-dialog-provider>
    </n-message-provider>
  </n-config-provider>
</template>

<style>
/* 全局样式，或者可以移到 style.css */

#app-container {
  max-width: 1600px;
  width: 100%;
  margin: 0 auto; /* 水平居中 */
  padding: 1rem; /* 容器内边距 */
  box-sizing: border-box; /* 确保 padding 不会增加总宽度 */
  min-height: 100vh; /* 至少占据整个视口高度，确保内容不足时背景也能撑满 */
  /* background-color: #ffffff; /* 给容器一个明确的白色背景，以覆盖可能的 body 背景 */
  /* overflow-y: auto;  /* 这个通常由浏览器自动处理，除非内部有特殊布局 */
}
</style>
