import { createApp } from 'vue'
import naive from 'naive-ui' // 引入 Naive UI
import './style.css'       // 全局样式
import App from './App.vue' // App.vue 将是 provider 的宿主

const app = createApp(App)
app.use(naive) // 全局安装 Naive UI，使得 NMessageProvider, NDialogProvider 等组件可用
app.mount('#app')