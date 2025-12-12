# LangGraph Agent Frontend

React 前端应用，连接到 LangGraph Agent 后端。

## 安装和运行

1. 进入 frontend 目录：
```bash
cd frontend
```

2. 安装依赖：
```bash
npm install
```

3. 启动开发服务器：
```bash
npm run dev
```

应用将在 `http://localhost:3000` 运行。

## 注意事项

- 确保后端 `server.py` 正在 `http://localhost:8000` 运行
- 前端会自动连接到 `/agent/stream` 端点进行流式响应
- 如果遇到 CORS 问题，后端已经配置了 CORS 中间件
- 前端使用 Vite 代理配置，开发时自动转发 `/agent` 请求到后端

## 构建生产版本

```bash
npm run build
```

构建的文件会在 `dist` 目录中。

