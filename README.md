# Soul TalkBuddy Backend

AI社交对话教练后端服务

## 部署到 Render

1. 在 GitHub 创建新仓库 `soul-backend`
2. 推送此代码到仓库
3. 在 Render 创建 Web Service，连接该仓库
4. 配置环境变量：
   - `MODELSCOPE_TOKEN`: 你的 ModelScope Token
   - `MODEL_BASE_URL`: `https://api-inference.modelscope.cn/v1`
   - `QWEN_MODEL_NAME`: `Qwen/Qwen3-8B`
   - `PYTHON_VERSION`: `3.11.0`

## 本地运行

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```
