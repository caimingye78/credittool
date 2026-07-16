# 信用卡效益最大化助手 (CreditTool)

支持手机银行截图上传 + 视觉大模型自动解析优惠活动，双推荐模式，帮助最大化国内信用卡收益。

**仓库地址**：https://github.com/caimingye78/credittool

## 主要功能
- 卡片管理（支持几十张卡）
- **截图上传解析**（推荐）
- 文字 + 大模型结构化
- 详细推荐 + 弹性策略推荐
- 本地 SQLite 数据库
- 一键导出 Excel
- 仪表盘

## 在线使用（部署后）
部署成功后地址会是：  
`https://credittool-xxxxx.streamlit.app`

## 部署到 Streamlit Cloud（推荐）

1. 本仓库已经准备好
2. 打开 https://share.streamlit.io 并用 GitHub 登录
3. New app → 选择 `caimingye78/credittool` → Main file path 填 `app.py`
4. 点击 Deploy
5. 部署完成后，进入 App Settings → Secrets，添加以下内容：

```toml
OPENAI_API_KEY = "你的API密钥"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-4o"