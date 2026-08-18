# 全景人像消除

基于 **SAM 点选粗 Mask + 笔刷精修 + 可切换修复模型** 的全景图人像消除 Demo。上传全景原图，点击人像自动生成 Mask（可笔刷微调），由 LaMa 或 PowerPaint 修复，结果保持原图分辨率。

## 功能特点

- **SAM 半自动**：在预览图上点击人像（可多点；支持负点排除背景），自动生成粗 Mask
- **笔刷精修**：SAM 结果写入画板，可用笔刷/橡皮继续修边
- **模型切换**：同一份 Mask，界面上切换 LaMa / PowerPaint，不必重新点选
- **保持原图分辨率**：推理可降采样或只跑 ROI，结果仅在 Mask 区域贴回原图
- **Mask 膨胀 / 推理边长可调**
- **样例图**：内置 `resource/` 全景样例

## 技术栈

| 依赖 | 用途 |
|------|------|
| Gradio | Web 交互界面 |
| segment-anything | SAM ViT-B 点选分割 |
| simple-lama-inpainting | LaMa 图像修复（本进程） |
| IOPaint 1.6.0（旁路 `iopaint-bench`） | PowerPaint 图像修复（独立进程） |
| PyTorch / OpenCV / Pillow | 推理与图像处理 |

## 项目结构

```
消除人像/
├── app.py              # Gradio Demo 主程序
├── backends/           # 修复模型插件（LaMa / PowerPaint）
├── setup.sh            # 创建虚拟环境并安装依赖
├── start.sh            # 启动服务
├── requirements.txt    # Python 依赖列表（不含 IOPaint）
├── resource/           # 样例全景图
├── output/             # 消除结果：`原图名_模型_时间.jpg`
└── .cache/             # 模型缓存（LaMa / SAM，已 gitignore）
```

## 环境要求

- **Python 3.12**（3.13 与部分依赖不兼容）
- macOS / Linux；LaMa 默认 **CPU**；PowerPaint 建议 **MPS / CUDA**
- 使用 PowerPaint 前，旁路目录 `../iopaint-bench` 需已 `./setup.sh`（可先跑过一次 `./run_compare.sh powerpaint` 把权重下好）

## 快速开始

```bash
./setup.sh
./start.sh
```

浏览器打开：**http://127.0.0.1:7860**

### 使用步骤

1. 上传全景原图（或点选样例）
2. 在「标注区」点人像 → 自动出红色粗 Mask（可继续加点或切换负点 / 矩形框）
3. 可选：在画板笔刷精修
4. 选修复模型（LaMa 或 PowerPaint），调节「Mask 膨胀」「推理最长边」后点「开始消除」
5. 结果：`output/原图名_模型_时间.jpg`（Mask 为同名 `_mask.png`）

首次运行会下载：
- LaMa 权重（约 196MB）
- SAM ViT-B 权重（约 375MB）→ `.cache/sam/`
- PowerPaint 首次约 4GB（落到 `iopaint-bench/.cache/`，不进本目录 `.venv`）

## 修复模型怎么切

界面左侧「修复模型」单选即可。SAM / 笔刷 / 贴回原图不变，只换「填洞」后端。

| 模型 | 跑在哪 | 怎么修 | 适合 |
|------|--------|--------|------|
| **LaMa** | 本 Demo `.venv` | 整图降采样 | 默认、快、稳 |
| **PowerPaint** | 旁路 `iopaint-bench/.venv` 常驻进程 | 按 Mask 裁 ROI 再修 | 门口/结构更清晰；大面积人可能编出新人 |

以后加 SDXL 等同理：在 `backends/` 加一个类，注册进 `BACKENDS`。不要把 IOPaint 装进本目录 `.venv`。

PowerPaint 首次点击会加载模型（约 1～2 分钟），之后进程常驻，再点只需几十秒。

## 配置说明

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `PORT` | `7860` | 服务端口 |
| `LAMA_DEVICE` | `cpu` | LaMa 设备：`cpu` / `cuda` / `mps` |
| `SAM_DEVICE` | 同 `LAMA_DEVICE` | SAM 设备 |
| `POWERPAINT_DEVICE` | `mps` | PowerPaint 设备 |
| `MAX_INFER_SIDE` | `1536` | LaMa 推理最长边 |
| `POWERPAINT_MAX_SIDE` | `768` | PowerPaint ROI 最长边 |
| `POWERPAINT_MARGIN` | `256` | ROI 相对 Mask 外扩像素 |
| `INPAINT_MODEL` | `LaMa` | 启动时默认模型 |
| `IOPAINT_BENCH` | `../iopaint-bench` | 旁路评测台路径 |

```bash
LAMA_DEVICE=cpu SAM_DEVICE=mps ./start.sh
POWERPAINT_DEVICE=mps ./start.sh
PORT=7861 MAX_INFER_SIDE=2048 ./start.sh
```

## 处理流程

1. SAM：点击坐标映射到原图 → 在最长边≤1024 的图上分割 → Mask 放大回原图，并写入画板
2. （可选）笔刷精修画板 Mask
3. 按所选模型修复：
   - LaMa：Mask 膨胀后按「推理最长边」降采样，整图修复后羽化贴回
   - PowerPaint：按 Mask 外扩裁 ROI → 旁路进程推理 → 仅 Mask 区域贴回原图
4. 输出原分辨率结果

## 注意事项

- Mac 上 LaMa 建议最长边约 **1536**，过大易被系统因内存杀掉
- PowerPaint 不要喂整幅 6720 全景，界面会自动裁 ROI；16GB 机器建议最长边 **768**
- TorchScript LaMa 在 MPS 上可能块状伪影，默认 CPU；SAM 可尝试 `SAM_DEVICE=mps`
- PowerPaint 在部分门口更清晰，但大面积人、近景人可能编出新人或碎块，不能无脑替换 LaMa
- `.venv` / `.cache` / `output` 不纳入 Git
