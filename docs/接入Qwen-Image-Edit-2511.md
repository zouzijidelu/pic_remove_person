# 服务器现状与接入 Qwen-Image-Edit-2511

本文说明当前 GPU 机上的全景人像消除服务怎么跑，以及若要在流程里增加 **Qwen-Image-Edit-2511** 应按什么方式接。不改现有 LaMa / PowerPaint 代码，只作为实施说明。

官方模型：[Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)（Apache-2.0）。

---

## 1. 当前服务器项目总结

### 1.1 部署了什么

| 角色 | 路径 | Git | 进程 |
|------|------|-----|------|
| 业务 Demo | `/root/pano-inpaint` | `pic_remove_person` | systemd `pano-lama`，端口 **7860** |
| PowerPaint 旁路 | `/root/iopaint-bench` | `iopaint-bench` | 由 Demo 拉起的独立 Python worker |

访问：内网 `http://10.2.143.130:7860`。机器：Ubuntu 20.04，RTX **3090 24GB**，驱动 CUDA 13。

GitHub HTTPS / SSH 均可。示例：

```bash
git remote set-url origin https://github.com/zouzijidelu/pic_remove_person.git
# 或：git@github.com:zouzijidelu/pic_remove_person.git
git pull
```

Hugging Face 走镜像：`HF_ENDPOINT=https://hf-mirror.com`。

### 1.2 处理链路（不变的部分）

```
上传全景（约 6720×3360）
    → SAM 点选 / 矩形框 / 笔刷  →  原尺寸 Mask（只存在服务端内存）
    → 选修复模型
    → 只在 Mask 区域填洞，结果贴回原分辨率
```

**SAM、笔刷、贴回原图不要为新模型重写。** 只换「填洞」后端。

### 1.3 现有两个填洞模型

| 模型 | 跑在哪 | 怎么修 | 特点 |
|------|--------|--------|------|
| **LaMa** | Demo `.venv`（Python 3.12）本进程 | 整图降到最长边 ≤2048 | 快、稳，默认 |
| **PowerPaint** | `iopaint-bench/.venv`（Python 3.11）旁路进程 | 按 Mask 裁 ROI 再修 | 结构有时更好，大面积人可能编出新人 |

IOPaint **只作为 PowerPaint 的加载器**，不要并进 Demo 的 `.venv`（Gradio / Pillow 会打架）。

启动时 `WARMUP=1` 会预加载 SAM、LaMa、PowerPaint worker，避免用户第一次点消除再等下载。

### 1.4 并发

单进程、单卡、Gradio `queue(max_size=2)`。适合 **1～2 人试用**。显存不是瓶颈（LaMa+SAM 约 3.4GB），GPU 推理是串行的。

### 1.5 生产方向（已达成共识）

- 前端以后用 **React**，Gradio 仅 Demo。
- 对外 **一个 API**；对内 LaMa / 扩散模型 **分进程**，不要把 IOPaint 整包装进主服务。
- 不要用 IOPaint 网页当产品。

---

## 2. Qwen-Image-Edit-2511 是什么、和现流程差在哪

Qwen-Image-Edit-2511 是 **指令编辑模型**（文生图式改图），不是 LaMa 那种「只看 Mask 填纹理」。

典型调用（diffusers）：

```python
from diffusers import QwenImageEditPlusPipeline
pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    torch_dtype=torch.bfloat16,
)
out = pipe(image=roi, prompt="Remove the person and fill with coherent background", ...).images[0]
```

接到本项目时必须：

1. **仍然用 SAM 的 Mask**（定位人在哪），不要让用户只打字、不标人。
2. **只把 ROI 送进 Qwen**（与 PowerPaint 相同：`crop_roi` + `paste_roi`），禁止喂 6720 整幅全景。
3. 用 prompt 表达「去掉人、补背景」，必要时把 Mask 作为约束（若所用 pipeline 支持 `mask` 则传入；不支持则 ROI + 强 prompt + 仅 Mask 区域贴回）。

贴回策略与 PowerPaint 一致：模型可以改整块 ROI，但 **只把 Mask 内像素写回原图**，避免墙、家具被改掉。

---

## 3. 3090 上怎么选精度（必读）

一块 24GB，且 **SAM + LaMa 已占约 3～4GB**。PowerPaint worker 若常驻还要再占约 4～8GB。

| 精度 | 大约显存 | 3090 能否与现服务共存 |
|------|----------|------------------------|
| BF16 原版 | ~40GB | **不能**，单卡装不下 |
| FP8 | ~20GB | 勉强；需 **卸掉或不要常驻 PowerPaint**，SAM+LaMa 可留 |
| 4bit / NF4 | ~16–20GB | 同样建议 **Qwen 单独 worker，按需加载** |

结论：

- **不要**把 Qwen 装进 Demo `.venv`，也不要和 IOPaint 混在一个环境。
- **不要**启动时 `WARMUP` 同时常驻：SAM + LaMa + PowerPaint + Qwen。
- 推荐：**第三个旁路进程**，第一次选「Qwen」再加载；或与 PowerPaint **互斥加载**（用谁加载谁，另一个释放显存）。

量化权重可参考社区 FP8 / NF4 仓库；上线前用同一批难例和 LaMa / PowerPaint 并排对比。

磁盘：权重大约 15～40GB，预留足够空间，下载走 `hf-mirror.com`。

---

## 4. 推荐接法（照抄 PowerPaint 旁路）

```
消除人像/app.py          SAM + 界面 + 贴回
消除人像/backends/lama.py           本进程
iopaint-bench/                      PowerPaint worker（已有）
qwen-edit-bench/                    【新建】Qwen worker + 独立 .venv
消除人像/backends/qwen_edit.py      【新建】像 powerpaint.py 一样调 worker
```

### 4.1 旁路目录建议

新建仓库或目录 `qwen-edit-bench/`（与 `pano-inpaint`、`iopaint-bench` 并列），例如 `/root/qwen-edit-bench`。

最少文件：

- `.venv/`（独立，Python 3.11 或 3.12 均可，以 diffusers 文档为准）
- `edit_worker.py`：stdin JSON 行协议，与 `inpaint_worker.py` 同类  
  请求：`image` / `mask` / `output` / `prompt` / `max_side`  
  就绪：`{"ok": true, "event": "ready"}`
- `setup.sh`、`configs/qwen_fast.json`

主服务通过环境变量指向它：

```bash
ENABLE_QWEN_EDIT=1
QWEN_EDIT_DEVICE=cuda
QWEN_EDIT_MAX_SIDE=768          # 与 PowerPaint 类似，宁小勿整图
QWEN_EDIT_MARGIN=256
QWEN_EDIT_BENCH=/root/qwen-edit-bench
QWEN_EDIT_PROMPT="Remove the person. Reconstruct the background only. Do not add new people."
```

### 4.2 Demo 里改哪些文件

只动「注册 + 一个 backend」，不要改 SAM。

1. `backends/qwen_edit.py`  
   - 复制 `backends/powerpaint.py` 结构：`crop_roi` → worker → `paste_roi`。  
   - `name = "Qwen-Edit"`。  
   - `warmup()` 里 `_ensure_worker()`（可选；3090 建议默认 **不** 随服务预热，避免和 PowerPaint 抢显存）。

2. `backends/__init__.py`  
   - 与 `ENABLE_POWERPAINT` 相同，增加 `ENABLE_QWEN_EDIT`，为真且 worker 的 python 存在时注册。

3. `app.py`  
   - `_model_controls()` 为 `Qwen-Edit` 增加滑条（ROI 最长边 512–1024）。  
   - 界面 Radio 会自动出现新选项（`list_backend_names()`）。

4. `server.env` / `server.env.example`  
   - 加上面一组变量。`WARMUP` 对 Qwen 建议 `0` 或单独 `WARMUP_QWEN=0`。

协议对齐现有 worker，React 以后只打一个 FastAPI，由网关按 `model=qwen` 转发到该进程。

### 4.3 Worker 内推理要点

- `QwenImageEditPlusPipeline.from_pretrained(...)`，`device=cuda`。
- 3090：优先 FP8 或 NF4；BF16 不要上这台机。
- `enable_model_cpu_offload()` 可降显存、会变慢，可作保底。
- `num_inference_steps`：完整 50 步太慢；先试 20，或官方 Lightning/少步配置。
- 输入图：ROI RGB；prompt 用环境变量，必要时把「白色为要消除」写进 prompt。
- 输出 resize 回 ROI 尺寸，主进程 `paste_roi` 只贴 Mask。
- 缓存目录：`HF_HOME=/root/qwen-edit-bench/.cache/huggingface`，不要写进 Demo 的 `.cache`。

### 4.4 不要做的

- 不要 `pip install` Qwen / 新版 diffusers 进 `/root/pano-inpaint/.venv`。
- 不要 `pip install` 进 `/root/iopaint-bench/.venv`（钉死 IOPaint 1.6.0 + 旧 diffusers）。
- 不要对 6720 全景直接 `pipe(image=full_pano)`。
- 不要与 PowerPaint 两个大模型同时 `warmup` 常驻（24GB 很容易 OOM）。

---

## 5. 建议实施顺序

1. **本机或 130 上单独脚本**  
   用一张 ROI（从现有 `crop_roi` 导出）+ 固定 prompt 跑通 `QwenImageEditPlusPipeline`，确认显存峰值（`nvidia-smi`）和质量。
2. **写成 `edit_worker.py`**  
   与 `inpaint_worker.py` 同样的 JSON 行协议，方便 Demo 和以后的 API 共用。
3. **`backends/qwen_edit.py` + 环境变量开关**  
   Demo 里第三个 Radio：LaMa / PowerPaint / Qwen-Edit。
4. **难例对比**  
   与 `iopaint-bench` 一样：同一 Mask，出并排图。看门口、大面积人、近景人；Qwen 也可能「编内容」，不能默认替换 LaMa。
5. **生产**  
   React → 一个 API → 按模型名转发 LaMa 本进程 / PowerPaint worker / Qwen worker。

---

## 6. 和现有仓库的关系

| 仓库 | 是否改 |
|------|--------|
| `pic_remove_person`（Demo） | 只加 `backends/qwen_edit.py` 和注册、env |
| `iopaint-bench` | **不必改**，继续只服务 PowerPaint |
| 新 `qwen-edit-bench` | 新建，独立 Git / 独立 `.venv` |

服务器 Git 可用 HTTPS 或 SSH。权重用 `hf-mirror.com`；大文件不要进 Git。

---

## 7. 一句话

当前服务是 **SAM 出 Mask + 可插拔填洞后端**。Qwen-Image-Edit-2511 应作为 **第三个旁路 worker（ROI + prompt + 贴回 Mask）**，在 3090 上用量化版、按需加载，不要塞进现有两个 venv，也不要当唯一默认模型。
