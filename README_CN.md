# ComfyUI-See-through

一个将 [See-through](https://github.com/shitagaki-lab/see-through) 集成到 ComfyUI 的插件 — 从单张动漫插画自动分解出可操作的 2.5D 分层模型，带深度排序，可直接用于 Live2D 工作流。

[English](README.md)

论文：[arxiv:2602.03749](https://arxiv.org/abs/2602.03749)（已被 ACM SIGGRAPH 2026 条件接收）

## 功能特性

- **单图分层分解** — 输入一张动漫角色图片，输出多达 24 个语义透明图层（头发、面部、眼睛、服装、饰品等）
- **深度估计** — 通过微调的 Marigold 为每层自动生成深度图，确定正确的绘制顺序
- **智能拆分** — 眼睛、耳朵、手套自动拆分为左/右；头发通过深度聚类拆分为前/后
- **PSD 导出** — 直接在浏览器下载分层 PSD 文件（前端 ag-psd 实现，无需 Python 依赖）
- **深度 PSD** — 可单独导出深度 PSD，用于 3D/视差工作流
- **预览输出** — 合成重建预览作为标准 ComfyUI IMAGE 输出
- **HuggingFace 自动下载** — 首次使用时自动从 HuggingFace 下载模型
- **显存优化** — 标签嵌入缓存、文本编码器卸载、分组卸载、可配置深度分辨率，适配低显存显卡

## 节点说明

| 节点 | 说明 |
|------|------|
| **SeeThrough Load LayerDiff Model** | 加载 LayerDiff SDXL 管线（图层生成） |
| **SeeThrough Load Depth Model** | 加载 Marigold 深度估计管线 |
| **SeeThrough Decompose** | 完整管线：LayerDiff + Marigold 深度 + 后处理 |
| **SeeThrough Save PSD** | 保存图层 PNG + 元数据；通过浏览器按钮下载 PSD |

## 安装

将此仓库克隆到 ComfyUI 的 `custom_nodes` 目录：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jtydhr88/ComfyUI-See-through.git
```

安装依赖：

```bash
cd ComfyUI-See-through
pip install -r requirements.txt
```

重启 ComfyUI，**SeeThrough** 节点将出现在 `SeeThrough` 分类下。

### 依赖

仅需在 ComfyUI 基础之上额外安装 4 个 Python 包：

- `diffusers` — Hugging Face 扩散管线
- `accelerate` — 模型加载加速
- `opencv-python` — 图像处理
- `scikit-learn` — 基于深度的图层拆分（KMeans 聚类）

### 模型

首次使用时自动从 HuggingFace 下载：

| 模型 | HuggingFace 仓库 | 用途 |
|------|-------------------|------|
| LayerDiff 3D | `layerdifforg/seethroughv0.0.2_layerdiff3d` | 基于 SDXL 的透明图层生成 |
| Marigold Depth | `24yearsold/seethroughv0.0.1_marigold` | 动漫微调的单目深度估计 |

也可手动下载模型放到 `ComfyUI/models/SeeThrough/` 目录。

## 使用方法

### 基本工作流

1. 添加 **SeeThrough Load LayerDiff Model** 和 **SeeThrough Load Depth Model** 节点
2. 添加 **SeeThrough Decompose** 节点 — 连接两个模型和一个 **Load Image** 节点
3. 添加 **SeeThrough Save PSD** — 连接 `parts` 输出
4. 添加 **Preview Image** — 连接 `preview` 输出
5. 运行工作流
6. 点击 Save PSD 节点上的 **Download PSD** 按钮生成并下载 PSD 文件

### 示例工作流

`workflows/` 目录中提供了预设工作流：

| 工作流 | 分辨率 | 步数 | 左右拆分 | 说明 |
|--------|--------|------|----------|------|
| `seethrough-basic.json` | 1280 | 30 | 是 | 标准质量，推荐使用 |
| `seethrough-highres.json` | 2048 | 50 | 是 | 高质量 + 保存预览图 |
| `seethrough-fast.json` | 1024 | 15 | 否 | 快速预览，质量较低 |

将 `.json` 文件拖入 ComfyUI 即可加载工作流。

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `seed` | 42 | 随机种子，用于结果复现 |
| `resolution` | 1280 | 处理分辨率（图像会居中填充为正方形） |
| `num_inference_steps` | 30 | 扩散去噪步数（越多质量越好，速度越慢） |
| `tblr_split` | true | 是否将对称部位（眼睛、耳朵、手套）拆分为左/右 |
| `cache_tag_embeds` | true | 预计算并缓存标签嵌入，然后卸载文本编码器以节省显存 |
| `group_offload` | false | 启用分组卸载，大幅降低峰值显存（实际占用约 0.2GB，预留约 7GB），但速度**慢 2–3 倍**。需要 `diffusers>=0.37.0` |
| `resolution_depth` | -1 | 深度推理分辨率。-1 表示与图层相同。设低（如 720）可节省显存并加速深度估计 |

### 显存优化指南

**大多数用户（12GB+ 显存）：** 默认设置即可。`cache_tag_embeds=true` 已默认开启，节省约 2GB 显存且不影响速度，无需其他更改。

**低显存用户（8–12 GB）：** 按以下顺序尝试，对速度影响从小到大：

1. **`cache_tag_embeds=true`**（默认已开启）— 缓存文本嵌入并卸载文本编码器，节省约 2GB 显存，无速度损失
2. **`resolution_depth=720`** — 以较低分辨率进行深度推理，然后放大回原始分辨率，质量损失很小
3. **降低 `resolution`** — 例如使用 1024 而非 1280，同时减少显存占用和计算量
4. **`group_offload=true`** — 最后手段。按需将单个模型块移入/移出 GPU，峰值实际占用降至约 0.2GB，但由于频繁 CPU↔GPU 传输**速度慢 2–3 倍**。需要 `pip install diffusers>=0.37.0`

#### 实测数据（RTX 5090，steps=30，`cache_tag_embeds=true`）

**`group_offload` 开启 vs 关闭（resolution=1280）：**

| 阶段 | group_offload=OFF | group_offload=ON |
|------|-------------------|------------------|
| UNet+VAE 加载后 | 7.94 GB | 0.21 GB |
| LayerDiff 峰值（实际占用 / 预留） | 7.95 GB / 13.69 GB | 0.21 GB / 7.31 GB |
| Marigold 峰值 | 2.49 GB | 0.07 GB |
| **总耗时** | **138 秒** | **385 秒（慢 2.8 倍）** |

**不同分辨率对比（group_offload=OFF）：**

| 分辨率 | LayerDiff 峰值（实际占用 / 预留） | Marigold 峰值 | 总耗时 | 建议最低显存 |
|--------|----------------------------------|---------------|--------|-------------|
| 1280 | 7.95 GB / 13.69 GB | 2.49 GB | 138 秒 | ~16 GB |
| 2048 | 7.96 GB / 22.56 GB | 2.59 GB | 382 秒 | ~24 GB |

## 输出图层

分解产生的语义图层包括：

**身体部位：** 前发、后发、颈部、上衣、手套、下装、腿饰、鞋子、尾巴、翅膀、物件

**头部部位：** 头饰、面部、虹膜、眉毛、眼白、睫毛、眼镜、耳朵、耳饰、鼻子、嘴巴

每个图层都是带透明度的 RGBA 图像，定位在画布上的正确位置。

## 致谢

本插件封装了 [shitagaki-lab](https://github.com/shitagaki-lab) 的 [See-through](https://github.com/shitagaki-lab/see-through) 研究项目。

PSD 生成使用浏览器端的 [ag-psd](https://github.com/nicasiomg/ag-psd) 库。

## 许可证

MIT
