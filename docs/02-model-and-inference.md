# 02 · 模型与推理

> [← 01 架构](01-architecture.md) | [返回首页](README.md) | 下一章:[03 流式管线](03-streaming-pipeline.md)

---

## 1. 模型谱系

ATC Copilot 端侧运行的语音模型,其完整来源链如下:

```
openai/whisper-large-v2  (基座, ~1.5B 参数)
        │  在 ATCO2 + ATCOSIM 上微调 (Apache-2.0 开源权重)
        ▼
whisper-large-v2-atco2-asr-atcosim  (HF safetensors, 2.9 GB, F32)
  · 评估 WER 5.55,12644 训练步,4×GPU,lr 1e-5
        │  GGML 转换 (F16)
        ▼
GGML F16  (~1.5 GB)
        │  whisper-quantize Q5_1
        ▼
flightai-asr-v1-q5_1.bin  (~1.1 GB)  ←─ iOS 端主推理模型
        +
flightai-asr-v1-encoder.mlmodelc  ←─ Core ML 编码器 (ANE), 6-bit 调色板量化
```

> ⚠️ **来源准确性说明**:`atc-python/models/` 下的 2.9 GB 权重(`flightai-asr-v1-hf` / `-mlx`)其 model card(`config.json` 的 `_name_or_path = openai/whisper-large-v2`、`.cache/huggingface/download/*` 下载记录、自动生成的 `whisper-large-v2-atco2-asr-atcosim` model card)表明它是**从 HuggingFace 获取的开源微调权重**(Apache-2.0),而非本项目从零训练。本项目的工程贡献在于**端侧部署优化**:量化、Core ML/ANE 编码器转换、命名规整与构建流水线。文档如实记录谱系,便于合规归属(Apache-2.0 要求保留许可与版权声明)。

## 2. 量化:为什么是 Q5_1

| 格式 | 体积 | 端侧可行性 |
|------|------|-----------|
| F32(原始) | 2.9 GB | ✗ 内存超限 |
| F16 | ~1.5 GB | △ 偏大 |
| **Q5_1** | **~1.1 GB** | ✓ 主选 |

Q5_1 是 GGML 的 5-bit 量化(每块带缩放 + 最小值)。实测在 ATCO2 测试集上,Q5_1 相对 F16 的 WER 退化可忽略,而体积下降约 27%,是端侧内存与精度的最佳平衡点。

> ⚠️ **容易混淆的点**:上表只对比**主模型**的量化格式。端侧运行时实际**同时加载两个配套文件,缺一不可**:
>
> | 文件 | 体积 | 量化 | 跑在 | 角色 |
> |------|------|------|------|------|
> | `flightai-asr-v1-q5_1.bin` | ~1.1 GB | Q5_1 | Metal / GPU | 主模型(含解码器) |
> | `flightai-asr-v1-encoder.mlmodelc` | ~458 MB | 6-bit 调色板 | **ANE** | 编码器(梅尔频谱 → 隐状态) |
>
> "用上 ANE 的模型"指的就是后者 —— Core ML 编码器跑在 Apple Neural Engine 上。两个文件靠命名约定配对(见 §3.2),缺了编码器会静默回退 GPU。

## 3. Core ML 编码器:ANE 加速的关键与陷阱

### 3.1 编码器/解码器分工

- **编码器(Encoder)→ Core ML / ANE**:把梅尔频谱编码为隐状态,计算密集、可并行,放 ANE 跑功耗与延迟最优。
- **解码器(Decoder)→ Metal / GPU**:自回归逐 token 解码,启用 Flash Attention。

### 3.2 命名约定陷阱(必读)

whisper.cpp 通过 `whisper_get_coreml_path_encoder` **由 `.bin` 文件名推导 Core ML 编码器路径**,推导规则是**剥离末尾的量化后缀**(`-q5_1`):

```
flightai-asr-v1-q5_1.bin   →   配对   →   flightai-asr-v1-encoder.mlmodelc
                  ↑ 剥离此后缀
```

> ⚠️ **静默失败**:若编码器命名不匹配,whisper.cpp **不会报错**,而是**静默回退到 GPU 编码**,RTF 显著变差。验证方式是检查启动日志中是否出现:
> ```
> whisper_init_state: Core ML model loaded
> ```
> 没有这一行 = 没在用 ANE。

### 3.3 ANE 加载的内存陷阱(0x20004)

未压缩的 F16 large 编码器(约 1.2 GB)在 iPhone ANE 上**无法加载**,报 `Program load failure 0x20004`。

**解决方案**:对编码器做 **6-bit k-means 调色板量化**(`--palettize 6`),把编码器压到约 458 MB,即可在 ANE 上正常加载。这是把 large-v2 跑上端侧 ANE 的必要步骤,不是可选优化。

## 4. whisper.cpp 集成(LibWhisper.swift)

`WhisperContext` 是对 whisper.cpp 的 actor 封装,核心是 `fullTranscribe(...)`。

### 4.1 上下文创建

```swift
static func createContext(path: String) throws -> WhisperContext {
    var params = whisper_context_default_params()
#if targetEnvironment(simulator)
    params.use_gpu = false          // 模拟器无法加载 1.1GB 模型,跳过 GPU
#else
    params.flash_attn = true        // 真机启用 Flash Attention
#endif
    ...
}
```

> **模拟器限制**:模拟器的 Metal 工作集为 0、CPU_REPACK 会 OOM,无法加载 1.1 GB 的 ATC 模型。因此模拟器上 `builtInModelUrl` 返回 nil,跳过自动加载 —— **基准测试与真实转录必须在真机上跑**。

### 4.2 预览模式(preview)

为压低每个 pass 的成本,预览模式做了三项裁剪:

```swift
if preview {
    params.no_timestamps = true   // 不解码无人显示的时间戳 token(~74ms)
    params.max_tokens    = 96     // 给重复循环兜底,防止单 pass 卡死
}
```
加上 `single_segment = true`,预览 pass 比完整 pass 明显更快,而预览文本本就是瞬态的,精度损失无关紧要。

### 4.3 温度回退:全局关闭

```swift
// 温度回退在所有路径上禁用。100 条 ATCO2 测试切片的 A/B 实测:
// 开/关 WER 几乎一致 (19.8% vs 19.9%),但"开"会产生严重幻觉
// ("china southern" → "air berlin") 和数秒级的重试尖峰。
params.temperature_inc = 0
```

> **这是一个经过 A/B 验证的反直觉决策**。whisper 默认在解码失败时升温重试以求稳健,但在 ATC 这种短促、术语密集的音频上,温度回退不仅无收益,反而引入幻觉和延迟尖峰。**全局 `temperature_inc = 0`**。

### 4.4 VAD 的取舍

完整提交 pass 在应用层已做 RMS 静音分段,因此关闭 whisper 内置的 Silero VAD(`useVAD: false`)—— Silero 在已分段的短句上会额外耗费 50–800ms,得不偿失。

## 5. 端侧性能特征

| 项 | 实测 | 备注 |
|----|------|------|
| 单 pass 推理 | ~1.2–1.5s | ANE 编码器 + Metal 解码器 |
| 完整提交 RTF | ~0.37 | 提交 pass 延迟 / 音频时长 |
| 热管理 | 每个预览 pass 后 150ms 喘息 | 背靠背推理会让 ANE/GPU 100% 占空比,长会话热降频 |

> **散热是真实约束**。背靠背推理把 ANE/GPU 拉满,长时间会话会热降频。预览 pass 后插入 150ms 喘息,在打字机动画仍在揭示文本时几乎无感,却能持续散热。充电 + 跑基准会叠加环境热负载,实测 RTF 会被环境热降频拖高。

---

> 下一章:[03 · 流式转录管线](03-streaming-pipeline.md)
