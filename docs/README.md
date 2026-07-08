# ATC Copilot 技术文档

> 离线、端侧运行的空中交通管制(ATC)无线电实时转录 iOS 应用。
> 隶属 FlightAI —— ATC 转录是航空自主化路线图的第一步。

本 wiki 面向工程读者,系统性地说明 ATC Copilot 的架构、模型、推理管线与关键工程权衡。每一处看似"反直觉"的设计(关闭温度回退、缓冲区派生静音计时、零推理提交等)背后都有可复现的实测依据,文档中一并记录。

---

## 文档目录

| 章节 | 内容 |
|------|------|
| [01 · 系统架构](01-architecture.md) | 模块划分、数据流、线程模型、技术选型 |
| [02 · 模型与推理](02-model-and-inference.md) | 模型谱系、量化、Core ML/ANE 加速、whisper.cpp 集成、命名约定 |
| [03 · 流式转录管线](03-streaming-pipeline.md) | 滑动窗口、静音断句、增量预览、稳定前缀、零推理提交、自适应打字机 |
| [04 · 文本规范化与高亮](04-text-normalization.md) | 数字/音标字母/停机位归一化、安全关键元素高亮、关键指令条 |
| [05 · 说话人标注](05-speaker-labeling.md) | GE2E 声纹编码器、Core ML 转换、相邻分段 |
| [06 · 构建与部署](06-build-and-deploy.md) | 模型构建流水线、xcframework、TestFlight/App Store 发布 |
| [Internal guide](INTERNAL_GUIDE.md) | English internal map of runtime functions, files, and team workflows |

---

## 一分钟概览

- **运行形态**:100% 离线、端侧推理,音频与文本不出设备。无服务器、无账号、无网络依赖。
- **语音模型**:基于 `openai/whisper-large-v2` 在 ATCO2 + ATCOSIM 语料上微调的开源权重(Apache-2.0),本项目对其做端侧优化 —— Q5_1 量化(2.9 GB → ~1.1 GB)+ Core ML 编码器(Apple Neural Engine 加速)。
- **推理后端**:[whisper.cpp](https://github.com/ggerganov/whisper.cpp),Core ML 编码器跑 ANE,Metal 解码器跑 GPU,启用 Flash Attention。
- **核心体验**:实时逐词显示(LocalAgreement-2 等价的稳定前缀)、安全关键元素(跑道/高度/频率/应答机/QNH)自动高亮、当前指令状态条。
- **目标平台**:iPhone(iPhone-only),实测设备 iPhone 17 Pro Max。

## 关键性能指标(端侧,iPhone 17 Pro Max)

| 指标 | 数值 | 说明 |
|------|------|------|
| RTF(实时因子) | ~0.37 | 单次完整提交 pass,Core ML ANE 编码器 |
| WER(词错误率) | ~14–19% | ATCO2 测试集随机抽样,原始 whisper 输出 vs 参考 |
| 模型体积 | ~1.1 GB | Q5_1 量化主模型 + Core ML 编码器 |
| 声纹模型 | 5.7 MB | GE2E 编码器 Core ML(FP16) |

## 术语表

| 术语 | 含义 |
|------|------|
| **ATC** | Air Traffic Control,空中交通管制 |
| **ATCO2 / ATCOSIM** | 两个公开 ATC 语音语料(真实录音 / 模拟录音) |
| **ANE** | Apple Neural Engine,苹果神经网络引擎 |
| **RTF** | Real-Time Factor,推理耗时 / 音频时长,<1 为快于实时 |
| **WER** | Word Error Rate,词错误率 |
| **PTT** | Push-To-Talk,无线电半双工按键通话 |
| **GE2E** | Generalized End-to-End loss,说话人声纹编码训练方法 |
| **稳定前缀** | 连续两次推理 pass 一致的词前缀,等价 LocalAgreement-2 |

---

*本文档随代码演进维护。若文档与代码冲突,以代码为准,并请同步更新本 wiki。*
