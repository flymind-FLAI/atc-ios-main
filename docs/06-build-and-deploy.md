# 06 · 构建与部署

> [← 05 说话人标注](05-speaker-labeling.md) | [返回首页](README.md)

---

## 1. 模型构建流水线

`build_ios_model.sh` 是一键流水线,把微调权重转成 iOS 端可加载的量化模型 + Core ML 编码器。

```
HF safetensors / MLX weights
        │  ① 转 GGML F16
        ▼
   GGML F16 (.bin)
        │  ② whisper-quantize Q5_1
        ▼
flightai-asr-v1-q5_1.bin  ──┐
        │  ③ 生成 Core ML 编码器        ④ 6-bit 调色板量化
        ▼                              ▼
flightai-asr-v1-encoder.mlmodelc (ANE 可加载, ~458MB)
```

> **命名约定(关键)**:`ENCODER_STEM` 必须剥离 `.bin` 的量化后缀(`-q5_1`),使 `flightai-asr-v1-q5_1.bin` 与 `flightai-asr-v1-encoder.mlmodelc` 配对。否则 whisper.cpp 静默回退 GPU。详见 [02 章 · 3.2](02-model-and-inference.md)。

### 声纹模型构建

```bash
/opt/miniconda3/bin/python3 convert_speaker_encoder.py
# → whisper.swiftui.demo/Resources/models/speaker-id-v1.mlmodelc (5.7MB)
```

## 2. 资源与 git 策略

| 资源 | 体积 | git | 来源 |
|------|------|-----|------|
| `flightai-asr-v1-q5_1.bin` | ~1.1 GB | gitignore | `build_ios_model.sh` 重新生成 |
| `flightai-asr-v1-encoder.mlmodelc` | ~458 MB | gitignore | 同上 |
| `ggml-silero-v5.1.2.bin` | 小 | gitignore | VAD 模型 |
| `speaker-id-v1.mlmodelc` | 5.7 MB | **纳入 git** | `convert_speaker_encoder.py` |
| `whisper.xcframework` | <100MB/文件 | **纳入 git** | 预编译,免去新克隆重编 whisper.cpp |
| `samples/*.wav` | ~4.9 MB | 纳入 git | 30 条 ATCO2 测试切片 |

`.gitignore` 关键规则:
```gitignore
*.bin
*.mlmodelc/
# 例外:5.7MB 声纹模型纳入 git
!whisper.swiftui.demo/Resources/models/speaker-id-v1.mlmodelc/
```

## 3. 基准测试(端侧)

> ⚠️ **必须真机**:模拟器加载不了 1.1 GB 模型(Metal 工作集 0、CPU_REPACK OOM)。

基准测试(Settings → Benchmark)的核心设计是:**把打包的 ATCO2 切片当作麦克风实时输入,逐帧喂进与真机完全相同的流式管线**,而非一次性整段转录。

```
随机抽 10 条 ATCO2 切片
   │  feedSamples():按实时速率(40ms/帧)追加进 simulatedSamples
   │  + 同时播放该切片音频
   │  feedSilence(1.0s):喂静音 → 触发 end-of-transmission 提交
   ▼
真实流式管线(滑动窗口预览 / 静音断句 / 提交 / 声纹分段)
   ▼
汇总:总音频时长 · 平均端到端延迟 · WER
```

> **为什么这样设计**:一个"一次性整段转录"的基准测试会绕过它本应测量的流式行为(预览、断句、提交、声纹分段),毫无意义。注入式回放让基准测试走的是用户真实经历的代码路径。
>
> 实现上引入 `simulatedSamples`(注入缓冲)+ `currentSamples()`(数据源抽象),`streamActive` 统一驱动麦克风录音与基准回放两种模式;`recorder.getSamples()` 的所有调用点都改为 `currentSamples()`。

汇总指标改为 **总音频 / 平均端到端延迟 / WER** —— 流式下提交多为零推理的即时提交,RTF 已非有意义指标,真正反映体验的是"末词到落定"的端到端延迟。

## 4. 离线配置(ATCConfig)

`ATCConfigManager` 管理领域配置(whisper 初始提示词、拼接/拼写纠错词表)。配置可内置,也支持从 URL 下载并缓存,便于在不发版的前提下迭代领域词表。注意:模型本身完全离线,配置下载是可选的运维通道。

## 5. 发布(TestFlight / App Store)

### 5.1 构建号与设备族

- 主 app target:`TARGETED_DEVICE_FAMILY = "1"`(**iPhone-only**)。
  > 改为 iPhone-only 的原因:App Store 要求 13″ iPad 截图是因为 target 声明了 iPad 支持;ATC Copilot 是手机使用场景,限制为 iPhone 既免去 iPad 截图要求,也避免 iPad UI 适配的审核风险。
- 每次上传 TestFlight/App Store,`CURRENT_PROJECT_VERSION` 必须递增(同一 `MARKETING_VERSION` 下构建号不可复用)。

### 5.2 归档与上传

```bash
xcodebuild -scheme WhisperCppDemo -destination 'generic/platform=iOS' \
  -configuration Release archive -archivePath /tmp/ATCCopilot.xcarchive \
  -allowProvisioningUpdates

xcodebuild -exportArchive -archivePath /tmp/ATCCopilot.xcarchive \
  -exportOptionsPlist ExportOptions.plist -allowProvisioningUpdates
# ExportOptions.plist: method=app-store-connect, destination=upload
```

### 5.3 麦克风用途说明

`Info.plist` 的 `NSMicrophoneUsageDescription` 必须如实说明用途(捕获 ATC 无线电音频用于端侧转录,不录制不上传)—— 缺失或含糊是常见拒审原因。

## 6. 已知问题

| 问题 | 状态 | 备注 |
|------|------|------|
| 真机 `EXC_BREAKPOINT`(后台线程) | 待加固 | 疑似 `identifySpeaker` 的 `Task.detached` 跨线程捕获非 Sendable 的 `speakerID`,或后台 Core ML 与 whisper 推理争用。偶发,根因未最终定位。见 [05 章 · 3.2](05-speaker-labeling.md) |
| 声纹阈值 0.82 偏高 | 待标定 | 强失真音频上可能误切/漏切,见 [05 章 · 5](05-speaker-labeling.md) |
| App 图标 | 待设计 | 现用机翼麦克风图标 |

## 7. 待办与资产备份

> ⚠️ **关键资产备份**:`atc-python/models/flightai-asr-v1-hf`(2.9 GB)是端侧模型的上游来源。代码、量化模型、Core ML 均可由它重新生成,务必对其(及其来源归属)做好备份与许可记录(Apache-2.0,见 [02 章 · 1](02-model-and-inference.md))。

---

> [返回首页](README.md)
