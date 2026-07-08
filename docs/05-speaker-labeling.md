# 05 · 说话人标注

> [← 04 文本规范化](04-text-normalization.md) | [返回首页](README.md) | 下一章:[06 构建与部署](06-build-and-deploy.md)

---

说话人标注在每条提交的转录前标注 **ATC / PLT**(管制员 / 飞行员),基于声纹差异区分。该功能从 Mac 版 `atc_app.py` 的 `SpeakerTracker` 移植而来。

## 1. 设计选择:相邻分段,而非身份库

> **关键设计**:不维护说话人身份库,只保留**上一句的声纹**。每条新语句与上一句比较,余弦相似度低于阈值即判定"换人",标签翻转。

为什么是相邻分段而非身份跟踪:

- ATC 无线电是 **PTT 半双工**,一来一回交替进行。"声音变了就翻转标签"的规则天然产生 ATC↔PLT 的交替,符合实际通话结构。
- **无线电音频失真严重**,长期维护的声纹质心会漂移、不可靠;而相邻两句的对比相对稳健。

```swift
final class SpeakerTracker {
    private let threshold: Float    // 0.82
    private let labels = ["ATC", "PLT"]
    private var lastEmbedding: [Float]?
    private var labelIndex = 0

    func identify(embedding emb: [Float]) -> String {
        defer { lastEmbedding = emb }
        guard let last = lastEmbedding else { labelIndex = 0; return labels[0] }
        let sim = cosine(emb, last)
        if sim < threshold { labelIndex = (labelIndex + 1) % labels.count }  // 换人 → 翻转
        return labels[labelIndex]
    }
}
```

> ℹ️ **当前状态**:声纹**底层照常计算**(基准测试与麦克风提交路径都会算),但转录行 UI 上的 ATC/PLT 标签**暂时隐藏**。功能代码与 `TranscriptEntry.speaker` 字段保留,便于后续换一种展示方式时直接复用。

## 2. 声纹编码器:GE2E → Core ML

### 2.1 模型来源

声纹嵌入用 [resemblyzer](https://github.com/resemble-ai/Resemblyzer) 的 GE2E VoiceEncoder(3 层 LSTM-256,输出 256 维 L2 归一化嵌入)。转换脚本:`convert_speaker_encoder.py`。

### 2.2 关键技巧:把梅尔前端烤进 Core ML

为了让 Swift 端**零 DSP 代码**,把 librosa 的梅尔频谱前端(`n_fft=400, hop=160, n_mels=40, power=2`)作为**固定卷积核**烤进 Core ML 模型(DFT-as-Conv1d):

```python
# DFT 实部/虚部 → 两个固定 Conv1d (201×400, stride 160),核为 hann·cos / -hann·sin
# 功率谱 = conv_re² + conv_im²
# 梅尔滤波器组 → 固定 Linear (201→40)
# 之后接 ve.lstm / linear / relu,L2 归一化
```

模型输入是 `[1, 25840]` 的原始 16kHz 波形(1.615s → 恰好 160 梅尔帧,无填充),输出 `[1, 256]` 嵌入。

### 2.3 转换正确性验证

转换脚本内置奇偶校验:对同一输入,自定义 `SpeakerNet`(含烤进的梅尔前端)与 resemblyzer 原版的余弦相似度必须 > 0.999。实测结果:

```
parity vs resemblyzer: cosine = 1.000000
CoreML FP16 vs resemblyzer cosine = 1.000000
```

模型最终编译为 `speaker-id-v1.mlmodelc`,仅 **5.7 MB**,纳入 git。

## 3. Swift 端推理(SpeakerID)

### 3.1 整句嵌入

镜像 resemblyzer 的 `embed_utterance`:1.615s 窗口、0.8s 步进滑动,逐窗推理后平均、再归一化。音量先归一到约 -30 dBFS(近似 `preprocess_wav`),跳过 VAD 裁剪(应用层 RMS 门已界定语音范围)。

```swift
static let windowSamples = 25_840    // 1.615s → 160 梅尔帧
private static let hopSamples = 12_800  // 0.8s
// 滑窗逐个 embedWindow,vDSP 累加平均,最后 L2 归一化
```

所有向量运算用 Accelerate(`vDSP_*`),嵌入耗时约 10–30ms。

### 3.2 离主线程执行

```swift
private func identifySpeaker(in samples: [Float]) async -> String? {
    guard samples.count >= minUtteranceSamples, let speakerID else { return nil }
    let emb = await Task.detached(priority: .userInitiated) {
        speakerID.embedUtterance(samples)
    }.value
    return emb.map { speakerTracker.identify(embedding: $0) }
}
```

> ⚠️ **已知风险**:`Task.detached` 跨线程捕获了非 `Sendable` 的 `speakerID`,且后台 Core ML 推理可能与 whisper 推理线程争用。真机曾出现 `EXC_BREAKPOINT`(后台线程)的偶发崩溃,根因尚未最终定位,属待加固项(见 [06 章 · 已知问题](06-build-and-deploy.md#已知问题))。

## 4. 会话生命周期

- **每次开始录音 / 每次基准测试运行**:`speakerTracker.reset()` —— 第一个声音 = ATC,之后按声纹变化翻转。
- **麦克风路径**:三条提交路径(即时、完整窗口、停止收尾)都会算声纹。
- **基准测试路径**:启用 `labelSpeaker`,把连续播放的不同切片当作相邻语句做分段验证。

## 5. 阈值标定备注

`threshold = 0.82` 沿用 Mac 版参数。但离线实测发现,在 ATCO2 这类强失真无线电音频上:

- 同一录音前后两半的相似度常落在 0.67–0.83;
- 不同录音之间有时高达 0.889。

即阈值 0.82 在强失真音频上存在"把同一个人切成两段"或"漏切"的风险。真机实测如发现标签乱跳,建议把阈值下调(如 0.75)或加入最短时长门槛。

---

> 下一章:[06 · 构建与部署](06-build-and-deploy.md)
