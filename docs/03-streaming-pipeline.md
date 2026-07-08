# 03 · 流式转录管线

> [← 02 模型与推理](02-model-and-inference.md) | [返回首页](README.md) | 下一章:[04 文本规范化](04-text-normalization.md)

---

流式管线是 ATC Copilot 的工程核心,目标是在单次推理就要 1.2–1.5 秒的硬约束下,做出"实时逐词"的体感。本章按数据流顺序拆解。

## 1. 总体状态机

转录循环(`startTranscriptionLoop`)以 80ms 为 tick 周期轮询,维护一个滑动窗口,在三种动作间切换:

```
        ┌──────────────────────────────────────┐
        │  每 80ms tick:                        │
        │  allSamples = currentSamples()        │
        │  silence = silenceDuration(allSamples)│
        └───────────────┬──────────────────────┘
                        │
         ┌──────────────┼───────────────┬─────────────────┐
         ▼ 未说话        ▼ 静音≥0.6s      ▼ 静音<0.3s        ▼ 否则
   推进窗口起点    commitUtterance   updateLiveText      等待
   (跟随重叠)      (提交+重置)        (预览)
```

关键状态变量:

| 变量 | 值 | 含义 |
|------|----|----|
| `windowStartSample` | 动态 | 当前窗口在全局缓冲中的起点 |
| `maxWindowSamples` | 28 × 16000 | 窗口上限(留在 whisper 30s 之内,一个 pass 一次编码) |
| `keepOverlapSamples` | 200 × 16 | 提交后保留 200ms 前置重叠 |
| `silenceRMSThreshold` | 0.008 | RMS 静音判定阈值(原 0.012,安静无线电检测偏晚) |
| `silenceCutoffSec` | 0.6 | 句末静音切分阈值 |
| `minPreviewGap` | 0.3 | 两次预览 START 之间的最小间隔 |
| `minUtteranceSamples` | 5120 | 0.32s,首次预览/提交的最小语音量 |

## 2. 缓冲区派生的静音计时(核心修复)

所有静音/时序**从音频缓冲本身派生**,绝不依赖循环 tick 计数。

```swift
// 从缓冲尾部向前扫描,以 80ms 块为单位,最多回扫 ~4.8s
nonisolated private static func silenceDuration(in samples: [Float], threshold: Float) -> Double {
    let block = 1280            // 80ms @ 16kHz
    let maxBlocks = 60          // 最多回扫 ~4.8s
    var end = samples.count, scanned = 0
    while end > 0, scanned < maxBlocks {
        let start = max(0, end - block)
        // 计算该块 RMS
        ...
        if rms > threshold { return Double(samples.count - end) / 16000.0 }
        end = start; scanned += 1
    }
    return Double(scanned) * 0.08
}
```

> **为什么这样设计**:推理 `await` 会阻塞轮询循环约 1.2 秒,任何 tick 计数的时钟在 await 期间都会停摆。而缓冲区是"地面真值"—— 即使刚刚经历了 1.2 秒的推理冻结,从缓冲尾部回扫得到的静音时长依然准确。这一改动消除了早期 ~0.5s/pass 的死区门控和 1–3s 的提交延迟。
>
> 函数标记 `nonisolated static`,操作传入的值拷贝,无副作用、可在任意上下文调用。

## 3. 实时预览:稳定前缀(LocalAgreement-2 等价)

预览的目标是让文本"稳定生长",而不是每个 pass 整段重写。

### 3.1 稳定前缀算法

对比当前 pass 与上一个 pass 的**逐词公共前缀**:一致的词进入"已确认"(实色,永不回退),其余为灰色"试探"尾巴。

```swift
// 连续两次 pass 一致的词数 = 稳定前缀长度
var stableN = 0
while stableN < min(pw.count, cw.count),
      pw[stableN].lowercased() == cw[stableN].lowercased() { stableN += 1 }
if stableN > targetConfirmedWords.count {
    targetConfirmedWords = Array(cw[0..<stableN])   // 采用当前 pass 的大小写
}
```

这等价于 whisper-streaming 论文中的 **LocalAgreement-2** 策略。

### 3.2 为什么是词数组而非字符切片

> ⚠️ **历史教训**:早期用字符偏移切片来截取稳定前缀。但 whisper 在不同 pass 间会抖动大小写和标点,字符偏移会把词从中间切断,产生乱码/重复文本。**改为基于词数组重建显示**,从结构上杜绝了重复 —— 显示永远是 `已确认词 + 试探词` 的拼接,不可能出现半个词。

## 4. 自适应打字机

推理每 ~1.2 秒落地一批新词。一次性全显得"块状";固定速率又显得"卡顿"。自适应打字机以"刚好在下一批到达前显示完"的速率逐词揭示:

```swift
// 把积压词在 ~1 秒内显示完,每词钳制在 30–180ms
let interval = min(0.18, max(0.03, 1.0 / Double(pending)))
```
积压少 → 从容打字;积压多 → 快速追赶。结果是 live 框近乎连续地逐词流出,几乎不增加感知延迟。

## 5. 提交:零推理是常态

这是"实时体感"的最后一块拼图,直接对标 Mac 版的做法。

### 5.1 即时提交(零额外推理)

```swift
private func commitUtterance(allSamples: [Float], silence: Double) async {
    let hypothesisWords = targetConfirmedWords + targetTentativeWords
    let speechEndSample = allSamples.count - Int(silence * 16000)
    let uncoveredTail = max(0, speechEndSample - lastPreviewWindowEnd)

    if !hypothesisWords.isEmpty, uncoveredTail < Int(0.3 * 16000) {
        // 最新预览假设已覆盖整句 → 直接复用,零推理提交
        let speakerLabel = await identifySpeaker(in: utterance)
        await finishLiveReveal()                     // 让 live 框先把字打完
        appendTranscript(normalizer.normalize(...), speaker: speakerLabel)
        resetLive()
    } else {
        await commitCurrentWindow()                  // 假设过期/缺尾 → 才付一次完整 pass
    }
}
```

**逻辑**:大多数情况下,提交时最新预览假设已经覆盖了整句话(`uncoveredTail < 0.3s`),此时**直接复用预览假设、零额外推理**就能提交。只有当语音在最后一次预览窗口之后又显著延续(尾部未覆盖),才付一次完整 pass。这把提交从"再跑一次 1.2s 推理"变成"瞬时"。

### 5.2 提交前交接

三条提交路径(即时提交、完整窗口提交、停止收尾)都先调用 `finishLiveReveal()` —— 以 35ms/词快速排空打字机,再停顿 150ms,让完成的行"落定"后再下移进转录区,避免 live 框在打字中途被抽走。

## 6. 半双工断句的合理性

ATC 无线电是 **PTT 半双工**:同一时刻只有一方在讲,说话人切换必然伴随静音间隙。因此**基于静音的分段**对 ATC 是恰当的 —— 一次"松开 PTT"就是一个自然的句子边界。

`silenceCutoffSec = 0.6` 经实测验证:真实 ATC 传输内部的停顿最长约 0.54s,0.6s 阈值能在不切断单次传输的前提下,可靠地识别传输结束。

## 7. 显示布局与方向

- **LiveBox**:1→3 行动态高度(`LiveTextHeightKey` PreferenceKey),超过 3 行内部滚动并跟随最新词。
- **TranscriptList**:**最新在顶部**(逆序),新传输落在紧贴 live 框的下方,阅读视线始终停在顶部,无需向下追。
- 组件间距统一由主 `VStack(spacing: 10)` 驱动,避免各组件零散 padding 导致的视觉不齐。

---

> 下一章:[04 · 文本规范化与高亮](04-text-normalization.md)
