# 01 · 系统架构

> [← 返回首页](README.md) | 下一章:[02 · 模型与推理](02-model-and-inference.md)

---

## 1. 设计目标与约束

ATC Copilot 的架构由四条硬约束决定:

1. **完全离线**:音频与转录文本不得离开设备。无网络调用、无遥测、无账号。这既是隐私承诺,也是航空场景的现实需求(座舱/塔台网络不可靠)。
2. **实时体感**:从"听到"到"屏幕出字"必须接近实时。ATC 通话语速快、信息密度高,延迟会让工具失去意义。
3. **端侧算力受限**:iPhone 的 ANE/GPU 算力和散热都有限。单次 whisper-large-v2 推理约 1.2–1.5 秒,管线必须围绕这个延迟做工程设计,而不是假装它不存在。
4. **安全关键可读性**:跑道、高度、频率等飞行员必须复诵的要素,要在视觉上立即可辨。

## 2. 模块划分

```
┌─────────────────────────────────────────────────────────────┐
│                         UI 层 (SwiftUI)                       │
│  ContentView · LiveBox · TranscriptList · KeyInstructionStrip │
└───────────────────────────────┬─────────────────────────────┘
                                 │ @Published 状态绑定
┌───────────────────────────────▼─────────────────────────────┐
│                    WhisperState (@MainActor)                  │
│  流式状态机 · 滑动窗口 · 静音断句 · 提交逻辑 · 显示节奏控制    │
└──────┬──────────────────┬───────────────────┬───────────────┘
       │                  │                   │
┌──────▼──────┐  ┌────────▼────────┐  ┌───────▼────────┐
│  Recorder   │  │ WhisperContext  │  │   SpeakerID    │
│  音频采集    │  │  (actor)        │  │  声纹推理       │
│  环形缓冲    │  │  whisper.cpp    │  │  Core ML       │
└─────────────┘  └────────┬────────┘  └────────────────┘
                          │
                 ┌────────▼─────────┐
                 │  ATCNormalizer   │
                 │  文本规范化/高亮  │
                 └──────────────────┘
```

### 2.1 各模块职责

| 模块 | 文件 | 职责 |
|------|------|------|
| **ContentView / UI** | `whisper.swiftui.demo/UI/ContentView.swift` | SwiftUI 视图层,纯展示 + 用户交互 |
| **WhisperState** | `whisper.swiftui.demo/Models/WhisperState.swift` | 核心状态机:采集→窗口→推理→断句→提交→显示的全流程编排 |
| **WhisperContext** | `whisper.cpp.swift/LibWhisper.swift` | whisper.cpp 的 Swift actor 封装,串行化推理调用 |
| **Recorder** | `whisper.swiftui.demo/Utils/Recorder.swift` | AVAudioEngine 采集,线程安全的样本环形缓冲 |
| **ATCNormalizer** | `whisper.swiftui.demo/Models/ATCNormalizer.swift` | ATC 专用文本规范化、安全元素高亮、关键指令抽取 |
| **SpeakerID** | `whisper.swiftui.demo/Models/SpeakerID.swift` | 声纹嵌入(Core ML)+ 相邻分段说话人标注 |
| **ATCConfigManager** | `whisper.swiftui.demo/Models/ATCConfigManager.swift` | 领域配置(初始提示词、纠错词表)加载与缓存 |

## 3. 线程与并发模型

并发模型是这个项目最容易出错、也最关键的部分。核心原则:

### 3.1 三个隔离域

- **`WhisperState` 运行在 `@MainActor`**:所有 UI 状态(`@Published`)和流式状态机都在主 actor 上,串行执行,天然无数据竞争。
- **`WhisperContext` 是独立 `actor`**:whisper.cpp 有"同一时刻只能被一个线程访问"的约束,用 actor 串行化所有推理调用。`fullTranscribe` 是 `await` 调用,期间主 actor 让出,UI 不卡。
- **`Recorder` 用锁保护**:音频回调在 AVAudioEngine 的后台线程,样本缓冲用 `NSLock`(`samplesLock`)保护,`getSamples()` 返回值拷贝。

### 3.2 关键洞察:推理 await 期间主 actor 空闲

单次推理阻塞 `WhisperContext` actor 约 1.2 秒,但由于它是独立 actor,`await` 期间 **主 actor 是空闲的** —— 打字机动画、缓冲区静音扫描、UI 更新都能继续跑。这是整个"实时体感"得以成立的基础。

> ⚠️ **历史教训**:早期版本用 tick 计数来估算静音时长和推理节奏。但推理 await 会冻结轮询循环,tick 计数的时钟在 await 期间停摆,导致每个 pass 约 0.5 秒的"死区门控"和 1–3 秒的提交延迟。修复方案是**所有静音/时序都从音频缓冲本身派生**(见 [03 章](03-streaming-pipeline.md)),而非依赖循环 tick。

## 4. 数据流:从麦克风到屏幕

```
麦克风 → Recorder 环形缓冲 ──┐
                            │ (基准测试时由 simulatedSamples 注入)
                            ▼
              ┌─── 转录循环 (80ms tick) ───┐
              │   1. 读取缓冲              │
              │   2. 缓冲区派生静音检测     │
              │   3. 判断:预览 / 提交?    │
              └────────┬───────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼ 预览          ▼ 提交(静音>0.6s)
   updateLiveText   commitUtterance
   (single_segment, (复用假设 / 完整pass)
    无时间戳,        + 声纹标注
    无温度回退)      + 文本规范化
        │                │
        ▼                ▼
   稳定前缀更新      appendTranscript
   自适应打字机      + 关键指令抽取
        │                │
        ▼                ▼
   LiveBox(逐词)   TranscriptList(已提交)
```

## 5. 技术选型理由

| 决策 | 选择 | 理由 |
|------|------|------|
| 推理后端 | whisper.cpp | 成熟的 C++ 实现,原生支持 Core ML 编码器(ANE)+ Metal 解码器,体积小、可量化 |
| 编码器加速 | Core ML / ANE | large-v2 编码器在 ANE 上跑,显著降低 RTF 与功耗 |
| 解码器 | Metal / GPU | 自回归解码用 GPU + Flash Attention |
| 量化 | Q5_1 | 在 WER 几乎无损的前提下把模型压到 ~1.1 GB,适配端侧内存 |
| UI 框架 | SwiftUI | 声明式、与 `@Published` 状态绑定流畅 |
| 声纹编码 | resemblyzer GE2E → Core ML | 把 librosa 梅尔前端烤进 Core ML,Swift 端零 DSP 代码 |

## 6. 仓库结构

```
atc-ios/
├── whisper.swiftui.demo/
│   ├── Models/          # WhisperState, ATCNormalizer, SpeakerID, ATCConfigManager
│   ├── UI/              # ContentView 及子视图
│   ├── Utils/           # Recorder, 音频解码
│   └── Resources/
│       ├── models/      # 量化模型 .bin、Core ML 编码器、声纹模型(gitignore 大文件)
│       └── samples/     # ATCO2 测试切片 + test_clips.json 参考文本
├── whisper.cpp.swift/   # LibWhisper.swift(whisper.cpp 的 Swift 封装)
├── whisper.xcframework/  # 预编译的 whisper 框架(纳入 git)
├── build_ios_model.sh   # 模型构建一键流水线
├── convert_speaker_encoder.py  # 声纹编码器 → Core ML 转换脚本
└── docs/                # 本 wiki
```

> **大文件策略**:`*.bin` 与 `*.mlmodelc/`(模型)被 gitignore,可由 `build_ios_model.sh` 重新生成;唯一例外是 5.7 MB 的声纹模型 `speaker-id-v1.mlmodelc`(小,纳入 git)。`whisper.xcframework`(所有文件 <100 MB)纳入 git,让新克隆无需重编 whisper.cpp 即可构建。

---

> 下一章:[02 · 模型与推理](02-model-and-inference.md)
