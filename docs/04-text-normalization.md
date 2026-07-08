# 04 · 文本规范化与高亮

> [← 03 流式管线](03-streaming-pipeline.md) | [返回首页](README.md) | 下一章:[05 说话人标注](05-speaker-labeling.md)

---

ATC 通话有一套高度程式化的口语约定:数字逐位念、字母用音标(alpha/bravo)、频率念到小数。`ATCNormalizer` 把 whisper 的原始口语转写规整成飞行员习惯阅读的紧凑形式,并标注安全关键要素。

## 1. 双形态显示原则

> **重要设计**:live 框显示 **原始口语**("alpha charlie"、"two seven left"),转录区显示 **规范化形式**("AC"、"27 left")。
>
> 理由:实时阶段用户在"听",原样回显最贴合听感、最快建立信任;提交后用户在"读",规范化形式信息密度更高。规范化只在提交落定时发生。

## 2. 规范化流水线

`normalize(_:)` 的处理顺序(`whisper.swiftui.demo/Models/ATCNormalizer.swift`):

```swift
var words = splitWords(trimmed)
words = applyJoinCorrections(words)              // 词拼接纠错 (fox trot → foxtrot)
words = applyPerWordCorrections(words)           // 拼写 + 模糊纠错
words = applyMultiWordAirlineSubstitution(words) // singapore airlines → SIA
words = applySingleWordAirlineSubstitution(words)// speedbird → BAW
words = applyContextDigitCorrections(words)      // 上下文相关 to→two
words = removeNoiseWords(words)
words = applyTerminologyReplacements(words)
words = applyWaypointUppercase(words)
words = applyPhoneticLetterSubstitution(words)   // alfa oscar → AO
let cased = normalizeCase(words.joined(...))     // ICAO/ATC 大小写
return mergeStandIdentifiers(normalizeNumbers(cased))  // 数字 + 停机位黏合
```

## 3. 关键规范化规则

### 3.1 数字归一化

把口语数字串转为阿拉伯数字,保留前导零、处理 teens/tens/multiplier 与小数:

- `"two seven"` → `27`(逐位拼接)
- `"five thousand"` → `5,000`(千分位)
- `"one one eight decimal one"` → `118.1`(频率小数)

频率特判:5–6 位数字且前三位落在 118–137(航空频段)时,在第 3 位后插入小数点。

### 3.2 音标字母折叠

连续音标词折叠为无空格字母串:

- `"alfa oscar"` → `AO`
- `"taxi via alfa charlie"` → `taxi via AC`

### 3.3 上下文数字纠错(需双侧约束)

> ⚠️ **历史教训**:`"runway two four to the right"` 曾被错误纠成 `"runway 242"` —— 把 "to" 单侧纠成 "two"。修复:对 `to/too/for` 这类词要求**两侧都像数字**才纠正(`needsBothSides` 集合)。

### 3.4 停机位/机位编号黏合

口语 `"position five foxtrot"` 经数字与音标转换后变成 `"position 5 F"`,需要重新黏合:

- `"position 5 F"` → `position 5F`
- `"gate 12 A"` → `gate 12A`
- `"stand A 12"` → `stand A12`

仅在 `stand/position/gate/spot/bay/apron/parking/ramp` 这些**上下文词后几个词内**触发,因此 `"descend 5,000 feet"`、`"runway 27 left"` 不会被误黏合。

```swift
static let standContextWords: Set<String> = [
    "stand", "position", "gate", "spot", "bay", "apron", "parking", "ramp"
]
// 数字+单字母 或 单字母+数字 → 紧凑标识
if aNum, bLetter { return ac + bc.uppercased() + bt }   // 5 F → 5F
if aLetter, bNum { return ac.uppercased() + bc + bt }   // A 12 → A12
```

## 4. 安全关键元素高亮(ATCHighlighter)

高亮的设计意图是 **ICAO 强制复诵项 / "错过会出事"的要素**,共三类:

| 类别 | 颜色 | 覆盖 |
|------|------|------|
| `.runway`(跑道) | 橙 | 跑道号、hold short、cleared to land/takeoff、LUAW、go-around |
| `.vertical`(垂直/横向/速度/气压) | 薄荷绿 | 高度/FL、航向、速度、QNH/altimeter |
| `.comms`(通信) | 天蓝 | 频率、squawk |

### 4.1 实现要点

- 高亮在 **提交时计算一次**,缓存进 `TranscriptEntry.display`(`AttributedString`)。
  > ⚠️ **性能教训**:早期在每次 SwiftUI render 都跑一遍正则,导致整个列表在每次实时更新时重扫,是明显的性能杀手。改为 init 时计算一次并缓存。
- 正则按优先级排序,重叠匹配时**最早起点优先、同起点高优先级优先、丢弃重叠**。
- 连接词白名单容忍 ASR 的口语啰嗦:`(?:and|to|at|altitude|maintain){0,3}`,使 `"descend to altitude 4,000 feet"` 也能命中。

### 4.2 为什么停机位不高亮

> **专业判断**:停机位/滑行道**不纳入**安全关键高亮。理由:(1) 它不是强制复诵的安全关键项,染成同样醒目的颜色会稀释跑道/高度/频率的视觉权重;(2) 地面阶段真正的安全项 `hold short`(防跑道侵入)已归在 runway 橙类。停机位只需正确**黏合显示**即可。

## 5. 关键指令条(ATCKeyState)

转录区上方的 chips 维护"当前分配给我的指令状态"—— 每个安全关键类别一个槽,保存最近一次听到的值,新传输只覆盖它提到的槽,镜像飞行员脑中维护的指令状态。

```swift
enum Kind: Int, CaseIterable, Comparable {
    case runway, altitude, heading, speed, frequency, squawk, qnh
}
// extract() 返回 [Kind: String],如 "RWY 27L"、"↓ FL100"
```

- 标签形如 `RWY 27L`、`↓ 3,000`、`HDG 270`、`118.1`。
- 箭头(↑/↓)用动词形态判断(`climbing`/`descending` 的 `hasPrefix`)。
- 与高亮共享配色:runway→橙,altitude/heading/speed/qnh→薄荷,frequency/squawk→天蓝。

## 6. 幻觉过滤

`isHallucination(_:)` 在提交时拦截 whisper 的重复/退化输出:词数 >4 且唯一词占比 <35% 判为幻觉;单词长度 >25 字符判为幻觉;并检测周期性重复子串。实时部分形态(半句)不做此过滤,避免误伤。

## 7. 语料驱动的回归验证

规范化与高亮有一套基于 871 条参考的回归测试。开发期用独立的 Swift 文件编译运行:

```bash
swiftc -o /tmp/selftest /tmp/main.swift \
  whisper.swiftui.demo/Models/ATCNormalizer.swift \
  whisper.swiftui.demo/Models/ATCConfigManager.swift
```
(测试文件须命名为 `main.swift` 以支持顶层代码。)每次改动归一化/高亮规则后跑一遍,确保不回退。

---

> 下一章:[05 · 说话人标注](05-speaker-labeling.md)
