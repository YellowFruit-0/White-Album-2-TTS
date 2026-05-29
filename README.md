# 白色相簿2 GPT-SoVITS TTS 模型 & OpenAI Compatible TTS 服务端

bilibili @ [黄水果天下第一](https://space.bilibili.com/535122654)

基于 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 训练的《白色相簿2》冬马和纱 / 小木曾雪菜 TTS 模型，以及配套的 OpenAI 兼容 TTS 服务端。可直接使用模型合成语音，也适用于酒馆聊天。

## 功能介绍

### 1. TTS 模型
使用《白色相簿2》游戏解包语音训练了**冬马和纱** 与 **小木曾雪菜**分别的TTS模型。
实测效果接近角色真实声线。训练语言为日语，推理建议使用日语。
可直接在 GPT-SoVITS 整合包的 TTS 推理 WebUI 中使用。

### 2. OpenAI 兼容 TTS 服务端
通过 server.py 建立 OpenAI Compatible TTS 服务端，可以接入各种已支持 OpenAI TTS API 的客户端(酒馆等)。
支持输入文本并根据预设角色或自定义模型进行语音合成。

### 3. LLM 增强功能: 翻译与情感识别
*此功能适合接入酒馆聊天的TTS功能，实现类似galgame的对话*
可以通过接入本地或在线大模型 API 实现：
**自动翻译**：可将输入文本识别提取对话内容(删除旁白等描述)，利用 LLM 翻译为日语后合成语音。
**识别情感**：可识别输入文本的情感色彩，并根据情绪特征自动切换对应的参考音频和参考文本，使生成的语音情感更加丰富真实。
*本人也在着手撰写白色相簿2的人物卡，后续如果效果好也会开源发布。*


## 使用方法

### 模型使用
与标准的 GPT-SoVITS 模型使用流程完全一致：
1. 下载 `GPT_weights_v2Pro` 和 `SoVMT_weights_v2Pro` 中你需要的角色文件夹（`setsuna` 或 `kazusa`）。
2. 将文件夹直接放置到你的 **GPT-SoVITS 整合包根目录** 下的同名文件夹中。
3. 在推理 WebUI 中即可直接加载使用。

### TTS 服务器使用
> **前提条件**：请确保本地已有可正常运行的 GPT-SoVITS 环境或整合包。

#### 一键启动
1. 下载本项目全部文件，并将所有内容复制到 **GPT-SoVITS 整合包根目录** 下。
2. 运行 `install_requirements.bat` 安装必要的 Python 依赖。
3. 根据需求双击运行 `start_server_setsuna.bat` (雪菜) 或 `start_server_kazusa.bat` (冬马)。

> [!CAUTION]
> 默认启动参数**没有启用**“对话提取”和“情感识别”功能。如需开启，请编辑对应的 `start_server_xxx.bat` 文件并**正确配置** 接入大模型的 API URL、Port 及 API Key。注意必须使用 OpenAI Compatible格式。
**配置示例: 接入大模型启用翻译和情感识别**
```start_server_setsuna.bat
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"
set "PATH=%SCRIPT_DIR%\runtime;%PATH%"
echo "White Album 2 TTS Project by 黄水果天下第一"
echo "关于server.py的详细配置参数见下，可按照需要修改 start_server.bat 的内容进行配置。"
runtime\python.exe server.py -h
runtime\python.exe server.py --base_path %~dp0 --character "set" --url http://localhost:1234/v1(改成你要使用的大模型api接入口) --key 要使用的API-Key(未设置可不加此参数)
pause```

#### 高级使用
配置`start_server_xxx.bat`中`server.py`的传入参数。或者自行配置好 gpt-sovits python 虚拟环境后直接带参数运行`server.py`，此方式注意要先安装requirements_server.txt依赖
**基本用法示例：**
```bash
python server.py -b "D:\GPT-SoVITS-Root" -c set -u "http://localhost:1234/v1" -k "your_api_key" -m "which_llm_model_to_use"
```
## 参数配置说明
---

| 参数 | 短参数 | 说明 | 默认值 |
| :--- | :--- | :--- | :--- |
| `--base_path` | `-b` | GPT-SoVITS 整合包（或环境）的根目录路径 | 必需给出 |
| `--device` | `-d` | 使用的 GPU 序号 (整数) | `0` |
| `--listen_port` | `-p` | TTS 服务端监听端口 | `5000` |
| `--url` | `-u` | OpenAI 兼容 LLM 服务端的地址 (在线或本地) | `http://localhost:1234/v1` |
| `--key` | `-k` | 访问上述服务端的 API Key | `not-needed` |
| `--model` | `-m` | 翻译和情感识别使用的模型名称，若不指定，则使用`models.list()` 返回的第一个 | - |
| `--character` | `-c` | 选择角色预设：`set`(雪菜) 或 `kaz`(冬马) | `set` |
| `--no_translate` | `-t` | 设置该项后，将不对传入文本进行对话提取及日语翻译 | 默认执行翻译 |
| `--no_recognition`| `-r` | 设置该项后，将不再根据情感自动选择参考音频 | 默认执行识别 |
| `--use_full_precision` | `-f` | 使用全精度 (f32) 模式。合成音频效果更好但显存占用翻倍 | - |
| `--gpt_model` | - | 手动指定 GPT 模型路径 | 根据角色自动选择 |
| `--sovits_model` | - | 手动指定 SoVITS 模型路径 | 根据角色自动选择 |
| `--ref_audio` | - | 手动指定参考音频路径 | - |
| `--ref_text` | - | 参考音频对应的文本内容 | - |
| `--ref_language` | - | 参考音频的语言 (`中文`, `英文`, `日文`) | `日文` |

---

## 许可协议 (License)

本项目可免费用于个人、教育和研究目的。但是，**未经作者明确的书面许可，严禁将本项目或其衍生品用于任何商业目的**。

注意：本项目的部分内容引用了 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 的代码，该部分代码遵循 MIT License 协议。所有原始版权声明均予以保留。

From GPT-SoVITS:
MIT License
Copyright (c) 2024 RVC-Boss
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
