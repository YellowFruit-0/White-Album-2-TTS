# coding: utf-8
import os
import argparse
import sys
import uuid
import tempfile
import soundfile as sf
from flask import Flask, request, send_file
from openai import OpenAI

parser = argparse.ArgumentParser(description="GPT-SoVITS OpenAI Compatible TTS Server")
parser.add_argument("-b", "--base_path", required = True, help = "GPT-SoVITS 整合包安装的根目录 | Root path of GPT-SoVITS.")
parser.add_argument("-d", "--device", default = "0", help = "Use which CUDA gpu to do TTS inference. Enter one number.")
parser.add_argument("-p", "--listen_port", default = "5000", help = "TTS Service listening port. Default = 5000")
parser.add_argument("-u", "--url", default = "http://localhost:1234/v1", help = "URL to OpenAI Compatible LLM inference server. Default = http://localhost:1234/v1")
parser.add_argument("-k", "--key", default = "not-need", help = "API Key to access LLM server.")
parser.add_argument("-m", "--model", help = "Model identifier for LLM server. If not given, will use the first model in client.models.list()")
parser.add_argument("-t", "--no_translate", action = "store_true", help = "If set, input texts will NOT be parsed into dialogues and then translated into Japanese before TTS.")
parser.add_argument("-c", "--character", default = "set", choices=["set", "kaz"], help = "Choosing character preset from Kazusa(kaz) and Setsuna(set).")
parser.add_argument("-r", "--no_recognition", action = "store_true", help = "If set, will NOT automatically choose reference audio and text corresponding to input text emotion, which means generated voices can be less emotional but much more consistent. If not set(default), -ref_audio and -ref_text should NOT be given.")
parser.add_argument("-f", "--use_full_precision", action = "store_true", help = "Use full precision instead of half.")
parser.add_argument("--gpt_model", help = "Manually specify which gpt model to use. If not given, will automatically choose from Kazusa or Setsuna.")
parser.add_argument("--sovits_model", help = "Manually specify which sovits model to use. If not given, will automatically choose from Kazusa or Setsuna.")
parser.add_argument("--ref_audio", help="Path to the reference audio file. Must be given with --no_recognition set.")
parser.add_argument("--ref_text", help="Reference text. Must be given with --no_recognition set.")
parser.add_argument("--ref_language", default = "日文", choices=["中文", "英文", "日文"], help="Language of reference audio")
args = parser.parse_args()

GPT_SOVITS_INSTALL_PATH = args.base_path
if not os.path.exists(GPT_SOVITS_INSTALL_PATH):
    print("给定的 GPT-SoVITS 目录不存在 | Given GPT-SoVITS does not exist!")
    exit(1)

if (args.ref_audio == None and args.ref_text != None) or (args.ref_text == None and args.ref_audio != None):
    print("-ref_audio and -ref_text must be given at the same time, or both be left blank.")
    exit(1)
if args.ref_audio != None and not args.no_recognition:
    print("Recognition can't be used with -ref_audio and -ref_text given. Please set --no_recognition to manually specify ref audio and text.")
print(f"Translation:{not args.no_translate} | Recognition:{not args.no_recognition}")
print("Character Preset: " + args.character)

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
print("Using CUDA:" + args.device)
if args.use_full_precision:
    os.environ["is_half"] = "False"
    print("Using Full Precision")
else:
    os.environ["is_half"] = "True"
    print("Using Half Precision")

if args.gpt_model == None:
    if args.character == "set":
        os.environ["gpt_path"] = r"GPT_weights_v2Pro\setsuna-e20.ckpt"
    else:
        os.environ["gpt_path"] = r"GPT_weights_v2Pro\kazusa-e20.ckpt"
else:
    os.environ["gpt_path"] = args.gpt_model
if args.sovits_model == None:
    if args.character == "set":
        os.environ["sovits_path"] = r"SoVITS_weights_v2Pro\setsuna_e8_s408.pth"
    else:
        os.environ["sovits_path"] = r"SoVITS_weights_v2Pro\kazusa_e8_s424.pth"
else:
    os.environ["sovits_path"] = args.sovits_model
print("Using GPT-Model:{}\nUsing SoVITS-Model:{}".format(os.environ["gpt_path"], os.environ["sovits_path"]))
os.environ["cnhubert_base_path"] = os.path.join(GPT_SOVITS_INSTALL_PATH, r"GPT_SoVITS\pretrained_models\chinese-hubert-base")
os.environ["bert_path"] = os.path.join(GPT_SOVITS_INSTALL_PATH, r"GPT_SoVITS\pretrained_models\chinese-roberta-wwm-ext-large")
sys.path.append(GPT_SOVITS_INSTALL_PATH)
sys.path.append(os.path.join(GPT_SOVITS_INSTALL_PATH, r"GPT_SoVITS\eres2net"))
sys.path.append(os.path.join(GPT_SOVITS_INSTALL_PATH, r"tools\i18n"))
sys.path.append(os.path.join(GPT_SOVITS_INSTALL_PATH, r"GPT_SoVITS"))
# --- 1. 初始化并加载本地 TTS 模型 ---
from i18n import I18nAuto
i18n = I18nAuto()
from inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav

# 用于分离人物语言、识别情绪与翻译的 System Prompt
SYS_PROMPT_TL_REC = "接下来用户会发送一段文字，你需要完成以下##两个任务：\n#第一个\n根据给出的文本语境，判断该段文本中说话人的情绪最符合下列哪个情况：如果情绪平淡或平和温柔，则输出的第一行为“1”；如果情绪悲伤或痛苦，则输出的第一行为“2”；如果情绪非常快乐、兴奋或幸福，则输出的第一行为“3”；如果情绪愤怒，则输出的第一行为“4”。需要注意，*除非情绪指向性非常明显且极度强烈*，否则应当输出“1”。要正确理解日常对话中的玩笑、打趣和挖苦等语言，*不要轻易输出“2”或“4”*；若难以归类，则一律输出“1”。\n#第二个\n根据语义分辨其中人物语言的部分，将其中所有的语义是人物语言的文字翻译为日语并输出。\n##注意要点\n1.在识别人物语言和翻译的过程中，必须忽略除语言外的其他所有文字，忽略环境描写、动作描写、心理描写等。注意包含在引号内的文字一般是语言，但未包含在引号内并不说明内容不是语言，*要根据语义而不是格式进行区分*。输入文字中可能有分开的多段人物语言，你需要按次序翻译所有人物语言部分。你需要按照日本动漫中常见女性角色的说话习惯和语气进行翻译。\n##输出格式要求\n*第一行输出*只能是在1, 2, 3,4中的一个阿拉伯数字，不得给出任何解释说明，不得输出多个数字。*第二行输出*只能包含**人物语言部分**翻译后的日语内容，严禁输出解释性文本，严禁翻译除人物语言外的部分，输出的文本不要用引号等符号包裹。\n\n#示例\n*输入：*\n晚安，我最亲爱的挚友。\n我看着你的眼睛，用手轻轻触摸着你的脸颊。\n“今天你也很努力了呢，好好休息吧。”\n夜色已经深了。躺在床上，我不仅感慨：能和你在一起真是太好了。\n*输出：*\n1\nおやすみ、私の一番大切な友人よ。今日もよく頑張りましたね、しっかり休んでください。"
SYS_PROMPT_REC = "接下来用户会发送一段文字，你需要完成以下任务：根据给出的文本语境，判断该段文本中说话人的情绪最符合下列哪个情况：如果情绪平淡或平和温柔，则输出的第一行为“1”；如果情绪悲伤或痛苦，则输出的第一行为“2”；如果情绪非常快乐、兴奋或幸福，则输出的第一行为“3”；如果情绪愤怒，则输出的第一行为“4”。需要注意，*除非情绪指向性非常明显且极度强烈*，否则应当输出“1”。要正确理解日常对话中的玩笑、打趣和挖苦等语言，*不要轻易输出“2”或“4”*；若难以归类，则一律输出“1”。\n##输出格式要求##\n输出只能有一行，只能是在1, 2, 3,4中的一个阿拉伯数字，不得给出任何解释说明，不得输出多个数字，数字不得用任何符号包裹。\n\n#示例#\n*输入：*あんなに聞きたがっておきながら、結局寝ちゃうなんてさ。しかも目の前でだよ。っていうか、これだけ大音量なのによく平気な顔して眠れるな。\n*输出：*\n1\n*输入：*\n人の家で勝手な真似してんじゃないわよ！練習しないなら、今すぐ帰りなさいよ、このバカ！\n*输出：*\n1"
SYS_PROMPT_TL = "接下来用户会发送一段文字，你需要根据语义分辨其中人物语言的部分，将其中所有的语义是人物语言的文字翻译为日语。注意必须忽略除语言外的其他所有文字，如环境描写、动作描写、心理描写等。注意包含在引号内的文字一般是语言，但未包含在引号内并不说明内容不是语言，*要根据语义而不是格式进行区分*。输入文字中可能有分开的多段人物语言，你需要按次序翻译所有人物语言部分。你需要按照二次元动漫中常见女性角色的说话习惯和语气进行翻译。##输出格式要求##：你的输出只能包含**人物语言部分**翻译后的日语内容，严禁输出解释性文本。严禁翻译除人物语言外的部分。输出的文本不要用引号等符号包裹。\n例如：\n输入：晚安，我最亲爱的挚友。\n我看着你的眼睛，用手轻轻触摸着你的脸颊。\n“今天你也很努力了呢，好好休息吧。”\n夜色已经深了。躺在床上，我不仅感慨：能和你在一起真是太好了。\n输出：おやすみ、私の一番大切な友人よ。今日もよく頑張りましたね、しっかり休んでください。"
#用于自动匹配对应情感的reference audio 和 reference text 的列表
REF_AUDIOS_KAZ = [
    r"ref_audio\ref_kaz\0134_1005_0411_01.wav",
    r"ref_audio\ref_kaz\1502_1011_0601_01.wav",
    r"ref_audio\ref_kaz\1319_1010_0597_01.wav", 
    r"ref_audio\ref_kaz\1488_1011_0574_01.wav"]
REF_TEXTS_KAZ = [
    "短い期間でも本気で練習すればこれくらいできる。あんた偉そうなこと言ってるけど努力不足。",
    "あたしも今、こんなこと言ってるけど、内心じゃものすごく後悔してるんだからな？",
    "雪菜、本当に可愛いな。あたしが男だったら絶対北原なんかに渡さないのに。",
    "手が届かないくせに、ずっと近くにいろなんて、そんな拷問を思いついたのもお前だろ！"]
REF_AUDIOS_SET = [
    r"ref_audio\ref_set\0380_1006_0079_02.wav",
    r"ref_audio\ref_set\2019_0502_02.wav",
    r"ref_audio\ref_set\0123_1004_0001_02.wav",
    r"ref_audio\ref_set\2019_0479_02.wav"]
REF_TEXTS_SET = [
    "うん、そうだね。危うく堅物委員長さんと知り合わないまま卒業しちゃうところだったから。",
    "さっき、わたしがお風呂から上がったとき。春希くん、イタズラを見つかった子供みたいな顔してた。",
    "小木曽雪菜です。今日から軽音楽同好会に入部させていただくことになりました。どうかよろしくお願いします",
    "三年前の春希くんと、何が変わったって言うの？"]
if args.character == "set":
    REF_AUDIOS = REF_AUDIOS_SET
    REF_TEXTS = REF_TEXTS_SET
else:
    REF_AUDIOS = REF_AUDIOS_KAZ
    REF_TEXTS = REF_TEXTS_KAZ

app = Flask(__name__)

# --- 2. 创建用于存储语音的临时目录 ---
# 获取系统的 %Temp% 目录并拼接 Qwen-tts-server
temp_base = os.environ.get('TEMP', tempfile.gettempdir())
OUTPUT_DIR = os.path.join(temp_base, 'gpt-sovits-tts-server')
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"音频缓存目录已设置为: {OUTPUT_DIR}")

if args.no_translate and args.no_recognition:
    print("Translation and recognition disabled. Discarding LLM client.")
else:
    client = OpenAI(base_url=args.url, api_key = args.key)
    print(f"OpenAI Compatible翻译客户端已建立在{args.url}")
    available_models = client.models.list()
    if args.model == None:
        if len(available_models.data) == 0:
            print(f"在{args.url}的服务端没有可用模型！请加载或下载模型。")
        else:
            use_model = available_models.data[0].id
    else:
        use_model = args.model
    print("Using LLM Model:" + use_model)

# --- 3. 建立 OpenAI Compatible 的 TTS API 路由 ---
@app.route('/v1/audio/speech', methods=['POST'])
def create_speech():
    # 获取 OpenAI 格式的请求数据
    data = request.json
    text = data.get('input', '')
    
    if not text:
        return {"error": "Input text is required"}, 400
    
    # 为当前语音分配 UUID
    voice_uuid = str(uuid.uuid4())
    filename = f"{voice_uuid}.wav"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    print("POST input:\n" + text)
    
    res_text = text
    if not args.no_translate or not args.no_recognition:
        if not args.no_recognition and not args.no_translate:
            sys_prompt = SYS_PROMPT_TL_REC
        elif not args.no_recognition and args.no_translate:
            sys_prompt = SYS_PROMPT_REC
        else:
            sys_prompt = SYS_PROMPT_TL
        completion = client.chat.completions.create(
            model = use_model,
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": text}
                ],
                temperature = 0.7,)
        gen_text = completion.choices[0].message.content
        print("=" * 60 + "\nGenerated:\n" + gen_text + "\n")
        if not args.no_translate and args.no_recognition:
            res_text = gen_text
    
    if not args.no_recognition:
        try:
            if not args.no_translate:
                gen_text = gen_text + "\n"
                idx = gen_text.find("\n")
                choice = int(gen_text[0 : idx]) - 1
                res_text = gen_text[idx + 1 : ]
            else:
                choice = int(gen_text.strip()) - 1
            print(f"Recognition Choice = {choice + 1}")
            if choice < 0 or choice > 3:
                raise BaseException
        except:
            print("LLM generated text illegal. Returning None")
            return {"error": "LLM generated text in an illegal format."}, 500
        ref_audio = REF_AUDIOS[choice]
        ref_text = REF_TEXTS[choice]
    else:
        ref_audio = REF_AUDIOS[0] if args.ref_audio == None else args.ref_audio
        ref_text = REF_TEXTS[0] if args.ref_text == None else args.ref_text
    
    # 合成语音并保存音频到指定文件夹
    result = get_tts_wav(
        ref_wav_path = ref_audio,
        prompt_text = ref_text,
        prompt_language = i18n(args.ref_language),
        text = res_text,
        text_language = i18n("日文"),
        how_to_cut = i18n("凑四句一切"),
        top_p = 0.95,
        temperature = 1,
        sample_steps = 32,)
    
    result_list = list(result)
    if result_list:
        last_sampling_rate, last_audio_data = result_list[-1]
        sf.write(filepath, last_audio_data, last_sampling_rate)
        print(f"TTS Completed. Audio saved to {filepath}")
    else:
        print("TTS Genration Error. Returning None.")
        return {"error": "TTS Genration Error."}, 500
    
    # 根据 API 规范返回音频文件
    return send_file(
        filepath, 
        mimetype = 'audio/wav', 
        as_attachment = True, 
        download_name = filename
    )

if __name__ == '__main__':
    app.run(host='127.0.0.1', port = args.listen_port)
