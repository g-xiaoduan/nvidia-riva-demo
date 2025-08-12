import re

import requests
import wave
import sounddevice as sd
import riva.client
import subprocess
from riva.client.proto import riva_asr_pb2
import numpy as np
import io
import IPython.display as ipd
import grpc

# ========== ASR 部分 ==========
auth = riva.client.Auth(uri="localhost:50051")
asr_service = riva.client.ASRService(auth)
tts_service = riva.client.SpeechSynthesisService(auth)
language = "en-US"
# language = "zh-CN"

def speech_to_text(content):
    config = riva.client.RecognitionConfig(
        language_code=language,
        max_alternatives=1,
        enable_automatic_punctuation=True,
        audio_channel_count=1
    )
    response = asr_service.offline_recognize(content, config)
    print(response)
    if response.results and response.results[0].alternatives:
        return response.results[0].alternatives[0].transcript
    else:
        return "[未识别到语音]"

# ========== LLM 部分 ==========
def ask_ollama(prompt):
    r = requests.post("http://localhost:11434/api/generate",
                      json={"model": "qwen3:30b-a3b",
                            "prompt": prompt,
                            "stream": False})
    return r.json()["response"]

def clean_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

def filter_tts_string(input_str):
    """
    过滤字符串中的表情符号、中文和其他非ASCII字符

    参数:
        input_str (str): 需要过滤的原始字符串

    返回:
        str: 过滤后的字符串，只包含基本ASCII字符
    """
    # 1. 移除所有emoji表情符号
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # 中文符号
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # 变体选择器
                               u"\u3030"
                               "]+", flags=re.UNICODE)

    # 2. 移除所有非ASCII字符（包括中文）
    filtered_str = emoji_pattern.sub(r'', input_str)
    filtered_str = filtered_str.encode('ascii', 'ignore').decode('ascii')

    # 3. 可选：移除多余空格和特殊标点
    filtered_str = re.sub(r'[^\w\s.,!?]', '', filtered_str)
    filtered_str = ' '.join(filtered_str.split())  # 移除多余空格

    return filtered_str
# ========== TTS 部分 ==========

def text_to_speech(text, out_file='reply.wav'):
    print("收到回复" + text)

    sample_rate_hz = 44100
    req = {
        "language_code": language,
        "encoding": riva.client.AudioEncoding.LINEAR_PCM,
        "sample_rate_hz": sample_rate_hz,
        # "voice_name": "Mandarin-CN.Female-1"
        "voice_name": "English-US.Female-1"
    }
    req["text"] = text
    resp = tts_service.synthesize(**req)
    # 转 numpy
    audio_samples = np.frombuffer(resp.audio, dtype=np.int16)
    # 直接播放
    sd.play(audio_samples, samplerate=sample_rate_hz)
    sd.wait()
    with open(out_file, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate_hz)
        wf.writeframes(resp.audio)
    return out_file
#
# # ========== 官方示例测试 ==========
# if __name__ == "__main__":
#     path = "es-US_sample.wav"
#     with io.open(path, 'rb') as fh:
#         content = fh.read()
#     ipd.Audio(path);
#     config = riva.client.RecognitionConfig(
#         language_code="en-US",
#         max_alternatives=1,
#         enable_automatic_punctuation=True,
#         audio_channel_count=1
#     )
#
#     # ASR Inference call with Recognize
#     response = asr_service.offline_recognize(content, config)
#     asr_best_transcript = response.results[0].alternatives[0].transcript
#     print("ASR Transcript without Word Boosting:", asr_best_transcript)
#
#     sample_rate_hz = 44100
#     req = {
#         "language_code": "en-US",
#         "encoding": riva.client.AudioEncoding.LINEAR_PCM,
#         "sample_rate_hz": sample_rate_hz,
#         "voice_name": "English-US.Female-1"
#     }
#
#     req["text"] = asr_best_transcript
#     resp = tts_service.synthesize(**req)
#
#     # 转 numpy
#     audio_samples = np.frombuffer(resp.audio, dtype=np.int16)
#
#     # 直接播放
#     sd.play(audio_samples, samplerate=sample_rate_hz)
#     sd.wait()
#
#     # 同时存文件
#     with wave.open("reply.wav", "wb") as wf:
#         wf.setnchannels(1)
#         wf.setsampwidth(2)
#         wf.setframerate(sample_rate_hz)
#         wf.writeframes(resp.audio)
#
#     print("TTS 音频已保存为 reply.wav")
#
# # ========== 测试代码 ==========
# if __name__ == "__main__":
#     voices = tts_service.list_voices()
#     for v in voices.voices:
#         print(f"Name: {v.name}, Lang: {v.language_code}, Gender: {v.ssml_gender}")
#
# ========== 主程序 ==========
if __name__ == "__main__":
    # 录音
    print("Speak now...")
    fs = 16000
    duration = 5
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    wav_file = 'input.wav'
    with wave.open(wav_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio.tobytes())

    with io.open(wav_file, 'rb') as fh:
        content = fh.read()
    # ipd.Audio(wav_file)
    # 语音转文字
    text = speech_to_text(content)
    print("You said:", text)

    # 问 LLM
    reply = ask_ollama(text)
    reply = clean_think(reply)
    reply = filter_tts_string(reply)
    print("LLM replied:", reply)

    # 文字转语音
    text_to_speech(reply)