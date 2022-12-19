import json
import os
import librosa
import soundfile as sf
from text import cleaners


def make_json(speaker):  # 提取目标人物json
    # 获取原始json数据
    f = open('./resultv32.json', 'r', encoding='utf-8')
    js = f.read()
    f.close()
    # 格式化
    json_all = json.loads("%s" % js)
    sd = {}
    for i in json_all:
        try:
            if json_all[i]["npcName"] == speaker and json_all[i]["language"] == 'CHS':
                sd[i] = {'text': json_all[i]["text"], 'file': json_all[i]["fileName"]}
        except Exception as e:
            print(f'在处理{i}时出现异常=>{e}')
    d = json.dumps(sd, sort_keys=False, indent=4, separators=(',', ':'), ensure_ascii=False)
    f = open('./' + speaker + '.json', 'w', encoding='utf-8')
    f.write(d)
    f.close()


def move_file(speaker):
    f = open(f'./{speaker}.json', 'r', encoding='utf-8')
    js = f.read()
    f.close()
    all = json.loads("%s" % js)
    k = 0
    f = 0
    j = len(all)
    for i in all:
        try:
            src_sig, sr = sf.read('./' + all[i]['file'])  # name是要 输入的wav 返回 src_sig:音频数据  sr:原采样频率
            dst_sig = librosa.resample(src_sig, sr, 22050)  # resample 入参三个 音频数据 原采样频率 和目标采样频率
            sf.write(f'./wavs/{i}.wav', dst_sig, 22050)  # 写出数据  参数三个 ：  目标地址  更改后的音频数据  目标采样数据
            k += 1
            print(f'已完成：{k},总数：{j},失败:{f}')
            break
        except Exception as e:
            print(f'出现异常。error=>{all[i]["file"]}')
            f += 1


def make_filelist(speaker):
    f = open(f'./{speaker}.json', 'r', encoding='utf-8')
    js = f.read()
    f.close()
    all = json.loads("%s" % js)
    fl = ''
    for i in all:
        try:
            fl = fl + 'wavs/' + i + '.wav|' + cleaners.chinese_cleaners2(all[i]['text']) + '\n'
        except Exception as e:
            print(f'处理{i}({all[i]["text"]})出异常=>{e}')
    ff = open('./filelists/nahida.txt', 'w', encoding='utf-8')
    ff.write(fl)
    ff.close()


make_filelist('纳西妲')