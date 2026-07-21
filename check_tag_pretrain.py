import json
import re
import os
from collections import Counter


corrupt_emo = ["confirm", "cal calm", "calming", "cal", "calmm", "fe fearful", "calmed", "calmb", "calml", "calms", "calmp", "calmd", "calmy", "joyful", "caln", "apologetic", "hopeful", "wonder", "cal m", "feared", "calme", "calм", "cheerful", "sorrowful", "nervous", "wistful", "feel", "sa", "feel sad", "feel amazed", "furious", "reflective", "calrm", "calr", "calh" ,"skeptical", "calmod", "cal下m", "calmo", "craving2", "romance2", "calm2", "relief", "nostalgia2", "entrancement2", "admiration3", "satisfaction3", "relief3", "empathic pain", "reserved2", "aesthetic appreciation", "aesthetic appreciation1", "boredom3", "amusement3", "empathic pain1", "entrancement3", "amusement", "reserved1", "awkwardness3", "interest", "craving3", "adoration3", "nostalgia3", "admiration", "entrancement", "satisfaction", "romance3",  "aesthetic appreciation2", "shyness1", "nostalgia", "aesthetic appreciation3", "empathic pain2", "calm3", "awkwardness", "craving", "shyness3", "adoration", "shyness2", "romance", "empathic pain3", "恐惧", "adoration1", "adoration2", "adoration3"]


emo_map = {
    'anger3': 'angry3', 'anger2': 'angry2', 'anger1': 'angry1', 'anger': 'angry',
    'fear3': 'fearful3', 'fear2': 'fearful2', 'fear1': 'fearful1', 'fear': 'fearful',
    'horror3': 'fearful3', 'horror2': 'fearful2', 'horror1': 'fearful1', 'horror': 'fearful',
    'sadness3': 'sad3', 'sadness2': 'sad2', 'sadness1': 'sad1', 'sadness': 'sad',
    'calmness3': 'calm3', 'calmness2': 'calm2', 'calmness1': 'calm1', 'calmness': 'calm', "calmed": "calm",
    'excitement3': 'excited3', 'excitement2': 'excited2', 'excitement1': 'excited1', 'excitement': 'excited',
    'surprise3': 'surprised3', 'surprise2': 'surprised2', 'surprise1': 'surprised1', 'surprise': 'surprised',
    'disgust3': 'disgusted3', 'disgust2': 'disgusted2', 'disgust1': 'disgusted1', 'disgust': 'disgusted',
}



def extract_tags(text):

    """提取所有类型的tag"""
    tags = []
    
    # 【】中文方括号
    zh_tags = re.findall(r'【(.*?)】', text)
    for tag in zh_tags:
        tags.append(('【】', tag))
    
    # [] 英文方括号
    en_tags = re.findall(r'\[(.*?)\]', text)
    for tag in en_tags:
        tags.append(('[]', tag))
    
    # <> 尖括号（包含成对的开闭标签）
    angle_tags = re.findall(r'<(.*?)>', text)
    for tag in angle_tags:
        tags.append(('<>', tag))
    
    return tags




def get_all_tags_set(text):

    """提取文本中所有tag的内容集合（用于过滤判断）"""
    # 【】中文方括号
    zh_tags = set(re.findall(r'【(.*?)】', text))
    # [] 英文方括号
    en_tags = set(re.findall(r'\[(.*?)\]', text))
    # <> 尖括号，只取标签名（去掉 / 前缀和属性）
    angle_tags = set(re.findall(r'<(.*?)>', text))
    # 标准化尖括号标签名（去掉斜杠和属性）
    angle_tag_names = set()
    for tag in angle_tags:
        tag_name = tag.lstrip('/').split()[0]  # 去掉 / 和属性
        angle_tag_names.add(tag_name)
    
    return zh_tags, en_tags, angle_tag_names




def should_discard(text):

    """
    判断样本是否应该丢弃：
    规则：只有 [breath] 和 <stress></stress> 标签的样本丢弃
    即：除了 breath 和 stress 之外没有其他任何有效标签
    """
    zh_tags, en_tags, angle_tag_names = get_all_tags_set(text)
    
    
    # 允许保留的标签白名单
    allowed_en_tags = {'breath', 'hold'}
    allowed_angle_tags = {'stress', '/stress'}  # 兼容写法
    allowed_angle_names = {'stress'}             # 标准化后的名称
    
    # 检查是否存在白名单之外的标签
    has_other_en = len(en_tags - allowed_en_tags) > 0  # 有除breath之外的英文标签
    has_other_angle = len(angle_tag_names - allowed_angle_names) > 0  # 有除stress之外的尖括号标签
    
    has_meaningful_tag = has_other_en or has_other_angle
    
    # 如果没有任何有意义的标签（只剩 breath / stress 或没有标签）-> 丢弃
    return not has_meaningful_tag




def main(jsonl_paths):

    # 分类统计
    zh_counter = Counter()    # 【】
    en_counter = Counter()    # []
    angle_counter = Counter() # <>
    
    total = 0
    discarded_only_breath_stress = 0  # 只有breath/stress标签被丢弃的数量
    emo_discarded = Counter()                  # neutral样本总数
    emo_limit = 6000               # neutral最多保留条数


    kept_entries = []  # 最终保留的条目


    for jsonl_path, jsonl_type in jsonl_paths:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
    
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                entry = json.loads(line)
    
                # 用于处理Emotion数据
                if jsonl_type == "emotion":
                    if entry.get('emo_degree', "") == "unknown":
                        degree = ""
                    elif entry.get('emo_degree', "") == "low":
                        degree = "1"
                    elif entry.get('emo_degree', "") == "high":
                        degree = "3"
                    else:
                        degree = ""
                        # print(entry['emo_degree'])
                    entry['text'] = "【" + entry['emotion'].lower() + degree + "】" + entry['txt_raw']
                # 用于处理副语言数据
                elif jsonl_type == "paralinguistic":
                    entry['text'] = entry['gt_text']

                # 情感标签归一化
                for key in emo_map:
                    entry['text'] = entry['text'].replace(f"【{key}】", f"【{emo_map[key]}】")

                # 去掉breath
                entry['text'] = entry['text'].replace("[breath]", "")
                    
                    
                text = entry.get('text', '')
                total += 1
    
    
                # ── 规则1：只有 [breath] 和 <stress> 标签的样本丢弃 ──
                if jsonl_type != "emotion":
                    if should_discard(text):
                        discarded_only_breath_stress += 1
                        continue

                # ── 规则2：情绪类别平衡 ──
                text = entry.get('text', '')
                tags = extract_tags(text)
                keep_flag = True
                for tag_type, tag_content in tags:
                        
                    if tag_type == '【】':
                        # 丢弃一些syntax错误的emotion
                        if tag_content in corrupt_emo:
                            continue
                        if zh_counter[tag_content] > emo_limit:
                            emo_discarded[tag_content] += 1
                            keep_flag = False
                            continue
                        zh_counter[tag_content] += 1
                    elif tag_type == '[]':
                        en_counter[tag_content] += 1
                    elif tag_type == '<>':
                        angle_counter[tag_content] += 1
                        
                if keep_flag:
                    kept_entries.append(entry)

    # ── 打印统计结果 ──

    print(f"\n{'='*50}")
    print(f"原始总条目数:                {total}")
    print(f"丢弃（仅breath/stress标签）: {discarded_only_breath_stress}")
    print(f"丢弃过多情绪:            {emo_discarded}")
    print(f"最终保留条目数:              {len(kept_entries)}")
    print(f"{'='*50}")


    print(f"\n📌 【】中文方括号 Tag 分布（共{sum(zh_counter.values())}个）:")

    print(f"{'-'*40}")
    for tag, count in zh_counter.most_common():
        print(f"  【{tag}】: {count}")


    print(f"\n📌 [] 英文方括号 Tag 分布（共{sum(en_counter.values())}个）:")

    print(f"{'-'*40}")
    for tag, count in en_counter.most_common():
        print(f"  [{tag}]: {count}")


    print(f"\n📌 <> 尖括号 Tag 分布（共{sum(angle_counter.values())}个）:")

    print(f"{'-'*40}")
    for tag, count in angle_counter.most_common():
        print(f"  <{tag}>: {count}")


    # # ── 可选：将过滤后的数据写出 ──
    # output_path = "./balanced_data_pretrain_wo_breath.jsonl"
    # with open(output_path, 'w', encoding='utf-8') as f_out:
    #     for entry in kept_entries:
    #         if 'wav_path' in entry:
    #             entry['audio'] = "/home/ma-user/work/dataset/csh_bj/instruct_tts/" + entry['wav_path'] if not "/home/ma-user/" in entry["wav_path"] else entry["wav_path"]
    #         elif "audio" in entry:
    #             entry['audio'] = "/home/ma-user/work/dataset/csh_bj/instruct_tts/" + entry['audio'] if not "/home/ma-user/" in entry["audio"] else entry["audio"]
    #         else:
    #             entry['audio'] = "/home/ma-user/work/dataset/csh_bj/BB03/" + entry['audio_path']
    #         entry['ref_audio'] = entry['audio']
    #         if '/home/ma-user/' not in entry['audio']:
    #             print(entry.keys())
    #         if not os.path.exists(entry['audio']):
    #             continue
    #         # del entry['audio_path']
    #         f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
    # print(f"\n✅ 过滤后数据已保存至: {output_path}")





if __name__ == '__main__':
    
    import sys
    jsonl_paths = [
            ('/home/ma-user/work/dataset/csh_bj/BB03/BB03_51h_cleaned.jsonl', "bb03"),
            # ('/home/ma-user/work/dataset/csh_bj/jiangziyue/Qwen3-TTS/script/data/minimax_level.jsonl', "emotion"),
            # ('/home/ma-user/work/dataset/csh_bj/jiangziyue/Qwen3-TTS/script/data/qwen3.jsonl', "emotion"),
            # ('/home/ma-user/work/dataset/csh_bj/jiangziyue/Qwen3-TTS/script/data/aopeng_bj4_obs_0.jsonl', 'aopeng'),
            # ('/home/ma-user/work/dataset/csh_bj/jiangziyue/Qwen3-TTS/script/data/emilia_nv_selected.jsonl', "paralinguistic"),
         ]

    main(jsonl_paths)
