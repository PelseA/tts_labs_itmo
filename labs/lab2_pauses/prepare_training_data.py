"""Создание обучающих данных для предсказателя пауз — лабораторная работа №2.

Скрипт объединяет нормализованные метаданные из лабораторной работы №1 с результатами выравнивания слов (MFA) и записывает в файл `data/RUSLAN_pause_metadata.csv` по одной строке на каждое слово::

    id|label|label_raw|duration|is_last_word|is_pause_after|pause_duration|set

Фразы, идентификатор которых оканчивается на 0 или 5, попадают в набор `test`, остальные — в `train`.

Запуск из директории лабораторной работы::

    python prepare_training_data.py
"""
import csv
import glob
import numpy as np
import os
import pandas as pd
from praatio import textgrid
import tqdm

RUSLAN_META = '../../data/metadata_RUSLAN_22200_normalized_byPelse.csv'
ALIGN_DIR = '../../data/RUSLAN_align_byPelse_attempt_2_v2/'
RESULT_PATH = '../../data/RUSLAN_pause_metadata_byPelse.csv'

def read_text_grids(ruslan: pd.DataFrame, align_root: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:   
    """Считывание MFA-файлов TextGrid для каждого высказывания в `ruslan`.

    Args:
        ruslan: Метаданные со столбцами `id` и `nrm`.
        align_root: Директория с файлами `{id}.TextGrid`.

    Returns:
        Интервалы слов (`label`, `duration`, `id`; для пауз метка ``""``), интервалы
        фонем (те же столбцы; для пауз ``"<SIL>"``) и строку фонем, объединенных
        пробелами, для каждой строки метаданных (``""``, если файл TextGrid отсутствует).
    """
    word_docs = []
    phn_docs = []
    phoneme_sequences = []
    for wave_id, text in tqdm.tqdm(ruslan[['id', 'nrm']].values):
        try:
            tg = textgrid.openTextgrid(os.path.join(align_root, wave_id+'.TextGrid'), True)
        except:
            phoneme_sequences.append('')
            continue
        words, phones = tg.tiers

        words = words.entries
        for i in words:
            word_docs.append({'label':i.label, 'duration':i.end-i.start, 'id': wave_id})
        
        phoneme_sequences.append(' '.join([p.label for p in phones.entries]))
        phones = phones.entries
        for i in phones:
            if i.label=='':
                phn_docs.append({'label':'<SIL>', 'duration':i.end-i.start, 'id': wave_id})
            else:
                phn_docs.append({'label':i.label, 'duration':i.end-i.start, 'id': wave_id})
    word_df = pd.DataFrame(word_docs)
    phn_df = pd.DataFrame(phn_docs)
    return word_df, phn_df, phoneme_sequences

def align_text_and_textgrid(tokens: pd.DataFrame, text: str) -> pd.DataFrame:
    """Прикрепить исходную текстовую форму каждого выровненного слова как `label_raw`.

    Метки MFA написаны строчными буквами и не содержат знаков препинания. `label_raw` восстанавливает регистр
    и добавляет знаки препинания после каждого слова; текст перед первым словом
    до первого слова. Интервалы тишины получают ``"<SIL>"``.

    Args:
        токены: интервалы между словами одного высказывания, возвращаемые :func:`read_text_grids`.
        текст: Нормализованный текст одного и того же высказывания.

    Returns:
        `tokens` со столбцом `label_raw` или `tokens` без изменений, если слово не
        найдено в `text` — такие высказывания позже опускаются.
    """
    # 1. Защита от SettingWithCopyWarning: создаем чистую копию объекта
    tokens = tokens.copy()
    
    raw_tokens = []
    text_lower = text.lower()
    previous_word = -1
    
    for t, d, i in tokens[['label', 'duration', 'id']].values:
        if t == '': # Empty, SIL token
            raw_tokens.append('<SIL>')
            continue
        splits = text_lower.split(t, maxsplit=1)
        
        if len(splits) == 1: # Слово не найдено в тексте
            # 2. Оптимизация вывода: пишем краткую ошибку, НЕ выводим весь датафрейм tokens целиком
            print(f'Error aligning for ID {i}! Word "{t}" not found in text.')
            return tokens 
            
        if previous_word == -1: 
            raw_tokens.append(text[:len(splits[0]+t)].strip())
        else:
            raw_tokens[previous_word] += text[:len(splits[0])].strip()
            raw_tokens.append(text[len(splits[0]):len(splits[0] + t)])
        text_lower = splits[1]
        text = text[len(splits[0] + t):]
        previous_word = len(raw_tokens)-1
        
    if len(text) and (previous_word >= 0):
        raw_tokens[previous_word] += text.strip()
        
    tokens['label_raw'] = raw_tokens
    return tokens

def add_pause_labels(align: pd.DataFrame) -> pd.DataFrame:
    """Отметить слова, после которых следует пауза, и указать длительность паузы.

    Добавляет столбцы `is_last_word`, `is_pause_after` и `pause_duration` (в секундах). Для строк,
    соответствующих тишине, устанавливаются значения ``False`` / ``0.0``; тишина перед первым словом игнорируется.

    Args:
        align: Упорядоченные временные интервалы слов одного высказывания.

    Returns:
        `align` с тремя дополнительными столбцами меток.
    """
    align = align.copy() # Защита от SettingWithCopyWarning: создаем чистую копию объекта
    is_last_word = []
    pause_after = []
    pause_duration = []
    last_word = -1
    for idx, (label, dur) in enumerate(align[['label', 'duration']].values):
        if label == '':
            if last_word>=0:
                pause_after[last_word] = True
                pause_duration[last_word] = dur
            pause_after.append(False)
            pause_duration.append(0.)
            is_last_word.append(False)
        else:
            pause_after.append(False)
            pause_duration.append(0.)
            is_last_word.append(False)
            last_word = idx
    if last_word>=0:
        is_last_word[last_word] = True
    align['is_last_word'] = is_last_word
    align['is_pause_after'] = pause_after
    align['pause_duration'] = pause_duration 
    return align
    

def main() -> None:
    """Read metadata and alignments, label pauses, split into train/test and save."""
    ruslan =pd.read_csv(f'{RUSLAN_META}', sep='|', names=['id', 'raw', 'nrm'], quoting=csv.QUOTE_NONE)
    word_df, _, _ = read_text_grids(ruslan, ALIGN_DIR)

    
    #Additional normalization to ensure good alignment
    word_df.label = word_df.label.str.replace('‐', '-') # differen hyphen symbols
    word_df.label = word_df.label.str.replace('‑', '-')
    ruslan.nrm = ruslan.nrm.str.replace('‐', '-')
    ruslan.nrm = ruslan.nrm.str.replace('‑', '-')
    ruslan.nrm = ruslan.nrm.str.replace('’', "'") # Mfa replaces ’ by '
    ruslan.nrm = ruslan.nrm.str.replace('\\((.*?)\\)', '[bracketed]', regex=True) # Mfa spells () and <> text as [bracketed]
    ruslan.nrm = ruslan.nrm.str.replace('\\<(.*?)\\>', '[bracketed]', regex=True)

    #Aligning TextGrid-normalised tokens and text, including punctuation
    aligns = []
    for n, i in tqdm.tqdm(ruslan[['nrm', 'id']].values):
        tokens = word_df[word_df.id==i]
        aligns.append(align_text_and_textgrid(tokens, n))

    # --- БЛОК ВИЗУАЛИЗАЦИИ НЕСКОЛЬКИХ СТРОК ALIGNS ---
    print("\n=== Пример исходных датафреймов aligns до обработки пауз ===")
    for idx, df_example in enumerate(aligns[:2]): # Смотрим на первые 2 датафрейма
        print(f"\n[Датафрейм #{idx + 1}] Всего строк: {len(df_example)}")
        # Метод head(5) покажет первые 5 строк каждого датафрейма
        try:
            # Если вы в Jupyter/Colab, display() сделает красивую интерактивную табличку
            display(df_example.head(10)) 
        except NameError:
            # Если запускаете просто скриптом .py в терминале, сработает print()
            print(df_example.head(10))
    print("============================================================\n")
    # -------------------------------------------------

    # Creating labels for pause predictor training
    aligns = [add_pause_labels(a) for a in aligns]

    # Creating dataframe
    pause_df = pd.concat(aligns)
    
    pause_df = pause_df[pause_df.label_raw!='<SIL>'] # Removing pause tokens -- all information about pauses is in word tokens now
    pause_df = pause_df[pause_df.label_raw.notna()] # Removing all the files, who failed to be aligned
    pause_df = pause_df.reset_index(drop=True) # Reseting the index after filtering
    
    pause_df.is_last_word = pause_df.is_last_word.astype(int) # Casting bool fields into int
    pause_df.is_pause_after = pause_df.is_pause_after.astype(int) #


    # Splitting into train and test deterministically: All the files with index, ending with 0 or 5 is considered test
    pause_df['set'] = 'train'
    pause_df.loc[pause_df.id.str.split('_', expand=True)[0].astype(int)%5==0, 'set'] = 'test'
    pause_df.to_csv(f'{RESULT_PATH}', sep='|', index=False, header=True, quoting=csv.QUOTE_NONE)
    
if __name__=='__main__':
    main()