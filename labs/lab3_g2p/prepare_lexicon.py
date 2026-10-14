"""Build the lab 3 training data from the MFA alignment — lab 3.

Reads the TextGrids downloaded in lab 2 and writes two files.

`data/tokens_phones.csv` — **the training table**, one row per aligned token, in sentence
order. Same shape as lab 2's `RUSLAN_pause_metadata.csv`, so the two labs read alike::

    id|token_id|token|phones|set

`data/lexicon.csv` — the pronunciation dictionary, aggregated over the training fold::

    word|phones|count

The training table is the primary artifact: it keeps the sentence grouping, so a
word-level model, a sentence-level model and anything contextual can all be built from it.
`lexicon.csv` is a convenience summary for lookup and for word-level training, and it
necessarily throws the context away.

Run from the lab directory::

    python prepare_lexicon.py
"""
import csv
import glob
import os
import pandas as pd
import unicodedata
import shutil
from collections import Counter, defaultdict
import urllib.request

from praatio import textgrid
import tqdm

ALIGN_DIR = '../../data/RUSLAN_align_byPelse_attempt_2_v2/'
NORMALIZED_META = '../../data/metadata_RUSLAN_22200_normalized_byPelse.csv'
EXISTING_DICT_PATH = '../../models/mfa/russian_mfa.dict' # у меня уже скачан словарь
# Пути к файлам, которые скрипт создаст автоматически
TOKENS_PATH = 'data/tokens_phones.csv'
TOKENS_SIL_PATH = 'data/tokens_phones_sil.csv'
LEXICON_PATH = 'data/lexicon.csv'
DICT_PATH = 'data/russian_mfa.dict'

SIL = '<SIL>'
LENGTH_MARK = 'ː'
TEST_EVERY = 5  # lab 2's split rule: test iff the utterance number divides by 5


def phones_of_token(phones: list[tuple[float, float, str]],
                    start: float, end: float) -> list[str]:
    """Собирает фонемы, относящиеся к одному слову по его временным границам."""
    """Collect the phones belonging to one word interval.

    Membership is decided by the phone's midpoint rather than its boundaries: word and
    phone tiers are written independently and a shared boundary can differ in the last
    decimal.

    Args:
        phones: `(start, end, label)` of every phone interval of the utterance.
        start: Word interval start, seconds.
        end: Word interval end, seconds.

    Returns:
        Phone labels in time order, NFC-normalized with the length mark collapsed to match
        `configs/phones_mfa.txt`. An empty label (silence) becomes ``"<SIL>"``.
    """
    out = []
    for phone_start, phone_end, label in phones:
        if start <= (phone_start + phone_end) / 2 < end:
            label = unicodedata.normalize('NFC', label) if label else SIL
            out.append(label.replace(LENGTH_MARK, ''))
    return out

def read_kept_ids(path: str) -> set[str] | None:
    """Читает ID предложений, успешно прошедших нормализацию.
    Идентификаторы высказываний, прошедшие фильтр `lab 1` (или `None`, если файл отсутствует).

    Выравнивание было построено на 'сыром' тексте, поэтому оно содержит высказывания,
    которые нормализатор бы отсеял, — в том числе три аббревиатуры, написанные заглавными буквами,
    для которых `mfa g2p` сгенерировал транскрипции (`кпсс` → `k r ə f ɨ s̪`).
    Ограничение результатов от `lab 1` позволяет их исключить.
    """
    if not os.path.isfile(path):
        print(f'{path} не найден: обрабатываем все доступные файлы с разметкой')
        return None
    with open(path, encoding='utf-8') as fh:
        # Исправлено: берем первый элемент (ID) и чистим его от пробелов/переносов
        return {line.split('|', 1)[0].strip() for line in fh if line.strip()}

def read_alignment(align_root: str, keep: set[str] | None, keep_sil=False) -> list[dict]:
    """Читает каждый TextGrid из `align_root` и преобразует в записи, соответствующие отдельным токенам.
    
    Args:
        align_root: Directory with `{id}.TextGrid` files.
        keep: Utterance ids to keep, or None for all.
        keep_sil: retains `<SIL>` tokens if true, else drops them
    
    Returns:
        One dict per aligned token with keys `id`, `token_id`, `token`, `phones`, `set`.
    """
    rows = []
    all_paths = sorted(glob.glob(os.path.join(align_root, '*.TextGrid')))
    
    if not all_paths:
        print(f"Внимание! В папке {align_root} не найдено .TextGrid файлов. Проверьте путь.")
        return rows

    # Первый проход: ищем и запоминаем ID предложений, содержащих 'spn'
    bad_utterances = set()
    for path in all_paths:
        utterance_id = os.path.basename(path)[:-len('.TextGrid')]
        try:
            tg = textgrid.openTextgrid(path, True)
            _, phone_tier = tg.tiers
            if any(p.label == 'spn' for p in phone_tier.entries):
                bad_utterances.add(utterance_id)
        except Exception:
            continue
    
    print(f"Найдено и будет удалено предложений с 'spn': {len(bad_utterances)}")

    # Второй проход: основной сбор данных
    for path in tqdm.tqdm(all_paths, desc="Парсинг TextGrid"):
        utterance_id = os.path.basename(path)[:-len('.TextGrid')]
        
        if (keep is not None and utterance_id not in keep) or (utterance_id in bad_utterances):
            continue
            
        try:
            tg = textgrid.openTextgrid(path, True)
        except Exception as exc:
            print(f'Не удалось прочитать {utterance_id}: {exc}')
            continue
            
        word_tier, phone_tier = tg.tiers
        phones = [(p.start, p.end, p.label) for p in phone_tier.entries]
        
        try:
            num_id = int(utterance_id.split('_')[0])
            fold = 'test' if num_id % TEST_EVERY == 0 else 'train'
        except (ValueError, IndexError):
            fold = 'train'

        token_id = 0
        for entry in word_tier.entries:
            token_label = entry.label.strip()
            
            if token_label == '':
                # Если это пауза и нам НЕ нужны паузы — пропускаем
                if not keep_sil:
                    continue
                # Если паузы НУЖНЫ, даем токену имя SIL, чтобы его не удалил фильтр пустых строк
                current_token = SIL
                current_phones = SIL
            else:
                current_token = unicodedata.normalize('NFC', token_label).lower()
                current_phones = ' '.join(phones_of_token(phones, entry.start, entry.end))
                
            rows.append({
                'id': utterance_id,
                'token_id': token_id,
                'token': current_token, 
                'phones': current_phones,
                'set': fold,
            })
            token_id += 1
            
    return rows


def build_lexicon(rows: list[dict]) -> dict[str, Counter]:
    """ Собирает частотный словарь произношений строго по train-выборке.
    Подсчитывайте количество вариантов произношения для каждого токена только на обучающей выборке.

    Фонема `spn` обозначает слово, для которого у системы выравнивания не нашлось варианта произношения: 
    этот интервал имеет длительность, но не имеет реальной транскрипции, поэтому соответствующая строка не несет лексической информации.

    Если выполнять этот подсчет для каждого фолда, то тестовые слова попадут в словарь, а итоговый показатель PER потеряет смысл.
    """
    lexicon: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        if row['set'] != 'train' or row['token'] == '':
            continue
        if not row['phones'] or row['phones'] == SIL:
            continue
            
        # Возвращаем split() и собираем обратно, как в первоисточнике
        phones_list = row['phones'].split()
        phones_normalized = ' '.join(phones_list)
        
        lexicon[row['token']][phones_normalized] += 1
    return lexicon

def main() -> None:
    """Read the alignment, write the training table and the lexicon."""
    # 1. Создаем папку для артефактов
    os.makedirs('data', exist_ok=True)
    
    # 2. Копируем локальный словарь в папку data, если его там ещё нет
    if os.path.exists(EXISTING_DICT_PATH):
        if not os.path.exists(DICT_PATH):
            print(f"Копирование словаря из {EXISTING_DICT_PATH} в {DICT_PATH}...")
            shutil.copy(EXISTING_DICT_PATH, DICT_PATH)
        else:
            print("Словарь MFA уже находится в папке data.")
    else:
        print(f"Внимание! Файл по пути {EXISTING_DICT_PATH} не найден. Проверьте путь к модели.")

    # 3. Читаем валидные ID из ЛР1
    kept_ids = read_kept_ids(NORMALIZED_META)

    # 4. Генерируем данные без пауз (для транскриптора)
    print("\nСбор данных без пауз...")
    rows_no_sil = read_alignment(ALIGN_DIR, kept_ids, keep_sil=False)
    
    # 5. Генерируем данные с паузами (для акустической модели)
    print("\nСбор данных с паузами...")
    rows_with_sil = read_alignment(ALIGN_DIR, kept_ids, keep_sil=True)
    
    # Сохраняем таблицы токенов
    df_no_sil = pd.DataFrame(rows_no_sil)
    df_with_sil = pd.DataFrame(rows_with_sil)
    
    # Исключаем пустые токены
    df_no_sil = df_no_sil[df_no_sil['token'].str.strip() != '']
    df_with_sil = df_with_sil[df_with_sil['token'].str.strip() != '']

    # Исключаем строки, где токен РЕАЛЬНО пустой (не равен ни слову, ни тегу <SIL>)
    df_no_sil = df_no_sil[df_no_sil['token'].str.strip() != '']
    df_with_sil = df_with_sil[df_with_sil['token'].str.strip() != '']

    # Дополнительная страховка: если вдруг в phones_no_sil просочились пустые значения
    df_no_sil = df_no_sil[df_no_sil['phones'].str.strip() != '']

    df_no_sil.to_csv(TOKENS_PATH, encoding='utf-8', sep='|', index=False)
    df_with_csv = df_with_sil.to_csv(TOKENS_SIL_PATH, encoding='utf-8', sep='|', index=False)

    print(f'\nСохранено токенов без пауз: {len(df_no_sil)}')
    print(f'Сохранено токенов с паузами: {len(df_with_sil)}')

    # 6. Строим и сохраняем лексикон по требованиям ТЗ (token|phones|frequency)
    lexicon = build_lexicon(rows_no_sil)
    
    with open(LEXICON_PATH, 'w', encoding='utf-8', newline='') as fh:
        writer = csv.writer(fh, delimiter='|', quoting=csv.QUOTE_NONE)
        writer.writerow(['token', 'phones', 'frequency']) # Заголовки строго по ТЗ
        for token in sorted(lexicon):
            for phones, count in lexicon[token].most_common():
                writer.writerow([token, phones, count])
                
    variants = sum(1 for counts in lexicon.values() if len(counts) > 1)
    print(f'Словарь собран: уникальных слов в train: {len(lexicon)}, из них омографов с разными произношениями: {variants}')

if __name__ == '__main__':
    main()
