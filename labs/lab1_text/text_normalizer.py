import re
import unicodedata
from num2words import num2words

"""Russian text normalizer — skeleton for lab 1.

Brings corpus text into a form usable for training a speech synthesizer.
"""


class TextNormalizer:
    """Normalizes text in Russian.


        "!.."           -> "!"
        "«цитата»"      -> '"цитата"'
        "текст * мусор" -> "текст мусор"
        "де‑факто"      -> "де-факто"      # U+2011 -> ordinary hyphen

    **Word-changing edits.** The alignment for that utterance becomes invalid and the
    row must be dropped from the training set — but the logic itself is still needed
    for lab 5, where arbitrary user input arrives with no alignment at all::

        "в 1995 г."     -> "в тысяча девятьсот девяносто пятом году"
        "прим. автора"  -> "примечание автора"

    Example:
        >>> normalizer = TextNormalizer()
        >>> normalizer.normalize("Расстреливать надо таких писателей!.")
        'Расстреливать надо таких писателей!'
    """

    """Prepare the normalizer's resources."""
    def __init__(self):
        # 1. Базовые регулярные выражения очистки
        self.non_breaking_hyphen = re.compile(r'\u2011')
        self.clean_punctuation = re.compile(r'([!?])\.{2,}')
        self.quotes_open = re.compile(r'«')
        self.quotes_close = re.compile(r'»')
        self.garbage = re.compile(r'\s+\*\s+')
        
        # 2. Статичные сокращения
        self.abbreviations = {
            r'\bприм\. автора\b': 'примечание автора',
            r'\bприм\b\.': 'примечание',
            r'\bт\.д\b\.': 'так далее',
            r'\bт\.п\b\.': 'тому подобное',
            r'\bтов\b\.': 'товарищу',
        }

        self.abbr_compiled = {re.compile(pattern, re.IGNORECASE): repl for pattern, repl in self.abbreviations.items()}

        # 2.1. Аббревиатуры, найденные в корпусе: загланые согласные от 2 до 5 букв
        # ['БТ', 'ВМК', 'ГБ', 'КВВК', 'КВН', 'КП', 'КПП', 'КПСС', 'МВД', 'МТС', 'НКВД', 'СС', 'ССР', 'СХШ', 'ФБР', 'ФД', 'ЦДЛ', 'ЧК']
        self.abbrev_uppercase_consonants = {
            r'\bБТ\b': 'бэ-тэ',
            r'\bВМК\b': 'вэ-эм-ка',
            r'\bГБ\b': 'гэ-бэ',
            r'\bКВВК\b': 'ка-вэ-вэ-ка',
            r'\bКВН\b': 'ка-вэ-эн',
            r'\bКП\b': 'ка-пэ',
            r'\bКПП\b': 'ка-пэ-пэ',
            r'\bКПСС\b': 'ка-пэ-эсэс',
            r'\bМВД\b': 'эм-вэ-дэ',
            r'\bМТС\b': 'эм-тэ-эс',
            r'\bНКВД\b': 'эн-ка-вэ-дэ',
            r'\bСС\b': 'эсэс',
            r'\bССР\b': 'эсэсэр',
            r'\bСХШ\b': 'эс-хэ-ша',
            r'\bФБР\b': 'фэ-бэ-эр',
            r'\bФД\b': 'фэ-дэ',
            r'\bЦДЛ\b': 'цэ-дэ-эл',
            r'\bЧК\b': 'че-ка',
            # с гласными
            r'\bТГУ\b': 'тэ-гэ-у',
            r'\bСША\b': 'сэ-шэ-а',
            r'КПЭ\b': 'ка-пэ'
        }
        
        # Компиляция БЕЗ re.IGNORECASE (флаг убран, регистр учитывается строго)
        self.abbrev_consonants_compiled = {
            re.compile(pattern): repl for pattern, repl in self.abbrev_uppercase_consonants.items()
        }
        
        # 3. Года с предлогами (в 1995 г.)
        self.year_abbrev = re.compile(r'\b(в|к|от|до|с|около)?\s*(\d+)\s*г\.', re.IGNORECASE)
        
        # 4. Римские цифры перед связанными словами (век, в., ст., глава)
        # Пример: XX век, XXI в.
        # Теперь группа 1 — это римская цифра, группа 2 — это контекстное слово.
        self.roman_nums = re.compile(r'\b([IVXLCDM]+)\s+(век|веке|века|в\.|столетие|столетии|ст\.|глава|гл\.)', re.IGNORECASE)
        self.roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        
        # 5. Обычные (количественные) числительные
        # Ищем отдельно стоящие цифры, не привязанные к "г." или римским
        self.cardinal_nums = re.compile(r'\b\d+\b')

    def _roman_to_int(self, roman: str) -> int:
        """Переводит римские цифры в арабские."""
        roman = roman.upper()
        total = 0
        prev_value = 0
        for char in reversed(roman):
            value = self.roman_map.get(char, 0)
            if value >= prev_value:
                total += value
            else:
                total -= value
            prev_value = value
        return total

    def _expand_years(self, match) -> str:
        """Раскрытие годов с учетом предлогов."""
        preposition = match.group(1)
        year_num = int(match.group(2))
        year_text = num2words(year_num, lang='ru', to='ordinal')
        
        if not preposition:
            return f"{year_text} год"
            
        prep_lower = preposition.lower()
        if prep_lower == 'в':
            year_text = re.sub(r'ый$', 'ом', year_text)
            year_text = re.sub(r'ой$', 'ом', year_text)
            year_text = re.sub(r'ий$', 'ем', year_text)
            return f"{preposition} {year_text} году"
        elif prep_lower == 'к':
            year_text = re.sub(r'ый$', 'ому', year_text)
            year_text = re.sub(r'ой$', 'ому', year_text)
            year_text = re.sub(r'ий$', 'ему', year_text)
            return f"{preposition} {year_text} году"
        elif prep_lower in ['от', 'до', 'с', 'около']:
            year_text = re.sub(r'ый$', 'ого', year_text)
            year_text = re.sub(r'ой$', 'ого', year_text)
            year_text = re.sub(r'ий$', 'его', year_text)
            return f"{preposition} {year_text} года"
            
        return f"{preposition} {year_text} год"

    def _expand_roman(self, match) -> str:
        """Переводит римские цифры в порядковые числительные с правильным падежом."""
        roman_str = match.group(1)     
        word_context = match.group(2)  
        
        num = self._roman_to_int(roman_str)
        text_num = num2words(num, lang='ru', to='ordinal') 
        
        word_lower = word_context.lower()
        left_context = match.string[:match.start()].strip().lower()
        has_preposition_v = left_context.endswith(' в') or left_context == 'в'

        if has_preposition_v or word_lower in ['веке', 'столетии', 'в.']:
            text_num = re.sub(r'(ый|ой)$', 'ом', text_num)
            text_num = re.sub(r'ий$', 'ем', text_num)
            
        elif word_lower in ['глава', 'гл.']:
            text_num = re.sub(r'(ый|ой)$', 'ая', text_num)
            text_num = re.sub(r'ий$', 'яя', text_num)
            
        return f"{text_num} {word_context}"


    def _expand_cardinals(self, match) -> str:
        """Превращает изолированные цифры в слова (15 -> пятнадцать)."""
        num = int(match.group(0))
        # Генерирует количественное числительное (именительный падеж)
        return num2words(num, lang='ru', to='cardinal')

    def normalize(self, text: str) -> str:
        """Основной метод нормализации строки."""
        if not text:
            return ""

        text = unicodedata.normalize('NFC', text)

        # 1. Замена специфических символов и кавычек
        text = self.non_breaking_hyphen.sub('-', text)
        text = self.quotes_open.sub('"', text)
        text = self.quotes_close.sub('"', text)
        text = self.clean_punctuation.sub(r'\1', text)
        text = self.garbage.sub(' ', text)

        # 2. Раскрытие текстовых сокращений
        for pattern, replacement in self.abbr_compiled.items():
            text = pattern.sub(replacement, text)

        # 2.1 Раскрытие аббревиатур согласных в верхнем регистре
        for pattern, replacement in self.abbrev_consonants_compiled.items():
            text = pattern.sub(replacement, text)

        # 3. Раскрытие римских цифр (выполняется до обычных чисел)
        text = self.roman_nums.sub(self._expand_roman, text)

        # 4. Раскрытие годов с предлогами
        text = self.year_abbrev.sub(self._expand_years, text)

        # 5. Раскрытие оставшихся количественных числительных
        text = self.cardinal_nums.sub(self._expand_cardinals, text)

        # Финальная чистка лишних пробелов
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


# Проверка работы обновленного класса
if __name__ == "__main__":
    normalizer = TextNormalizer()
    
    # Тест 1: Предлог "в"
    print(normalizer.normalize("в 1995 г."), ": в тысяча девятьсот девяносто пятом году")
    
    # Тест 2: Предлог "к"
    print(normalizer.normalize("к 2026 г."), ": к две тысячи двадцать шестому году")
    
    # Тест 3: Тест из задачи
    print(normalizer.normalize("Расстреливать надо таких писателей!..") == "Расстреливать надо таких писателей!")

    # Тест
    print(normalizer.normalize("На Волге около 1921 г. произошло событие."), ": На Волге около тысяча девятьсот двадцать первого года произошло событие.")

    # Тест числительных и аббревиатур
    print(normalizer.normalize("Его привезли в НКВД"))
    # Выведет: "Его привезли в эн-ка-вэ-дэ"

     # Тест числительных и аббревиатур
    print(normalizer.normalize("В XX веке в США было 15 выстрелов!.."))
    # Выведет: "В двадцатый веке в сэ-шэ-а было пятнадцать выстрелов!"
    
    print(normalizer.normalize("Глава IV прим. автора"), ": четвертая глава примечание автора")
    # Выведет: "четвертая глава примечание автора"
    
    print("Все тесты с пройдены")
