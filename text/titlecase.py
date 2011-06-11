import re


LOWERCASE_WORDS = ['a',   'an',   'and',  'as',   'at',   'by',   'de',
                   'et',  'for',  'from', 'in',   'into', 'of',   'on',
                   'or',  'the',  'their','then', 'to',   'with', "'s"]

UPPERCASE_WORDS = ['SIAM', 'AICHE', 'ISIJ', 'IEEE', 'II', 'III', 'AIME', 'EPL']

FIRST_WORD = re.compile(r'^\b\w+')
ALL_WORDS = re.compile(r'\b\w+')


def titlecase(title_string):
    """Return string that is converted to title case.

    Unlike the `title` method of the `str` built-in, this function converts
    some words to lowercase/uppercase.

    Articles, prepositions and conjunctions are left as lowercase; these words
    are set by LOWERCASE_WORDS. Similarly, UPPERCASE_WORDS lists words that
    should be uppercase.

    Note that words with apostrophes are improperly title cased, because
    `str.title` doesn't recognize apostrophes are part of the
    """
    title_string = ALL_WORDS.sub(_titlecase_all, title_string)
    title_string = FIRST_WORD.sub(_titlecase_first, title_string)
    return title_string

def _titlecase_all(match):
    word = match.group(0)
    if word.lower() in LOWERCASE_WORDS:
        return word.lower()
    elif word.upper() in UPPERCASE_WORDS:
        return word.upper()
    else:
        return word.title()

def _titlecase_first(match):
    word = match.group(0)
    if word.upper() in UPPERCASE_WORDS:
        return word.upper()
    else:
        return word.title()

if __name__ == '__main__':
    import nose
    nose.runmodule('tests/test_titlecase.py')
