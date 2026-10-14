"""Text-side tests (no torch needed): python -m pytest corpus/tests"""
import pytest

from corpus.hawaiian import HawaiianClassifier, assign_stress, syllabify, to_ipa, PRIMARY, SECONDARY
from corpus.normalize import normalize, paired


@pytest.mark.parametrize('n,spoken', [
    ('405', 'four oh five'), ('280', 'two eighty'), ('101', 'one oh one'), ('500', 'five hundred'),
    ('1960', 'nineteen sixty'), ('2005', 'two thousand five'), ('2000', 'two thousand'),
    ('1600', 'sixteen hundred'), ('12345', 'one twenty three forty five'),
])
def test_paired(n, spoken):
    assert paired(n) == spoken


@pytest.mark.parametrize('raw,spoken', [
    ('In 0.3 mi, turn L onto W 5th St.', 'In point three miles, turn left onto West fifth Street.'),
    ('Take I-405 N toward Sacramento.', 'Take I four oh five North toward Sacramento.'),
    ('Merge onto H-1 W.', 'Merge onto H one West.'),
    ('Turn right onto St. Charles Ave.', 'Turn right onto Saint Charles Avenue.'),
    ('Turn left onto Main St. Then keep R.', 'Turn left onto Main Street. Then keep right.'),
    ('Head south on Dr. Martin Luther King Jr. Blvd', 'Head south on Doctor Martin Luther King Junior Boulevard'),
    ('Turn onto 500 South then 700 East.', 'Turn onto five hundred South then seven hundred East.'),
    ('Take FM 1960 W for 1/2 mi', 'Take F M nineteen sixty West for a half mile'),
    ('Take Exit 23B toward US-101 S/Ventura Fwy.', 'Take Exit twenty three B toward U S one oh one South, Ventura Freeway.'),
    ('Arrive at 1 Aloha Tower Dr, Honolulu, HI 96813.', 'Arrive at one Aloha Tower Drive, Honolulu, Hawaii nine six eight one three.'),
])
def test_nav_normalize(raw, spoken):
    assert normalize(raw, nav=True) == spoken


def test_prose_keeps_letters():
    assert normalize('Vitamin E costs $3.50.') == 'Vitamin E costs three dollars and fifty cents.'


def _stress(word):
    s = syllabify(word)
    assign_stress(s)
    return ['1' if x.stress == PRIMARY else '2' if x.stress == SECONDARY else '0' for x in s]


@pytest.mark.parametrize('word,pattern', [
    ('Kalākaua', '0210'),                           # ka.ˌlā.ˈkau.a
    ('Kalanianaʻole', '0202010'),                   # ka.ˌla.ni.ˌa.na.ˈʻo.le
    ('Waikīkī', '221'),
    ('Kāneʻohe', '2010'),
    ('Hawaiʻi', '010'),
    ('Honolulu', '2010'),
])
def test_stress(word, pattern):
    assert ''.join(_stress(word)) == pattern


def test_registers():
    clf = HawaiianClassifier(['Kalanianaʻole', 'Kona'], english_vocab=['kona', 'pewaukee', 'main'], trusted=['Kona'])
    assert clf.register('Kalanianaʻole') == 'strict'
    assert clf.register('Kalanianaole') == 'local'
    assert clf.register('Kona') == 'local'
    assert clf.register('Pewaukee') is None       # CV-shaped but not Hawaiian
    assert clf.register('Main') is None
    # local reading keeps the marked form's syllable breaks: no "ao" diphthong
    assert 'aʊ' not in clf.ipa('Kalanianaole')


def test_no_unlearned_symbols():
    for w in ('Kalanianaʻole', 'Hawaiʻi', 'Kāneʻohe', 'ʻEwa'):
        ipa = to_ipa(w)
        assert not set(ipa) & set('ˈˌʔɐ'), ipa
