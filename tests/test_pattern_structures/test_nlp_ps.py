from paspailleur.pattern_structures.nlp_ps import SynonymPS, AntonymPS


#####################
#  Test SynonymPS  ##
#####################

def test_synonyms_init_():
    ps = SynonymPS(n_synonyms=5)
    assert ps.n_synonyms == 5


def test_get_synonyms():
    ps = SynonymPS(n_synonyms=5)
    assert ps.get_synonyms('world') == {'creation', 'cosmos', 'universe', 'world', 'existence'}


def test_synonyms_preprocess_data():
    ps = SynonymPS(n_synonyms=5)
    assert next(ps.preprocess_data(['hello world'])) == ps.get_synonyms('hello') | ps.get_synonyms('world')

#####################
#  Test AntonymPS  ##
#####################


def test_antonyms_init_():
    ps = AntonymPS(n_antonyms=5)
    assert ps.n_antonyms == 5


def test_get_antonyms():
    ps = AntonymPS(n_antonyms=5)
    assert ps.get_antonyms('good') == {'evilness', 'evil', 'ill', 'bad', 'badness'}


def test_antonyms_preprocess_data():
    ps = AntonymPS(n_antonyms=5)
    assert next(ps.preprocess_data(['good boy'])) == ps.get_antonyms('good') | ps.get_antonyms('boy')
