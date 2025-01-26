# Autor: Aleksander Szymczyk, Andrzej Tokajuk
# Data utworzenia: 26.01.2025

import pytest
import pandas as pd
from cart_tree import CartTree
from unittest.mock import patch


@pytest.fixture
def setup_data():
    data = {
    'Wiek': [21, 21, 26, 14, 14, 29],
    'Płeć': ['m', 'm', 'm', 'k', 'k', 'k'],
    'Czas_spedzony_na_stronie_(min)': [13, 13, 14, 10, 10, 19],
    'Urządzenie': ['komputer', 'komputer', 'komputer', 'tablet', 'tablet', 'smartfon'],
    'Kliknięto_w_reklamę': ['tak', 'tak', 'tak', 'nie', 'nie', 'tak']
    }
    return pd.DataFrame(data)

@pytest.fixture
def cart_tree():
    return CartTree(max_depth=5, min_samples_split=2)


def test_split_categorical_data(setup_data, cart_tree):
    splits = cart_tree.split_categorical(setup_data, 'Urządzenie')
    assert splits == ['Urządzenie', ({'komputer'}, {'tablet', 'smartfon'}), ({'tablet'}, {'smartfon', 'komputer'}), ({'smartfon'}, {'tablet', 'komputer'})]


def test_split_categorical_data_repeats(setup_data, cart_tree):
    splits = cart_tree.split_categorical(setup_data, 'Płeć')
    assert splits == ['Płeć', ({'m'}, {'k'})]


def test_split_numerical_data(setup_data, cart_tree):
    splits = cart_tree.split_numerical(setup_data, 'Wiek')
    assert splits == [('Wiek', 17.5), ('Wiek', 23.5), ('Wiek', 27.5)]


def test_calculate_gini_categorical(setup_data, cart_tree):
    gini_value = cart_tree.calculate_gini(setup_data, ['Płeć', ({'m'}, {'k'})], 'Kliknięto_w_reklamę')
    assert gini_value == pytest.approx(0.222, rel=1e-2)


def test_calculate_gini_numerical(setup_data, cart_tree):
    gini_value = cart_tree.calculate_gini(setup_data, ('Czas_spedzony_na_stronie_(min)', 16.5), 'Kliknięto_w_reklamę')
    assert gini_value == pytest.approx(0.4)


def test_make_split(setup_data, cart_tree):
    with patch('random.choices', return_value=[('Czas_spedzony_na_stronie_(min)', 16.5), ['Płeć', ({'m'}, {'k'})]]):
        best_split = cart_tree.choose_best_split(setup_data, 'Kliknięto_w_reklamę')
    assert best_split == ['Płeć', ({'m'}, {'k'})]


# def test_build_tree(setup_data, cart_tree):
#     cart_tree.build_tree(setup_data, 'Kliknięto_w_reklamę')
#     assert cart_tree.root.feature == 'Czas_spedzony_na_stronie_(min)'
#     assert cart_tree.root.condition == 16.5
#     assert cart_tree.root.left.label == 'tak'
#     assert cart_tree.root.right.label == 'nie'

