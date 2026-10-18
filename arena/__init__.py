"""The model arena: a catalogue of algorithms, a fair way to train them, and a ranking that
respects statistical noise.

Importing this package is cheap on purpose. Heavy libraries (LightGBM, CatBoost) are only
imported when a model of that kind is actually built.
"""
