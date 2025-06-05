from paspailleur.pattern_structures import built_in_patterns as bip
from paspailleur.pattern_structures.pattern_structure import PatternStructure


def load_titanic(min_atom_support: int = 80) -> PatternStructure:
    """
    Return a PatternStructure for Titanic dataset.

    Parameters
    ----------
    min_atom_support: int, optional
        Minimal support of an atomic pattern. Defaults to 80 (about 10% of the data).


    Returns
    -------
    ps: PatternStructure
    """
    import pandas as pd

    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv'
    df_full = pd.read_csv(url, index_col=0)

    df_full['Embarked'] = df_full['Embarked'].map({'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'})
    df_full['Survived'] = df_full['Survived'].map(['No', 'Yes'].__getitem__)
    df_full['Known Age'] = (~df_full['Age'].isna()).map(['No', 'Yes'].__getitem__)
    df_full['Known Cabin'] = (~df_full['Cabin'].isna()).map(['No', 'Yes'].__getitem__)
    df_full = df_full.rename(columns={'Pclass': 'Passenger Class', 'SibSp': '# Siblings and Spouses', 'Parch': '# Parents and Children'})

    columns_to_consider = [
        'Survived', 'Known Age', 'Known Cabin', 'Sex', 'Embarked',  # categorical columns
        'Passenger Class', 'Age', '# Siblings and Spouses', '# Parents and Children', 'Fare',  # numerical columns
        'Name',  # textual column
    ]

    df = df_full[columns_to_consider].copy()

    class SurvivedPattern(bip.CategorySetPattern):
        Universe = ('No', 'Yes')

    class KnownAgePattern(bip.CategorySetPattern):
        Universe = ('No', 'Yes')

    class KnownCabinPattern(bip.CategorySetPattern):
        Universe = ('No', 'Yes')

    class SexPattern(bip.CategorySetPattern):
        Universe = ('female', 'male')

    class EmbarkedPattern(bip.CategorySetPattern):
        Universe = ('Southampton', 'Cherbourg', 'Queenstown')

    class PassengerClassPattern(bip.IntervalPattern):
        BoundsUniverse = (1, 2, 3)

    class AgePattern(bip.IntervalPattern):
        BoundsUniverse = (0, 20, 40, 60, 80)

    class NSiblingsPattern(bip.IntervalPattern):
        BoundsUniverse = (0, 1, 2, 8)

    class NParentsPattern(bip.IntervalPattern):
        BoundsUniverse = (0, 1, 2, 6)

    class FarePattern(bip.IntervalPattern):
        BoundsUniverse = (0, 30, 100, 300, 515)

    class NamePattern(bip.NgramSetPattern):
        ...

    class TitanicPassengerPattern(bip.CartesianPattern):
        DimensionTypes = {
            'Survived': SurvivedPattern,
            'Known Age': KnownAgePattern,
            'Known Cabin': KnownCabinPattern,
            'Sex': SexPattern,
            'Embarked': EmbarkedPattern,
            'Passenger Class': PassengerClassPattern,
            'Age': AgePattern,
            '# Siblings and Spouses': NSiblingsPattern,
            '# Parents and Children': NParentsPattern,
            'Fare': FarePattern,
            'Name': NamePattern
        }

    df['Age'] = df['Age'].fillna(AgePattern.get_min_pattern())
    df['Embarked'] = df['Embarked'].fillna(EmbarkedPattern.get_min_pattern())

    ps = PatternStructure(TitanicPassengerPattern)
    ps.fit(df.to_dict('index'), min_atom_support=min_atom_support)
    return ps
