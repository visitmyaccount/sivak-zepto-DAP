# Titanic EDA Results

## Load and profile

- Source used in this run: committed CSV fallback
- Shape: 891 rows x 15 columns

### `df.info()`

```text
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 15 columns):
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   survived     891 non-null    int64
 1   pclass       891 non-null    int64
 2   sex          891 non-null    object
 3   age          714 non-null    float64
 4   sibsp        891 non-null    int64
 5   parch        891 non-null    int64
 6   fare         891 non-null    float64
 7   embarked     889 non-null    object
 8   class        891 non-null    object
 9   who          891 non-null    object
 10  adult_male   891 non-null    bool
 11  deck         203 non-null    object
 12  embark_town  889 non-null    object
 13  alive        891 non-null    object
 14  alone        891 non-null    bool
dtypes: bool(2), float64(2), int64(4), object(7)
memory usage: 92.4+ KB
```

### `df.describe(include="all")`

```text
             count unique          top freq       mean        std   min     25%      50%   75%       max
survived     891.0    NaN          NaN  NaN   0.383838   0.486592   0.0     0.0      0.0   1.0       1.0
pclass       891.0    NaN          NaN  NaN   2.308642   0.836071   1.0     2.0      3.0   3.0       3.0
sex            891      2         male  577        NaN        NaN   NaN     NaN      NaN   NaN       NaN
age          714.0    NaN          NaN  NaN  29.699118  14.526497  0.42  20.125     28.0  38.0      80.0
sibsp        891.0    NaN          NaN  NaN   0.523008   1.102743   0.0     0.0      0.0   1.0       8.0
parch        891.0    NaN          NaN  NaN   0.381594   0.806057   0.0     0.0      0.0   0.0       6.0
fare         891.0    NaN          NaN  NaN  32.204208  49.693429   0.0  7.9104  14.4542  31.0  512.3292
embarked       889      3            S  644        NaN        NaN   NaN     NaN      NaN   NaN       NaN
class          891      3        Third  491        NaN        NaN   NaN     NaN      NaN   NaN       NaN
who            891      3          man  537        NaN        NaN   NaN     NaN      NaN   NaN       NaN
adult_male     891      2         True  537        NaN        NaN   NaN     NaN      NaN   NaN       NaN
deck           203      7            C   59        NaN        NaN   NaN     NaN      NaN   NaN       NaN
embark_town    889      3  Southampton  644        NaN        NaN   NaN     NaN      NaN   NaN       NaN
alive          891      2           no  549        NaN        NaN   NaN     NaN      NaN   NaN       NaN
alone          891      2         True  537        NaN        NaN   NaN     NaN      NaN   NaN       NaN
```

## Missing values before cleaning

- `age`: 19.87%
- `embarked`: 0.22%
- `deck`: 77.22%
- `embark_town`: 0.22%

### Decisions

- `deck`: 77.22% missing, so the column was dropped because filling more than 30% would be unreliable.
- `age`: 19.87% missing, so values were filled with the median (28.00) under the 5%-30% rule.
- `embarked`: 0.22% missing, so affected rows were dropped under the under-5% rule.
- `embark_town`: 0.22% missing, so affected rows were dropped under the under-5% rule.

The cleaned EDA copy contains 889 rows and 14 columns.

## Univariate findings

- Age IQR limits: 2.50 to 54.50; outliers: **65**.
- Fare IQR limits: -26.76 to 65.66; outliers: **114**.
- Fare mean: 32.097; median: 14.454; mode: 8.050.
- Fare is **right-skewed** because mean > median > mode. A small number of very high fares pull the mean upward.

## Survival rates

### By sex

- female: 0.740
- male: 0.189

### By passenger class

- Class 1: 0.626
- Class 2: 0.473
- Class 3: 0.242

### By sex and passenger class

- female, class 1: 0.967
- female, class 2: 0.921
- female, class 3: 0.500
- male, class 1: 0.369
- male, class 2: 0.157
- male, class 3: 0.135

## Correlation matrix

The matrix uses exactly `survived`, `pclass`, `age`, `sibsp`, `parch`, and `fare`. The derived boolean columns `adult_male` and `alone` are excluded.

```text
          survived  pclass    age  sibsp  parch   fare
survived     1.000  -0.336 -0.070 -0.034  0.083  0.255
pclass      -0.336   1.000 -0.337  0.082  0.017 -0.548
age         -0.070  -0.337  1.000 -0.233 -0.171  0.094
sibsp       -0.034   0.082 -0.233  1.000  0.415  0.161
parch        0.083   0.017 -0.171  0.415  1.000  0.218
fare         0.255  -0.548  0.094  0.161  0.218  1.000
```

The two strongest absolute off-diagonal pairs are:

- `pclass` and `fare`: -0.548
- `sibsp` and `parch`: 0.415

The class/fare relationship reflects the much higher fares paid in the lower-numbered passenger classes. The positive `sibsp`/`parch` relationship suggests that passengers travelling with siblings or a spouse often also travelled with parents or children as part of a family group.

## Four-chart data story

1. **Sex and class:** First-class female survival was 96.7%, compared with 13.5% for third-class males. Sex is the clearest separation, while passenger class adds another strong difference inside each group.
2. **Age and survival:** Median age was 28.0 for survivors and 28.0 for non-survivors. The boxes overlap greatly, so age alone does not separate the outcomes as clearly as sex and class.
3. **Age, fare, and survival:** Survivors paid a median fare of 26.00, versus 10.50 for non-survivors. Higher-fare observations contain more survivors, which agrees with the class pattern, although both outcomes occur across the age range.
4. **Family size:** Among family sizes with at least 10 passengers, size 4 had the highest survival rate at 72.4%. Travelling completely alone or in very large groups was less favorable than travelling with a small family group.

## Exploratory standardization check

This check uses the full cleaned EDA copy only and is not passed into the modeling pipeline.

```text
      before_mean  before_std  after_mean  after_std
age     29.315152   12.984932    0.000000   1.000000
fare    32.096681   49.697504    0.000000   1.000000
```

After z-score standardization, age and fare both have means close to 0 and standard deviations of 1.
