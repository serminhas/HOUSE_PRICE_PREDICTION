# import pickle
# from helpers.data_prep import *
# from helpers.eda import *
# from helpers.helpers import *
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    classification_report, mean_squared_error, RocCurveDisplay
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor  # cok degiskenli aykiri deger yakalama yontemi
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, \
    RobustScaler  # standartlastirma, donusturme fonksiyonlari
# Siniflandirma Algoritmalari
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import warnings

warnings.simplefilter(action="ignore")  # uyari mesajlarini engeller

pd.set_option("display.max_columns", None)  # 20
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.width", 500)  # 170
pd.set_option('display.expand_frame_repr', False)

df_test = pd.read_csv("HOUSE_PRICE_PREDICTION/test.csv")
df_train = pd.read_csv("HOUSE_PRICE_PREDICTION/train.csv")
df_test_backup = df_test.copy()
df_train_backup = df_train.copy()
df_test.head()
df_train.head()
df_test.shape
df_train.shape
df_test.info()
df_train.info()


######################
# GENEL GOZLEM
######################
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Info #####################")
    print(dataframe.info())
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Head #####################")
    print(dataframe.head())
    print("##################### Tail #####################")
    print(dataframe.tail())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("##################### Describe #####################")
    print(dataframe.describe().T)
    print("##################### Info #####################")
    print(dataframe.info())


check_df(df_test)

check_df(df_train)

######################################################
# GOREV-1: KESIFCI VERI ANALIZI (EDA)
######################################################
# Adim 1: Train ve Test veri setlerini okutup birleştiriniz. Birleştirdiğiniz veri üzerinden ilerleyiniz.
df = pd.concat([df_test, df_train])
df.shape
df.tail()
df.isnull().sum()
df.head()
df["Neighborhood"].nunique()
df["Neighborhood"].unique()


# Adim 2: Numerik ve kategorik değişkenleri yakalayınız

def grab_col_names(dataframe, cat_th=25, car_th=30):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and col != "SalePrice"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


check_df(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df)
# cat_th=10, car_th=20 iken;
# Observations: 2919
# Variables: 81
# cat_cols: 52
# num_cols: 28
# cat_but_car: 1
# num_but_cat: 10
# ----------------------------------------------------
# cat_th=25, car_th=30 iken;
# Observations: 2919
# Variables: 81
# cat_cols: 58
# num_cols: 23
# cat_but_car: 0
# num_but_cat: 15

# Adim 3: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)
df.head()
df.info()  # tip hatasi yok gibi.

df["GarageYrBlt"] = pd.to_numeric(df["GarageYrBlt"], errors="coerce")  # --> zaten float ama '' olarak string
#                                                                           bosluklar varsa NaN atmasi icin yaptim.

df_GarageYrBlt_control = df[df["GarageYrBlt"].isna()]
df_GarageYrBlt_control.shape  # 159 adet bos deger var, bu bosluklari mean ile dolduracagiz. Devami --> 353. satirda.

# # 159 NaN var, bu satirlari droplayacagim. --> Update: DROPLAMA SAKIN!! test verisi de eksiliyor :@
# df.dropna(subset=['GarageYrBlt'], inplace=True)
# df["GarageYrBlt"] = df["GarageYrBlt"].astype("int")
# df["GarageYrBlt"].dtype

df["GarageYrBlt"].nunique()
df["GarageYrBlt"].max()  # 2207 --> aykiri degerleri baskilarken gidecek.

df.shape
# (2760, 81) --> drop yapmadiginda olmasi gereken 2919!

# Adim 4: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# cat_th=25, car_th=30
# Observations: 2919
# Variables: 81
# cat_cols: 58
# num_cols: 22
# cat_but_car: 0
# num_but_cat: 15

##################################
# NUMERIK DEGISKEN ANALIZI
##################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
        plt.pause(5)


df.describe().T
num_summary(df, num_cols)


##################################
# NUMERIK DEGISKENLERIN HEDEF DEGISKENE GORE ANALIZI
##################################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, "SalePrice", col)


# Adim 5: Kategorik değişkenler ile hedef değişken incelemesini yapınız.

##################################
# KATEGORİK DEĞİŞKENLERİN ANALİZİ
##################################

def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))


cat_summary(df, "SalePrice")


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df, "SalePrice", col)

# Adim 6: Aykırı gözlem var mı inceleyiniz.
df.describe().T
df["GarageYrBlt"].max()

# Adim 7: Eksik gözlem var mı inceleyiniz.
df.isnull().sum()
# MSZoning            4 --> mode ile doldur
# LotFrontage       475 --> mean ile doldur
# Alley            2584 --> mode ile doldur --> ek kontrol --> bulunmadigi icin NA gorunuyor, direkt "Yok" yazdir.
# Utilities           2 --> mode ile doldur
# Exterior1st         1 --> mode ile doldur
# Exterior2nd         1 --> mode ile doldur
# MasVnrType         24 --> mode ile doldur
# MasVnrArea         23 --> mean ile doldur
# BsmtQual           81 --> mode ile doldur --> ek kontrol --> bulunmadigi icin NA gorunuyor, direkt "Yok" yazdir.
# BsmtCond           82 --> mode ile doldur --> ek kontrol --> bulunmadigi icin NA gorunuyor, direkt "Yok" yazdir.
# BsmtExposure       82 --> mode ile doldur --> ek kontrol --> bulunmadigi icin NA gorunuyor, direkt "Yok" yazdir.
# BsmtFinType1       79 --> mode ile doldur --> ek kontrol --> bulunmadigi icin NA gorunuyor, direkt "Yok" yazdir.
# BsmtFinSF1          1 --> mean ile doldur --> ek kontrol --> bulunmadigi icin NA gorunuyor, direkt "Yok" yazdir.
# BsmtFinType2       80 --> mode ile doldur --> ek kontrol --> bulunmadigi icin NA gorunuyor, direkt "Yok" yazdir.
# BsmtFinSF2          1 --> mean ile doldur
# BsmtUnfSF           1 --> mean ile doldur
# TotalBsmtSF         1 --> mean ile doldur
# Electrical          1 --> mode ile doldur
# BsmtFullBath        2 --> mean ile doldur
# BsmtHalfBath        2 --> mean ile doldur
# KitchenQual         1 --> mode ile doldur
# Functional          2 --> mode ile doldur
# FireplaceQu      1420 --> mode ile doldur --> ek kontrol --> bulunmadigi icin NA gorunuyor, direkt "Yok" yazdir.
# GarageType        157 --> mode ile doldur --> ek kontrol --> bulunmadigi icin NA gorunuyor, direkt "Yok" yazdir.
# GarageYrBlt       159 --> mean ile doldur --> ek kontrol --> bulunmadigi icin NA gorunuyor, direkt "Yok" yazdir.
# GarageFinish      159 --> mode ile doldur --> ek kontrol --> bulunmadigi icin NA gorunuyor, direkt "Yok" yazdir.
# GarageCars          1 --> mean ile doldur
# GarageArea          1 --> mean ile doldur
# GarageQual        159 --> mode ile doldur --> ek kontrol --> bulunmadigi icin NA gorunuyor, direkt "Yok" yazdir.
# GarageCond        159 --> mode ile doldur --> ek kontrol --> bulunmadigi icin NA gorunuyor, direkt "Yok" yazdir.
# PoolQC           2909 --> mode ile doldur --> ek kontrol --> bulunmadigi icin NA gorunuyor, direkt "Yok" yazdir.
# Fence            2348 --> mode ile doldur --> ek kontrol --> bulunmadigi icin NA gorunuyor, direkt "Yok" yazdir.
# MiscFeature      2814 --> mode ile doldur --> ek kontrol --> bulunmadigi icin NA gorunuyor, direkt "Yok" yazdir.
# SaleType            1 --> mode ile doldur
# SalePrice        1459 --> train verisinden gelen degisken, tahmin yaparken droplanacak.

df["PoolQC"].isna().sum()  # --> 10 adet evde bu ozellik belirtilmis, 2909 adedi bos.
df["PoolQC"].shape

######################################################
# GOREV-2: FEATURE ENGINEERING
######################################################
# Adim 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.

# -------------------------
# Eksik Degerleri Doldurma
# -------------------------
df.isnull().sum()
# MSZoning            4 --> mode ile doldur
df["MSZoning"] = df["MSZoning"].fillna(df["MSZoning"].mode()[0])
# LotFrontage       475 --> mean ile doldur
df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].mean())
# Alley            2584 --> mode ile doldur --> ek kontrol --> bulunmadigi icin NA gorunuyor, direkt "Yok" yazdir.
# Utilities           2 --> mode ile doldur
df["Utilities"] = df["Utilities"].fillna(df["Utilities"].mode()[0])
# Exterior1st         1 --> mode ile doldur
df["Exterior1st"] = df["Exterior1st"].fillna(df["Exterior1st"].mode()[0])
# Exterior2nd         1 --> mode ile doldur
df["Exterior2nd"] = df["Exterior2nd"].fillna(df["Exterior2nd"].mode()[0])
# MasVnrType         24 --> mode ile doldur
df["MasVnrType"] = df["MasVnrType"].fillna(df["MasVnrType"].mode()[0])
# MasVnrArea         23 --> mean ile doldur
df["MasVnrArea"] = df["MasVnrArea"].fillna(df["MasVnrArea"].mean())
# BsmtQual           81 --> mode ile doldur --> ek kontrol --> bulunmadigi icin NA gorunuyor, direkt "Yok" yazdir.
# BsmtCond           82 --> mode ile doldur --> ek kontrol --> bulunmadigi icin NA gorunuyor, direkt "Yok" yazdir.
# BsmtExposure       82 --> mode ile doldur --> ek kontrol --> bulunmadigi icin NA gorunuyor, direkt "Yok" yazdir.
# BsmtFinType1       79 --> mode ile doldur --> ek kontrol --> bulunmadigi icin NA gorunuyor, direkt "Yok" yazdir.
# BsmtFinSF1(int)     1 --> mean ile doldur --> ek kontrol --> bulunmadigi icin NA gorunuyor, direkt "0" yazdir.
df["BsmtFinSF1"].fillna(0, inplace=True)
# BsmtFinType2       80 --> mode ile doldur --> ek kontrol --> bulunmadigi icin NA gorunuyor, direkt "Yok" yazdir.
# BsmtFinSF2          1 --> mean ile doldur
df["BsmtFinSF2"] = df["BsmtFinSF2"].fillna(df["BsmtFinSF2"].mean())
# BsmtUnfSF           1 --> mean ile doldur
df["BsmtUnfSF"] = df["BsmtUnfSF"].fillna(df["BsmtUnfSF"].mean())
# TotalBsmtSF         1 --> mean ile doldur
df["TotalBsmtSF"] = df["TotalBsmtSF"].fillna(df["TotalBsmtSF"].mean())
# Electrical          1 --> mode ile doldur
df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])
# BsmtFullBath        2 --> mean ile doldur
df["BsmtFullBath"] = df["BsmtFullBath"].fillna(df["BsmtFullBath"].mean())
# BsmtHalfBath        2 --> mean ile doldur
df["BsmtHalfBath"] = df["BsmtHalfBath"].fillna(df["BsmtHalfBath"].mean())
# KitchenQual         1 --> mode ile doldur
df["KitchenQual"] = df["KitchenQual"].fillna(df["KitchenQual"].mode()[0])
# Functional          2 --> mode ile doldur
df["Functional"] = df["Functional"].fillna(df["Functional"].mode()[0])
# FireplaceQu      1420 --> mode ile doldur --> ek kontrol --> bulunmadigi icin NA gorunuyor, direkt "Yok" yazdir.
# GarageType        157 --> mode ile doldur --> ek kontrol --> bulunmadigi icin NA gorunuyor, direkt "Yok" yazdir.
# GarageYrBlt(int)       159 --> mean ile doldur --> ek kontrol --> bulunmadigi icin NA gorunuyor, direkt "0" yazdir.
df["GarageYrBlt"].fillna(0, inplace=True)
# GarageFinish      159 --> mode ile doldur --> ek kontrol --> bulunmadigi icin NA gorunuyor, direkt "Yok" yazdir.
# GarageCars          1 --> mean ile doldur
df["GarageCars"] = df["GarageCars"].fillna(df["GarageCars"].mean())
# GarageArea          1 --> mean ile doldur
df["GarageArea"] = df["GarageArea"].fillna(df["GarageArea"].mean())
# GarageQual        159 --> mode ile doldur --> ek kontrol --> bulunmadigi icin NA gorunuyor, direkt "Yok" yazdir.
# GarageCond        159 --> mode ile doldur --> ek kontrol --> bulunmadigi icin NA gorunuyor, direkt "Yok" yazdir.
# PoolQC           2909 --> mode ile doldur --> ek kontrol --> bulunmadigi icin NA gorunuyor, direkt "Yok" yazdir.
# Fence            2348 --> mode ile doldur --> ek kontrol --> bulunmadigi icin NA gorunuyor, direkt "Yok" yazdir.
# MiscFeature      2814 --> mode ile doldur --> ek kontrol --> bulunmadigi icin NA gorunuyor, direkt "Yok" yazdir.
# SaleType            1 --> mode ile doldur
df["SaleType"] = df["SaleType"].fillna(df["SaleType"].mode()[0])
# SalePrice        1459 --> train verisinden gelen degisken, tahmin yaparken droplanacak.

# NA degerleri toplu degistirecegim;
df.isna().sum()
df.loc[:, df.columns != "SalePrice"] = df.loc[:, df.columns != 'SalePrice'].fillna("Yok")
num_cols

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Observations: 2919
# Variables: 81
# cat_cols: 58
# num_cols: 22
# cat_but_car: 0
# num_but_cat: 15

# ------------------------------
# Aykiri Gozlemler icin Islemler
# ------------------------------

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


# Kontrol
for col in num_cols:
    print(col, check_outlier(df, col))


def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Aykırı Değer Analizi ve Baskılama İşlemi
for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

# Adim 2: Rare Encoder uygulayınız.

df.columns
df["SaleCondition"].value_counts()


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)
        plt.pause(5)


for col in cat_cols:
    cat_summary(df, col)


#############################################
# Rare Encoding
#############################################
# 1. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi;

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


rare_analyser(df, "SalePrice", cat_cols)


# 2. Rare encoder yazacağız.

# Nadir sınıfların tespit edilmesi
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


new_df = rare_encoder(df, 0.01)  # ben 0.15 yapip bakacagim --> 0.01 idi --> BURADA KALDIM
rare_analyser(new_df, "SalePrice", cat_cols)

# ---------- RARE SINIFLANDIRMASI ----------

rare_categories = {
    "MiscFeature": {
        "rare_values": ["Gar2", "TenC"],
        "target_variable": "MiscFeature_Rare"
    }
}

for column, config in rare_categories.items():
    rare_values = config["rare_values"]
    target_variable = config["target_variable"]
    df[column] = np.where(df[column].isin(rare_values), target_variable, df[column])

# Kontrol;
df.groupby("MiscFeature")["SalePrice"].mean()

# ---------- RARE SINIFLANDIRMASI BITTI ----------

# Adim 3: Yeni değişkenler oluşturunuz.

# Kategorik degiskenler arasindaki iliskiyi gormek icin ki-kare testi;
from scipy.stats import chi2_contingency

for col in cat_cols:
    crosstab = pd.crosstab(df["SalePrice"], df[col])
    chi2, p_value, dof, expected = chi2_contingency(crosstab)
    print(f"{col}:")
    print("Chi-square test statistic:", chi2)
    print("p-value:", p_value)
    if p_value < 0.05:
        print(
            "Anlamlılık düzeyi 0.05'ten küçük olduğu için H0 hipotezi reddedildi. Yani, iki değişken arasında anlamlı bir ilişki vardır.\n")
    else:
        print(
            "Anlamlılık düzeyi 0.05'ten büyük olduğu için H0 hipotezi kabul edilir. Yani, iki değişken arasında anlamlı bir ilişki yoktur.\n")

################################################
# Kategorik Degiskenlerden Uretilen Degiskenler
################################################

df["NEW_SeasonSold"] = df["MoSold"].replace({1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
                                             6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall",
                                             11: "Fall", 12: 'Winter'})

df["MoSold"].sort_values(ascending=False).value_counts()
cat_cols
df["Street"].unique()

# Numerik degiskenler arasindaki iliskiyi gormek icin korelasyon analizi;
df.corr()
num_cols

# Korelasyon Matrisi
# f, ax = plt.subplots(figsize=[18, 13])
# sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
# ax.set_title("Correlation Matrix", fontsize=20)
# plt.show(block=True)
# plt.pause(5)

df.corrwith(df["SalePrice"]).sort_values(ascending=False)

# 0.10'dan küçük: zayıf korelasyon
# 0.10-0.30 arası: orta düzeyde korelasyon
# 0.30'dan büyük: güçlü korelasyon

################################################
# Numerik Degiskenlerden Uretilen Degiskenler
################################################

# Yasam alani buyuklugune gore kalite puani;
df["NEW_Qual_LivArea"] = df["OverallQual"] * df["GrLivArea"]  # corr: 0.83

# Total SF
df["NEW_TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]  # corr: 0.80

# df["NEW_TotalBsmtSF_d_LotArea"] = df["TotalBsmtSF"] / df["LotArea"]

# Evin toplam alani, SF cinsinden olanlar toplandi;
df["NEW_TotalArea"] = df["NEW_TotalSF"] + df["GarageArea"]  # corr: 0.828

#  yasam alani kalitesi * toplam alan (toplam alanin kalite skoru)--> 150523 olusturuldu.
df["NEW_TotalAreaQual"] = df["NEW_Qual_LivArea"] * df["NEW_TotalArea"]  # corr: 0.78

# Total Garage Size --> garaj kapasitesi;
df["NEW_TotalGarageSize"] = df["GarageCars"] * df["GarageArea"]  # corr: 0.66

# Evin toplam alani + garaj alani --> # 150523 olusturuldu.
df["NEW_TotalArea_n_Garage"] = df["NEW_TotalArea"] + df["NEW_TotalGarageSize"]  # corr: 0.822

# Evin kalite puani * garajin araba kapasitesi --> Evin kalite puani 160523 olusturuldu.
df["NEW_OverallQualScore"] = df["OverallQual"] * df["GarageCars"]  # corr: 0.81

# # Evin kalite puani * genel durumuna dair skoru # --> 160523 olusturuldu.
# df["NEW_OverallQualCond_Score"] = df["NEW_OverallQualScore"] * df["OverallCond"]  # corr: 0.77

# Evin genel kalitesi * genel durumuna dair skoru --> 160523 olusturuldu.
df["NEW_OverallQualCond_Score_2"] = df["OverallQual"] * df["OverallCond"]  # --> corr: 0.56 --> catboost feat. imp. iyi

# Evin butun odalari + banyolar dahil
df["NEW_TotalRmsAllIncluded"] = df["TotRmsAbvGrd"] + df["FullBath"] + df["HalfBath"]  # corr: 0.60

# Total Quality;
df["NEW_TotalQual"] = df[["OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtCond", "BsmtFinType1",
                          "BsmtFinType2", "HeatingQC", "KitchenQual", "Functional", "FireplaceQu", "GarageQual",
                          "GarageCond", "Fence"]].sum(axis=1)  # corr: 0.59
# Arazi genisligi;
df["NEW_LotSize"] = df["LotArea"] * df["LotFrontage"]  # corr: 0.37

df.head()

# ------------------
# SERMIN
# ------------------
df['NEWS_Total_Porch_Area'] = (df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df[
    'WoodDeckSF'])  # corr: 0.38 --> catboost feat. imp. iyi
df['NEWS_AbvGrArea'] = df['GrLivArea'] + df['TotRmsAbvGrd']  # corr: 0.71
df['NEWS_GarageYrCarsAr'] = (df['GarageYrBlt'] + df['GarageCars'] + df['GarageArea'])  # corr: 0.42

# --------- BU KISIM DEGERLENDIRILECEK FIYATA ETKISI NASIL OLCULEBILIR ----------
# Evin satildigi yas;
df["NEW_Age"] = df["YrSold"] - df["YearBuilt"]

df["NEW_RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

# ------------------
# MERVE
# ------------------
df.corrwith(df["SalePrice"]).sort_values(ascending=False)

# toplam banyo sayısı
df['NEWM_totalbath']=df['BsmtFullBath']+0.5*df['BsmtHalfBath']+df['FullBath']+0.5*df['HalfBath'] # corr: 0.63

# 2. -------------------------------
#yapı yaşı
# import datetime
# current_year = datetime.datetime.now().year -- nan geldi kontrol edecegiz
# -------------------------------
# df['NEWM_houseage']=current_year - df['YearBuilt']
# df['NEWM_houseage'].head()


# 3. -------------------------------
#cephe büyüklüğü oransal ifadesi
df['NEWM_forantage_ratio'] = df['LotFrontage']/df['LotArea'] # catboost basari biraz dustu.

# 4. -------------------------------
# #toplam katsayısı
unique_styles = df["HouseStyle"].unique()
df["NEWM_TotalFloors"] = df["HouseStyle"].apply(lambda x: int(x.split("Story")[0]) if "Story" in x else None)
# df["NEWM_TotalFloors"].dtype # --> catboost 24.757

# 5. -------------------------------
#TAKS Değeri
df['NEWM_BCR'] = df['1stFlrSF']/df['LotArea']

df['NEWS_Area/Lot'] = (df['TotalArea'] / df['LotArea']


# #KAKS Değeri
# df['NEWM_FAR']=df['NEW_BCR']*df["NEW_TotalFloors"]
#
# #toplam inşaat alanı
# df['NEWM_totalConsArea']=df['GrLivArea']+df['TotalBsmtSF']
#
# #yapının yenilenip yenilenmediği
# df['NEWM_renewal']=df.apply(lambda row: 1 if row["YearRemodAdd"] > row["YearBuilt"] else 0, axis=1)
#
# #yasam alanı ve kalitesi ilişkisi
# df['NEWM_quality']=df['OverallQual']*df['GrLivArea']
#
# #garaj için ort araç büyüklüğü
# df['NEWM_carArea']=df['GarageArea']/df['GarageCars']


# Adim 4: Encoding işlemlerini gerçekleştiriniz.

cat_cols, cat_but_car, num_cols = grab_col_names(df)


# ---------------
# label encoder --> binary kolonlara uygulanmis;
# ---------------
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtypes == "O" and len(df[col].unique()) == 2]

for col in binary_cols:
    label_encoder(df, col)


# ---------------
# one-hot encoder --> kateorik kolonlara uygulanmis;
# ---------------
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()

######################################################
# GOREV-3: MODEL KURMA
######################################################
# Adim 1: Train ve Test verisini ayırınız. (SalePrice değişkeni boş olan değerler test verisidir.)

#  Train ve Test verisini ayırınız. (SalePrice değişkeni boş olan değerler test verisidir.)
df_train = df[df["SalePrice"].notnull()]
df_test = df[df["SalePrice"].isnull()]

y = df_train["SalePrice"]
X = df_train.drop(["Id", "SalePrice"], axis=1)

# Adim 2: Train verisi ile model kurup, model başarısını değerlendiriniz.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

models = [('LR', LinearRegression()),
          # ("Ridge", Ridge()),
          # ("Lasso", Lasso()),
          # ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          # ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

"""
RMSE: 152739.3862 (LR) 
RMSE: 61541.2023 (KNN) 
RMSE: 42804.7959 (CART) 
RMSE: 30407.2135 (RF) 
RMSE: 26711.6325 (GBM) 
RMSE: 29674.3512 (XGBoost) 
RMSE: 30121.3988 (LightGBM) 
RMSE: 25347.498 (CatBoost) 

----------12.05.2023 sonuclari----------

RMSE: 7448297.1349 (LR) 
RMSE: 60706.6119 (KNN) 
RMSE: 38039.1857 (CART) 
RMSE: 29551.2914 (RF) 
RMSE: 26526.1386 (GBM) 
RMSE: 29130.7416 (XGBoost) 
RMSE: 29761.2054 (LightGBM) 
RMSE: 24799.2747 (CatBoost) 

----------16.05.2023 sonuclari----------

RMSE: 1788982.4519 (LR) 
RMSE: 41442.4856 (KNN) 
RMSE: 39947.3716 (CART) 
RMSE: 29101.958 (RF) 
RMSE: 25390.9303 (GBM) 
RMSE: 28703.551 (XGBoost) 
RMSE: 28191.956 (LightGBM) 
RMSE: 24538.2557 (CatBoost) 

"""

# Bonus-1: Hedef değişkene log dönüşümü yaparak model kurunuz ve rmse sonuçlarını gözlemleyiniz.
# Not: Log'un tersini (inverse) almayı unutmayınız.

# Log dönüşümünün gerçekleştirilmesi

df_train = df[df['SalePrice'].notnull()]
df_test = df[df['SalePrice'].isnull()]

y = np.log1p(df_train['SalePrice'])
X = df_train.drop(["Id", "SalePrice"], axis=1)

# ---------------------------
# C A T B O O S T
# ---------------------------
# Verinin eğitim ve test verisi olarak bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

# CatBoostRegressor modelinin oluşturulması ve eğitilmesi
catboost = CatBoostRegressor()
catboost.fit(X_train, y_train)

# Test verisi üzerinde tahmin yapılması
y_pred = catboost.predict(X_test)

# Tahmin sonuçlarının ters log dönüşümü
new_y = np.expm1(y_pred)
new_y_test = np.expm1(y_test)

# Hata metriği hesaplanması (RMSE)
rmse = np.sqrt(mean_squared_error(new_y_test, new_y))
print("RMSE:", rmse)

# Hiperparametre optimizasyonu
catboost_model = CatBoostRegressor(random_state=46)

rmse = np.mean(np.sqrt(-cross_val_score(catboost_model, X, y, cv=5, scoring="neg_mean_squared_error")))

catboost_params = {"iterations": [200, 500, 1000],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

catboost_gs_best = GridSearchCV(catboost_model,
                                catboost_params,
                                cv=5,
                                n_jobs=-1,
                                verbose=False).fit(X, y)

final_model = catboost_model.set_params(**catboost_gs_best.best_params_).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))

print("Optimized RMSE:", rmse)


# # ---------------------------
# # L I G T H - G B M
# # ---------------------------
# # # Verinin eğitim ve tet verisi olarak bölünmesi
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)
#
# # lgbm_tuned = LGBMRegressor(**lgbm_gs_best.best_params_).fit(X_train, y_train)
#
# lgbm = LGBMRegressor().fit(X_train, y_train)
# y_pred = lgbm.predict(X_test)
#
# y_pred
# # Yapılan LOG dönüşümünün tersinin (inverse'nin) alınması
# new_y = np.expm1(y_pred)
# new_y
# new_y_test = np.expm1(y_test)
# new_y_test
#
# np.sqrt(mean_squared_error(new_y_test, new_y))
# # RMSE: 30501.838185805045
#
# # Adim 3: Hiperparemetre optimizasyonu gerçekleştiriniz.
#
# lgbm_model = LGBMRegressor(random_state=46)
#
# rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model, X, y, cv=5, scoring="neg_mean_squared_error")))
#
# lgbm_params = {"learning_rate": [0.01, 0.1],
#                "n_estimators": [500, 1000, 1500],
#                "colsample_bytree": [0.5, 0.7, 1]
#                }
#
# lgbm_gs_best = GridSearchCV(lgbm_model,
#                             lgbm_params,
#                             cv=5,
#                             n_jobs=-1,
#                             verbose=True).fit(X, y)
#
# final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)
#
# rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))


# Adim 4: Değişken önem düzeyini inceleyeniz.

# feature importance
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show(block=True)
    plt.pause(5)
    if save:
        plt.savefig("importances.png")


# model = LGBMRegressor()
# model.fit(X, y)

model = CatBoostRegressor()
model.fit(X, y)

plot_importance(model, X, num=110)

# Bonus-2: Test verisinde boş olan salePrice değişkenlerini tahminleyiniz ve
# Kaggle sayfasına submit etmeye uygun halde bir dataframe oluşturup sonucunuzu yükleyiniz.

model = CatBoostRegressor()
model.fit(X, y)
predictions = model.predict(df_test.drop(["Id", "SalePrice"], axis=1))

dictionary = {"Id": df_test["Id"].astype(int), "SalePrice": predictions}
dfSubmission = pd.DataFrame(dictionary)
dfSubmission.to_csv("5th_HousePricePrediction_1705.csv", index=False)
