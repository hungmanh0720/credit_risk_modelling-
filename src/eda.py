from pyspark.sql import SparkSession, functions as F
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession, functions as F

DATA = '../data'
spark = SparkSession.builder.getOrCreate()


def load(name):
    # Ví dụ: load('bureau')
    return spark.read.csv(f'{DATA}/{name}.csv', header=True, inferSchema=True)


def null_report(df):
    total = df.count()
    rows = []

    for col_name, col_type in df.dtypes:
        n_null = df.filter(F.col(col_name).isNull()).count()
        pct_null = round(n_null / total * 100, 2)
        rows.append([col_name, col_type, n_null, pct_null])

    report = pd.DataFrame(rows, columns=['column', 'dtype', 'n_null', 'pct_null'])
    return report.sort_values('pct_null', ascending=False)

def boxplot_by_target(df, col):
    data = []
    for label in [0, 1]:
        rows = df.filter((F.col('TARGET') == label) & F.col(col).isNotNull()).select(col).collect()
        data.append([r[col] for r in rows])

    box = plt.boxplot(data, showfliers=False, patch_artist=True, widths=0.5)

    colors = ['tab:blue', 'tab:orange']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for median in box['medians']:
        median.set_color('white')
        median.set_linewidth(2)

    for i, values in enumerate(data):
        med = sorted(values)[len(values) // 2]
        plt.text(i + 1.28, med, f'{med:,.0f}', va='center')

    plt.xticks([1, 2], ['0 - Bình thường', '1 - Khó khăn chi trả'])
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    plt.grid(axis='y', alpha=0.3)
    plt.ylabel(col)
    plt.title(f'{col} theo nhãn')
    plt.show()

def target_rate(df, col):
    total = df.count()
    result = df.groupBy(col).agg(
        F.count('*').alias('n'),
        F.sum('TARGET').alias('n_1'),
        F.avg('TARGET').alias('rate'),
    )
    rows = result.orderBy(col).collect()

    names = []
    pct_0 = []
    pct_1 = []

    for r in rows:
        share = round(r['n'] / total * 100, 2)
        rate = round(r['rate'] * 100, 2)
        print(r[col], '| số đơn:', r['n'], '| % tổng:', share,
              '| số đơn khó khăn chi trả:', r['n_1'], '| % khó khăn chi trả:', rate)

        names.append(str(r[col]))
        pct_1.append(r['rate'] * 100)
        pct_0.append(100 - r['rate'] * 100)

    plt.bar(names, pct_0, label='0')
    plt.bar(names, pct_1, bottom=pct_0, label='1')
    plt.ylabel('Percentage')
    plt.xlabel(col)
    plt.title(f'Tỷ lệ khó khăn chi trả theo {col}')
    plt.legend(title='TARGET')
    plt.show()