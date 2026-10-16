import json
import math
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType

BASE = "data/processed/youtube_hinglish_clean.csv"
_model = None


@pandas_udf(StringType())
def predict_udf(comments: pd.Series) -> pd.Series:
    """Clean each comment and predict its sentiment with the saved TF-IDF model."""
    global _model
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    import joblib
    from src.cleaning import clean_text
    if _model is None:
        _model = joblib.load(os.path.join(ROOT, "models", "tfidf_lr.joblib"))
    cleaned = [clean_text(t) for t in comments.astype(str)]
    return pd.Series(_model.predict(cleaned))


def build_spark():
    return (SparkSession.builder.master("local[*]")
            .appName("hinglish-sentiment-lab")
            .config("spark.driver.memory", "4g")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate())


def read_csv(spark, path):
    return spark.read.csv(path, header=True, multiLine=True, escape='"')


def pandas_baseline(n=20000):
    import joblib
    sys.path.insert(0, ROOT)
    from src.cleaning import clean_text
    model = joblib.load("models/tfidf_lr.joblib")
    base = pd.read_csv(BASE)["comment"].astype(str)
    texts = pd.concat([base] * math.ceil(n / len(base)), ignore_index=True).head(n)
    t0 = time.time()
    model.predict([clean_text(t) for t in texts])
    return n / (time.time() - t0)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    n_target = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
    campaign = sys.argv[2] if len(sys.argv) > 2 else None
    spark = build_spark()
    spark.sparkContext.setLogLevel("ERROR")

    # ---- 1) scale test (synthetic replication) ----
    base = read_csv(spark, BASE).select(F.col("comment").alias("comment")).dropna()
    n_base = base.count()
    k = math.ceil(n_target / n_base)
    big = base.crossJoin(spark.range(k)).drop("id").repartition(16)
    print(f"base comments: {n_base}, copies: {k}, total rows: {n_base * k}")

    t0 = time.time()
    dist = big.withColumn("pred", predict_udf("comment")).groupBy("pred").count().toPandas()
    spark_secs = time.time() - t0
    spark_rate = (n_base * k) / spark_secs
    print(dist)
    print(f"Spark: {n_base * k} rows in {spark_secs:.1f}s = {spark_rate:,.0f} comments/s")

    pandas_rate = pandas_baseline()
    print(f"pandas (single process, 20,000 rows): {pandas_rate:,.0f} comments/s")

    Path("reports").mkdir(exist_ok=True)
    Path("reports/spark_benchmark.json").write_text(json.dumps({
        "rows": n_base * k, "spark_seconds": round(spark_secs, 1),
        "spark_comments_per_s": round(spark_rate),
        "pandas_comments_per_s": round(pandas_rate),
        "note": "synthetic replication of existing comments; measures pipeline throughput",
    }, indent=2))

    # ---- 2) campaign insights (real comments, model predictions) ----
    if campaign:
        camp = (read_csv(spark, campaign)
                .withColumn("likes", F.col("likes").cast("int"))
                .withColumn("pred", predict_udf("comment"))
                .withColumn("month", F.date_format(F.to_timestamp("published_at"), "yyyy-MM"))
                .cache())
        total = camp.count()
        mix = camp.groupBy("pred").count().toPandas()
        mix["share"] = (mix["count"] / total).round(3)
        print("\nPredicted sentiment mix:\n", mix.to_string(index=False))
        trend = camp.groupBy("month", "pred").count().orderBy("month").toPandas()
        print("\nBy month:\n", trend.to_string(index=False))
        top = (camp.orderBy(F.desc("likes")).select("pred", "likes", "comment")
               .limit(10).toPandas())
        print("\nMost-liked comments:\n", top.to_string(index=False))
        mix.to_csv("reports/campaign_sentiment_mix.csv", index=False)
        trend.to_csv("reports/campaign_sentiment_by_month.csv", index=False)

    spark.stop()