from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import csv

spark = SparkSession.builder.appName('Asin Counter').config("spark.python.worker.faulthandler.enabled", "true").getOrCreate()
df = spark.read.json('data/raw/Office_Products.jsonl')

counts = df.groupBy('parent_asin').count().select('parent_asin', col('count')).coalesce(1).sort('count', ascending=False)

with open("output/review_parent_asin_stat.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["parent_asin", "count"])
    for row in counts.toLocalIterator():
        writer.writerow([row["parent_asin"], row["count"]])