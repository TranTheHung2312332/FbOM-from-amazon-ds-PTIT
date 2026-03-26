import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


from pyspark.sql import SparkSession
import json
import csv

def extract_fields(obj, prefix=""):
    fields = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            fields.append(path)
            fields.extend(extract_fields(v, path))

    elif isinstance(obj, list):
        for item in obj:
            fields.extend(extract_fields(item, prefix+'__list_of'))

    return fields


spark = SparkSession.builder.appName('Fields Analysis').config("spark.python.worker.faulthandler.enabled", "true").getOrCreate()

sc = spark.sparkContext
rdd = sc.textFile("data/raw/meta_Office_Products.jsonl")

# Map phase
fields_rdd = rdd.map(json.loads).flatMap(lambda obj: extract_fields(obj)).map(lambda field1: (field1, 1))

# Reduce phase
field_counts = fields_rdd.reduceByKey(lambda a, b: a + b).sortBy(lambda x: (-x[1], x[0]))

# Test
print(field_counts.take(20))

# Save
rows = field_counts.collect()
with open("output/meta_fields_stat.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["field", "count"])
    for k, v in rows:
        writer.writerow([k, v])