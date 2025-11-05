# CODTECH INTERNSHIP - TASK 1 : BIG DATA ANALYSIS (REAL DATA)
# BY VARSHINI G S

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, desc
import time

# 1️⃣ Start Spark session
spark = SparkSession.builder.appName("RealData_BigDataAnalysis").getOrCreate()
print("\n🚀 Spark Session Started")

# 2️⃣ Load your real dataset (uploaded CSV file)
data_path = "Electric_Vehicle_Population_Data.csv"  # use your file name here

print(f"\n📂 Loading real dataset: {data_path}")
start = time.time()

df = spark.read.option("header", True).option("inferSchema", True).csv(data_path)

print(f"✅ Dataset loaded in {round(time.time() - start, 2)} sec")
print(f"Total Rows: {df.count()}\n")

# 3️⃣ Show sample rows and schema
print("📊 Sample Data:")
df.show(5)
print("\n📘 Schema Information:")
df.printSchema()

# 4️⃣ Clean data: drop missing rows and select numeric columns
print("\n🧹 Cleaning data...")
numeric_cols = [field.name for field in df.schema.fields if str(field.dataType) in ("IntegerType", "DoubleType", "LongType")]
if len(numeric_cols) >= 2:
    df_clean = df.select(*numeric_cols).dropna()
else:
    df_clean = df.na.drop()

print(f"✅ Cleaned dataset has {df_clean.count()} rows\n")

# 5️⃣ Basic big data analysis
print("📈 Running basic analysis...")

# Find average of each numeric column (if applicable)
summary = df_clean.select(
    *[avg(col(c)).alias(f"avg_{c}") for c in df_clean.columns if df_clean.schema[c].dataType.simpleString() in ["int", "double", "long"]]
)

# Count total records
record_count = df_clean.count()

print("\n✅ Basic Summary Statistics:")
summary.show()

# 6️⃣ Example aggregation (if 'category' or 'id' columns exist)
cols = [c.lower() for c in df_clean.columns]
if "category" in cols:
    print("\n🔹 Aggregation by 'category':")
    result = df_clean.groupBy("category").agg(
        count("*").alias("Total_Records"),
        *[avg(col(c)).alias(f"avg_{c}") for c in df_clean.columns if c != "category"]
    ).orderBy(desc("Total_Records"))
    result.show(10)
else:
    print("\n⚠️ No 'category' column found. Showing overall summary only.")

# 7️⃣ Save results to CSV file
output_path = "output/task1_results"
summary.write.mode("overwrite").option("header", True).csv(output_path)
print(f"\n💾 Results saved to folder: {output_path}")

# 8️⃣ Stop Spark
spark.stop()
print("\n🏁 Spark session stopped. Task 1 Completed Successfully!")
