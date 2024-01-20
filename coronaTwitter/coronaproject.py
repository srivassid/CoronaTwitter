from pyspark.sql import SparkSession
import time
from pyspark.sql import functions as F
from pyspark.sql.types import StructField, ArrayType, StringType, StructType
from pyspark.sql.functions import countDistinct
from pyspark.sql.functions import lower
from pyspark.sql.functions import to_timestamp
from pyspark.sql.types import DateType
from pyspark.sql.functions import year, month

def sparkSession():
    # Create SparkSession
    spark = SparkSession.builder \
      .master("local[1]") \
      .appName("tbcov") \
      .getOrCreate()
    return spark

def countries_tweets():
    start = time.time()
    spark = sparkSession()

    df = spark.read.format('csv').option("sep", "\t").option('header', 'true'). \
        load('/media/sid/Kingston/data/twittercorona/full_dataset/*/*')

    print(df.head())

    df = df.groupBy('user_loc_country_code').count().orderBy("count", ascending=False)

    print(df.show())

    df.coalesce(1).write.csv("/home/sid/Documents/PythonProjects/SparkTwitter/coronaTwitter/question_2.csv", header=True)

    print("Time is ",time.time() - start)


def migration():
    start = time.time()
    spark = sparkSession()
    spark.sparkContext.setLogLevel("ERROR")

    df = spark.read.format('csv').option("sep", "\t").option('header', 'true'). \
        load('/media/sid/Kingston/data/twittercorona/full_dataset/*/*')
    df = df.groupBy('user_id').agg(countDistinct("user_loc_city").alias('city_count')).orderBy('city_count',ascending=False)

    print(df.count())

    total = df.count()
    df = df.where('city_count >= 2')

    print(df.count())

    migrated = df.count()

    with open("/coronaTwitter/migrated.csv", 'w') as f:
        f.writelines("total " + str(total) + " migrated " +  str(migrated))

    print("Time is ", time.time() - start)

def health_impact():
    start = time.time()
    spark = sparkSession()
    spark.sparkContext.setLogLevel("ERROR")

    df = spark.read.format('csv').option("sep", "\t").option('header', 'true'). \
        load('/media/sid/Kingston/data/twittercorona/full_dataset/*/*')

    # print(df.select("tweet_text_named_entities").rdd.map(lambda x: x[0]).collect())

    user_schema = ArrayType(
        StructType([
            StructField("label", StringType(), True),
            StructField("term", StringType(), True),
        ])
    )

    df = (df.withColumn("tweet_text_named_entities", F.from_json("tweet_text_named_entities", user_schema))
           .selectExpr("inline(tweet_text_named_entities)","date_time"))
    print(df.show())

    df = df.withColumn("record_date", df['date_time'].cast(DateType()))
    print(df.show())

    df = df.select(year(df.record_date).alias('dt_year'), month(df.record_date).alias('dt_month'), df.term)
    print(df.show())

    df = df.withColumn("term", lower(df["term"]))
    smell_df = df.filter((df.term.contains('loss of smell')))

    print(smell_df.show())
    smell_df_count = smell_df.groupBy('dt_year','dt_month').count()

    smell_df_count.coalesce(1).write.csv("/home/sid/Documents/PythonProjects/SparkTwitter/coronaTwitter/health", header=True)


if __name__ == '__main__':
    # countries_tweets()
    # migration()
    health_impact()