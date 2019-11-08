## Spark ML exercise

You have already used [spark.ml](https://spark.apache.org/docs/latest/api/python/pyspark.ml) and spark SQL with most of the code given to you. The goal of this exercise is for you to build a machine learning pipeline entirely in spark from start to finish to familiarize yourself with the spark ML API. This dataset is small enough that you can run spark locally, but you are welcome to run on AWS if you desire.

### Getting Started

First, launch a jupyter notebook and start a spark context. Read your data like
`spark.read.json('data/restaurants.json.gz')`. Use `df.show()` and `df.printSchema()` to familiarize yourself with the data. This is a subset of the yelp restaurant dataset you may have worked with before.

You will want to build a machine learning model to predict the rating (`stars` column). This should be a regression model, so you can pick your favorite regressor from [pyspark.ml.regression](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#module-pyspark.ml.regression)

Don't forget to leave out a test set! You can split your data using the `df.randomSplit()` function.

### Building your pipeline

Check out the documentation for ml [pipeline](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.Pipeline) and [example](https://spark.apache.org/docs/latest/ml-pipeline.html#example-pipeline). Make sure to click on the python tab! You will be putting all of your transformation classes and your model into the pipeline, and performing all transformations in one step.

#### Feature Engineering
Your first stages in the pipeline should be feature engineering. 

Check out the datatypes of your columns. Remember that spark ML models only accept numerical data types as `DoubleType`. If you see any numerical columns that are not `DoubleType`, you will need to use `pyspark.sql.Column.astype` or `pyspark.sql.Column.cast` to change the datatypes. 
Changing a datatype is ok to do outside of a pipeline, because this is a 1-to-1 transformation and doesn't cause data leakage.

Once you have converted the data types of your columns, the rest of your changes will need to happen inside the pipeline. Make sure to check out [feature documentation](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#module-pyspark.ml.feature).

You should have some string columns, which will need to be processed in your pipeline! String categorical columns need to go through two stages, first indexing the strings with a `StringIndexer` object, and then one-hot encoding with the `OneHotEncoder` object.

#### Vector Assembler
Spark ML models require the input of only **one** feature column. You will need to turn a list of input columns into a vector using the `VectorAssembler` object as the second-to-last stage in your pipeline (unless you are scaling).

**If** you are using a model where scaling is necessary, you should use `StandardScaler` on the assembled column.

#### Model
The last stage in your pipeline should be your model.

Once you have added the model in, call `.fit()` on your pipeline object.

You can access the model portion of the pipeline by calling `.stages[-1]` on the fitted pipeline object. From there you can look at learned model parameters, e.g. linear regression coefficients.

### Evaluation
Use the [regression evaluator](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.evaluation.RegressionEvaluator) to see how your model performs on the training and test set. You should use `pipeline.transform()` on your test set. The goal of this exercise is to get a model to run using the pipeline syntax, so if this runs without error, you are done.  

### Extra Credit 1
The ml pipeline process loses a lot of interpretability. Play around with your `model.stages` and try writing some code to retain column information as you are building your pipeline.                                          

### Extra Credit 2
If you have time, use [cross-validation](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#module-pyspark.ml.tuning) to tune model hyper-parameters for better results.