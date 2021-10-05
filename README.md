# Decision Trees on Spark - Pridicting Hand Written Numbers

### Decision Trees on Spark
Let's setup Spark Colab environment.

```python
!pip install pyspark
!pip install -U -q PyDrive
!apt install openjdk-8-jdk-headless -qq
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
```

```python
Requirement already satisfied: pyspark in /usr/local/lib/python3.7/dist-packages (3.1.2)
Requirement already satisfied: py4j==0.10.9 in /usr/local/lib/python3.7/dist-packages (from pyspark) (0.10.9)
openjdk-8-jdk-headless is already the newest version (8u292-b10-0ubuntu1~18.04).
0 upgraded, 0 newly installed, 0 to remove and 37 not upgraded.
```

Now I am authenticate a Google Drive client to download the files that I will be processing in Spark job.

```python
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
```

```python
id='1aJrdYMVmmnUKYhLTlXtyB0FQ9gYJqCrs'
downloaded = drive.CreateFile({'id': id})
downloaded.GetContentFile('mnist-digits-train.txt')

id='1yLwxRaJIyrC03yxqbTKpedMmHEF86AAq'
downloaded = drive.CreateFile({'id': id})
downloaded.GetContentFile('mnist-digits-test.txt')
```


If you executed the cells above, you should be able to see the dataset we will use for this Colab under the "Files" tab on the left panel.


Next, let me import some of the common libraries needed for my task.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import pyspark
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
```
Let's initialize the Spark context.

```python
# create the session
conf = SparkConf().set("spark.ui.port", "4050")

# create the context
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()
```
```python
You can easily check the current version and get the link of the web interface. In the Spark UI, you can monitor the progress of your job and debug the performance bottlenecks (if your Colab is running with a local runtime).
```

```python
spark
```
```python
SparkSession - in-memory
SparkContext
Spark UI
Version
v3.1.2
Master
local[*]
AppName
pyspark-shell
```

If you are running this Colab on the Google hosted runtime, the cell below will create a ngrok tunnel which will allow you to still check the Spark UI.

```python
!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip ngrok-stable-linux-amd64.zip
get_ipython().system_raw('./ngrok http 4050 &')
!curl -s http://localhost:4040/api/tunnels | python3 -c \
    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
```

```python

--2021-10-05 10:56:17--  https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
Resolving bin.equinox.io (bin.equinox.io)... 54.161.241.46, 18.205.222.128, 52.202.168.65, ...
Connecting to bin.equinox.io (bin.equinox.io)|54.161.241.46|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 13832437 (13M) [application/octet-stream]
Saving to: ‘ngrok-stable-linux-amd64.zip.2’

ngrok-stable-linux- 100%[===================>]  13.19M  12.8MB/s    in 1.0s    

2021-10-05 10:56:18 (12.8 MB/s) - ‘ngrok-stable-linux-amd64.zip.2’ saved [13832437/13832437]

Archive:  ngrok-stable-linux-amd64.zip
replace ngrok? [y]es, [n]o, [A]ll, [N]one, [r]ename: yes
  inflating: ngrok                   
http://cd7c-35-204-80-15.ngrok.io
```


## Data Loading

![Alt text](/digit.png?raw=true "Title")

I will be using the famous MNIST database, a large collection of handwritten digits that is widely used for training and testing in the field of machine learning.The dataset has already been converted to the popular LibSVM format, where each digit is represented as a sparse vector of grayscale pixel values.

```python
training = spark.read.format("libsvm").load("mnist-digits-train.txt")
test = spark.read.format("libsvm").load("mnist-digits-test.txt")

# Cache data for multiple uses
training.cache()
test.cache()
```

```python
DataFrame[label: double, features: vector]
```

```python
training.show(truncate=False)
```

```python
|5.0  |(780,[152,153,154,155,156,157,158,159,160,161,162,163,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,231,232,233,234,235,236,237,238,239,240,241,260,261,262,263,264,265,266,268,269,289,290,291,292,293,319,320,321,322,347,348,349,350,376,377,378,379,380,381,405,406,407,408,409,410,434,435,436,437,438,439,463,464,465,466,467,493,494,495,496,518,519,520,521,522,523,524,544,545,546,547,548,549,550,551,570,571,572,573,574,575,576,577,578,596,597,598,599,600,601,602,603,604,605,622,623,624,625,626,627,628,629,630,631,648,649,650,651,652,653,654,655,656,657,676,677,678,679,680,681,682,683],[3.0,18.0,18.0,18.0,126.0,136.0,175.0,26.0,166.0,255.0,247.0,127.0,30.0,36.0,94.0,154.0,170.0,253.0,253.0,253.0,253.0,253.0,225.0,172.0,253.0,242.0,195.0,64.0,49.0,238.0,253.0,253.0,253.0,253.0,253.0,253.0,253.0,253.0,251.0,93.0,82.0,82.0,56.0,39.0,18.0,219.0,253.0,253.0,253.0,253.0,253.0,198.0,182.0,247.0,241.0,80.0,156.0,107.0,253.0,253.0,205.0,11.0,43.0,154.0,14.0,1.0,154.0,253.0,90.0,139.0,253.0,190.0,2.0,11.0,190.0,253.0,70.0,35.0,241.0,225.0,160.0,108.0,1.0,81.0,240.0,253.0,253.0,119.0,25.0,45.0,186.0,253.0,253.0,150.0,27.0,16.0,93.0,252.0,253.0,187.0,249.0,253.0,249.0,64.0,46.0,130.0,183.0,253.0,253.0,207.0,2.0,39.0,148.0,229.0,253.0,253.0,253.0,250.0,182.0,24.0,114.0,221.0,253.0,253.0,253.0,253.0,201.0,78.0,23.0,66.0,213.0,253.0,253.0,253.0,253.0,198.0,81.0,2.0,18.0,171.0,219.0,253.0,253.0,253.0,253.0,195.0,80.0,9.0,55.0,172.0,226.0,253.0,253.0,253.0,253.0,244.0,133.0,11.0,136.0,253.0,253.0,253.0,212.0,135.0,132.0,16.0])                                                                                                                                                                                                                                                                                                                                                             ||5.0  |(780,[152,153,154,155,156,157,158,159,160,161,162,163,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,231,232,233,234,235,236,237,238,239,240,241,260,261,262,263,264,265,266,268,269,289,290,291,292,293,319,320,321,322,347,348,349,350,376,377,378,379,380,381,405,406,407,408,409,410,434,435,436,437,438,439,463,464,465,466,467,493,494,495,496,518,519,520,521,522,523,524,544,545,546,547,548,549,550,551,570,571,572,573,574,575,576,577,578,596,597,598,599,600,601,602,603,604,605,622,623,624,625,626,627,628,629,630,631,648,649,650,651,652,653,654,655,656,657,676,677,678,679,680,681,682,683],[3.0,18.0,18.0,18.0,126.0,136.0,175.0,26.0,166.0,255.0,247.0,127.0,30.0,36.0,94.0,154.0,170.0,253.0,253.0,253.0,253.0,253.0,225.0,172.0,253.0,242.0,195.0,64.0,49.0,238.0,253.0,253.0,253.0,253.0,253.0,253.0,253.0,253.0,251.0,93.0,82.0,82.0,56.0,39.0,18.0,219.0,253.0,253.0,253.0,253.0,253.0,198.0,182.0,247.0,241.0,80.0,156.0,107.0,253.0,253.0,205.0,11.0,43.0,154.0,14.0,1.0,154.0,253.0,90.0,139.0,253.0,190.0,2.0,11.0,190.0,253.0,70.0,35.0,241.0,225.0,160.0,108.0,1.0,81.0,240.0,253.0,253.0,119.0,25.0,45.0,186.0,253.0,253.0,150.0,27.0,16.0,93.0,252.0,253.0,187.0,249.0,253.0,249.0,64.0,46.0,130.0,183.0,253.0,253.0,207.0,2.0,39.0,148.0,229.0,253.0,253.0,253.0,250.0,182.0,24.0,114.0,221.0,253.0,253.0,253.0,253.0,201.0,78.0,23.0,66.0,213.0,253.0,253.0,253.0,253.0,198.0,81.0,2.0,18.0,171.0,219.0,253.0,253.0,253.0,253.0,195.0,80.0,9.0,55.0,172.0,226.0,253.0,253.0,253.0,253.0,244.0,133.0,11.0,136.0,253.0,253.0,253.0,212.0,135.0,132.0,16.0])                                                                                                                                                                                                                                                                                                                                                             |
```

```Python
training.printSchema()
```

```python

root
 |-- label: double (nullable = true)
 |-- features: vector (nullable = true)
 
```

```python
test.printSchema()
```

```python

root
 |-- label: double (nullable = true)
 |-- features: vector (nullable = true)
 
```


First of all, find out how many instances we have in our training / test split.

```python
print(training.count())
print(test.count())
```
```python
60000
10000
```

Now train a Decision Tree on the training dataset using Spark MLlib.

I am using the Python example on this documentation page: https://spark.apache.org/docs/latest/ml-classification-regression.html#decision-tree-classifier

```python
# importing the Decision Tree Classifier libraries
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# defining DecisionTreeClassifier
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")

# Model fitted by DecisionTreeClassifier.
model = dt.fit(training)

predictions = model.transform(test)
```

With the Decision Tree I just induced on the training data, predict the labels of the test set. Printing the predictions for the first 10 digits, and comparing them with the labels.

```python
predictions.select("prediction", "label").show(10)
```
```python

+----------+-----+
|prediction|label|
+----------+-----+
|       7.0|  7.0|
|       2.0|  2.0|
|       8.0|  1.0|
|       0.0|  0.0|
|       9.0|  4.0|
|       1.0|  1.0|
|       5.0|  4.0|
|       6.0|  9.0|
|       6.0|  5.0|
|       9.0|  9.0|
+----------+-----+
only showing top 10 rows
```

The small sample above looks good, but not great!

Let's dig deeper. Computing the accuracy of the model, using the MulticlassClassificationEvaluator from MLlib.

```python
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g " % accuracy)
```
```python
Accuracy = 0.6795 
```
Finding out the max depth of the trained Decision Tree, and its total number of nodes.

```python
print(model)
```
```python
DecisionTreeClassificationModel: uid=DecisionTreeClassifier_ad6f4018c453, depth=5, numNodes=61, numClasses=10, numFeatures=780
```

It appears that the default settings of the Decision Tree implemented in MLlib did not allow us to train a very powerful model!

Before starting to train a Decision Tree, I can tune the max depth it can reach by using the setMaxDepth() method. Train 21 different DTs, varying the max depth from 0 to 20, endpoints included (i.e., [0, 20]). For each value of the parameter, I am going to print the accuracy achieved on the test set, and the number of nodes contained in the given DT.

IMPORTANT: this parameter sweep can take 30 minutes or more, depending on how busy is your Colab instance. Notice how the induction time grows super-linearly!


```python
def train_dt(training, test, max_depth):
    dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=max_depth)
    model = dt.fit(training)
    predictions = model.transform(test)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    return accuracy

accs = []
for d in range(21):
    accs.append(train_dt(training, test, d))
```
Just making some changes to plot the Accuracy of the model

```python
y = accs
x = range(0,21)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
y =pd.Series(y)
```
```python
fig, ax = plt.subplots()
ax.plot(x,y)

def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)

annot_max(x,y)


ax.set_ylim(-0.3,1.5)
plt.show()
```
![Alt text](/acc.png?raw=true "Title")

```python
# Stopping Spark Environment
sc.stop()
```

It appears that the model is performing 88% accuracy at Max_Depth of 15. Actually, not bad :D







