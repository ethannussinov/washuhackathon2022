
from datetime import datetime
import psycopg2
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas.io.sql as sqlio
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


conn = psycopg2.connect(DATEBASE_URL)
print(conn)


#     cur.execute("""
#     CREATE TABLE woodpecker_data (
#     customer_id int,
#     spending_location String,
#     spending_time datetime,
#     spending_amount double,
# );
#     CREATE TABLE user_data (
#     username String,
#     id int,
#     hashed_password 
#     )

#     """)
    # test = cur.fetchall()
    # conn.commit()
    # print(test)
#function that you get all the transactions from the database

def create_woodpecker_data():
    with conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE woodpecker_data (
        customer_id INT,
        spending_location STRING,
        spending_time DATETIME,
        spending_amount DOUBLE,
        );
        """)
        test = cur.fetchall()
        conn.commit()
        print(test)

def woodpecker_stuff(customer_id, spending_location, spending_time, spending_amount):
        with conn.cursor() as cur:
            SQL = "INSERT INTO woodpecker_data (customer_id, spending_location, spending_time, spending_amount) VALUES (%s, %s, %s, %s)"
            data = (customer_id, spending_location, spending_time, spending_amount)
            cur.execute(SQL, data)
            # cur.execute(f"""
            # INSERT INTO woodpecker_data (customer_id, spending_location, spending_time, spending_amount)
            # VALUES ({customer_id}, {spending_location}, {spending_time}, {spending_amount});
            # """)
            test = cur.fetchall()
            conn.commit()
            print(test)

create_woodpecker_data()
woodpecker_stuff(0, "Walmart", datetime.now(), 100)

def regressionModel():
    # Read the data

    X = sqlio.read_sql_query("SELECT * FROM woodpecker_data", conn)

    df = pd.get_dummies(df, columns=['spending_location', 'spending_time'])
    X = pd.read_csv('train.csv', index_col='Id')
    X_test_full = pd.read_csv('test.csv', index_col='Id')

    # Remove rows with missing target, separate target from predictors
    X.dropna(axis=0, subset=['spending_amount'], inplace=True)
    y = X.spending_amount              
    X.drop(['spending_amount'], axis=1, inplace=True)

    # To keep things simple, we'll use only numerical predictors
    X = X.select_dtypes(exclude=['object'])
    X_test = X_test_full.select_dtypes(exclude=['object'])

    # Break off validation set from training data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                        random_state=0)

    # Define the model
    my_model = DecisionTreeRegressor()

    # Fit the model
    my_model.fit(X_train, y_train)

    # Get predictions
    predictions = my_model.predict(X_valid)
    print("Mean Absolute Error : " + str(mean_absolute_error(predictions, y_valid)))