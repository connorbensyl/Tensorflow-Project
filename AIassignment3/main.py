
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#prune irrelevant columns
df = pd.read_csv('aug_train.csv')
df = df.drop(['enrollee_id', 'city'], axis=1)

#replace strings with numbers, using numpy.random.uniform when we have ranges
df['gender'] = df['gender'].replace(['Male','Female','Other',None],[0,1,2,3])
df['relevent_experience'] = df['relevent_experience'].replace([None,'Has relevent experience', 'No relevent experience'],[0,1,2])
df['enrolled_university'] = df['enrolled_university'].replace([None,'no_enrollment', 'Full time course', 'Part time course'],[0,1,3,2])
df['education_level'] = df['education_level'].replace([None,'High School','Graduate','Masters','Phd','Primary School'],[0,1,2,3,4,5])
df['major_discipline'] = df['major_discipline'].replace([None,'STEM','Business Degree','Humanities','Arts','No Major','Other'],[0,1,2,3,4,5,6])
df['experience'] = df['experience'].replace([None,'>20','<1'],[0,random.uniform(20,45),random.uniform(0,1)])
df['company_size'] = df['company_size'].replace([None,'<10','Oct-49','50-99','100-500','500-999','1000-4999','5000-9999','10000+'],
                                                [0,random.uniform(0,10),random.uniform(10,49),random.uniform(50,99),random.uniform(100,500),random.uniform(500,999),
                                                random.uniform(1000,4999),random.uniform(5000,9999),random.uniform(10000,20000)])

df['company_type'] = df['company_type'].replace([None, 'Pvt Ltd', 'Early Stage Startup', 'Funded Startup', 'Public Sector', 'NGO','Other'],
                                                [0,1,2,3,4,5,6])
df['last_new_job'] = df['last_new_job'].replace([None,'never','>4'],[0,1,random.uniform(4,20)])

df['experience'] = pd.to_numeric(df['experience'])
df['company_size'] = pd.to_numeric(df['company_size'])
df['last_new_job'] = pd.to_numeric(df['last_new_job'])
#unsure of how to incorporate validation
validation = np.array(df[2874:].drop(['target']),axis=1)

df = df.drop(df.index[2874:])
data = np.array(df.drop(['target'],axis=1))
target = np.array(df['target'])
x_train, x_test, y_train, y_test = train_test_split(data, target,train_size=0.8235,test_size=0.1765)
input_neurons = 11                                            #0.8235 and 0.1765 = training data and testing data after removing validation data
np.set_printoptions(threshold=np.inf)

x_train = np.asarray(x_train).astype('float32')
y_train = np.asarray(y_train).astype('float32')
x_test = np.asarray(x_test).astype('float32')
y_test = np.asarray(y_test).astype('float32')
output_neurons = 1 #I know the assignment pdf says to have 2 output neurons, but it raises an error when I make this 2
hidden_neurons = 22
batch_size = 10
input_layer = keras.Input(shape=(input_neurons))
#build the model
first_hidden_layer = layers.Dense(hidden_neurons,activation='relu')(input_layer)
second_hidden_layer = layers.Dense(hidden_neurons,activation='relu')(first_hidden_layer)
third_hidden_layer = layers.Dense(hidden_neurons,activation='softmax')(second_hidden_layer)
output_layer = layers.Dense(output_neurons,activation='sigmoid')(third_hidden_layer)
model = keras.Model(inputs=input_layer,outputs=output_layer)
model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),optimizer=keras.optimizers.SGD(lr=0.001),metrics='accuracy')
model.fit(x_train,y_train,batch_size=batch_size,verbose=2)
#model.evaluate(x_test,y_test,batch_size=batch_size,verbose=2)
model.predict(validation,verbose=2,batch_size=batch_size)