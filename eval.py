import settings
import model
import utils
import tensorflow as tf
import numpy as np
import time
import os
from random import shuffle
config='config=tf.ConfigProto(log_device_placement=True)'
sess=tf.InteractiveSession()
model=model.Model(training=False)
saver=tf.train.Saver()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
try:
    saver.restore(sess,'./model/model/model.ckpt')
    print('load from past checkpoint')
except Exception as e:
    print(" not found checkpoint")
last_time=time.time()
list_file_data= os.listdir("./data_tensor_test/")
confusion_matrix=np.asarray([[0,0],[0,0]])
for f in list_file_data:
    data=utils.load_data('./data_tensor_test/'+f)
    shuffle(data)
    for batch in data:
        ind=0
        acc_avg_batch=0
        confusion_matrix_tf=sess.run([model.confusion_matrix],feed_dict={model.sentence1:batch[0],model.sentence2:batch[1],model.sentence3:batch[2],model.y_true:batch[3]})
        confusion_matrix=np.add(confusion_matrix,confusion_matrix_tf[0])
        print(confusion_matrix)
        print("Accuracy : {} , Precision : {}, Recall : {}".format(model.acc(confusion_matrix),model.precision(confusion_matrix),model.recall(confusion_matrix)))
fw=open("result.txt","w")
fw.write("Accuracy : {} , Precision : {}, Recall : {}".format(model.acc(confusion_matrix),model.precision(confusion_matrix),model.recall(confusion_matrix)))
fw.close()