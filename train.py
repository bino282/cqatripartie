import settings
import model
import utils
import tensorflow as tf
import numpy as np
import time
import os
from random import shuffle
sess=tf.InteractiveSession()
model=model.Model()
saver=tf.train.Saver()
sess.run(tf.global_variables_initializer())
try:
    saver.restore(sess,'./model/model.ckpt')
    print('load from past checkpoint')
except Exception as e:
    print(" not found checkpoint")
for i in range(settings.epoche):
    last_time=time.time()
    total_loss=0
    list_file_data= os.listdir("./data_tensor/")
    index=0
    for f in list_file_data:
        data=utils.load_data('./data_tensor/'+f)
        shuffle(data)
        for batch in data:
            loss,_=sess.run([model.cost,model.optimizer],feed_dict={model.sentence1:batch[0],model.sentence2:batch[1],model.sentence3:batch[2],model.y_true:batch[3]})
            total_loss+=loss
            print("run on batch : "+str(index)+' , loss: '+str(loss))
            index+=1
    print('epoch: '+str(i+1)+', loss: '+str(total_loss/index)+' , s/epoche: '+str(time.time()-last_time))
    saver.save(sess,'./model/model.ckpt')