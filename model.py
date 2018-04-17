import tensorflow as tf
import settings
import numpy as np
class Model:
	def __init__(self,training=True,type=3):
		self.sentence1_size=settings.sentence1_size
		self.sentence2_size=settings.sentence2_size
		self.sentence1= tf.placeholder(tf.float32, shape=[None,self.sentence1_size[0]*self.sentence1_size[1]], name='x1')
		self.sentence2=tf.placeholder(tf.float32,shape=[None,self.sentence2_size[0]*self.sentence2_size[1]],name='x2')
		self.num_channels=settings.num_channels
		self.num_filter1=settings.num_filter1
		self.num_filter2=settings.num_filter2
		self.shape_filter1=settings.shape_filter1
		self.shape_filter2=settings.shape_filter2
		self.weights1=tf.Variable(tf.truncated_normal([self.shape_filter1,self.shape_filter1,self.num_channels,self.num_filter1],stddev=0.05))
		self.weights2=tf.Variable(tf.truncated_normal([self.shape_filter2,self.shape_filter2,self.num_filter1,self.num_filter2],stddev=0.05))
		self.size_out_cnn=settings.size_out_cnn
		self.matrix12=tf.Variable(tf.truncated_normal([self.size_out_cnn,self.size_out_cnn],stddev=0.05))
		self.size_hidden_layer=settings.size_hidden_layer
		self.size_output=settings.size_output
		self.num_classes=settings.num_classes
		self.y_true = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='y_true')
		if(type==3):
			self.sentence3_size=settings.sentence3_size
			self.matrix13=tf.Variable(tf.truncated_normal([self.size_out_cnn,self.size_out_cnn],stddev=0.05))
			self.matrix23=tf.Variable(tf.truncated_normal([self.size_out_cnn,self.size_out_cnn],stddev=0.05))
			self.sentence3=tf.placeholder(tf.float32,shape=[None,self.sentence3_size[0]*self.sentence3_size[1]],name='x3')
			self.net=self.build_network(self.sentence1,self.sentence2,self.sentence3,self.num_classes)
		if(type==2):
			self.net=self.build_netword_2line(self.sentence1,self.sentence2,self.num_classes)
		if(training):
			self.batch=tf.Variable(0)
			self.cost=self.loss_layers(self.net,self.y_true)
			self.learning_rate = tf.train.exponential_decay(settings.learning_rate, self.batch * settings.batch_size, settings.decay_step, settings.decay_rate, True)
			self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost, global_step = self.batch)
		else:
			self.confusion_matrix=self.confusion_matrix(self.net,self.y_true)
	
	def build_network(self,sentence1,sentence2,sentence3,num_outputs,keep_prob = settings.dropout,training = True):

		sentence1=tf.reshape(sentence1, [-1, self.sentence1_size[0],self.sentence1_size[1],self.num_channels])
		net1 = tf.pad(sentence1, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]))
		net1=tf.nn.conv2d(net1,filter=self.weights1,strides=[1, 1, 1, 1],padding='SAME')
		net1=tf.nn.relu(net1)
		net1=tf.nn.conv2d(net1,filter=self.weights2,strides=[1, 1, 1, 1],padding='SAME')
		net1=tf.nn.relu(net1)
		net1=tf.nn.max_pool(net1,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
		net1=tf.contrib.layers.flatten(net1)
		net1=tf.layers.dense(net1,self.size_out_cnn,activation=tf.nn.relu)
		net1 = tf.layers.dropout(net1, rate = keep_prob, training = training)

		sentence2=tf.reshape(sentence2, [-1, self.sentence2_size[0],self.sentence2_size[1],self.num_channels])
		net2 = tf.pad(sentence2, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]))
		net2=tf.nn.conv2d(net2,filter=self.weights1,strides=[1, 1, 1, 1],padding='SAME')
		net2=tf.nn.relu(net2)
		net2=tf.nn.conv2d(net2,filter=self.weights2,strides=[1, 1, 1, 1],padding='SAME')
		net2=tf.nn.relu(net2)
		net2=tf.nn.max_pool(net2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
		net2=tf.contrib.layers.flatten(net2)
		net2=tf.layers.dense(net2,self.size_out_cnn,activation=tf.nn.relu)
		net2 = tf.layers.dropout(net2, rate = keep_prob, training = training)

		sentence3=tf.reshape(sentence3, [-1, self.sentence3_size[0],self.sentence3_size[1],self.num_channels])
		net3 = tf.pad(sentence3, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]))
		net3=tf.nn.conv2d(net3,filter=self.weights1,strides=[1, 1, 1, 1],padding='SAME')
		net3=tf.nn.relu(net3)
		net3=tf.nn.conv2d(net3,filter=self.weights2,strides=[1, 1, 1, 1],padding='SAME')
		net3=tf.nn.relu(net3)
		net3=tf.nn.max_pool(net3,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
		net3=tf.contrib.layers.flatten(net3)
		net3=tf.layers.dense(net3,self.size_out_cnn,activation=tf.nn.relu)
		net3 = tf.layers.dropout(net3, rate = keep_prob, training = training)

		simi12=tf.matmul(tf.matmul(net1,self.matrix12),tf.transpose(net2))
		simi12=tf.diag_part(simi12)
		simi12=tf.reshape(simi12,[-1,1])

		simi13=tf.matmul(tf.matmul(net1,self.matrix13),tf.transpose(net3))
		simi13=tf.diag_part(simi13)		
		simi13=tf.reshape(simi13,[-1,1])

		simi23=tf.matmul(tf.matmul(net2,self.matrix23),tf.transpose(net3))
		simi23=tf.diag_part(simi23)
		simi23=tf.reshape(simi23,[-1,1])

		net=tf.concat([simi13,net1,simi12,net2,simi23,net3],1)
		net=tf.layers.dense(net,self.size_hidden_layer,activation=tf.nn.relu)
		net = tf.layers.dropout(net, rate = keep_prob, training = training)
		net=tf.layers.dense(net,num_outputs)
		return net
	def build_netword_2line(self,sentence1,sentence2,num_outputs,keep_prob = settings.dropout,training = True):

		sentence1=tf.reshape(sentence1, [-1, self.sentence1_size[0],self.sentence1_size[1],self.num_channels])
		net1 = tf.pad(sentence1, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]))
		net1=tf.nn.conv2d(net1,filter=self.weights1,strides=[1, 1, 1, 1],padding='SAME')
		net1=tf.nn.conv2d(net1,filter=self.weights2,strides=[1, 1, 1, 1],padding='SAME')
		net1=tf.nn.max_pool(net1,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
		net1=tf.contrib.layers.flatten(net1)
		net1=tf.layers.dense(net1,self.size_out_cnn,activation=tf.nn.relu)
		net1 = tf.layers.dropout(net1, rate = keep_prob, training = training)

		sentence2=tf.reshape(sentence2, [-1, self.sentence2_size[0],self.sentence2_size[1],self.num_channels])
		net2 = tf.pad(sentence2, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]))
		net2=tf.nn.conv2d(net2,filter=self.weights1,strides=[1, 1, 1, 1],padding='SAME')
		net2=tf.nn.conv2d(net2,filter=self.weights2,strides=[1, 1, 1, 1],padding='SAME')
		net2=tf.nn.max_pool(net2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
		net2=tf.contrib.layers.flatten(net2)
		net2=tf.layers.dense(net2,self.size_out_cnn,activation=tf.nn.relu)
		net2 = tf.layers.dropout(net2, rate = keep_prob, training = training)

		simi12=tf.matmul(tf.matmul(net1,self.matrix12),tf.transpose(net2))
		simi12=tf.diag_part(simi12)
		simi12=tf.reshape(simi12,[-1,1])

		net=tf.concat([net1,simi12,net2],1)
		net=tf.layers.dense(net,self.size_hidden_layer,activation=tf.nn.relu)
		net = tf.layers.dropout(net, rate = keep_prob, training = training)
		net = tf.layers.dense(net,num_outputs)

		return net

	def loss_layers(self,net,y_true):
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=net,labels=y_true)
		cost = tf.reduce_mean(cross_entropy)
		return cost
	def accuracy(self,net,y_true):
		y_pred = tf.nn.softmax(net)
		y_pred_cls = tf.argmax(y_pred, dimension=1)
		y_true_cls = tf.argmax(y_true, dimension=1)
		correct_prediction = tf.equal(y_pred_cls, y_true_cls)
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		return accuracy
	def confusion_matrix(self,net,y_true):
		y_pred = tf.nn.softmax(net)
		confusion_matrix = tf.confusion_matrix(tf.argmax(y_true,dimension=1),tf.argmax(y_pred,dimension=1))
		return confusion_matrix

	def recall(self,confusion_matrix):
		return confusion_matrix[0,0]/(confusion_matrix[1,0]+confusion_matrix[0,0])
		
	def precision(self,confusion_matrix):
		return confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])

	def acc(self,confusion_matrix):
		return (confusion_matrix[0,0]+confusion_matrix[1,1])/(confusion_matrix[0,0]+confusion_matrix[1,1]+confusion_matrix[0,1]+confusion_matrix[1,0])