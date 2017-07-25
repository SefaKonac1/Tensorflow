import tensorflow as tf
import numpy as np


def graph_builder():
	
	g = tf.Graph()
	with g.as_default():
		c = tf.Variable(20,name = "c")
		assert c.graph is g

		d = tf.add(c,5,name = "d")
		e = tf.multiply(d,c,name = "e")

		with tf.Session() as sess:
		
			init = tf.global_variables_initializer()
			sess.run(init)
			writer = tf.train.SummaryWriter('./my_graph',sess.graph)
		
		g.add_to_collection(c,5)
	
        

graph()










       
