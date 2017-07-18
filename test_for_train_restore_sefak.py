import tensorflow as tf

sess=tf.Session()    
#First let's load metagraph
saver = tf.train.import_meta_graph('my_test_model-1000.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))


# Access saved Variables directly
print(sess.run('d:0'))


graph = tf.get_default_graph()
a = graph.get_tensor_by_name("a:0")
b = graph.get_tensor_by_name("b:0")
feed_dict ={a:[3.0,4.2],b:[1.0,2.4]}
 
#Now, access the op that you want to run. 
e = graph.get_tensor_by_name("e:0")
 
print(sess.run(e,feed_dict))

sess.close()
