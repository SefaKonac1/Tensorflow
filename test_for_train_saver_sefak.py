import tensorflow as tf


def graph_builder():

    # Create some input, whose dimension is 2 , as placeholder and variable
    a = tf.placeholder(tf.float32,shape = [2],name = "a")
    b = tf.placeholder(tf.float32,shape = [2],name = "b")
    d = tf.Variable([.3,-.02],name = "d")

    #create feed dict for placeholders.

    feed_dict = {a:[2,3],b:[4,5]}

    # create graph c and e
 	    
    c = tf.add(a,b,name = "c")
    e = tf.multiply(c,d,name = "e")

    # begining session

    with tf.Session() as sess:
	
	# occur initializer for variables.
        
        init = tf.global_variables_initializer()
        sess.run(init)
        
	# create saver variable
        saver = tf.train.Saver()
        
        print sess.run(e,feed_dict)
  
 
        #Now, save the graph
        saver.save(sess, 'my_test_model',global_step = 1000)
        
    
    
graph_builder()    
