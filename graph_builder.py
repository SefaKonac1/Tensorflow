import tensorflow as tf
import numpy as np


def graph_builder(op,inputs,output):
    graph = tf.get_default_graph()

    s = tf.Variable(5,name = "s")
    x = tf.Variable(2.2,name = inputs[0])
    y = tf.Variable(2.1,name = inputs[1])
    z = tf.add(x,y,name = output[0])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(z.name)
    print(y.name)
    print(x.name)
    print(z.eval(session = sess))
    saver = tf.train.Saver()
    for op in graph.get_operations():
        #print(str(op.name))
        if(op.name == "sefa"):
            print("bulunduuuuuuuuuuu")
    #print(output_graph_def)

    saver.save(sess, 'model/my_test_model')

graph_builder("+",["a","b"],["c"])
