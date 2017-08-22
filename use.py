import argparse
import tensorflow as tf
from load import load_graph
from graph_builder import graph_builder

if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="model/frozen_model.pb", type=str, help="Frozen model file to import")
    args = parser.parse_args()

    # We use our "load_graph" function
    graph = load_graph(args.frozen_model_filename)
    print(graph)
    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(str(op.name))
        if(op.name == "a"):
            print("bulunduuuuuuuuuuu")

    graph_builder("+",["d","c"],["g"])
        
    #print(output_graph_def)

    # We access the input and output nodes
    #x = graph.get_tensor_by_name('prefix/Placeholder/inputs_placeholder:0')
    #y = graph.get_tensor_by_name('prefix/Accuracy/predictions:0')

    # We launch a Session
    #with tf.Session(graph=graph) as sess:
        # Note: we didn't initialize/restore anything, everything is stored in the graph_def
    #    y_out = sess.run(y, feed_dict={
    #        x: [[3, 5, 7, 4, 5, 1, 1, 1, 1, 1]] # < 45
    #    })
    #    print(y_out) # [[ False ]] Yay, it works!
