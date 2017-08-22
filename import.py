import os
import argparse
import tensorflow as tf
from tensorflow.python.framework import graph_util

dir = os.path.dirname(os.path.realpath(__file__))
print(__file__)

def freeze_graph(model_folder):

    # we retrieve our checkpoint fullpath
    print (model_folder)
    checkpoint = tf.train.get_checkpoint_state(model_folder,"checkpoint")
    print(checkpoint)
    input_checkpoint = checkpoint.model_checkpoint_path

    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])

    print(absolute_model_folder)
    # NOTE: Create frozen model file in folder.
    output_graph = absolute_model_folder + "/frozen_model.pb"
    print(output_graph)

    #clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True
    #import the meta graph
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # NOTE: we retrieve the protobuf graph definition

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    output_node_names = "c"
    with tf.Session() as sess:

        saver.restore(sess,input_checkpoint)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,# NOTE: the grap_def is used to retrieve the nodes
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        )
        with tf.gfile.GFile(output_graph, "wb") as f:

             f.write(output_graph_def.SerializeToString())
			


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('model' , type=str, help="Model folder to export")
	args = parser.parse_args()
	print(args)


	freeze_graph(args.model)
