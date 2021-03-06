from preprocess.graph_preprocess import generate_ast, generate_graph
from preprocess.data_preprocsee import generate_pair_data

dataset_name = 'googlejam4_src'

# generate_pair_data(dataset_name)
file2ast, file2tokens = generate_ast(dataset_name)
file2graph, file2tokenIdx = generate_graph(file2ast, file2tokens)