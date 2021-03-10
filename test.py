from preprocess.graph_preprocess import generate_ast, generate_graph

dataset_name = 'bigclonebenchdata'

file2ast, file2tokens = generate_ast(dataset_name)
file2graph, file2tokenIdx = generate_graph(file2ast, file2tokens)