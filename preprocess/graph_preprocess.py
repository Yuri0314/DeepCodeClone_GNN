import os
import javalang

from javalang.ast import Node
from tqdm import tqdm
from treelib import Tree

from preprocess.edge_process import add_edges


def parse_node(node):
    """
    解析返回当前结点的token和children列表
    :param node: 当前结点
    :return: 当前结点的token和children列表
    """
    token = ''
    children = []
    if isinstance(node, Node):
        token = node.__class__.__name__
        children = node.children
    elif isinstance(node, str):
        if not str(node).startswith('/**'):
            token = node

    def expand(nested_list):  # 拉平嵌套的children
        for item in nested_list:
            if isinstance(item, (list, set)):
                for sub_item in expand(item):
                    yield sub_item
            else:
                yield item

    return token, list(expand(children))


def convert_to_ast(origin_root):
    """
    将javalang提取出的ast树进行过滤转换，并统计所有token
    :param origin_root: javalang提取出的ast树根节点
    :return: 转换后的ast树根节点，token集合
    """
    tree = Tree()
    token_set = set()
    node_num = 0

    def create_tree(origin_node, parent=None):
        nonlocal node_num
        token, children = parse_node(origin_node)
        if token == '':
            return
        tree.create_node(token, node_num, parent=parent, data=origin_node)
        token_set.add(token)
        parent = node_num
        node_num += 1
        for child in children:
            create_tree(child, parent)

    create_tree(origin_root)
    return tree, token_set


def generate_ast(dataset_name):
    """
    生成java文件的ast树及所有token的词典
    :param dataset_name: 输入数据集名称
    :return: 对应java文件的ast树及tokens集合
    """
    dir = os.path.join('./DataSet', dataset_name)
    paths = []
    asts = []
    tokens_set = set()
    if dataset_name == 'googlejam4_src':
        for root, dirs, files in os.walk(dir):
            if len(files) == 0:
                continue
            for file in files:
                file = os.path.join(root, file)
                with open(file, 'r', encoding="UTF-8") as f:
                    code_text = f.read()
                    ast = javalang.parse.parse(code_text).types[0]
                    ast, tokens = convert_to_ast(ast)
                    asts.append(ast)
                    tokens_set.update(tokens)
                    paths.append(file)
    elif dataset_name == 'bigclonebenchdata':
        files = os.listdir(dir)
        for file in tqdm(files):
            file = os.path.join(dir, file)
            with open(file, 'r', encoding="UTF-8") as f:
                code_text = f.read()
                tokens = javalang.tokenizer.tokenize(code_text)
                parser = javalang.parser.Parser(tokens)
                ast = parser.parse_member_declaration()
                ast, tokens = convert_to_ast(ast)
                asts.append(ast)
                tokens_set.update(tokens)
                paths.append(file)

    file2ast = dict(zip(paths, asts))
    all_tokens = list(tokens_set)
    tokens_idx = range(len(all_tokens))
    token2idx = dict(zip(all_tokens, tokens_idx))
    return file2ast, token2idx


# 图的forward()输入为dgl.graph和[N*dim]的输入图结点向量
def generate_graph(file2ast, token2idx):
    file2tokenIdx = dict()
    file2graph = dict()
    for file, ast in tqdm(file2ast.items(), desc='Graph generating'):
        # 先获取对应graph的结点index列表，用于模型输入
        idx_list = []
        nodes = ast.nodes
        for i in range(ast.size()):
            idx_list.append([token2idx[nodes[i].tag]])
        file2tokenIdx[file] = idx_list

        # 再获取对应graph的表示
        edges = []
        edge_types = []
        add_edges(ast, edges, edge_types, extra_edge=True)
        graph = [edges, edge_types]
        file2graph[file] = graph

    return file2graph, file2tokenIdx


if __name__ == '__main__':
    dataset_name = 'googlejam4_src'
    file2ast, file2tokens = generate_ast(dataset_name)
    file2graph, file2tokenIdx = generate_graph(file2ast, file2tokens)
