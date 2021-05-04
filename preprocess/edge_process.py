edge_type_idx = {'Child': 0, 'Parent': 1, 'Sibling': 2,
                 'Token': 3, 'VarReference': 4, 'VarInvocation': 5,
                 'VarUseChain': 6, 'WhileLoop': 7, 'DoLoop': 8,
                 'ForLoop': 9, 'IfBranch': 10, 'ElseBranch': 11,
                 'SwitchCase': 12, 'Assignment': 13, 'ReturnTo': 14,
                 'ParameterList': 15}


def add_edges(ast, edges, edge_types, structural_edge=False, semantic_edge=False):
    root_id = ast.root
    add_child_edges(ast, root_id, edges, edge_types)
    add_parent_edges(ast, root_id, edges, edge_types)
    if structural_edge:  # 增加结构型边
        add_sibling_edges(ast, root_id, edges, edge_types)
        add_token_edges(ast, root_id, edges, edge_types)
        add_loop_edges(ast, edges, edge_types)
        add_if_else_edges(ast, edges, edge_types)
        add_switch_edges(ast, edges, edge_types)
        add_return_edges(ast, edges, edge_types)
        add_paramList_edges(ast, edges, edge_types)
    if semantic_edge:   # 增加语义型边
        add_assign_edges(ast, edges, edge_types)
        add_var_edges(ast, edges, edge_types)


def add_child_edges(ast, root_id, edges, edge_types):
    for child_id in ast.is_branch(root_id):
        edges.append((root_id, child_id))
        edge_types.append([edge_type_idx['Child']])
        add_child_edges(ast, child_id, edges, edge_types)


def add_parent_edges(ast, root_id, edges, edge_types):
    for child_id in ast.is_branch(root_id):
        edges.append((child_id, root_id))
        edge_types.append([edge_type_idx['Parent']])
        add_parent_edges(ast, child_id, edges, edge_types)


def add_sibling_edges(ast, root_id, edges, edge_types):
    child_ids = ast.is_branch(root_id)
    for i in range(len(child_ids) - 1):
        edges.append((child_ids[i], child_ids[i + 1]))
        edge_types.append([edge_type_idx['Sibling']])
        edges.append((child_ids[i + 1], child_ids[i]))
        edge_types.append([edge_type_idx['Sibling']])
    for child_id in child_ids:
        add_sibling_edges(ast, child_id, edges, edge_types)


def add_token_edges(ast, root_id, edges, edge_types):
    leaves = ast.leaves(root_id)
    for i in range(len(leaves) - 1):
        cur_id = leaves[i].identifier
        next_id = leaves[i + 1].identifier
        edges.append((cur_id, next_id))
        edge_types.append([edge_type_idx['Token']])
        edges.append((next_id, cur_id))
        edge_types.append([edge_type_idx['Token']])


def add_var_edges(ast, edges, edge_types):
    accessed = set() # 访问过的结点id

    def add_var_edge(subtree_id, var_id, var_name):
        nonlocal accessed
        search_tree = ast.subtree(subtree_id)
        use_chain = []

        # 先添加变量reference的边
        reference_nodes = search_tree.filter_nodes(lambda node: node.tag == 'MemberReference')
        for reference_node in reference_nodes:
            if reference_node.data.member != var_name:
                continue
            children = ast.children(reference_node.identifier)
            for child in children:
                if isinstance(child.data, str) and child.data == var_name and child.identifier not in accessed:
                    edges.append((var_id, child.identifier))
                    edge_types.append([edge_type_idx['VarReference']])
                    use_chain.append(child.identifier)

        # 再添加变量进行方法调用的边
        invocation_nodes = search_tree.filter_nodes(lambda node: node.tag == 'MethodInvocation')
        for invocation_node in invocation_nodes:
            if invocation_node.data.qualifier != var_name:
                continue
            children = ast.children(invocation_node.identifier)
            for child in children:
                if isinstance(child.data, str) and child.data == var_name and child.identifier not in accessed:
                    edges.append((var_id, child.identifier))
                    edge_types.append([edge_type_idx['VarInvocation']])
                    use_chain.append(child.identifier)

        # 最后添加变量使用链的边
        for i in range(len(use_chain) - 1):
            edges.append((use_chain[i], use_chain[i + 1]))
            edge_types.append([edge_type_idx['VarUseChain']])
            edges.append((use_chain[i + 1], use_chain[i]))
            edge_types.append([edge_type_idx['VarUseChain']])

        accessed.update(use_chain)

    nodes = list(ast.filter_nodes(lambda node: node.tag == 'VariableDeclarator'))

    # 先处理局部变量
    for node in nodes:
        local_var_node = ast.parent(node.identifier)
        if local_var_node.tag != 'LocalVariableDeclaration':
            continue
        # 获取变量声明的结点id
        dec_children = ast.children(node.identifier)
        for dec_child in dec_children:
            if isinstance(dec_child.data, str):
                var_id = dec_child.identifier
        add_var_edge(ast.parent(local_var_node.identifier).identifier, var_id, node.data.name)

    # 再处理普通变量
    for node in nodes:
        var_node = ast.parent(node.identifier)
        if var_node.tag != 'VariableDeclaration':
            continue
        tmp_node = ast.parent(var_node.identifier)
        # 获取变量声明的结点id
        dec_children = ast.children(node.identifier)
        for dec_child in dec_children:
            if isinstance(dec_child.data, str):
                var_id = dec_child.identifier
        add_var_edge(ast.parent(tmp_node.identifier).identifier, var_id, node.data.name)

    ast_type = ast.get_node(ast.root).tag
    if ast_type == 'ClassDeclaration':
        # 最后根据是否为类的ast树来判断是否处理成员变量
        for node in nodes:
            field_node = ast.parent(node.identifier)
            if field_node.tag != 'FieldDeclaration':
                continue
            # 获取变量声明的结点id
            dec_children = ast.children(node.identifier)
            for dec_child in dec_children:
                if isinstance(dec_child.data, str):
                    var_id = dec_child.identifier
            add_var_edge(ast.parent(field_node.identifier).identifier, var_id, node.data.name)


def add_loop_edges(ast, edges, edge_types):
    nodes = ast.filter_nodes(lambda node: node.tag == 'WhileStatement')
    for node in nodes:
        children = ast.children(node.identifier)
        edges.append((children[0].identifier, children[1].identifier))
        edge_types.append([edge_type_idx['WhileLoop']])
        edges.append((children[1].identifier, children[0].identifier))
        edge_types.append([edge_type_idx['WhileLoop']])

    nodes = ast.filter_nodes(lambda node: node.tag == 'DoStatement')
    for node in nodes:
        children = ast.children(node.identifier)
        edges.append((children[0].identifier, children[1].identifier))
        edge_types.append([edge_type_idx['DoLoop']])
        edges.append((children[1].identifier, children[0].identifier))
        edge_types.append([edge_type_idx['DoLoop']])

    nodes = ast.filter_nodes(lambda node: node.tag == 'ForStatement')
    for node in nodes:
        children = ast.children(node.identifier)
        edges.append((children[0].identifier, children[1].identifier))
        edge_types.append([edge_type_idx['ForLoop']])
        edges.append((children[1].identifier, children[0].identifier))
        edge_types.append([edge_type_idx['ForLoop']])


def add_if_else_edges(ast, edges, edge_types):
    nodes = ast.filter_nodes(lambda node: node.tag == 'IfStatement')
    for node in nodes:
        children = ast.children(node.identifier)
        edges.append((children[0].identifier, children[1].identifier))
        edge_types.append([edge_type_idx['IfBranch']])
        if len(children) == 3:
            edges.append((children[0].identifier, children[2].identifier))
            edge_types.append([edge_type_idx['ElseBranch']])


def add_switch_edges(ast, edges, edge_types):
    nodes = ast.filter_nodes(lambda node: node.tag == 'SwitchStatement')
    for node in nodes:
        children = ast.children(node.identifier)
        for child in children:
            if child.tag != 'SwitchStatementCase':
                var_id = child.identifier
                break
        for child in children:
            if child.tag != 'SwitchStatementCase':
                continue
            edges.append((var_id, child.identifier))
            edge_types.append([edge_type_idx['SwitchCase']])


def add_assign_edges(ast, edges, edge_types):
    nodes = ast.filter_nodes(lambda node: node.tag == 'Assignment')
    for node in nodes:
        children = ast.children(node.identifier)
        edges.append((children[1].identifier, children[2].identifier))
        edge_types.append([edge_type_idx['Assignment']])
        edges.append((children[2].identifier, children[1].identifier))
        edge_types.append([edge_type_idx['Assignment']])


def add_return_edges(ast, edges, edge_types):
    nodes = ast.filter_nodes(lambda node: node.tag == 'MethodDeclaration')
    for node in nodes:
        search_tree = ast.subtree(node.identifier)
        return_nodes = search_tree.filter_nodes(lambda node: node.tag == 'ReturnStatement')
        for return_node in return_nodes:
            edges.append((return_node.identifier, node.identifier))
            edge_types.append([edge_type_idx['ReturnTo']])


def add_paramList_edges(ast, edges, edge_types):
    nodes = ast.filter_nodes(lambda node: node.tag == 'MethodDeclaration')
    for node in nodes:
        param_list = [child.identifier for child in ast.children(node.identifier) if child.tag == 'FormalParameter']
        for i in range(len(param_list) - 1):
            for j in range(i + 1, len(param_list)):
                edges.append((param_list[i], param_list[j]))
                edge_types.append([edge_type_idx['ParameterList']])
                edges.append((param_list[j], param_list[i]))
                edge_types.append([edge_type_idx['ParameterList']])