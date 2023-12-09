import stanza
from collections import defaultdict
import copy
from tqdm import tqdm
from anytree import Node, RenderTree
from anytree.exporter import DotExporter


be_verb = ['am', 'are', 'is', 'was', 'were']

nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse', use_gpu=True, model_dir='/home/zizheng/projects/exp-gen/coreNLP_models/stanford-corenlp-4.4.0')
class dependencyNode:
    def __init__(self, _id, word, parent, children, relation, upos, xpos):
        self.id = _id
        self.word = word
        self.parent = parent
        self.children = children
        self.relation = relation
        self.upos = upos
        self.xpos = xpos

        self.keep_in_extraction = False
        self.replace_mark = False
        self.includes_answer = False

    @classmethod
    def build_from_list(cls, parsing_list):
        node_list = [dependencyNode(0, 'ROOT', None, [], '', '', '')] + [
            dependencyNode(self_id + 1, word, None, [], relation, upos, xpos)
            for self_id, (word, _, relation, upos,
                          xpos) in enumerate(parsing_list)
        ]
        for self_id, (_, head_id, _, _, _) in enumerate(parsing_list):
            node_list[head_id].children.append(node_list[self_id + 1])
            node_list[self_id + 1].parent = node_list[head_id]

        #     print(f'{node_list[self_id+1].word} IS CHILD OF {node_list[head_id].word}')
        # breakpoint()
        return node_list[0]

    def is_head(self, node):
        res = False
        if node in self.children:
            return True
        for child in self.children:
            res = res or child.is_head(node)
            if res:
                return True
        return False

    def is_child(self, node):
        self_node = self
        while self_node.parent is not None:
            if self_node.parent == node:
                return True
            self_node = self_node.parent
        return False


    def _includes_answer_nodes(self, answer):
        for child in self.children:
            child._includes_answer_nodes(answer)

        if answer in self.word:
            self.includes_answer = True
        elif len(self.children) > 0 and any([child.includes_answer for child in self.children]):
            self.includes_answer = True

    def _merge_hint_nodes(self, answer):
        # breakpoint()
        for child in self.children:
            child._merge_hint_nodes(answer)

        if self.word != answer:
            merge_list = [child for child in self.children if child.word.lower() != answer.lower()] + [self]
            merge_list = sorted(merge_list, key=lambda x: x.id)
            self.word = ' '.join([node.word for node in merge_list])
            
            self.children = [child for child in self.children if child.includes_answer]

            for child in self.children:
                if child.includes_answer and child.word.lower() != answer.lower():
                    self.id = child.id
                    self.relation = child.relation
                    self.xpos = child.xpos
                    self.upos = child.upos
                    self.children = child.children
                    break

            # for node in merge_list:
            #     if node.id != self.id:
            #         del node

    def merge_hint_nodes(self, answer):
        self._includes_answer_nodes(answer)
        self._merge_hint_nodes(answer)


    def extract_subtree(self, nodes=[]):
        lca = self.lca_multi_nodes(nodes)
        lca.keep_in_extraction = True

        for node in nodes:
            while node != lca:
                node.keep_in_extraction = True
                node = node.parent

        def cut(node):
            children = []
            for child in node.children:
                if child.keep_in_extraction:
                    children.append(child)
                    cut(child)
            node.children = children
            return node
        
        lca = cut(lca)
        return lca

    def search_by_word(self, word, closest_node=None):
        # if word == 'been':
        #     breakpoint()
        if closest_node is not None:
            candidates = []
        if self.word.lower() == word.lower() or (word == 'be' and self.word in be_verb):
            return self
        q = [c for c in self.children]
        # breakpoint()
        while len(q) > 0:
            node = q[0]
            # print(node.word, word)
            if node.word.lower() == word.lower() or (word == 'be' and node.word in be_verb):
                if closest_node is None:
                    return node
                else:
                    candidates.append((node, node.distance(closest_node)))
            q = q[1:] + node.children

        candidates = sorted(candidates, key=lambda x: x[1])
        if len(candidates) == 0:
            return None
        return candidates[0][0]

    def distance(self, node):
        lca = self.lca(self, node)
        dis1 = self.distance_to_root()
        dis2 = node.distance_to_root()
        dis3 = lca.distance_to_root()
        return dis1 + dis2 - dis3

    def distance_to_root(self):
        res = 0
        node = self
        while node.id != 0:
            res += 1
            node = node.parent
        return res

    @classmethod
    def lca(cls, node1, node2):
        parents1 = [node1]
        while node1.parent is not None:
            parents1.append(node1.parent)
            node1 = node1.parent

        while node2 is not None:
            if node2 in parents1:
                return node2
            node2 = node2.parent
        return None

    @classmethod
    def lca_multi_nodes(cls, nodes):
        if len(nodes) == 1:
            return nodes[0]
        res = cls.lca(nodes[0], nodes[1])
        for node in nodes[2:]:
            res = cls.lca(res, node)
        return res

    def has_same_value(self, node):
        # return self.relation == node.relation and self.upos == node.upos and self.xpos == node.xpos
        return self.relation == node.relation and self.upos == node.upos
        return self.relation == node.relation

    def _has_sub_tree(self, root, node, res_recoder=[]):
        q = [(root, node)]
        res_recoder = []
        while len(q) > 0:
            n1, n2 = q[0]
            res_recoder.append((n1, n2))
            for c1 in n1.children:
                for c2 in n2.children:
                    if c1.has_same_value(c2):
                        q.append((c1, c2))
            q = q[1:]
        # breakpoint()
        if len(res_recoder) == node.length():
            return res_recoder
        else:
            return None


    def has_sub_tree(self, node):
        possible_roots = []
        # root = None
        q = [self]
        while len(q) > 0:
            n = q[0]
            if n.has_same_value(node):
                possible_roots.append(n)
            q = q[1:] + n.children
        # breakpoint()
        for root in possible_roots:
            res = self._has_sub_tree(root, node)
            if res is not None:
                return res
        return None

    def is_same_with(self, node):
        q = [(self, node)]
        while len(q) > 0:
            n1, n2 = q[0]
            if n1.has_same_value(n2) and len(n1.children) == len(n2.children):
                used = set()
                for c1 in n1.children:
                    for c2 in n2.children:
                        if c1.has_same_value(c2) and c2 not in used:
                            q.append((c1, c2))
                            used.add(c2)
            else:
                return False
            q = q[1:]
        return True

    def print_tree(self):
        print(f'{self.word}(|{self.relation}, {self.upos}, {self.xpos})', end=' ')
        q = [(c, 0, self.word) for c in self.children]
        last_head = ''
        last_depth = -1
        while len(q) > 0:
            node, depth, head = q[0]
            if depth != last_depth:
                print()
            else:
                if head != last_head:
                    print('||', end=' ')
            print(f'{node.word}({head}|{node.relation}, {node.upos}, {node.xpos})', end=' ')
            q = q[1:] + [(c, depth + 1, node.word) for c in node.children]
            last_depth = depth
            last_head = head
        print()

    def print_sentence(self, return_value=False):
        node = self
        node_list = [node]
        q = node.children
        while len(q) > 0:
            node = q[0]
            node_list.append(node)
            q = q[1:] + node.children

        node_list = sorted(node_list, key=lambda x: x.id)
        words = [node.word for node in node_list if node.id != 0]
        if return_value:
            return words
        else:
            print(words)

    def length(self):
        res = -1 if self.id == 0 else 0
        q = [self]
        while len(q) > 0:
            n = q[0]
            q = q[1:] + n.children
            res += 1
        return res

    def _to_text(self, show_word, show_id):
        res = f'r:{self.relation}\nupos:{self.upos}\nxpos:{self.xpos}'
        if show_word:
            res = f'w:{self.word}\n' + res
        if show_id:
            res = f'id:{self.id}\n' + res
        return res

    def show(self, output_path, answer, show_word=False, show_id=False):
        def set_color_shape(node):
            attrs = []
            attrs += [f'color={node.color}'] if hasattr(node, 'color') else []
            attrs += [f'shape={node.shape}'] if hasattr(node, 'shape') else []
            return ', '.join(attrs)

        root = Node(self._to_text(show_word, show_id), color='red' if self.word in answer.split() else 'black')
        q = [(root, node) for node in self.children]
        while len(q) > 0:
            parent, node = q[0]
            new_node = Node(node._to_text(show_word, show_id), color='red' if node.word in answer.split() else 'black', parent=parent)
            q = q[1:] + [(new_node, n) for n in node.children]
        DotExporter(root, nodeattrfunc=set_color_shape).to_picture(output_path)


def parse_sentences(sentences, one_sent=False, print_out=False):
    doc = nlp(sentences)
    if one_sent:
        parsing_list = [(word.text, word.head, word.deprel, word.upos, word.xpos)
                    for sent in doc.sentences for word in sent.words]
        return parsing_list
    else:
        parsing_lists = []
        for idx, sent in enumerate(doc.sentences):
            parsing_list = [(word.text, word.head, word.deprel, word.upos,
                            word.xpos) for word in sent.words]
            parsing_lists.append(parsing_list)
            if print_out:
                print(f'Sentence {idx}')
                print(*[
                    f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}\tupos: {word.upos}\txpos: {word.xpos}'
                    for word in sent.words
                ])
        return parsing_lists


def parse2tree(parsing_lists, print_out=False):
    trees = []
    for idx, parsing_list in enumerate(parsing_lists):
        tree = dependencyNode.build_from_list(parsing_list)
        trees.append(tree)
        if print_out:
            print(f'Sentence {idx}')
            tree.print_tree()
    return trees

