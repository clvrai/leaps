from collections import OrderedDict 


def fetch_mapping(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    dsl2prl_mapping = OrderedDict()
    prl2dsl_mapping = OrderedDict()
    dsl_tokens = []
    prl_tokens = []
    for line in lines:
        tokens = [t for t in line.strip().split(' ') if not t == '']
        assert len(tokens) == 2
        token_dsl = tokens[0]
        token_prl = tokens[1]
        dsl_tokens.append(token_dsl)
        dsl2prl_mapping[token_dsl] = token_prl
        if not token_prl == '#':
            prl_tokens.append(token_prl)
            prl2dsl_mapping[token_prl] = token_prl
    return dsl2prl_mapping, prl2dsl_mapping, dsl_tokens, prl_tokens

if __name__ == "__main__":
    fetch_mapping('mapping_karel2prl.txt')
