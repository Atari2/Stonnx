import requests
import bs4
import re
import argparse

BASE_REF_URL = 'https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_'
BASE_URL = 'https://onnx.ai/onnx/operators/onnx__{}.html'

def normalize_name(name: str):
    # FIXME: doesn't work names like MatMul where the individual words composing the name are not long enough
    #        for ONNX to use underscores (e.g. MatMul becomes matmul instead of mat_mul
    #        but BatchNormalization becomes batch_normalization)
    return ''.join([('_' if i != 0 else '') + r.lower() if r.isupper() else r for i, r in enumerate(name)])

def get_version_list(opname: str):
    url = BASE_URL.format(opname)
    r = requests.get(url)
    soup = bs4.BeautifulSoup(r.text, 'html.parser')
    versions = set()
    lopname = opname.lower()
    version_re = re.compile(rf'#{lopname}-(\d+)')
    for a in soup.find_all('a'):
        if a.get('href') is not None:
            if ver := version_re.match(a.get('href')):
                versions.add(int(ver.group(1)))
    return sorted(versions)

PRELUDE = """use crate::{
    onnx::NodeProto,
    utils::{ArrayType, BoxResult, OperationResult},
};
"""

def make_rust_file(opname: str, versions: list[int]):
    normalized = normalize_name(opname)
    lopname = opname.lower()
    refurl = BASE_REF_URL + normalized + '.py'
    with open(f'src/operators/{lopname}.rs', 'w') as f:
        f.write(PRELUDE)
        f.write('\n\n')
        f.write(f'const _OPSET_VERSIONS: [i64; {len(versions)}] = {versions};')
        f.write('\n\n')
        f.write(f'/// {refurl}\n')
        f.write(f'/// {BASE_URL.format(opname)}\n')
        f.write(f"""pub fn {normalized}(
    _inputs: &[&ArrayType],
    _node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperationResult> {{
    todo!("{opname}")
}}
""")
    with open('src/operators/mod.rs', 'a') as f:
        f.write(f'pub mod {lopname};\n')
    
argparser = argparse.ArgumentParser()
argparser.add_argument('-o', '--opname', type=str, required=True)

args = argparser.parse_args()

version_list = get_version_list(args.opname)
make_rust_file(args.opname, version_list)