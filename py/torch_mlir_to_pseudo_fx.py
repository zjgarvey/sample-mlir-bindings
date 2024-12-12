import torch_mlir
import torch_mlir._mlir_libs
from torch_mlir.ir import Module, Context, WalkResult
from torch_mlir.extras.fx_importer import TORCH_DTYPE_TO_MLIR_TYPE_ASM
from torch_mlir.dialects import torch as torch_d
from pathlib import Path
from typing import Dict, Any, List, Union
import argparse


MLIR_TYPE_ASM_TO_TORCH_DTYPE = {value : key for key, value in TORCH_DTYPE_TO_MLIR_TYPE_ASM.items()}

def load_module_from_path(file_path: Union[Path, str]) -> Module:
    # register the torch dialect with a context
    ctx = Context()
    torch_d.register_dialect(ctx)

    # parse the file contents as an mlir module
    with open(file_path, "r") as f:
        m = Module.parse(f.read(), ctx)
    return m

def hacky_get_dialect_resources(m: Module) -> Dict[str,str]:
    m_lines : List[str] = m.operation.get_asm().split("\n")
    dialect_resources = dict()
    key_idx_start = None
    key_idx_end = None
    for idx, l in enumerate(m_lines):
        if l == "{-#":
            key_idx_start = idx
        if l =="#-}":
            key_idx_end = idx
    assert key_idx_start is not None and key_idx_end is not None
    for line in m_lines[key_idx_start+3:key_idx_end-2]:
        line = line.lstrip(" ")
        key = line[0:line.find(":")]
        value = line[line.find('"')+1:-1]
        dialect_resources[key] = value
    return dialect_resources

def get_pseudo_fx_graph(m: Module) -> List[str]:
    # get the func op
    dialect_resources = hacky_get_dialect_resources(m)
    func_op = m.body.operations[0]
    instruction_strings: List[str] = []

    # print(f'func op info:')
    # for attribute in func_op.attributes:
    #     print(f'\t{attribute.name} : {attribute.attr}')
    func_name = func_op.attributes["sym_name"].value
    instruction_strings.append(f"{func_name}():")
    for r in func_op.regions:
        for b in r.blocks:
            for arg in b.arguments:
                # print(f'\targument name : {arg.get_name()} with type : {arg.type}')
                num_uses = 0
                for use in arg.uses:
                    num_uses += 1
                    # print(f'\t\t{arg.get_name()} is used by: {use.owner.results[0].get_name()}')
                instruction_strings.append(
                    f'\t{arg.get_name()} : [num_users={num_uses}] = placeholder[target={arg.get_name().lstrip("%")}][type_expr={arg.type}]'
                )

    # for storing references to constants and lists in the IR
    constant_associator: Dict[str, Any] = dict()
    list_associator: Dict[str, List] = dict()

    def process_op(op):
        """This function will look at a torch-mlir op and try to generate an instruction string in a pseudo fx_graph style"""
        if op.name == "func.func":
            # do nothing
            return WalkResult(0)
        if op.name == "func.return":
            instruction_strings.append(
                f'\treturn {op.operands[0].get_name().lstrip("%")}'
            )
            return WalkResult(0)
        result = op.result
        if op.name.startswith("torch.constant"):
            # don't print constants: store them in a dict
            attr = None if len(op.attributes) == 0 else op.attributes[0].attr.value
            constant_associator[result.get_name()] = attr
            return WalkResult(0)
        if op.name == "torch.prim.ListConstruct":
            # don't print lists: store them in a dict
            l = []
            for opd in op.operands:
                if opd.get_name() in constant_associator.keys():
                    l.append(constant_associator[opd.get_name()])
                else:
                    l.append(opd.get_name())
            list_associator[result.get_name()] = l
            return WalkResult(0)
        view = op.opview
        # the opview often contains some extra info.
        # this is probably better than string matching the op name:
        if isinstance(view, torch_d.ValueTensorLiteralOp):
            try:
                value = torch_mlir.ir.DenseResourceElementsAttr(view.attributes["value"])
                lt_index = str(value).find("<")
                gt_index = str(value).find(">")
                handle_name = str(value)[lt_index+1:gt_index]
                blob_hex_string = dialect_resources[handle_name]
                print(f'{result.get_name()} with type {value.type} has resource string: {blob_hex_string}')
                # Dense resources have a ranked tensor type (which has some better python bindings than value tensor type)
                # Here is an example of getting the dims and dtype, with the slightly tedious caveat that signedness
                # of dtypes is not recognized, so si32 and ui32 tensors would both be stored as i32. For this reason,
                # it might be better to get the dtype from !torch.vtensor<[shape],dtype> string parsing, with a TODO to add
                # python bindings for ValueTensorType and NonValueTensorType...
                ranked_type = torch_mlir.ir.RankedTensorType(value.type)
                shape = [dim for dim in ranked_type.shape]
                print(f'shape = {shape}')
                builtin_dtype_asm = str(ranked_type.element_type)
                try: 
                    torch_dtype = MLIR_TYPE_ASM_TO_TORCH_DTYPE[builtin_dtype_asm]
                except KeyError:
                    torch_dtype = MLIR_TYPE_ASM_TO_TORCH_DTYPE['s'+builtin_dtype_asm]
                print(f'torch_dtype = {torch_dtype}\n')
                return WalkResult(0)
            except ValueError:
                #pass
                print(f"{value} is not a dense resource elements attr!")
            
            # If it is not a dense resource elements attr, then do something else:
            try:
                value = torch_mlir.ir.DenseElementsAttr(view.attributes["value"])
                # do something
                print(f'found a DenseElementsAttr : {value}\n')
                return WalkResult(0)
            except ValueError:
                # pass
                print(f"{value} is not a dense elements attr!")

            print(f"{value} unhandled\n") 
            return WalkResult(1)

        result_type = result.type
        # TODO: find something that converts IR types back to torch types
        result_name = result.get_name()
        # get the "args" for the op
        inputs = []
        for opd in op.operands:
            opd_name = opd.get_name()
            if opd_name in constant_associator.keys():
                inputs.append(constant_associator[opd_name])
                continue
            if opd_name in list_associator.keys():
                inputs.append(tuple(list_associator[opd_name]))
                continue
            inputs.append(opd_name)
        # make a string from args:
        arg_str = "("
        for i in inputs:
            arg_str += f"{i}, "
        arg_str += ")"
        # get number of uses:
        num_uses = 0
        for use in result.uses:
            num_uses += 1
        # get keyword args (attributes ?)
        attrs = dict()
        for attribute in op.attributes:
            attrs[attribute.name] = attribute.attr
        # generate the fake instruction string
        instruction_strings.append(
            f"\t{result_name} : [num_users = {num_uses}] = call_function[target={op.name}](args={arg_str}, kwargs={attrs})[type_expr={result_type}]"
        )
        return WalkResult(0)

    func_op.walk(process_op)
    return instruction_strings


def main(args):
    m = load_module_from_path(args.input_file)
    pseudo_fx = get_pseudo_fx_graph(m)
    f = None
    if args.output_file is not None:
        f = open(args.output_file, "w")
    print(*pseudo_fx, sep="\n", file=f)
    if f is not None:
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("torch_mlir_to_pseudo_fx.py")
    parser.add_argument("input_file", type=str)
    parser.add_argument("-o", "--output_file", type=str)
    args = parser.parse_args()
    main(args)
