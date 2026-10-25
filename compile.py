#compile.py -- compile an Emmy IR file to a PTX file
#region imports
import sys # for command line arguments
from pathlib import Path # for file paths
import json # for JSON data
import logging # for logging
from emmy.compiler import Graph, Node
from emmy.compiler import InputOp, ConstantOp
from emmy.compiler.ir.kernel import KernelOp
#endregion
ptx_config = json.loads(Path("ptx_config.json").read_text())
logging.basicConfig(level=logging.INFO)

class PTXGenerator:
    def __init__(self, graph: Graph, version_config: dict):
        self.graph = graph
        self.version_config = version_config
        self.ptx_lines = []
        self.initialize_ptx_lines()
    
    def initialize_ptx_lines(self) -> None:
        self._write("comment", "This is a generated PTX file by the compile.py script from the Emmy IR file")
        self._write("line_break",1)
        self._write("version", self.version_config["version"])
        self._write("target", self.version_config["target"])
        self._write("address_size", self.version_config["address_size"])
        self._write("line_break",1)
        # get kernel name from graph
        for node_name, node in self.graph.nodes.items():
            if isinstance(node.op, KernelOp):
                kernel_name = node.op.name
                break
        self._write("kernel_name", kernel_name)
    
    def _write(self, key: str, value: str) -> None:
        # config keys
        if key == "version" or key == "target" or key == "address_size":
            self.ptx_lines.append(f"{key} {value}")
        # line break key
        elif key == "line_break":
            self.ptx_lines.append("\n" * value)
        # kernel name key
        elif key == "kernel_name":
            self.ptx_lines.append(f".visible .entry {value}(")
        elif key == "comment":
            self.ptx_lines.append(f"/*{value}*/")
        # invalid key
        else:
            raise NotImplementedError(f"Invalid key: {key}")
    
    def generate_ptx(self) -> str:
        for node_name, node in self.graph.nodes.items():
            self.generate_node_ptx(node)
        return "\n".join(self.ptx_lines)
    
    def generate_node_ptx(self, node: Node) -> str:
        op = node.op
        print(f"op_name: {type(op).__name__}")
        if isinstance(op, InputOp):
            self.ptx_lines.append(f"// InputOp: {node.op}")
        elif isinstance(op, ConstantOp):
            self.ptx_lines.append(f"// ConstantOp: {node.op}")
        elif isinstance(op, KernelOp):
            for name, tensor in node.op.inputs.items():
                logging.info(
                    "Input %s has dtype %s",
                    name,
                    tensor.dtype.name
                )
            for name, tensor in node.op.outputs.items():
                logging.info(
                    "Output %s has dtype %s",
                    name,
                    tensor.dtype.name
                )
        else:
            raise NotImplementedError(f"Node {node.op} has an invalid op: {op}")
        return "\n".join(self.ptx_lines)

def compile(ir_dict: dict) -> str:
    """
    Compile an Emmy IR file to a PTX file.

    Args:
        ir_path: The path to the Emmy IR file.

    Returns:
        The PTX code.
    """
    try:
        # Step 1: Use the existing IR Dict and convert them to an Emmy IR Graph from the emmy.compiler module
        graph = Graph.from_dict(ir_dict)
        # Step 2: Convert the Emmy IR Graph to a PTX code
        version_config = {
            "version": f"{ptx_config['version']['major']}.{ptx_config['version']['minor']}",
            "target": ptx_config['target'],
            "address_size": ptx_config['address_size'],
        }
        ptx = PTXGenerator(graph, version_config).generate_ptx()
        return ptx
    except Exception as e:
        logging.error(f"error: {e}")
        return ""

#region main
if __name__ == "__main__":
    logging.info("Starting the compilation process...")
    logging.info("--------------------------------\n\n")
    ir_path = Path(sys.argv[1])
    ir_dict = json.loads(ir_path.read_text())
    ptx = compile(ir_dict)
    ptx_path = ir_path.with_suffix(".ptx")
    ptx_path.write_text(ptx)
    print(f"status: compiled {ir_path} to {ptx_path}")
    logging.info("Compilation process completed successfully!")
#endregion