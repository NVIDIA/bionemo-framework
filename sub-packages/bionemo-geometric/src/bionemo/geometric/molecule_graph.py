from torch_geometric.data import Data

class MoleculeGraph(Data):
    """A class for storing and processing molecular graph information"""
    def __init__(self, x_n: torch.tensor, edges: torch.tensor, x_e, x_g: torch.tensor):
        """Initializes MoleculeGraph data point object.

        Initializes MoleculeGraph data point with node attributes `x_n`, edges, edge attributes `e_n` and graph attributes `x_g`"""
        self.x_g = x_g
        super().__init__(x_n, x_n, x_e)


