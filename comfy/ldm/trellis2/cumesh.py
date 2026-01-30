# will contain every cuda -> pytorch operation

import torch
from typing import Dict


class TorchHashMap:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor, default_value: int):
        device = keys.device
        # use long for searchsorted
        self.sorted_keys, order = torch.sort(keys.long())
        self.sorted_vals = values.long()[order]
        self.default_value = torch.tensor(default_value, dtype=torch.long, device=device)
        self._n = self.sorted_keys.numel()

    def lookup_flat(self, flat_keys: torch.Tensor) -> torch.Tensor:
        flat = flat_keys.long()
        idx = torch.searchsorted(self.sorted_keys, flat)
        found = (idx < self._n) & (self.sorted_keys[idx] == flat)
        out = torch.full((flat.shape[0],), self.default_value, device=flat.device, dtype=self.sorted_vals.dtype)
        if found.any():
            out[found] = self.sorted_vals[idx[found]]
        return out

class Voxel:
    def __init__(
            self,
            origin: list,
            voxel_size: float,
            coords: torch.Tensor = None,
            attrs: torch.Tensor = None,
            layout: Dict = {},
            device: torch.device = 'cuda'
        ):
        self.origin = torch.tensor(origin, dtype=torch.float32, device=device)
        self.voxel_size = voxel_size
        self.coords = coords
        self.attrs = attrs
        self.layout = layout
        self.device = device

    @property
    def position(self):
        return (self.coords + 0.5) * self.voxel_size + self.origin[None, :]

    def split_attrs(self):
        return {
            k: self.attrs[:, self.layout[k]]
            for k in self.layout
        }

class Mesh:
    def __init__(self,
        vertices,
        faces,
        vertex_attrs=None
    ):
        self.vertices = vertices.float()
        self.faces = faces.int()
        self.vertex_attrs = vertex_attrs

    @property
    def device(self):
        return self.vertices.device

    def to(self, device, non_blocking=False):
        return Mesh(
            self.vertices.to(device, non_blocking=non_blocking),
            self.faces.to(device, non_blocking=non_blocking),
            self.vertex_attrs.to(device, non_blocking=non_blocking) if self.vertex_attrs is not None else None,
        )

    def cuda(self, non_blocking=False):
        return self.to('cuda', non_blocking=non_blocking)

    def cpu(self):
        return self.to('cpu')

    # TODO could be an option
    def fill_holes(self, max_hole_perimeter=3e-2):
        import cumesh
        vertices = self.vertices.cuda()
        faces = self.faces.cuda()

        mesh = cumesh.CuMesh()
        mesh.init(vertices, faces)
        mesh.get_edges()
        mesh.get_boundary_info()
        if mesh.num_boundaries == 0:
            return
        mesh.get_vertex_edge_adjacency()
        mesh.get_vertex_boundary_adjacency()
        mesh.get_manifold_boundary_adjacency()
        mesh.read_manifold_boundary_adjacency()
        mesh.get_boundary_connected_components()
        mesh.get_boundary_loops()
        if mesh.num_boundary_loops == 0:
            return
        mesh.fill_holes(max_hole_perimeter=max_hole_perimeter)
        new_vertices, new_faces = mesh.read()

        self.vertices = new_vertices.to(self.device)
        self.faces = new_faces.to(self.device)

    # TODO could be an option
    def simplify(self, target=1000000, verbose: bool=False, options: dict={}):
        import cumesh
        vertices = self.vertices.cuda()
        faces = self.faces.cuda()

        mesh = cumesh.CuMesh()
        mesh.init(vertices, faces)
        mesh.simplify(target, verbose=verbose, options=options)
        new_vertices, new_faces = mesh.read()

        self.vertices = new_vertices.to(self.device)
        self.faces = new_faces.to(self.device)

class MeshWithVoxel(Mesh, Voxel):
    def __init__(self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        origin: list,
        voxel_size: float,
        coords: torch.Tensor,
        attrs: torch.Tensor,
        voxel_shape: torch.Size,
        layout: Dict = {},
    ):
        self.vertices = vertices.float()
        self.faces = faces.int()
        self.origin = torch.tensor(origin, dtype=torch.float32, device=self.device)
        self.voxel_size = voxel_size
        self.coords = coords
        self.attrs = attrs
        self.voxel_shape = voxel_shape
        self.layout = layout

    def to(self, device, non_blocking=False):
        return MeshWithVoxel(
            self.vertices.to(device, non_blocking=non_blocking),
            self.faces.to(device, non_blocking=non_blocking),
            self.origin.tolist(),
            self.voxel_size,
            self.coords.to(device, non_blocking=non_blocking),
            self.attrs.to(device, non_blocking=non_blocking),
            self.voxel_shape,
            self.layout,
        )
