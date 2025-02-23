from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    add_self_loops,
    coalesce,
    get_ppr,
    is_undirected,
    scatter,
    sort_edge_index,
    to_dense_adj,
)


@functional_transform('khopgdc')
class GDC(BaseTransform):

    def __init__(
            self,
            self_loop_weight: float = 1.,
            normalization_in: str = 'sym',
            normalization_out: str = 'col',
            diffusion_kwargs: Dict[str, Any] = dict(method='ppr', alpha=0.15),
            sparsification_kwargs: Dict[str, Any] = dict(
                method='threshold',
                avg_degree=64,
            ),
            exact: bool = True,
    ) -> None:
        self.self_loop_weight = self_loop_weight
        self.normalization_in = normalization_in
        self.normalization_out = normalization_out
        self.diffusion_kwargs = diffusion_kwargs
        self.sparsification_kwargs = sparsification_kwargs
        self.exact = exact

        if self_loop_weight:
            assert exact or self_loop_weight == 1

    @torch.no_grad()
    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        edge_index = data.edge_index
        N = data.num_nodes
        assert N is not None

        if data.edge_attr is None:
            edge_weight = torch.ones(edge_index.size(1),
                                     device=edge_index.device)
        else:
            edge_weight = data.edge_attr
            assert self.exact
            assert edge_weight.dim() == 1

        if self.self_loop_weight:
            edge_index, edge_weight = add_self_loops(
                edge_index, edge_weight, fill_value=self.self_loop_weight,
                num_nodes=N)

        edge_index, edge_weight = coalesce(edge_index, edge_weight, N)

        if self.exact:
            edge_index, edge_weight = self.transition_matrix(
                edge_index, edge_weight, N, self.normalization_in)
            diff_mat = self.diffusion_matrix_exact(edge_index, edge_weight, N,
                                                   **self.diffusion_kwargs)
            edge_index, edge_weight = self.sparsify_dense(
                diff_mat, **self.sparsification_kwargs)
        else:
            edge_index, edge_weight = self.diffusion_matrix_approx(
                edge_index, edge_weight, N, self.normalization_in,
                **self.diffusion_kwargs)
            edge_index, edge_weight = self.sparsify_sparse(
                edge_index, edge_weight, N, **self.sparsification_kwargs)

        edge_index, edge_weight = coalesce(edge_index, edge_weight, N)
        edge_index, edge_weight = self.transition_matrix(
            edge_index, edge_weight, N, self.normalization_out)

        data.edge_index = edge_index
        data.edge_attr = edge_weight

        return data

    def transition_matrix(
            self,
            edge_index: Tensor,
            edge_weight: Tensor,
            num_nodes: int,
            normalization: str,
    ) -> Tuple[Tensor, Tensor]:

        if normalization == 'sym':
            row, col = edge_index
            deg = scatter(edge_weight, col, 0, num_nodes, reduce='sum')
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        elif normalization == 'col':
            _, col = edge_index
            deg = scatter(edge_weight, col, 0, num_nodes, reduce='sum')
            deg_inv = 1. / deg
            deg_inv[deg_inv == float('inf')] = 0
            edge_weight = edge_weight * deg_inv[col]
        elif normalization == 'row':
            row, _ = edge_index
            deg = scatter(edge_weight, row, 0, num_nodes, reduce='sum')
            deg_inv = 1. / deg
            deg_inv[deg_inv == float('inf')] = 0
            edge_weight = edge_weight * deg_inv[row]
        elif normalization is None:
            pass
        else:
            raise ValueError(
                f"Transition matrix normalization '{normalization}' unknown")

        return edge_index, edge_weight

    def diffusion_matrix_exact(
        self,
        edge_index: Tensor,
        edge_weight: Tensor,
        num_nodes: int,
        method: str,
        **kwargs: Any,
    ) -> Tensor:
   
        if method == 'ppr':
            # α (I_n + (α - 1) A)^-1
            edge_weight = (kwargs['alpha'] - 1) * edge_weight
            edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                     fill_value=1,
                                                     num_nodes=num_nodes)
            mat = to_dense_adj(edge_index, edge_attr=edge_weight).squeeze()
            diff_matrix = kwargs['alpha'] * torch.inverse(mat)

        elif method == 'heat':
            # exp(t (A - I_n))
            edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                     fill_value=-1,
                                                     num_nodes=num_nodes)
            edge_weight = kwargs['t'] * edge_weight
            mat = to_dense_adj(edge_index, edge_attr=edge_weight).squeeze()
            undirected = is_undirected(edge_index, edge_weight, num_nodes)
            diff_matrix = self.__expm__(mat, undirected)
        elif method == 'gaussian':
              # 使用 K-hop 高斯扩散核心
              adj_matrix = to_dense_adj(edge_index, edge_attr=edge_weight).squeeze()
              k = kwargs.get('k', 1)  # 获取 k 值，默认为1
              sigma = kwargs.get('sigma', 1)  # 获取 sigma 值，标准正态分布默认值为1

              # 初始化结果矩阵为邻接矩阵（1-hop）
              k_hop_matrix = adj_matrix.clone()
              current_matrix = adj_matrix.clone()

              # 计算 K-hop 高斯扩散矩阵
              for i in range(1, k):
                  # 计算下一个幂次的邻接矩阵
                  current_matrix = torch.matmul(current_matrix, adj_matrix)

                  if sigma is not None:
                      # 计算高斯权重
                      weight = torch.exp(- (torch.tensor(i ** 2, dtype=torch.float32) / (2 * sigma ** 2)))
                      # 将权重应用到矩阵上并累加
                      k_hop_matrix += weight * current_matrix
                  else:
                      # 没有 sigma 参数，不进行高斯加权，直接累加
                      k_hop_matrix += current_matrix

              # 对高斯权重进行归一化
              if sigma is not None:
                  total_weight = sum(torch.exp(- (torch.tensor(i ** 2, dtype=torch.float32) / (2 * sigma ** 2))) for i in range(1, k))
                  k_hop_matrix /= total_weight

              diff_matrix = k_hop_matrix
        elif method == 'kppr':
            # 计算 PPR 和 k-hop 扩散矩阵，然后相乘
            k = kwargs.get('k', 2)
            alpha = kwargs.get('alpha', 0.15)
            adj_matrix = to_dense_adj(edge_index, edge_attr=edge_weight).squeeze()
            ppr_matrix = self.compute_ppr_matrix(edge_index, edge_weight, num_nodes, alpha)
            khop_matrix = self.compute_k_hop_weighted_matrix(adj_matrix, k)
            diff_matrix = ppr_matrix * khop_matrix

        elif method == 'kheat':
            # 计算 Heat 和 k-hop 扩散矩阵，然后相乘
            k = kwargs.get('k', 2)
            t = kwargs.get('t', 1.0)
            adj_matrix = to_dense_adj(edge_index, edge_attr=edge_weight).squeeze()
            heat_matrix = self.compute_heat_matrix(edge_index, edge_weight, num_nodes, t)
            khop_matrix = self.compute_k_hop_weighted_matrix(adj_matrix, k)
            diff_matrix = heat_matrix * khop_matrix

        else:
            raise ValueError(f"Exact GDC diffusion '{method}' unknown")

        return diff_matrix

    def compute_ppr_matrix(self, edge_index: Tensor, edge_weight: Tensor, num_nodes: int, alpha: float) -> Tensor:
        """计算 PPR 扩散矩阵。"""
        edge_weight = (alpha - 1) * edge_weight
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=1, num_nodes=num_nodes)
        mat = to_dense_adj(edge_index, edge_attr=edge_weight).squeeze()
        ppr_matrix = alpha * torch.inverse(mat)
        return ppr_matrix


    def compute_heat_matrix(self, edge_index: Tensor, edge_weight: Tensor, num_nodes: int, t: float) -> Tensor:
        """计算 Heat 扩散矩阵。"""
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=-1, num_nodes=num_nodes)
        edge_weight = t * edge_weight
        mat = to_dense_adj(edge_index, edge_attr=edge_weight).squeeze()
        undirected = is_undirected(edge_index, edge_weight, num_nodes)
        heat_matrix = self.__expm__(mat, undirected)
        return heat_matrix

    def compute_k_hop_weighted_matrix(self, adj_matrix: Tensor, k: int, sigma: float = None) -> Tensor:
        """
        计算带有高斯权重的 k-hop 范围内的邻域矩阵。
        如果 sigma 为 None，则不执行高斯加权。
        """
        # 初始化结果矩阵为邻接矩阵（1-hop）
        k_hop_matrix = adj_matrix.clone()

        # 当前矩阵表示累积到的 k-hop
        current_matrix = adj_matrix.clone()

        # 计算从 2-hop 到 k-hop 的所有矩阵并累加
        if k == 1:
            current_matrix = torch.matmul(current_matrix, adj_matrix)
            k_hop_matrix += current_matrix
            k_hop_matrix = torch.where(k_hop_matrix > 0, torch.ones_like(k_hop_matrix),
                                       torch.zeros_like(k_hop_matrix))
            
        else:
            for i in range(1, k):
                # 计算下一个幂次的邻接矩阵
                current_matrix = torch.matmul(current_matrix, adj_matrix)

                if sigma is not None:
                    # 计算高斯权重，将 float 转换为 Tensor
                    weight = torch.exp(-torch.tensor((i + 1 - 1) ** 2) / (2 * sigma ** 2))
                    # 将权重应用到矩阵上并累加
                    k_hop_matrix += weight * current_matrix
                else:
                    # 如果没有 sigma 参数，不进行高斯加权，直接累加
                    k_hop_matrix += current_matrix
                    #k_hop_matrix = torch.where(k_hop_matrix > 0, torch.ones_like(k_hop_matrix),
                                               #torch.zeros_like(k_hop_matrix))



        return k_hop_matrix

    def diffusion_matrix_approx(  # noqa: D417
            self,
            edge_index: Tensor,
            edge_weight: Tensor,
            num_nodes: int,
            normalization: str,
            method: str,
            **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:

        if method == 'ppr':
            if normalization == 'sym':
                # Calculate original degrees.
                _, col = edge_index
                deg = scatter(edge_weight, col, 0, num_nodes, reduce='sum')

            edge_index, edge_weight = get_ppr(
                edge_index,
                alpha=kwargs['alpha'],
                eps=kwargs['eps'],
                num_nodes=num_nodes,
            )

            if normalization == 'col':
                edge_index, edge_weight = sort_edge_index(
                    edge_index.flip([0]), edge_weight, num_nodes)

            if normalization == 'sym':
                row, col = edge_index
                deg_inv = deg.sqrt()
                deg_inv_sqrt = deg.pow(-0.5)
                deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
                edge_weight = deg_inv[row] * edge_weight * deg_inv_sqrt[col]
            elif normalization in ['col', 'row']:
                pass
            else:
                raise ValueError(
                    f"Transition matrix normalization '{normalization}' not "
                    f"implemented for non-exact GDC computation")

        elif method == 'heat':
            raise NotImplementedError(
                'Currently no fast heat kernel is implemented. You are '
                'welcome to create one yourself, e.g., based on '
                '"Kloster and Gleich: Heat kernel based community detection '
                '(KDD 2014)."')
        else:
            raise ValueError(f"Approximate GDC diffusion '{method}' unknown")

        return edge_index, edge_weight

    def sparsify_dense(  # noqa: D417
            self,
            matrix: Tensor,
            method: str,
            **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:

        assert matrix.shape[0] == matrix.shape[1]
        N = matrix.shape[1]

        if method == 'threshold':
            if 'eps' not in kwargs.keys():
                kwargs['eps'] = self.__calculate_eps__(matrix, N,
                                                       kwargs['avg_degree'])

            edge_index = (matrix >= kwargs['eps']).nonzero(as_tuple=False).t()
            edge_index_flat = edge_index[0] * N + edge_index[1]
            edge_weight = matrix.flatten()[edge_index_flat]

        elif method == 'topk':
            k, dim = min(N, kwargs['k']), kwargs['dim']
            assert dim in [0, 1]
            sort_idx = torch.argsort(matrix, dim=dim, descending=True)
            if dim == 0:
                top_idx = sort_idx[:k]
                edge_weight = torch.gather(matrix, dim=dim,
                                           index=top_idx).flatten()

                row_idx = torch.arange(0, N, device=matrix.device).repeat(k)
                edge_index = torch.stack([top_idx.flatten(), row_idx], dim=0)
            else:
                top_idx = sort_idx[:, :k]
                edge_weight = torch.gather(matrix, dim=dim,
                                           index=top_idx).flatten()

                col_idx = torch.arange(
                    0, N, device=matrix.device).repeat_interleave(k)
                edge_index = torch.stack([col_idx, top_idx.flatten()], dim=0)
        else:
            raise ValueError(f"GDC sparsification '{method}' unknown")

        return edge_index, edge_weight

    def sparsify_sparse(  # noqa: D417
            self,
            edge_index: Tensor,
            edge_weight: Tensor,
            num_nodes: int,
            method: str,
            **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:

        if method == 'threshold':
            if 'eps' not in kwargs.keys():
                kwargs['eps'] = self.__calculate_eps__(
                    edge_weight,
                    num_nodes,
                    kwargs['avg_degree'],
                )

            remaining_edge_idx = (edge_weight >= kwargs['eps']).nonzero(
                as_tuple=False).flatten()
            edge_index = edge_index[:, remaining_edge_idx]
            edge_weight = edge_weight[remaining_edge_idx]
        elif method == 'topk':
            raise NotImplementedError(
                'Sparse topk sparsification not implemented')
        else:
            raise ValueError(f"GDC sparsification '{method}' unknown")

        return edge_index, edge_weight

    def __expm__(self, matrix: Tensor, symmetric: bool) -> Tensor:

        from scipy.linalg import expm

        if symmetric:
            e, V = torch.linalg.eigh(matrix, UPLO='U')
            diff_mat = V @ torch.diag(e.exp()) @ V.t()
        else:
            diff_mat = torch.from_numpy(expm(matrix.cpu().numpy()))
            diff_mat = diff_mat.to(matrix.device, matrix.dtype)
        return diff_mat

    def __calculate_eps__(
            self,
            matrix: Tensor,
            num_nodes: int,
            avg_degree: int,
    ) -> float:

        sorted_edges = torch.sort(matrix.flatten(), descending=True).values
        if avg_degree * num_nodes > len(sorted_edges):
            return -np.inf

        left = sorted_edges[avg_degree * num_nodes - 1]
        right = sorted_edges[avg_degree * num_nodes]
        return float(left + right) / 2.0