"""
Graph Neural Network para Predicción de Asistencias
==================================================

Implementa GNNs especializados que modelan las relaciones complejas entre
jugadores, equipos y oponentes para predecir asistencias de manera más precisa.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import logging
import numpy as np

from .base import BaseDLModel, MLPBlock, initialize_weights

logger = logging.getLogger(__name__)

# Simulamos torch_geometric para compatibilidad
try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    logger.warning("torch_geometric no disponible. Usando implementación personalizada de GNN.")


class PlayerTeamGNN(BaseDLModel):
    """
    Graph Neural Network que modela relaciones jugador-equipo-oponente.
    
    Características:
    - Nodos: jugadores, equipos, oponentes
    - Aristas: relaciones de pertenencia, matchups, sinergia
    - Múltiples tipos de convolución: GCN, GAT, GraphSAGE
    - Agregación jerárquica para diferentes niveles de información
    """
    
    def __init__(self, config):
        """
        Inicializa el modelo GNN.
        
        Args:
            config: Configuración del modelo (GNNConfig)
        """
        super(PlayerTeamGNN, self).__init__(config, "PlayerTeamGNN")
        
        self.node_features = config.node_features
        self.edge_features = config.edge_features
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_gnn_layers
        self.gnn_type = config.gnn_type
        
        # Embeddings iniciales
        self.node_embedding = nn.Linear(self.node_features, self.hidden_dim)
        self.edge_embedding = nn.Linear(self.edge_features, self.hidden_dim) if self.edge_features > 0 else None
        
        # Capas GNN
        self.gnn_layers = nn.ModuleList()
        
        if HAS_TORCH_GEOMETRIC:
            self._build_torch_geometric_layers(config)
        else:
            self._build_custom_gnn_layers(config)
        
        # Normalización por capas
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)
        ])
        
        # Attention para agregación de nodos
        self.node_attention = NodeAttention(self.hidden_dim)
        
        # Cabezas de predicción
        self._build_prediction_heads(config)
        
        # Inicializar pesos
        self.apply(initialize_weights)
        
        logger.info(f"GNN inicializado: tipo={self.gnn_type}, "
                   f"hidden_dim={self.hidden_dim}, layers={self.num_layers}")
    
    def _build_torch_geometric_layers(self, config):
        """Construye capas usando torch_geometric."""
        for i in range(self.num_layers):
            if self.gnn_type == "GCN":
                layer = GCNConv(self.hidden_dim, self.hidden_dim)
            elif self.gnn_type == "GAT":
                layer = GATConv(
                    self.hidden_dim, 
                    self.hidden_dim // config.num_attention_heads,
                    heads=config.num_attention_heads,
                    dropout=config.attention_dropout,
                    concat=True
                )
            elif self.gnn_type == "GraphSAGE":
                layer = SAGEConv(self.hidden_dim, self.hidden_dim)
            else:
                layer = GCNConv(self.hidden_dim, self.hidden_dim)
            
            self.gnn_layers.append(layer)
    
    def _build_custom_gnn_layers(self, config):
        """Construye capas GNN personalizadas."""
        for i in range(self.num_layers):
            if self.gnn_type == "GCN":
                layer = CustomGCNLayer(self.hidden_dim, self.hidden_dim)
            elif self.gnn_type == "GAT":
                layer = CustomGATLayer(
                    self.hidden_dim, 
                    self.hidden_dim,
                    num_heads=config.num_attention_heads,
                    dropout=config.attention_dropout
                )
            else:
                layer = CustomGCNLayer(self.hidden_dim, self.hidden_dim)
            
            self.gnn_layers.append(layer)
    
    def _build_prediction_heads(self, config):
        """Construye las cabezas de predicción."""
        
        # Cabeza principal de AST
        self.ast_predictor = nn.Sequential(
            MLPBlock(self.hidden_dim, 128, dropout=config.dropout_rate),
            MLPBlock(128, 64, dropout=config.dropout_rate),
            MLPBlock(64, 32, dropout=config.dropout_rate),
            nn.Linear(32, 1)
        )
        
        # Cabeza auxiliar para contexto de equipo
        self.team_context_predictor = nn.Sequential(
            MLPBlock(self.hidden_dim, 64, dropout=config.dropout_rate),
            nn.Linear(64, 5)  # Predice métricas de equipo
        )
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_features: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass del GNN.
        
        Args:
            node_features: Features de nodos [num_nodes, node_features]
            edge_index: Índices de aristas [2, num_edges]
            edge_features: Features de aristas [num_edges, edge_features]
            batch: Índices de batch [num_nodes]
            
        Returns:
            Predicciones de asistencias [batch_size, 1]
        """
        # Embedding inicial
        x = self.node_embedding(node_features)  # [num_nodes, hidden_dim]
        
        # Procesar edge features si existen
        if edge_features is not None and self.edge_embedding is not None:
            edge_attr = self.edge_embedding(edge_features)
        else:
            edge_attr = None
        
        # Aplicar capas GNN
        for i, (gnn_layer, norm_layer) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            # Residual connection
            residual = x
            
            # GNN layer
            if HAS_TORCH_GEOMETRIC and hasattr(gnn_layer, 'forward'):
                if edge_attr is not None and hasattr(gnn_layer, 'edge_dim'):
                    x = gnn_layer(x, edge_index, edge_attr=edge_attr)
                else:
                    x = gnn_layer(x, edge_index)
            else:
                x = gnn_layer(x, edge_index, edge_attr)
            
            # Normalización y residual
            x = norm_layer(x)
            x = F.relu(x + residual)
            x = F.dropout(x, p=self.config.dropout_rate, training=self.training)
        
        # Agregación por grafo/batch
        if batch is not None:
            if HAS_TORCH_GEOMETRIC:
                # Usar agregación de torch_geometric
                x = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
            else:
                # Agregación manual
                x = self._manual_batch_aggregation(x, batch)
        else:
            # Si no hay batch, asumir un solo grafo
            x = torch.mean(x, dim=0, keepdim=True)  # [1, hidden_dim]
        
        # Predicción final
        ast_prediction = self.ast_predictor(x)  # [batch_size, 1]
        
        return ast_prediction
    
    def _manual_batch_aggregation(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Agregación manual por batch."""
        batch_size = batch.max().item() + 1
        aggregated = []
        
        for i in range(batch_size):
            mask = (batch == i)
            if mask.sum() > 0:
                batch_nodes = x[mask]
                # Usar attention pooling para esta batch
                pooled, _ = self.node_attention(batch_nodes.unsqueeze(0))
                aggregated.append(pooled.squeeze(0))
            else:
                aggregated.append(torch.zeros(self.hidden_dim, device=x.device))
        
        return torch.stack(aggregated)
    
    def forward_with_attention(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                             edge_features: Optional[torch.Tensor] = None,
                             batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """Forward pass que retorna información de atención."""
        
        # Forward normal
        prediction = self.forward(node_features, edge_index, edge_features, batch)
        
        # Extraer información de atención (simplificado)
        attention_info = {
            'node_importance': torch.mean(node_features, dim=1).detach().cpu().numpy(),
            'prediction': prediction.detach().cpu().numpy()
        }
        
        return prediction, attention_info


class NodeAttention(nn.Module):
    """Módulo de atención para agregación de nodos."""
    
    def __init__(self, input_dim: int):
        super(NodeAttention, self).__init__()
        
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, num_nodes, input_dim] o [num_nodes, input_dim]
            mask: Máscara opcional
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Agregar dimensión de batch
        
        # Calcular scores de atención
        attention_scores = self.attention_net(x)  # [batch_size, num_nodes, 1]
        attention_scores = attention_scores.squeeze(-1)  # [batch_size, num_nodes]
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Aplicar atención
        weighted_output = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)
        
        return weighted_output, attention_weights


class CustomGCNLayer(nn.Module):
    """Implementación personalizada de Graph Convolution."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super(CustomGCNLayer, self).__init__()
        
        self.linear = nn.Linear(input_dim, output_dim)
        self.bias = nn.Parameter(torch.zeros(output_dim))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes (no usado en GCN básico)
        """
        # Transformación lineal
        x = self.linear(x)
        
        # Crear matriz de adyacencia
        num_nodes = x.size(0)
        adj_matrix = torch.zeros(num_nodes, num_nodes, device=x.device)
        
        # Llenar matriz de adyacencia
        row, col = edge_index
        adj_matrix[row, col] = 1.0
        adj_matrix[col, row] = 1.0  # Grafo no dirigido
        
        # Normalización simétrica
        degree = torch.sum(adj_matrix, dim=1)
        degree_inv_sqrt = torch.pow(degree + 1e-6, -0.5)
        degree_matrix_inv_sqrt = torch.diag(degree_inv_sqrt)
        
        normalized_adj = torch.mm(torch.mm(degree_matrix_inv_sqrt, adj_matrix), degree_matrix_inv_sqrt)
        
        # Convolución
        output = torch.mm(normalized_adj, x) + self.bias
        
        return output


class CustomGATLayer(nn.Module):
    """Implementación personalizada de Graph Attention."""
    
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super(CustomGATLayer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        
        # Proyecciones para Q, K, V
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        
        # Attention weights
        self.attention = nn.Parameter(torch.randn(num_heads, 2 * self.head_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
        """
        num_nodes = x.size(0)
        
        # Proyecciones
        q = self.query(x).view(num_nodes, self.num_heads, self.head_dim)
        k = self.key(x).view(num_nodes, self.num_heads, self.head_dim)
        v = self.value(x).view(num_nodes, self.num_heads, self.head_dim)
        
        # Calcular attention para cada arista
        row, col = edge_index
        
        # Concatenar features de nodos conectados
        edge_features = torch.cat([q[row], k[col]], dim=-1)  # [num_edges, num_heads, 2*head_dim]
        
        # Calcular scores de atención
        attention_scores = torch.sum(edge_features * self.attention.unsqueeze(0), dim=-1)  # [num_edges, num_heads]
        attention_scores = self.leaky_relu(attention_scores)
        
        # Softmax por nodo de destino
        attention_weights = torch.zeros(num_nodes, self.num_heads, device=x.device)
        for i in range(num_nodes):
            mask = (col == i)
            if mask.sum() > 0:
                scores = attention_scores[mask]
                weights = F.softmax(scores, dim=0)
                attention_weights[i] = torch.mean(weights, dim=0)
        
        # Aplicar attention
        output = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=x.device)
        
        for i in range(len(row)):
            src, dst = row[i], col[i]
            for h in range(self.num_heads):
                output[dst, h] += attention_weights[dst, h] * v[src, h]
        
        # Concatenar cabezas
        output = output.view(num_nodes, self.output_dim)
        
        return self.dropout(output)


class HierarchicalGNN(PlayerTeamGNN):
    """
    GNN Jerárquico que procesa información a múltiples niveles:
    - Nivel jugador: relaciones individuales
    - Nivel equipo: dinámicas de equipo
    - Nivel liga: patrones globales
    """
    
    def __init__(self, config):
        super(HierarchicalGNN, self).__init__(config)
        
        # GNNs especializados por nivel
        self.player_gnn = self._create_level_gnn("player")
        self.team_gnn = self._create_level_gnn("team")
        self.league_gnn = self._create_level_gnn("league")
        
        # Fusión de niveles
        self.level_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )
    
    def _create_level_gnn(self, level: str) -> nn.Module:
        """Crea un GNN especializado para un nivel específico."""
        if level == "player":
            # GNN enfocado en relaciones individuales
            return CustomGATLayer(self.hidden_dim, self.hidden_dim, num_heads=4)
        elif level == "team":
            # GNN enfocado en dinámicas de equipo
            return CustomGCNLayer(self.hidden_dim, self.hidden_dim)
        else:  # league
            # GNN enfocado en patrones globales
            return CustomGCNLayer(self.hidden_dim, self.hidden_dim)
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_features: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass jerárquico."""
        
        # Embedding inicial
        x = self.node_embedding(node_features)
        
        # Procesar en cada nivel
        player_output = self.player_gnn(x, edge_index)
        team_output = self.team_gnn(x, edge_index)
        league_output = self.league_gnn(x, edge_index)
        
        # Agregación por batch si es necesario
        if batch is not None:
            player_output = self._manual_batch_aggregation(player_output, batch)
            team_output = self._manual_batch_aggregation(team_output, batch)
            league_output = self._manual_batch_aggregation(league_output, batch)
        else:
            player_output = torch.mean(player_output, dim=0, keepdim=True)
            team_output = torch.mean(team_output, dim=0, keepdim=True)
            league_output = torch.mean(league_output, dim=0, keepdim=True)
        
        # Fusionar niveles
        fused_features = torch.cat([player_output, team_output, league_output], dim=1)
        fused_output = self.level_fusion(fused_features)
        
        # Predicción final
        ast_prediction = self.ast_predictor(fused_output)
        
        return ast_prediction 