"""Configuration schema for FSTHGT.

Copy this file or override these attributes from your own experiment launcher.
The release package intentionally does not include dataset paths, device IDs, or
training hyperparameter values.
"""

device = "cpu"

dataset = None
trainset_ratio = None
valset_ratio = None
testset_ratio = None
epochs = None
learning_rate = None
weight_decay = None
patience = None
seed = None

eps = 1e-6
window_size = None
pred_k = None
batch = None
time_step_in_day = None

khop = None
dynamic_top_k = None
random_walk_steps = None
laplacian_k = None

d_model = None
d_token = None
embed_dim = None
num_heads = None
self_attn_num_heads = None
dropout = None
num_layers = None
dim_forward = None

use_line_hyperedge = True
use_khop_hyperedge = True
use_dynamic_hyperedge = True
use_flow_aware = True
use_spatial_embedding = True
