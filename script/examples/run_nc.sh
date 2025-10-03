#To run QGCN2:

python main.py --task=nc --dataset=cora --model=QGCN2 --time_dim=14 --space_dim=2 --epochs=1000 --lr=0.005 \
 --weight_decay=1e-1  --seed= --using_riemannianAdam=True --act=tanh --patience=1000 --step_lr=1000 --dropout=0.0 --use_feats (2 layers, gcn encoder, residual) 

python main.py --task=nc --dataset=citeseer --model=QGCN2 --time_dim=8 --space_dim=8 --epochs=1000 --lr=0.005 \
 --weight_decay=1e-1 --seed= --using_riemannianAdam=True --act=tanh --patience=1000 --step_lr=1000 --dropout=0.0 --use_feats --normalize_feats (2 layers, linear encoder, no residual) 

python main.py --task=nc --dataset=pubmed --model=QGCN2 --time_dim=14 --space_dim=2 --epochs=1000 --lr=0.005 \
 --weight_decay=1e-1  --seed= --using_riemannianAdam=True --act=tanh --patience=1000 --step_lr=1000 --dropout=0.0 --use_feats --normalize_feats (2 layers, no residual, linear encoder) 

python main.py --task=nc --dataset=airport --model=QGCN2 --time_dim=15 --space_dim=1 --epochs=1000 --lr=0.01 \
 --weight_decay=5e-4  --seed= --using_riemannianAdam=True --act=relu --patience=1000 --step_lr=1000  --dropout=0.0 --use_feats (2 layers, no residual, gcn encoder) 

python main.py --task=nc --dataset=ogbn-arxiv --model=QGCN2 --time_dim=15 --space_dim=1 --epochs=1000 --lr=0.01 \
 --weight_decay=5e-7  --seed= --using_riemannianAdam=True --act=tanh --patience=1000 --step_lr=1000  --dropout=0.0 --use_feats (2 layers, no residual, no gcn encoder) 

python main.py --task=nc --dataset=fb100 --model=QGCN2 --time_dim=15 --space_dim=1 --epochs=1000 --lr=0.01 \
 --weight_decay=5e-7  --seed= --using_riemannianAdam=True --act=tanh --patience=1000 --step_lr=1000  --dropout=0.0 --use_feats (2 layers, no residual, no gcn encoder) 

python main.py --task=nc --dataset=twitch_gamer --model=QGCN2 --time_dim=16 --space_dim=0 --epochs=1000 --lr=0.001 \
 --weight_decay=5e-7  --seed= --using_riemannianAdam=True --act=relu --patience=1000 --step_lr=1000  --dropout=0.0 --use_feats (2 layers, no residual, no gcn encoder) 

# To run QGT

python main.py --task=nc --dataset=cora --model=QGT --trans_num_layers=3 --trans_num_heads=1 --epochs=1000 \
 --lr=0.005 --seed=1234 --patience=1000 --min_epoch=250  --act=tanh --time_dim=14 --space_dim=2  --graph_num_layers=3 \
 --graph_weight=0.8 --weight_decay=1e-1 --weight_decay_2=1e-1 --weight_decay_3=5e-2 --normalize_feats --dropout_time=0.2 \
 --dropout_space=0.2 --g_dropout_time=0.0 --g_dropout_space=0.0 --use_feats # gcn encoding

python main.py --task=nc --dataset=pubmed --model=QGT --trans_num_layers=2 --trans_num_heads=1 --epochs=1000 --lr=0.01 \
 --seed=1234 --patience=1000 --min_epoch=1000  --act=tanh --time_dim=14 --space_dim=2  --graph_num_layers=4  --graph_weight=0.8 \
 --weight_decay=1e-1 --weight_decay_2=1e-1 --weight_decay_3=5e-2 --normalize_feats  --dropout_time=0.3 --dropout_space=0.0 \
 --g_dropout_time=0.4 --g_dropout_space=0.0 --use_feats --trans_use_residual # gcn encoding

python main.py --task=nc --dataset=citeseer --model=QGT --trans_num_layers=3 --trans_num_heads=1 --epochs=1000 \
 --lr=0.002 --seed=1234 --patience=1000 --min_epoch=250  --act=tanh --time_dim=8 --space_dim=8  --graph_num_layers=2 \
 --graph_weight=0.8 --weight_decay=1e-1 --weight_decay_2=1e-1 --weight_decay_3=5e-2 --normalize_feats --dropout_time=0.2 \
 --dropout_space=0.2 --g_dropout_time=0.0 --g_dropout_space=0.0 --use_feats --trans_use_bn # gcn encoding

python main.py --task=nc --dataset=airport --model=QGT --trans_num_layers=1 --trans_num_heads=1 --epochs=1000 --lr=0.005 \
 --seed=1234 --patience=1000 --min_epoch=250  --act=relu --time_dim=15 --space_dim=1  --graph_num_layers=3 \
 --graph_weight=0.8 --weight_decay=5e-4 --weight_decay_2=5e-4 --weight_decay_3=5e-4 --dropout_time=0.0 \
 -dropout_space=0.0 --g_dropout_time=0.0 --g_dropout_space=0.0  --use_feats # gcn encoding

python main.py --task=nc --dataset=ogbn-arxiv --model=QGT --trans_num_layers=1 --trans_num_heads=1 --epochs=1000 \
 --lr=0.005 --seed=1234 --patience=1000 --min_epoch=250  --act=tanh --time_dim=15 --space_dim=1  --graph_num_layers=3 \
 --graph_weight=0.5 --weight_decay=5e-7 --weight_decay_2=5e-7 --weight_decay_3=5e-7 --dropout_time=0.0 \
 --dropout_space=0.0 --g_dropout_time=0.0 --g_dropout_space=0.0  --use_feats # gcn encoding

python main.py --task=nc --dataset=fb100 --model=QGT --trans_num_layers=1 --trans_num_heads=1 --epochs=1000 \
 --lr=0.001 --seed=1234 --patience=1000 --min_epoch=250  --act=tanh --time_dim=15 --space_dim=1  \
 --graph_num_layers=2 --graph_weight=0.9 --weight_decay=5e-12 --weight_decay_2=5e-12 --weight_decay_3=5e-12 \
 --dropout_time=0.4 --dropout_space=0.0 --g_dropout_time=0.4 --g_dropout_space=0.0 --normalize_feats --use_feats # gcn encoding

python main.py --task=nc --dataset=twitch_gamer --model=QGT --trans_num_layers=1 --trans_num_heads=1 --epochs=1000 \
 --lr=0.05 --seed=1234 --patience=1000 --min_epoch=250  --act=tanh --time_dim=16 --space_dim=0  --graph_num_layers=2 \
 --graph_weight=0.7 --weight_decay=5e-4 --weight_decay_2=5e-4 --weight_decay_3=5e-4 --dropout_time=0.4 \
 --dropout_space=0.0 --g_dropout_time=0.4 --g_dropout_space=0.0 --normalize_feats  --use_feats # gcn encoding


## Large graph
python main.py --task=nc --dataset=ogbn-products --model=QGT --trans_num_layers=1 --trans_num_heads=1 --epochs=200  --lr=0.01 \
 --seed=1234 --patience=200 --min_epoch=250  --act=tanh --time_dim=15 --space_dim=1  --graph_num_layers=3  --graph_weight=0.8 \
 --weight_decay=1e-4 --weight_decay_2=1e-4 --weight_decay_3=5e-4 --normalize_feats --dropout_time=0.0  --dropout_space=0.0 \
 --g_dropout_time=0.0 --g_dropout_space=0.0 --use_feats


python main.py --task=nc --dataset=ogbn-products --model=QGCN2 --time_dim=15 --space_dim=1 --epochs=200 --lr=0.005 --weight_decay=5e-4 \
 --seed=1234 --using_riemannianAdam=True --act=tanh --patience=200 --step_lr=1000 --dropout=0.0 --use_feats (2 layers, no res)
