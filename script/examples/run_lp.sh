# To run QGT

python run_lp.py --task=lp --dataset=cora --model=QGT --trans_num_layers=1 --trans_num_heads=1 --epochs=1000 \
 --lr=0.005 --seed=1234 --patience=1000 --min_epoch=250  --act=tanh --time_dim=14 --space_dim=2  --graph_num_layers=1 \
 --graph_weight=0.8 --weight_decay=1e-3 --weight_decay_2=1e-3 --weight_decay_3=5e-3 --normalize_feats --dropout_time=0.2 \
 --dropout_space=0.2 --g_dropout_time=0.0 --g_dropout_space=0.0 --use_hyperdecoder --use_feats # gcn encoding

 python run_lp.py --task=lp --dataset=pubmed --model=QGT --trans_num_layers=1 --trans_num_heads=1 --epochs=1000 \
 --lr=0.005 --seed=1234 --patience=1000 --min_epoch=250  --act=tanh --time_dim=14 --space_dim=2  --graph_num_layers=1 \
 --graph_weight=0.8 --weight_decay=1e-3 --weight_decay_2=1e-3 --weight_decay_3=5e-3 --normalize_feats --dropout_time=0.2 \
 --dropout_space=0.2 --g_dropout_time=0.0 --g_dropout_space=0.0 --use_hyperdecoder --use_feats # gcn encoding

 python run_lp.py --task=lp --dataset=citeseer --model=QGT --trans_num_layers=2 --trans_num_heads=1 --epochs=1000 \
 --lr=0.005 --seed=1234 --patience=1000 --min_epoch=250  --act=tanh --time_dim=8 --space_dim=8  --graph_num_layers=2 \
 --graph_weight=0.8 --weight_decay=1e-3 --weight_decay_2=1e-3 --weight_decay_3=5e-3 --normalize_feats --dropout_time=0.2 \
 --dropout_space=0.2 --g_dropout_time=0.0 --g_dropout_space=0.0 --use_hyperdecoder --use_feats # gcn encoding

 python run_lp.py --task=lp --dataset=airport --model=QGT --trans_num_layers=1 --trans_num_heads=1 --epochs=1000 --lr=0.01 \
 --seed=1234 --patience=1000 --min_epoch=250  --act=tanh --time_dim=15 --space_dim=1  --graph_num_layers=1 --graph_weight=0.9 \
 --weight_decay=1e-3 --weight_decay_2=1e-3 --weight_decay_3=5e-3 --dropout_time=0.3 --dropout_space=0.0 --g_dropout_time=0.0 \
 --g_dropout_space=0.0 --use_hyperdecoder --use_feats # gcn encoding


# To run QGCN2

python run_lp.py --task=lp --dataset=cora --model=QGCN2 --time_dim=14 --space_dim=2 --epochs=1000 --lr=0.005 --weight_decay=1e-1  \
 --seed=1234 --using_riemannianAdam=True --act=tanh --patience=1000 --step_lr=1000 --dropout=0.0 --normalize_feats --use_hyperdecoder --use_feats (gcn decoder, no res)

python run_lp.py --task=lp --dataset=citeseer --model=QGCN2 --time_dim=8 --space_dim=8 --epochs=1000 --lr=0.005 --weight_decay=1e-4  \
 --seed=1234 --using_riemannianAdam=True --act=tanh --patience=1000 --step_lr=1000 --dropout=0.3 --normalize_feats --use_hyperdecoder --use_feats (gcn decoder, no res)

python run_lp.py --task=lp --dataset=pubmed --model=QGCN2 --time_dim=14 --space_dim=2 --epochs=1000 --lr=0.005 --weight_decay=1e-4 \
 --seed=1234 --using_riemannianAdam=True --act=tanh --patience=1000 --step_lr=1000 --dropout=0.0 --normalize_feats --use_hyperdecoder --use_feats (gcn decoder, no res)

python run_lp.py --task=lp --dataset=airport --model=QGCN2 --time_dim=15 --space_dim=1 --epochs=1000 --lr=0.02 --weight_decay=1e-4  \
 --seed=1234 --using_riemannianAdam=True --act=relu --patience=1000 --step_lr=1000 --dropout=0.0 --normalize_feats --use_hyperdecoder --use_feats (linear decoder, no res)


 ###### ABLATION STUDY ######


# QGCN2

python run_lp.py --task=lp --dataset=tree1 --model=QGCN2 --time_dim=0 --space_dim=16 --epochs=1000 --lr=0.005 --weight_decay=1e-4 \
  --seed= --using_riemannianAdam=True --act=tanh --patience=1000 --step_lr=1000 --dropout=0.0  --use_hyperdecoder --use_feats (2 layers, no res, gcn encoder)

python run_lp.py --task=lp --dataset=tree2 --model=QGCN2 --time_dim=1 --space_dim=15 --epochs=1000 --lr=0.005 --weight_decay=1e-4  \
 --seed= --using_riemannianAdam=True --act=tanh --patience=1000 --step_lr=1000 --dropout=0.5  --use_hyperdecoder --use_feats (2 layers, no res, gcn encoder)

python run_lp.py --task=lp --dataset=tree3 --model=QGCN2 --time_dim=3 --space_dim=13 --epochs=1000 --lr=0.01 --weight_decay=1e-2 \
 --seed= --using_riemannianAdam=True --act=tanh --patience=1000 --step_lr=1000 --dropout=0.0 --use_hyperdecoder --use_feats (2 layers, no res, gcn encoder)


# QGT

python run_lp.py --task=lp --dataset=tree1 --model=QGT --trans_num_layers=1 --trans_num_heads=1 --epochs=1000 --lr=0.01 --seed= \
 --patience=1000 --min_epoch=250  --act=tanh --time_dim=0 --space_dim=16  --graph_num_layers=1 --graph_weight=0.9 --weight_decay=1e-3  \
 --weight_decay_2=1e-3 --weight_decay_3=5e-3 --dropout_time=0.0 --dropout_space=0.0 --g_dropout_time=0.0 --g_dropout_space=0.0 --use_hyperdecoder --use_feats (replace linear to gcn encoder)

python run_lp.py --task=lp --dataset=tree2 --model=QGT --trans_num_layers=1 --trans_num_heads=1 --epochs=1000 --lr=0.01 --seed= \
 --patience=1000 --min_epoch=250  --act=tanh --time_dim=1 --space_dim=15  --graph_num_layers=1 --graph_weight=0.8 --weight_decay=1e-3 \
 --weight_decay_2=1e-3 --weight_decay_3=5e-3 --dropout_time=0.0 --dropout_space=0.0 --g_dropout_time=0.0 --g_dropout_space=0.0 --use_hyperdecoder --use_feats (replace linear to gcn encoder)

python run_lp.py --task=lp --dataset=tree3 --model=QGT --trans_num_layers=1 --trans_num_heads=1 --epochs=1000 --lr=0.01 --seed= \
 --patience=1000 --min_epoch=250  --act=tanh --time_dim=3 --space_dim=13  --graph_num_layers=1 --graph_weight=0.8 --weight_decay=1e-3 \
 --weight_decay_2=1e-3 --weight_decay_3=5e-3 --dropout_time=0.0 --dropout_space=0.0 --g_dropout_time=0.0 --g_dropout_space=0.0 --use_hyperdecoder --use_feats (replace linear to gcn encoder)


## Large graph

python run_lp.py --task=lp --dataset=ogbl-vessel --model=QGT --trans_num_layers=1 --trans_num_heads=1 --epochs=200 --lr=0.01 --seed=1234 \
 --patience=200 --min_epoch=250  --act=tanh --time_dim=7 --space_dim=9  --graph_num_layers=1  --graph_weight=0.8 --weight_decay=1e-4 \
 --weight_decay_2=1e-4 --weight_decay_3=5e-4 --normalize_feats --dropout_time=0.0 --dropout_space=0.0 --g_dropout_time=0.0 \
 --g_dropout_space=0.0 --use_hyperdecoder --use_feats


python run_lp.py --task=lp --dataset=ogbl-vessel --model=QGCN2 --time_dim=7 --space_dim=9 --epochs=200 --lr=0.005 --weight_decay=5e-4  \
 --seed=1234 --using_riemannianAdam=True --act=tanh --patience=200 --step_lr=1000 --dropout=0.0 --use_feats --use_hyperdecoder (2 layers, no res)
