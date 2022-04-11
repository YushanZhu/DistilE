# WN18RR ComplEx 
## Teacher_dim = 500 Student_dim = 64

## preatrain a teacher model 
sh scripts/WordNet/ComplEx_WN_pretrain.sh

## start the first stage of distilling 
sh scripts/WordNet/ComplEx_WN_distil.sh

## start the second stage of distilling
sh scripts/WordNet/ComplEx_WN_distil_stage2.sh


