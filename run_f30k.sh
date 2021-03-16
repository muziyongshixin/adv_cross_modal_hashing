#CUDA_VISIBLE_DEVICES=3 python test.py --adversary_type=noun --adversary_num=10  --final_dims=2048 --resume=/S4/MI/liyz/saem_retrieval/runs/f30k_precomp/2020_05_21,18_13_57_icst10_adv_mix30_st-1,mg0.5/checkpoints/model_best.pth.tar --remark=+mix30_2048model_adv_noun --batch_size=64
#CUDA_VISIBLE_DEVICES=3 python test.py --adversary_type=num --adversary_num=10  --final_dims=2048 --resume=/S4/MI/liyz/saem_retrieval/runs/f30k_precomp/2020_05_21,18_13_57_icst10_adv_mix30_st-1,mg0.5/checkpoints/model_best.pth.tar --remark=+mix30_2048model_adv_num --batch_size=64
#CUDA_VISIBLE_DEVICES=3 python test.py --adversary_type=rela --adversary_num=10  --final_dims=2048 --resume=/S4/MI/liyz/saem_retrieval/runs/f30k_precomp/2020_05_21,18_13_57_icst10_adv_mix30_st-1,mg0.5/checkpoints/model_best.pth.tar --remark=+mix30_2048model_adv_rela --batch_size=64
#CUDA_VISIBLE_DEVICES=3 python test.py --adversary_type=mixed --adversary_num=10  --final_dims=2048 --resume=/S4/MI/liyz/saem_retrieval/runs/f30k_precomp/2020_05_21,18_13_57_icst10_adv_mix30_st-1,mg0.5/checkpoints/model_best.pth.tar --remark=+mix30_2048model_adv_mix10 --batch_size=64
#CUDA_VISIBLE_DEVICES=3 python test.py --adversary_type=mixed --adversary_num=30  --final_dims=2048 --resume=/S4/MI/liyz/saem_retrieval/runs/f30k_precomp/2020_05_21,18_13_57_icst10_adv_mix30_st-1,mg0.5/checkpoints/model_best.pth.tar --remark=+mix30_2048model_adv_mix30 --batch_size=64


# cross domain 测试， flickr30上训练 coco上测试
CUDA_VISIBLE_DEVICES=3 python test.py --final_dims=2048 --resume=runs/f30k_precomp/2020_05_15,00_07_00_icst5_f30k_rerank__+rephr_dim2048/checkpoints/checkpoint_28.pth.tar --remark=train_on_f30k_test_on_coco,+rephrase+concept_dim2048 --batch_size=64
CUDA_VISIBLE_DEVICES=3 python test.py --final_dims=2048 --resume=runs/f30k_precomp/2020_05_10,01_37_28_icst4_f30k_2000concept_dim2048/checkpoints/checkpoint_33.pth.tar   --remark=train_on_f30k_test_on_coco,+concept_dim2048 --batch_size=64
CUDA_VISIBLE_DEVICES=3 python test.py --final_dims=2048 --resume=runs/f30k_precomp/2020_05_12,20_05_35_icst5_f30k_rerank_noconcept_dim2048/checkpoints/checkpoint_21.pth.tar  --remark=train_on_f30k_test_on_coco,nothing_dim2048 --batch_size=64


# cross domain 测试， coco上训练 flickr30k上测试
CUDA_VISIBLE_DEVICES=3 python test.py --final_dims=2048 --resume=runs/coco_precomp/2020_04_22,00_29_24_icst1_fullconcept_2048d_woadvrephr/checkpoints/checkpoint_29.pth.tar  --remark=train_on_coco_test_on_flickr,+concept_dim2048 --batch_size=64
CUDA_VISIBLE_DEVICES=3 python test.py --final_dims=2048 --resume=runs/coco_precomp/2020_04_22,00_36_28_icst5_fullconcept_2048d+advrephr/checkpoints/checkpoint_22.pth.tar --remark=train_on_coco_test_on_flickr,+concept+rephrase_dim2048 --batch_size=64


CUDA_VISIBLE_DEVICES=3 python train.py --final_dims=2048 --need_concept_label=0 --need_rephrase_data=0 --adversary_num=0 --remark=train_on_coco_test_on_flickr,noconcept,norephre,dim2048

CUDA_VISIBLE_DEVICES=2 python train.py --final_dims=2048 --need_concept_label=1 --need_rephrase_data=0 --adversary_num=0 --remark=train_on_coco_test_on_flickr,+concept,norephre,dim2048

CUDA_VISIBLE_DEVICES=1 python train.py --final_dims=2048 --need_concept_label=1 --need_rephrase_data=1 --adversary_num=0 --remark=train_on_coco_test_on_flickr,+concept,+rephre,dim2048


CUDA_VISIBLE_DEVICES=1 python test.py --final_dims=2048 --need_concept_label=0 --need_rephrase_data=0 --adversary_num=0 --resume=/S4/MI/liyz/saem_retrieval/runs/f30k_precomp/2020_05_12,20_05_35_icst5_f30k_rerank_noconcept_dim2048/checkpoints/checkpoint_21.pth.tar --remark=test_flickr30k_noconcept_norephre