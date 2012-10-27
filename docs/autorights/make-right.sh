#!/bin/bash

PERL=perl5.10
AUTORIGHTS=../autorights/autorights.pl

# 2011-2012 g:
${PERL} ${AUTORIGHTS}  \
  ./sample_voc_config_override.m  \
  ./startup.m \
  ./compile.m \
  ./voc_config.m \
  ./voc_config_inriaperson.m \
  ./car_grammar/car_grammar_init.m \
  ./car_grammar/pascal_car_grammar.m \
  ./car_grammar/pascal_train_car_grammar.m \
  ./car_grammar/voc_config_car_grammar.m \
  ./person_grammar/add_head_parts.m \
  ./person_grammar/add_slab_parts.m \
  ./person_grammar/pascal_person_grammar.m \
  ./person_grammar/pascal_train_person_grammar.m \
  ./person_grammar/person_grammar_init.m \
  ./person_grammar/visualize_person_grammar_model.m \
  ./person_grammar/voc_config_person_grammar.m \
  ./fv_cache/fv_cache.h \
  ./fv_cache/mempool.h \
  ./fv_cache/model.h \
  ./fv_cache/obj_func.h \
  ./fv_cache/fv_cache.cc \
  ./fv_cache/obj_func.cc \
  ./fv_cache/fv_compile.m \
  ./fv_cache/fv_model_args.m \
  ./fv_cache/fv_obj_func.m \
  ./fv_cache/max_fv_dim.m \
  ./gdetect/fconvsse.cc \
  ./gdetect/get_detection_trees.cc \
  ./gdetect/loss_func.m \
  ./gdetect/loss_pyramid.m \
  ./gdetect/tree_mat_to_struct.m \
  ./gdetect/validate_levels.m \
  ./gdetect/write_zero_fv.m \
  ./vis/vis_derived_filter.m \
  ./vis/vis_grammar.m \
  --template=docs/autorights/autorights-notice-one.txt \
  --authors "Ross Girshick" \
  --years "2011-2012" \
  --program "voc-releaseX"

#2009-2012 g:
${PERL} ${AUTORIGHTS}  \
  ./star-cascade/cascade.cc \
  ./star-cascade/model.cc \
  ./star-cascade/model.h \
  ./star-cascade/timer.h \
  ./star-cascade/cascade_compile.m \
  ./star-cascade/cascade_data.m \
  ./star-cascade/cascade_detect.m \
  ./star-cascade/cascade_model.m \
  ./star-cascade/cascade_test.m \
  ./star-cascade/gdetect_pos_c.m \
  ./star-cascade/gdetect_pos_prepare_c.m \
  ./star-cascade/get_block_scores.m \
  ./star-cascade/grammar2simple.m \
  ./star-cascade/pca_of_hog.m \
  ./star-cascade/project.m \
  ./star-cascade/project_model.m \
  ./star-cascade/project_pyramid.m \
  ./demo_cascade.m \
  ./model/block_types.m \
  ./model/getopts.m \
  ./model/lr_root_model.m \
  ./model/mkpartfilters.m \
  ./model/model_add_block.m \
  ./model/model_add_def_rule.m \
  ./model/model_add_nonterminal.m \
  ./model/model_add_parts.m \
  ./model/model_add_struct_rule.m \
  ./model/model_add_symbol.m \
  ./model/model_add_terminal.m \
  ./model/model_create.m \
  ./model/model_get_block.m \
  ./model/model_merge.m \
  ./model/model_sort.m \
  ./model/model_types.m \
  ./utils/auc_ap_2007.m \
  ./utils/bootstrap/test_stats.m \
  ./utils/five2four.m \
  ./utils/model_attach_weights.m \
  ./utils/model_cmp.m \
  ./utils/model_norms.m \
  ./utils/procid.m \
  ./utils/reduceboxes.m \
  ./utils/report.m \
  ./utils/report_cmp.m \
  ./utils/rndtest.m \
  ./utils/showboxesc.m \
  ./utils/showposlat.m \
  ./utils/tic_toc_print.m \
  ./utils/viewerrors.m \
  ./features/flipfeat.m \
  ./features/getpadding.m \
  ./features/loc_feat.m \
  ./data/imreadx.m \
  ./gdetect/compute_overlaps.m \
  ./gdetect/gdetect.m \
  ./gdetect/gdetect_dp.m \
  ./gdetect/gdetect_parse.m \
  ./gdetect/gdetect_pos.m \
  ./gdetect/gdetect_pos_prepare.m \
  ./gdetect/gdetect_write.m \
  ./gdetect/imgdetect.m \
  ./gdetect/bounded_dt.cc \
  ./gdetect/compute_overlap.cc \
  ./train/train.m \
  --template=docs/autorights/autorights-notice-one.txt \
  --authors "Ross Girshick" \
  --years "2009-2012" \
  --program "voc-releaseX"

#2011-2012 g:
#2008-2010 gf:
${PERL} ${AUTORIGHTS}  \
  ./bbox_pred/bboxpred_data.m \
  ./bbox_pred/bboxpred_get.m \
  ./bbox_pred/bboxpred_input.m \
  ./bbox_pred/bboxpred_rescore.m \
  ./bbox_pred/bboxpred_train.m \
  ./context/context_data.m \
  ./context/context_labels.m \
  ./context/context_rescore.m \
  ./context/context_test.m \
  ./context/context_train.m \
  ./gdetect/fconv_var_dim.cc \
  ./gdetect/fconv_var_dim_MT.cc \
  ./vis/visualizemodel.m \
  ./utils/boxoverlap.m \
  ./model/root_model.m \
  ./data/pascal_data.m \
  ./demo.m \
  ./train/pascal_train.m \
  ./train/seed_rand.m \
  ./train/split.m \
  ./train/trainval.m \
  ./test/pascal_eval.m \
  ./test/pascal_test.m \
  ./train/croppos.m \
  ./train/lrsplit.m \
  ./pascal.m \
  ./process.m \
  ./test/clipboxes.m \
  --template=docs/autorights/autorights-notice-f.txt \
  --authors "Ross Girshick" \
  --years "2011-2012" \
  --program "voc-releaseX"

#2007-2012 gfr:
${PERL} ${AUTORIGHTS}  \
  ./features/features.cc \
  ./vis/visualizeHOG.m \
  ./vis/foldHOG.m \
  ./vis/HOGpicture.m \
  ./vis/showboxes.m \
  ./train/warppos.m \
  ./test/nms.m \
  ./train/subarray.m \
  ./features/color.m \
  ./features/featpyramid.m \
  --template=docs/autorights/autorights-notice-fr.txt \
  --authors "Ross Girshick" \
  --years "2011-2012" \
  --program "voc-releaseX"

#2007 f:
${PERL} ${AUTORIGHTS}  \
  ./features/resize.cc \
  ./gdetect/dt.cc \
  --template=docs/autorights/autorights-notice-one.txt \
  --authors "Pedro Felzenszwalb" \
  --years "2007" \
  --program "voc-releaseX"

