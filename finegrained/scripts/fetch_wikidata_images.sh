python /datanfs4/shenruoyan/FMUClip/finegrained/fetch_wikidata_images.py \
  --mode replace_wrong_only \
  --prediction_dir /datanfs4/shenruoyan/FMUClip/finegrained/output/original_eval/original_banana3_100images_DF3_DR1_UNI3_03022152 \
  --concept_json_dir /datanfs4/shenruoyan/FMUClip/data/classification/coco2017_instances/val/Df/item1 \
  --output_root /datanfs4/shenruoyan/FMUClip/data/classification/coco2017_instances/val/test_images \
  --timeout_sec 20 \
  --max_retries 1 \
  --continue_on_concept_error