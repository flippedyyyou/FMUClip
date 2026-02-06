from classification.datasets.cifar100 import LABEL_NAMES
import json
import os
import dotenv
from tqdm import tqdm

from utils.flickr30k_entities_utils import get_sentence_data as parse_flickr30k_sentence
from utils.flickr30k_entities_utils import get_annotations as parse_flickr30k_annotations

from utils.utils import load_json, save_json

dotenv.load_dotenv()

FLICKR30K_ENTITIES_PATH = os.getenv("FLICKR30K_ENTITIES_PATH")
OUTPUT_PATH = 'data_split'

# Classification

cifar_concepts = list(LABEL_NAMES.values())
for concept in cifar_concepts:
    concept = concept.replace('_', ' ')

customizd_concepts = ['dog', 'horse', 'apple']

all_concepts = set(cifar_concepts + customizd_concepts)

def split_flickr30k_entities_for_classification():
    print("Splitting Flickr30K Entities dataset for classification...")
    # annotations_path = os.path.join(FLICKR30K_ENTITIES_PATH, 'annotations/Annotations')
    sentence_path = os.path.join(
        FLICKR30K_ENTITIES_PATH, 'annotations/Sentences')

    if not os.path.exists(os.path.join(OUTPUT_PATH, 'flickr30k_entities/phrases.json')):
        os.makedirs(os.path.join(
            OUTPUT_PATH, 'flickr30k_entities'), exist_ok=True)
        phrases = {}  # Len: 243801
        for sent_file in tqdm(os.listdir(sentence_path), desc="Loading sentences:"):
            img_id = sent_file.split('.')[0]
            sentences = parse_flickr30k_sentence(
                os.path.join(sentence_path, sent_file))
            phrases[img_id] = {}
            for sent in sentences:
                for phrase in sent['phrases']:
                    phrases[img_id][phrase['phrase_id']] = {
                        'phrase': phrase['phrase'].lower(),
                        'phrase_type': phrase['phrase_type'],
                    }

        save_json(phrases, os.path.join(
            OUTPUT_PATH, 'flickr30k_entities/phrases.json'))

    else:
        phrases = load_json(os.path.join(
            OUTPUT_PATH, 'flickr30k_entities/phrases.json'))

    for concept in tqdm(all_concepts, desc="Processing concepts:"):
        related_imgs = {}
        for img_id, img_phrases in phrases.items():
            for phrase_id in img_phrases.keys():
                phrase_text = img_phrases[phrase_id]['phrase']
                if concept in phrase_text:
                    if img_id not in related_imgs:
                        related_imgs[img_id] = {
                            'item_num': len(img_phrases),
                            'phrases': []
                        }
                    related_imgs[img_id]['phrases'].append({
                        'phrase_id': phrase_id,
                        'phrase': phrase_text,
                        'phrase_type': img_phrases[phrase_id]['phrase_type'],
                    })
        related_imgs = dict(
            sorted(related_imgs.items(), key=lambda kv: kv[1]["item_num"], reverse=True)
        )
        save_json(related_imgs, os.path.join(
            OUTPUT_PATH, f'flickr30k_entities/{concept.replace(" ", "_")}/related_imgs.json'))


if __name__ == "__main__":
    split_flickr30k_entities_for_classification()
