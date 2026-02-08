import json
import os
import re
import dotenv
from tqdm import tqdm
from typing import Iterable

from utils.flickr30k_entities_utils import get_sentence_data as parse_flickr30k_sentence
from utils.flickr30k_entities_utils import get_annotations as parse_flickr30k_annotations
from classification.datasets.cifar100 import LABEL_NAMES


from utils.io import load_json, save_json, save_txt

dotenv.load_dotenv()

FLICKR30K_ENTITIES_PATH = os.getenv("FLICKR30K_ENTITIES_PATH")
OUTPUT_PATH = 'data'

# Classification

cifar_concepts = list(LABEL_NAMES.values())
for concept in cifar_concepts:
    concept = concept.replace('_', ' ')

customizd_concepts = ['dog', 'horse', 'apple']
all_concepts = set(cifar_concepts + customizd_concepts)


class DatasetProcessor:
    def __init__(
        self,
        raw_dataset_path: str,
        output_path: str,
        all_concepts: Iterable[str],
        target_concepts: Iterable[str] = None,
    ):
        self.raw_dataset_path = raw_dataset_path
        self.output_path = output_path
        self.all_concepts = all_concepts
        self.target_concepts = target_concepts

    def split_for_classification(self):
        raise NotImplementedError

    def split_for_retrieval(self):
        raise NotImplementedError


class Flikr30kEntitiesProcessor(DatasetProcessor):
    def _plural_pattern(self, word: str) -> str:
        # Basic English pluralization (simple heuristic for CIFAR labels).
        if word == "man":
            return "(?:man|men)"
        if word == "woman":
            return "(?:woman|women)"
        if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
            return f"(?:{re.escape(word)}|{re.escape(word[:-1])}ies)"
        if word.endswith(("s", "x", "z", "ch", "sh")):
            return f"(?:{re.escape(word)}|{re.escape(word)}es)"
        return f"(?:{re.escape(word)}|{re.escape(word)}s)"

    def _concept_in_phrase(self, concept: str, phrase_text: str) -> bool:
        # Match whole words; allow plural for the last word in a concept.
        _NON_WORD_RE = re.compile(r"[^\w]+", re.UNICODE)
        concept = _NON_WORD_RE.sub(" ", concept.lower()).strip()
        phrase_text = _NON_WORD_RE.sub(" ", phrase_text.lower()).strip()
        if not concept or not phrase_text:
            return False
        tokens = concept.split()
        if len(tokens) == 1:
            core = self._plural_pattern(tokens[0])
        else:
            # Only pluralize the last token to avoid over-matching.
            core = r"\s+".join(re.escape(t) for t in tokens[:-1])
            core = core + r"\s+" + self._plural_pattern(tokens[-1])
        pattern = r"\b" + core + r"\b"
        return re.search(pattern, phrase_text) is not None

    def split_for_classification(self):
        print("Splitting Flickr30K Entities dataset for classification...")
        # annotations_path = os.path.join(FLICKR30K_ENTITIES_PATH, 'annotations/Annotations')
        sentence_path = os.path.join(self.raw_dataset_path, 'annotations/Sentences')

        if not os.path.exists(os.path.join(self.output_path, 'classification', 'flickr30k_entities/meta/phrases.json')):
            phrases = {}  # Len: 243801
            for sent_file in tqdm(os.listdir(sentence_path), desc="Loading sentences"):
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
            save_json(phrases, os.path.join(self.output_path, 'classification', 'flickr30k_entities/meta/phrases.json'))

        else:
            phrases = load_json(os.path.join(self.output_path, 'classification',
                                'flickr30k_entities/meta/phrases.json'))

        meta_info = {}
        for concept in tqdm(all_concepts, desc="Processing concepts"):
            related_imgs = {}
            for img_id, img_phrases in phrases.items():
                for phrase_id in img_phrases.keys():
                    phrase_text = img_phrases[phrase_id]['phrase']
                    if self._concept_in_phrase(concept, phrase_text):
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
            if related_imgs:  # Save only if there are related images.
                related_imgs = dict(sorted(related_imgs.items(), key=lambda kv: kv[1]["item_num"], reverse=True))
                save_json(related_imgs, os.path.join(self.output_path, 'classification',
                        f'flickr30k_entities/meta/{concept.replace(" ", "_")}/related_imgs.json'))
                concept_meta_info = {
                    'all_concepts': {
                        'num': len(related_imgs),
                        'img_ids': list(related_imgs.keys()),
                    },
                    'ge_3': {
                        'num': sum(1 for info in related_imgs.values() if info['item_num'] >= 3),
                        'img_ids': [img_id for img_id, info in related_imgs.items() if info['item_num'] >= 3],
                    },
                    'ge_5': {
                        'num': sum(1 for info in related_imgs.values() if info['item_num'] >= 5),
                        'img_ids': [img_id for img_id, info in related_imgs.items() if info['item_num'] >= 5],
                    },
                    'ge_10': {
                        'num': sum(1 for info in related_imgs.values() if info['item_num'] >= 10),
                        'img_ids': [img_id for img_id, info in related_imgs.items() if info['item_num'] >= 10],
                    },
                    'ge_20': {
                        'num': sum(1 for info in related_imgs.values() if info['item_num'] >= 20),
                        'img_ids': [img_id for img_id, info in related_imgs.items() if info['item_num'] >= 20],
                    }
                }
                meta_info[concept.replace(" ", "_")] = concept_meta_info

                # TODO: Modify here to save different splits
                # Save ge_5 concepts for quick training
                ge_5_img_ids = concept_meta_info['ge_5']['img_ids']
                save_txt(ge_5_img_ids, os.path.join(self.output_path, 'classification',
                    f'flickr30k_entities/Df/item5+/{concept.replace(" ", "_")}.txt'))

        save_json(meta_info, os.path.join(self.output_path, 'classification',
                f'flickr30k_entities/meta/meta_info.json'))



if __name__ == "__main__":
    flickr30k_dataset=Flikr30kEntitiesProcessor(
        raw_dataset_path=FLICKR30K_ENTITIES_PATH,
        output_path=OUTPUT_PATH,
        all_concepts=all_concepts,
    )
    flickr30k_dataset.split_for_classification()
