import xml.etree.ElementTree as ET
import json
import os

def _xml_to_dict(elem):
    """递归把 XML 转成字典"""
    d = {}
    for child in elem:
        if len(child):
            value = _xml_to_dict(child)
        else:
            value = child.text

        if child.tag in d:
            if not isinstance(d[child.tag], list):
                d[child.tag] = [d[child.tag]]
            d[child.tag].append(value)
        else:
            d[child.tag] = value
    return d


def parse_xml_file(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    return {root.tag: _xml_to_dict(root)}

def load_json(file_path):
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif file_path.endswith('jsonl'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


def save_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

