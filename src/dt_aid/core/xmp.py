from __future__ import annotations

from pathlib import Path

from lxml import etree

from .tagging import hierarchical_ancestors

NS = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "lr": "http://ns.adobe.com/lightroom/1.0/",
    "xmp": "http://ns.adobe.com/xap/1.0/",
    "darktable": "http://darktable.sf.net/",
}

RDF = f"{{{NS['rdf']}}}"
DC = f"{{{NS['dc']}}}"
LR = f"{{{NS['lr']}}}"
X = f"{{{NS['x']}}}"


EMPTY_XMP = f"""<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="{NS['x']}">
  <rdf:RDF xmlns:rdf="{NS['rdf']}">
    <rdf:Description rdf:about=""
                     xmlns:dc="{NS['dc']}"
                     xmlns:lr="{NS['lr']}"
                     xmlns:xmp="{NS['xmp']}"
                     xmlns:darktable="{NS['darktable']}"/>
  </rdf:RDF>
</x:xmpmeta>
"""


def sidecar_path(image_path: Path) -> Path:
    return image_path.with_suffix(image_path.suffix + ".xmp")


def _load_or_new(path: Path) -> etree._ElementTree:
    if path.exists():
        parser = etree.XMLParser(remove_blank_text=False)
        return etree.parse(str(path), parser)
    root = etree.fromstring(EMPTY_XMP.encode("utf-8"))
    return etree.ElementTree(root)


def _description(tree: etree._ElementTree) -> etree._Element:
    root = tree.getroot()
    rdf = root.find(f"{RDF}RDF")
    if rdf is None:
        rdf = etree.SubElement(root, f"{RDF}RDF")
    desc = rdf.find(f"{RDF}Description")
    if desc is None:
        desc = etree.SubElement(rdf, f"{RDF}Description")
        desc.set(f"{RDF}about", "")
    return desc


def _set_bag(desc: etree._Element, qname: str, values: list[str]) -> None:
    existing = desc.find(qname)
    if existing is not None:
        desc.remove(existing)
    if not values:
        return
    container = etree.SubElement(desc, qname)
    bag = etree.SubElement(container, f"{RDF}Bag")
    for v in values:
        li = etree.SubElement(bag, f"{RDF}li")
        li.text = v


def _get_bag(desc: etree._Element, qname: str) -> list[str]:
    node = desc.find(qname)
    if node is None:
        return []
    bag = node.find(f"{RDF}Bag")
    if bag is None:
        return []
    return [li.text for li in bag.findall(f"{RDF}li") if li.text]


def read_subjects(image_path: Path) -> tuple[list[str], list[str]]:
    """Return (dc:subject, lr:hierarchicalSubject) from the sidecar, or ([],[])."""
    xmp = sidecar_path(image_path)
    if not xmp.exists():
        return ([], [])
    tree = _load_or_new(xmp)
    desc = _description(tree)
    return (_get_bag(desc, f"{DC}subject"), _get_bag(desc, f"{LR}hierarchicalSubject"))


def write_subjects(
    image_path: Path,
    *,
    flat_tags: list[str],
    hierarchical_tags: list[str],
) -> Path:
    """
    Write dc:subject (leaf labels) and lr:hierarchicalSubject (pipe-joined)
    to the image's sidecar, creating it if missing. Returns the sidecar path.
    """
    xmp = sidecar_path(image_path)
    tree = _load_or_new(xmp)
    desc = _description(tree)
    _set_bag(desc, f"{DC}subject", flat_tags)
    # expand ancestors so darktable shows the full hierarchy in its tag panel
    expanded: list[str] = []
    seen: set[str] = set()
    for t in hierarchical_tags:
        for anc in hierarchical_ancestors(t):
            if anc not in seen:
                seen.add(anc)
                expanded.append(anc)
    _set_bag(desc, f"{LR}hierarchicalSubject", expanded)
    xmp.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(xmp), xml_declaration=True, encoding="UTF-8", pretty_print=True)
    return xmp


def leaf_label(hierarchical_tag: str) -> str:
    return hierarchical_tag.rsplit("|", 1)[-1]
