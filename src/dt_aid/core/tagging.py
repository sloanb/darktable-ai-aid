from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum

from .config import ELEMENTS_MODEL_VERSION, FACES_MODEL_VERSION

SEP = "|"

PEOPLE_ROOT = "people"
UNKNOWN_ROOT = f"{PEOPLE_ROOT}{SEP}unknown"
AUTO_ROOT = "auto"
AUTO_OBJECT = f"{AUTO_ROOT}{SEP}object"
AUTO_SCENE = f"{AUTO_ROOT}{SEP}scene"
AUTO_ATTR = f"{AUTO_ROOT}{SEP}attr"
AUTO_META = f"{AUTO_ROOT}{SEP}_meta"

_slug_re = re.compile(r"[^a-z0-9\-]+")


class TagKind(str, Enum):
    PERSON = "person"
    PERSON_CLUSTER = "person_cluster"
    OBJECT = "object"
    SCENE = "scene"
    ATTR = "attr"
    META = "meta"


@dataclass(frozen=True)
class Tag:
    kind: TagKind
    value: str

    def __str__(self) -> str:
        return self.value


def slug(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_only.lower().replace(" ", "-").replace("_", "-")
    return _slug_re.sub("", lowered).strip("-")


def person_tag(name: str) -> Tag:
    return Tag(TagKind.PERSON, f"{PEOPLE_ROOT}{SEP}{slug(name)}")


def cluster_tag(cluster_id: int) -> Tag:
    return Tag(TagKind.PERSON_CLUSTER, f"{UNKNOWN_ROOT}{SEP}cluster-{cluster_id:03d}")


def object_tag(label: str) -> Tag:
    return Tag(TagKind.OBJECT, f"{AUTO_OBJECT}{SEP}{slug(label)}")


def scene_tag(label: str) -> Tag:
    return Tag(TagKind.SCENE, f"{AUTO_SCENE}{SEP}{slug(label)}")


def attr_tag(label: str) -> Tag:
    return Tag(TagKind.ATTR, f"{AUTO_ATTR}{SEP}{slug(label)}")


def faces_provenance_tag() -> Tag:
    return Tag(TagKind.META, f"{AUTO_META}{SEP}model-faces-{slug(FACES_MODEL_VERSION)}")


def elements_provenance_tag() -> Tag:
    return Tag(TagKind.META, f"{AUTO_META}{SEP}model-elements-{slug(ELEMENTS_MODEL_VERSION)}")


def is_managed_tag(name: str) -> bool:
    """True for tags this tool is allowed to add, remove, or overwrite."""
    return (
        name == PEOPLE_ROOT
        or name.startswith(PEOPLE_ROOT + SEP)
        or name == AUTO_ROOT
        or name.startswith(AUTO_ROOT + SEP)
    )


def dedup_tags(existing: list[str], new_tags: list[Tag]) -> list[Tag]:
    """Return new_tags minus anything already present in existing."""
    have = set(existing)
    out: list[Tag] = []
    seen: set[str] = set()
    for t in new_tags:
        if t.value in have or t.value in seen:
            continue
        seen.add(t.value)
        out.append(t)
    return out


def merge_managed(existing: list[str], managed_new: list[Tag]) -> list[str]:
    """
    Return the final tag list: keep all non-managed existing tags, replace
    managed tags with the new set. Used for XMP rewriting where we own the
    `people|*` and `auto|*` namespaces.
    """
    kept = [t for t in existing if not is_managed_tag(t)]
    new_values = [t.value for t in managed_new]
    # de-dup while preserving order
    seen: set[str] = set()
    merged: list[str] = []
    for v in (*kept, *new_values):
        if v not in seen:
            seen.add(v)
            merged.append(v)
    return merged


def hierarchical_ancestors(tag: str) -> list[str]:
    """'people|family|alice' -> ['people', 'people|family', 'people|family|alice']"""
    parts = tag.split(SEP)
    return [SEP.join(parts[: i + 1]) for i in range(len(parts))]
