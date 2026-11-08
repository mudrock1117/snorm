"""Metadata-only fixed-sample lookup for repository datasets."""
import random

from torch.utils.data import DataLoader, Subset

from detection.data import collate_fn


_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


_INDEX_ATTRIBUTES = (
    "ids",
    "image_ids",
    "image_set_index",
    "file_names",
    "filenames",
    "samples",
    "data_infos",
    "images",
)


def strip_image_extension(value):
    """Remove only a recognized image suffix; preserve dots inside sample IDs."""
    text = str(value)
    lower = text.lower()
    for extension in _IMAGE_EXTENSIONS:
        if lower.endswith(extension):
            return text[:-len(extension)]
    return text


def canonicalize_sample_id(value):
    """Return a path-free ID while preserving Foggy Cityscapes beta decimals."""
    file_name = str(value).replace("\\", "/").rsplit("/", 1)[-1]
    return strip_image_extension(file_name)


def _add_value(candidates, value):
    if value is None:
        return
    if isinstance(value, dict):
        for key in ("id", "image_id", "file_name", "filename", "path", "image_path"):
            if key in value:
                _add_value(candidates, value[key])
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _add_value(candidates, item)
        return
    if isinstance(value, (str, int)):
        text = str(value)
        candidates.add(text)
        file_name = text.replace("\\", "/").rsplit("/", 1)[-1]
        candidates.add(file_name)
        candidates.add(canonicalize_sample_id(file_name))


def dataset_index_candidates(dataset, index):
    """Return IDs without invoking dataset[index] or image transforms."""
    candidates = set()
    for name in _INDEX_ATTRIBUTES:
        values = getattr(dataset, name, None)
        if values is None:
            continue
        try:
            if len(values) > index:
                _add_value(candidates, values[index])
        except (TypeError, KeyError, IndexError):
            continue

    # Use metadata helpers only as a fallback. This avoids XML/JSON metadata
    # parsing when ids/file_names already expose the sample identity.
    if not candidates:
        for name in ("get_img_info", "get_image_info"):
            method = getattr(dataset, name, None)
            if callable(method):
                try:
                    _add_value(candidates, method(index))
                except Exception:
                    pass
    return candidates


def _domain_aliases(dataset_name):
    name = dataset_name.lower()
    aliases = {name}
    if "fog" in name:
        aliases.add("fog")
    if "night" in name:
        aliases.add("night")
    return aliases


def requested_ids_for_dataset(dataset_name, requested):
    """Route explicit domain IDs and obvious Fog/Night names before scanning."""
    aliases = _domain_aliases(dataset_name)
    selected = set()
    for raw in requested:
        value = raw.strip()
        if not value:
            continue
        if ":" in value:
            prefix, identifier = value.split(":", 1)
            if prefix.lower() in aliases:
                selected.add(identifier)
            continue
        lower = value.lower()
        if "fog" in lower and "fog" not in aliases:
            continue
        if "night" in lower and "night" not in aliases:
            continue
        selected.add(value)
    return selected


def find_dataset_indices(dataset, dataset_name, requested, example_limit=8):
    """Scan metadata only; return matches, examples, and requested IDs for domain."""
    domain_requested = requested_ids_for_dataset(dataset_name, requested)
    matches = {}
    examples = []
    for index in range(len(dataset)):
        candidates = dataset_index_candidates(dataset, index)
        if len(examples) < example_limit:
            examples.extend(
                item for item in sorted(candidates) if item not in examples
            )
            examples = examples[:example_limit]
        for identifier in domain_requested:
            identifier_candidates = {
                identifier,
                str(identifier).replace("\\", "/").rsplit("/", 1)[-1],
                canonicalize_sample_id(identifier),
            }
            if candidates.intersection(identifier_candidates):
                matches[identifier] = index
        if len(matches) == len(domain_requested):
            break
    return matches, examples, domain_requested


def build_subset_loader(dataset, indices):
    """Preserve original dataset __getitem__, transforms, meta and 8-item collate."""
    subset = Subset(dataset, list(indices))
    return DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )


def canonical_sample_id(dataset, index, dataset_name):
    candidates = dataset_index_candidates(dataset, index)
    if not candidates:
        raise RuntimeError(
            "dataset index %d exposes no metadata identifier" % int(index)
        )
    domain_tokens = []
    lower_name = dataset_name.lower()
    if "fog" in lower_name:
        domain_tokens.append("fog")
    if "night" in lower_name:
        domain_tokens.append("night")

    normalized = set(canonicalize_sample_id(value) for value in candidates)

    def score(value):
        lower = value.lower()
        return (
            1 if any(token in lower for token in domain_tokens) else 0,
            value.count("_"),
            len(value),
        )

    return max(normalized, key=score)


def stratified_sample_indices(length, count, seed, dataset_name):
    """One deterministic random index per evenly spaced dataset stratum."""
    length = int(length)
    count = min(int(count), length)
    if count <= 0:
        return []
    stable_offset = sum(ord(char) for char in str(dataset_name))
    generator = random.Random(int(seed) + stable_offset)
    indices = []
    for stratum in range(count):
        start = (stratum * length) // count
        end = ((stratum + 1) * length) // count
        if end <= start:
            index = start
        else:
            index = generator.randrange(start, end)
        indices.append(index)
    return sorted(set(indices))
