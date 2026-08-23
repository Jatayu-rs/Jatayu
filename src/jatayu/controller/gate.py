"""Stage 1: decide the task family from the inputs alone. Cannot hallucinate."""

from jatayu.schemas import ImageRef, Modality, TaskFamily


class AmbiguousInputError(ValueError):
    pass


def classify_family(images: list[ImageRef]) -> TaskFamily:
    if len(images) == 1:
        return TaskFamily.SINGLE_IMAGE
    if len(images) != 2:
        raise AmbiguousInputError(
            f"{len(images)} images given. Jatayu takes one image or a pair."
        )

    a, b = images
    mods = {a.modality, b.modality}
    if Modality.SAR in mods and any(m.is_optical_family for m in mods):
        return TaskFamily.CROSS_MODAL
    if a.acquired and b.acquired and a.acquired.date() == b.acquired.date():
        raise AmbiguousInputError(
            "Both images share a modality and a date — neither a cross-modal "
            "nor a temporal pair. Check you uploaded the right files."
        )
    return TaskFamily.BI_TEMPORAL
