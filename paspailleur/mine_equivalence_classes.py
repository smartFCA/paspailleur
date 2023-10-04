from .pattern_structures import AbstractPS
from bitarray import bitarray


def list_intents_via_Lindig_complex(data: list, pattern_structure: AbstractPS) -> list['PatternDescription']:
    """Get the list of intents of pattern concepts from `data` described by `pattern_structure` running Lindig algorithm
    from "Fast Concept Analysis" by Christian Lindig, Harvard University, Division of Engineering and Applied Sciences

    Parameters
    ----------
    data:
        list of objects described by a pattern structure
    pattern_structure:
        type of pattern structure related to data

    Returns
    -------
    Lattice_data_intents:
        list of intents of pattern concepts
    """

    class NotFound(Exception):
        pass

    def compute_bits_intersection(bits: list[bitarray], len_bitarray):
        if bits == []:
            return(bitarray([1 for _ in range(len_bitarray)]))
        bit = bits[0]
        for obj in bits[1:]:
            bit = bit & obj
        return(bit)

    def find_upper_neighbors(data: list, concept_extent: list, objects_indices: list):
        min_set = [obj for obj in objects_indices if obj not in concept_extent]
        concept_extent_values = [data[i] for i in concept_extent]
        neighbors = []

        for g in [obj for obj in objects_indices if obj not in concept_extent]:
            B1 = ps.intent(concept_extent_values + [data[g]])
            A1 = list(ps.extent(B1, data))
            if len(set(min_set) & set([obj for obj in A1 if obj not in concept_extent and obj not in [g]])) == 0:
                neighbors.append(A1)
            else:
                min_set.remove(g)
        return neighbors

    def find_next_concept_extent(data: list, concept_extent: list, List_extents: list):
        bin_col_names, rows = ps.binarize(data)
        next_concept_extent = None
        concept_extent_bit = compute_bits_intersection([rows[i] for i in concept_extent], len(rows[0]))
        for extent in List_extents:
            extent_bit = compute_bits_intersection([rows[i] for i in extent], len(rows[0]))
            if extent_bit < concept_extent_bit and (next_concept_extent is None or extent_bit > next_concept_extent_bit):
                next_concept_extent = extent
                next_concept_extent_bit = compute_bits_intersection([rows[i] for i in next_concept_extent], len(rows[0]))
        if next_concept_extent is not None:
            return next_concept_extent
        raise NotFound("Next concept not found in Lattice")

    ps = pattern_structure
    Lattice_data_extents = []  # extents set
    concept_extent = []  # Initial concept extent
    objects_indices = list(ps.extent([], data))
    Lattice_data_extents.append(concept_extent)  # Insert the initial concept extent into Lattice

    while True:
        for parent in find_upper_neighbors(data, concept_extent, objects_indices):
            if parent not in Lattice_data_extents:
              Lattice_data_extents.append(parent)

        try:
            concept_extent = find_next_concept_extent(data, concept_extent, Lattice_data_extents)
        except NotFound:
            break

    Lattice_data_intents = []
    for i in range(len(Lattice_data_extents)):
        Lattice_data_intents.append(ps.intent([data[j] for j in Lattice_data_extents[i]]))

    return Lattice_data_intents
