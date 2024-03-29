from typing import Dict, Final

# Please see https://github.com/crowdbotp/OpenTraj/tree/master/datasets/SDD for more information.
# These homographies (transformations from pixel values to world coordinates) were estimated,
# albeit most of them with high certainty. The certainty values indicate how reliable the
# estimate is (or is not). Some of these scales were estimated using google maps, others are a pure guess.
SDD_HOMOGRAPHY_SCALES: Final[Dict[str, Dict[str, float]]] = {
    "bookstore_0": {"certainty": 1.0, "scale": 0.038392063},
    "bookstore_1": {"certainty": 1.0, "scale": 0.039892913},
    "bookstore_2": {"certainty": 1.0, "scale": 0.04062433},
    "bookstore_3": {"certainty": 1.0, "scale": 0.039098596},
    "bookstore_4": {"certainty": 1.0, "scale": 0.0396},
    "bookstore_5": {"certainty": 0.9, "scale": 0.0396},
    "bookstore_6": {"certainty": 0.9, "scale": 0.0413},
    "coupa_0": {"certainty": 1.0, "scale": 0.027995674},
    "coupa_1": {"certainty": 1.0, "scale": 0.023224545},
    "coupa_2": {"certainty": 1.0, "scale": 0.024},
    "coupa_3": {"certainty": 1.0, "scale": 0.025524906},
    "deathCircle_0": {"certainty": 1.0, "scale": 0.04064},
    "deathCircle_1": {"certainty": 1.0, "scale": 0.039076923},
    "deathCircle_2": {"certainty": 1.0, "scale": 0.03948382},
    "deathCircle_3": {"certainty": 1.0, "scale": 0.028478209},
    "deathCircle_4": {"certainty": 1.0, "scale": 0.038980137},
    "gates_0": {"certainty": 1.0, "scale": 0.03976968},
    "gates_1": {"certainty": 1.0, "scale": 0.03770837},
    "gates_2": {"certainty": 1.0, "scale": 0.037272793},
    "gates_3": {"certainty": 1.0, "scale": 0.034515323},
    "gates_4": {"certainty": 1.0, "scale": 0.04412268},
    "gates_5": {"certainty": 1.0, "scale": 0.0342392},
    "gates_6": {"certainty": 1.0, "scale": 0.0342392},
    "gates_7": {"certainty": 1.0, "scale": 0.04540353},
    "gates_8": {"certainty": 1.0, "scale": 0.045191525},
    "hyang_0": {"certainty": 1.0, "scale": 0.034749693},
    "hyang_1": {"certainty": 1.0, "scale": 0.0453136},
    "hyang_10": {"certainty": 1.0, "scale": 0.054460944},
    "hyang_11": {"certainty": 1.0, "scale": 0.054992233},
    "hyang_12": {"certainty": 1.0, "scale": 0.054104065},
    "hyang_13": {"certainty": 0.0, "scale": 0.0541},
    "hyang_14": {"certainty": 0.0, "scale": 0.0541},
    "hyang_2": {"certainty": 1.0, "scale": 0.054992233},
    "hyang_3": {"certainty": 1.0, "scale": 0.056642},
    "hyang_4": {"certainty": 1.0, "scale": 0.034265612},
    "hyang_5": {"certainty": 1.0, "scale": 0.029655497},
    "hyang_6": {"certainty": 1.0, "scale": 0.052936449},
    "hyang_7": {"certainty": 1.0, "scale": 0.03540125},
    "hyang_8": {"certainty": 1.0, "scale": 0.034592381},
    "hyang_9": {"certainty": 1.0, "scale": 0.038031423},
    "little_0": {"certainty": 1.0, "scale": 0.028930169},
    "little_1": {"certainty": 1.0, "scale": 0.028543144},
    "little_2": {"certainty": 1.0, "scale": 0.028543144},
    "little_3": {"certainty": 1.0, "scale": 0.028638926},
    "nexus_0": {"certainty": 1.0, "scale": 0.043986494},
    "nexus_1": {"certainty": 1.0, "scale": 0.043316805},
    "nexus_10": {"certainty": 1.0, "scale": 0.043991753},
    "nexus_11": {"certainty": 1.0, "scale": 0.043766154},
    "nexus_2": {"certainty": 1.0, "scale": 0.042247434},
    "nexus_3": {"certainty": 1.0, "scale": 0.045883871},
    "nexus_4": {"certainty": 1.0, "scale": 0.045883871},
    "nexus_5": {"certainty": 1.0, "scale": 0.045395745},
    "nexus_6": {"certainty": 1.0, "scale": 0.037929168},
    "nexus_7": {"certainty": 1.0, "scale": 0.037106087},
    "nexus_8": {"certainty": 1.0, "scale": 0.037106087},
    "nexus_9": {"certainty": 1.0, "scale": 0.044917895},
    "quad_0": {"certainty": 1.0, "scale": 0.043606807},
    "quad_1": {"certainty": 1.0, "scale": 0.042530206},
    "quad_2": {"certainty": 1.0, "scale": 0.043338169},
    "quad_3": {"certainty": 1.0, "scale": 0.044396842},
}
