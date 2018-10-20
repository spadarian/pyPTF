# -*- coding: utf-8 -*-
"""Soil water content dataset.

The dataset contains 934 soil samples with clay, silt, sand content, and water
holding capacity at -33 and -1500 kPa.

References
----------
CIREN., 1996a. Estudio Agrológico VI Región. Descripciones de suelos, materiales y sı́mbolos.
Publicación No 114. Centro de Información de Recursos Naturales (CIREN), Santiago, Chile.

CIREN., 1996b. Estudio Agrológico Región Metropolitana. Descripciones de suelos, materiales y sı́mbolos.
Publicación No 115. Centro de Información de Recursos Naturales (CIREN), Santiago, Chile.

CIREN., 1997a. Estudio Agrológico V Región. Descripciones de suelos, materiales y sı́mbolos.
Publicación No 116. Centro de Información de Recursos Naturales (CIREN), Santiago, Chile.

CIREN., 1997b. Estudio Agrológico VII Región. Descripciones de suelos, materiales y sı́mbolos.
Publicación No 117. Centro de Información de Recursos Naturales (CIREN), Santiago, Chile.

CIREN., 1999. Estudio Agrológico VIII Región. Descripciones de suelos, materiales y sı́mbolos.
Publicación No 121. Centro de Información de Recursos Naturales (CIREN), Santiago, Chile.

CIREN., 2002. Estudio Agrológico IX Región. Descripciones de suelos, materiales y sı́mbolos.
Publicación No 122. Centro de Información de Recursos Naturales (CIREN), Santiago, Chile.

"""

from .base import _load_dataset


def load_water():
    """Load and return the soil water dataset."""
    return _load_dataset('water')
