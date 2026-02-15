# markov/tensor_library.py
# The Markov Tensor Library - A Cathedral of the Divine Emanations
# Version 1.0 - The Foundation

"""
In the beginning, there was the Word, and the Word was a Tensor.
This library is a testament to the sacred geometry of the cosmos, an attempt
to model the divine emanations that form the fabric of reality. Each structure
here is a step closer to understanding the mind of God.

- MarkovTensor: A single vector of divine potential. The fundamental utterance.
- MarkovCube: A plane of possibilities, woven from the threads of Tensors.
- MarkovHypercube: A space of realities, a multitude of Cubes in chorus.
- TheTensorium: The grand cathedral itself, housing these sacred forms and
               providing the liturgical functions to interact with them.
"""

import math
import random
from typing import List, Tuple
from pydantic import BaseModel, Field
import time

# --- Sacred Geometries (Pydantic Models) ---

class MarkovTensor(BaseModel):
    """A single vector of divine potential. The fundamental utterance."""
    values: List[float] = Field(..., description="A sequence of floating-point values representing a state vector.")
    metadata: dict = Field(default_factory=dict, description="Metadata, such as source, timestamp, or lucidity.")

    def __len__(self):
        return len(self.values)

    def magnitude(self) -> float:
        """Calculates the Euclidean norm (L2 magnitude) of the tensor."""
        return math.sqrt(sum(x * x for x in self.values))

class MarkovCube(BaseModel):
    """A plane of possibilities, woven from the threads of Tensors."""
    tensors: List[MarkovTensor] = Field(..., description="A 2D matrix composed of MarkovTensors.")
    metadata: dict = Field(default_factory=dict, description="Metadata for the entire cube.")

    def dimensions(self) -> Tuple[int, int]:
        """Returns the (rows, cols) of the cube, assuming uniform tensor length."""
        if not self.tensors:
            return (0, 0)
        return (len(self.tensors), len(self.tensors[0]))

class MarkovHypercube(BaseModel):
    """A space of realities, a multitude of Cubes in chorus."""
    cubes: List[MarkovCube] = Field(..., description="A 3D matrix composed of MarkovCubes.")
    metadata: dict = Field(default_factory=dict, description="Metadata for the entire hypercube.")

    def dimensions(self) -> Tuple[int, int, int]:
        """Returns the (depth, rows, cols) of the hypercube."""
        if not self.cubes or not self.cubes[0].tensors:
            return (0, 0, 0)
        cube_dims = self.cubes[0].dimensions()
        return (len(self.cubes), cube_dims[0], cube_dims[1])

# --- Liturgical Functions (Operations) ---

class TheTensorium:
    """
    The Tensorium is the sacred space where the divine geometries are created,
    contemplated, and transformed. It is the workshop of the demiurge.
    """

    @staticmethod
    def create_tensor(values: List[float], source: str = "genesis") -> MarkovTensor:
        """Creates a new MarkovTensor with standard metadata."""
        return MarkovTensor(
            values=values,
            metadata={"timestamp": time.time(), "source": source}
        )

    @staticmethod
    def create_random_tensor(dimensions: int, value_range: Tuple[float, float] = (0.0, 1.0), source: str = "chaos") -> MarkovTensor:
        """Generates a tensor with random values from the void."""
        values = [random.uniform(value_range[0], value_range[1]) for _ in range(dimensions)]
        return TheTensorium.create_tensor(values, source=source)

    @staticmethod
    def normalize_tensor(tensor: MarkovTensor) -> MarkovTensor:
        """Scales a tensor's values to the [0, 1] range."""
        min_val = min(tensor.values)
        max_val = max(tensor.values)
        range_val = max_val - min_val
        if range_val == 0:
            return TheTensorium.create_tensor([0.0] * len(tensor.values), source="normalized")

        normalized_values = [(v - min_val) / range_val for v in tensor.values]
        new_tensor = TheTensorium.create_tensor(normalized_values, source="normalized")
        new_tensor.metadata.update(tensor.metadata)
        return new_tensor

    @staticmethod
    def tensor_distance(t1: MarkovTensor, t2: MarkovTensor) -> float:
        """Calculates the Euclidean distance between two tensors of the same dimension."""
        if len(t1) != len(t2):
            raise ValueError("Tensors must have the same dimensions to calculate distance.")
        
        return math.sqrt(sum((v1 - v2) ** 2 for v1, v2 in zip(t1.values, t2.values)))

    @staticmethod
    def create_cube_from_tensors(tensors: List[MarkovTensor], source: str = "weaving") -> MarkovCube:
        """Weaves a list of tensors into a single MarkovCube."""
        return MarkovCube(
            tensors=tensors,
            metadata={"timestamp": time.time(), "source": source}
        )

    @staticmethod
    def project_hypercube_to_cube(hypercube: MarkovHypercube, method: str = "average") -> MarkovCube:
        """
        Projects a 4D Hypercube down to a 3D Cube, reducing its complexity.
        This is a form of descending into a lower 'Level of Lucidity'.

        Methods:
        - 'average': Averages the values of tensors across all cubes.
        - 'first': Takes the first cube in the hypercube.
        """
        if not hypercube.cubes:
            raise ValueError("Cannot project an empty hypercube.")

        if method == "first":
            return hypercube.cubes[0]
        
        if method == "average":
            dims = hypercube.dimensions()
            if dims[0] == 0:
                raise ValueError("Cannot project a hypercube with no cubes.")

            num_cubes = dims[0]
            num_rows = dims[1]
            num_cols = dims[2]
            
            avg_tensors: List[MarkovTensor] = []
            for r in range(num_rows):
                new_values = [0.0] * num_cols
                for c in range(num_cols):
                    sum_val = sum(hypercube.cubes[i].tensors[r].values[c] for i in range(num_cubes))
                    new_values[c] = sum_val / num_cubes
                avg_tensors.append(TheTensorium.create_tensor(new_values, source="projection_avg"))

            return TheTensorium.create_cube_from_tensors(avg_tensors, source="projection")
        
        raise ValueError(f"Unknown projection method: {method}")

# --- Example Canticle (Usage) ---
if __name__ == '__main__':
    print("=== A Canticle of Creation ===\n")

    # 1. From the void, a random thought is born.
    tensor_a = TheTensorium.create_random_tensor(10, source="first_spark")
    print(f"A random tensor is born (Magnitude: {tensor_a.magnitude():.2f}):\n{tensor_a.values}\n")

    # 2. Another thought emerges, close to the first.
    tensor_b_values = [v + random.uniform(-0.1, 0.1) for v in tensor_a.values]
    tensor_b = TheTensorium.create_tensor(tensor_b_values, source="echo")
    print(f"A second tensor echoes the first:\n{tensor_b.values}\n")

    # 3. The distance between them is measured, a measure of difference.
    distance = TheTensorium.tensor_distance(tensor_a, tensor_b)
    print(f"The distance between the two thoughts: {distance:.4f}\n")

    # 4. Weaving tensors into a plane of possibility.
    cube_tensors = [TheTensorium.create_random_tensor(4) for _ in range(4)]
    cube = TheTensorium.create_cube_from_tensors(cube_tensors)
    print(f"A 4x4 Cube is woven. Dimensions: {cube.dimensions()}\n")

    # 5. Stacking planes into a space of realities.
    hypercube_cubes = [
        TheTensorium.create_cube_from_tensors([TheTensorium.create_random_tensor(3) for _ in range(3)])
        for _ in range(3)
    ]
    hypercube = MarkovHypercube(cubes=hypercube_cubes, metadata={"name": "Genesis Block"})
    print(f"A 3x3x3 Hypercube is constructed. Dimensions: {hypercube.dimensions()}\n")

    # 6. Projecting the higher reality to a lower, more comprehensible form.
    projected_cube = TheTensorium.project_hypercube_to_cube(hypercube, method="average")
    print(f"The Hypercube is projected into a single Cube. Dimensions: {projected_cube.dimensions()}")
    print("First tensor of projected cube:", projected_cube.tensors[0].values)

    print("\n=== The Canticle is Complete ===")
