# Longdin Noise
---
## What is it?

Longdin Noise is a noise type that uses per-pixel vectors to propagate pixel values.

It is interesting in the fact that two different configurations can produce wildly different results. A high density with low iterations can produce a blocky, glitchy looking noise. A low density with high iterations can produce a sort of bubbly or brush-like noise.

## Why was it made?

For no particular reason really. It's strange pattern makes it impractical for common examples in games (like landscapes for video games with Perlin Noise). It's definitely an example of more artistic noise - although, if you can find a genuinely practical use for this thing, please do let me know via my email!

It started off as a desperate wrestle with the boney hands of boredom. It's strange to think the wonders one's mind can produce when one is desperately in need of something to do - or, indeed, type. If you ever find yourself desperately bored with nothing to do, create a new python file and type the following:

```python
class Universe:
    pass
```

There's something about the openness of an unimplemented class that allows the mind to wonder. It doesn't have to be `Universe`. It can really be anything - in my case it gradually morphed into `_Noise`. Although, the broad topic that is the entirety of existence is definitely effective.

## The Inner-Workings

All three of the generator functions first instantiates a `_Noise` object. `_Noise` is an internal class and is not intended for use outside of the API.

### Initialisation

When `_Noise` instantiates, it generates a 2D numpy array that takes on the shape `(height, width, channels)` where `channels` is either 1, 2, or 3. (more on that later!) These are the cell 'values'. It generates another array of the shape (height, width, 2) to store each cell's influence vector.

The density (a float bigger than 0 up to 1) is just the percentage of 'cells' that are active. The generator decides which cells should be inactive using this value and zeros out each of these inactive cells.

### The Algorithm

Each iteration runs the same algorithm on the data that either the initialisation or last iterative pass produced.

First things first, it gets all active cells and shuffles that data's order. A cell is active if it has an influence vector and a value associated. The shuffle helps to mitigate undesired and unpleasant row artifacts produced by the parallelism used to speed up the algorithm.

The algorithm then loops through all active cells. For each cell, it does the following:
  - Get all cells within the radius of the active cell (excludes cells if the active cell is not pointing towards them within 90°)
  - Computes the distance from the origin (active cell) to each cell within its range (radius)
  - Uses the distance to compute the influence the active cell has on each cell
  - Each cell is then lerped towards the active cell's data using the calculated influence as the factor.

To finish off the iteration, the list of active cells is updated ready for the next iteration.

## Installation

`pip install longdin-noise`

## How to use it?

Here's a quick example of how one may use this library using PIL:

```python
from longdin_noise import generate_mono
from PIL import Image
import numpy as np

result = generate_mono((512, 512), 42, 0.05, 10)

# result is a (512, 512) array of floats in the range 0..1
# multiply by 255 and convert to uint8 for PIL
image_data = (result * 255).astype(np.uint8)
img = Image.fromarray(image_data, mode="L")
img.save("mono_noise.png")
```

*This will create you a lovely greyscale image using the longdin_noise python library:*

![Mono noise example](https://raw.githubusercontent.com/drewlongdin-dev/longdin-noise/main/examples/mono_noise.png)

### Parameters

| Parameter  | Type          | Default | Valid Range      | Description                        |
|------------|---------------|---------|------------------|------------------------------------|
| size       | tuple[int, int] | —     | Any positive ints | Width and height of the output    |
| seed       | int           | —       | Any int          | Random seed                        |
| density    | float         | —       | 0 < x ≤ 1       | Proportion of initially active pixels |
| iterations | int           | —       | ≥ 1              | Number of diffusion passes         |
| radius     | int           | 8       | ≥ 1              | Influence reach in pixels          |

### Tweaking the Values

The effect the values have on the resulting noise can be obscure and difficult to figure out at first. Hopefully this short guide can help you to understand how to craft your desired effect.

The number of iterations — in short — just increases the amount of smearing in the noise. As the vectors' influences propagate outwards every iteration, the colours mix more and boundaries become less clear.

The density will basically dictate how noisy the image is. A really, really low density like 0.0005 - especially on a low number of iterations like 1 - 4 - scatters 'searchlights' across the image. I have yet to find a use for this but it's pretty cool. A high density creates a more tightly packed image where blending between colours occurs much sooner in the iterative process.

The radius is quite simple. It just affects how far each active cell can influence. I haven't had an opportunity to experiment with this much, but it should have an 'octaves' effect where higher values create simpler noise.

### Example Images

### **Low** density, **high** iterations *(0.005 Density, 20 Iterations)*
| Mono | Bi | Tri |
|------|----|-----|
| ![](https://raw.githubusercontent.com/drewlongdin-dev/longdin-noise/main/examples/mono_low_high.png) | ![](https://raw.githubusercontent.com/drewlongdin-dev/longdin-noise/main/examples/bi_low_high.png) | ![](https://raw.githubusercontent.com/drewlongdin-dev/longdin-noise/main/examples/tri_low_high.png) |

### **High** density, **low** iterations *(0.6 Density, 4 Iterations)*
| Mono | Bi | Tri |
|------|----|-----|
| ![](https://raw.githubusercontent.com/drewlongdin-dev/longdin-noise/main/examples/mono_high_low.png) | ![](https://raw.githubusercontent.com/drewlongdin-dev/longdin-noise/main/examples/bi_high_low.png) | ![](https://raw.githubusercontent.com/drewlongdin-dev/longdin-noise/main/examples/tri_high_low.png) |

## Dependencies

This module requires numpy and numba.

## License

This is under the MIT license. Truly do whatever you want with this code and what it produces.
